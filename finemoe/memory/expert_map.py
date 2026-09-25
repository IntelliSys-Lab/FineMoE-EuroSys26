"""Expert-map storage and retrieval from FineMoE sections 4.1–4.4."""

from pathlib import Path
from dataclasses import dataclass
from threading import RLock

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrajectoryState:
    """Incremental trajectory cosine products."""
    version: int = -1
    observed: int = 0
    dots: np.ndarray | None = None
    squared_norm: np.ndarray | None = None


@dataclass(frozen=True)
class MapSnapshot:
    """Immutable map snapshot for GPU search."""
    embedding: torch.Tensor
    probabilities: torch.Tensor
    array: np.ndarray
    prefix_norms: torch.Tensor
    order: np.ndarray
    mass: np.ndarray


def prefetch_rows(maps, scores, layers, top_k):
    """Combine batch predictions for cache updates and prefetching."""
    if not 1 <= top_k <= maps.shape[-1]:
        raise ValueError("top_k must be between 1 and the number of experts")
    probabilities = maps.detach().cpu().numpy()[:, layers]
    threshold = np.clip(1 - scores.detach().cpu().numpy(), 0, 1)[:, None, None]
    indices = np.argsort(-probabilities, axis=-1, kind="stable")
    values = np.take_along_axis(probabilities, indices, axis=-1)
    # Match ATen's cumulative-sum rounding.
    mass = values.cumsum(-1, dtype=np.float64).astype(values.dtype)
    count = np.clip((mass < threshold).sum(-1) + 1, top_k, maps.shape[-1])
    selected = np.zeros_like(probabilities, dtype=bool)
    np.put_along_axis(selected, indices, np.arange(maps.shape[-1]) < count[..., None], axis=-1)
    selected, probabilities = selected.any(0), probabilities.max(0)
    return probabilities, selected


class ExpertMapStore:
    """Bounded store of embeddings and full router probabilities."""

    def __init__(self, capacity, num_layers, num_experts, embed_dim,
                 prefetch_distance):
        if any(not isinstance(x, int) or x < 1 for x in
               (capacity, num_layers, num_experts, embed_dim)):
            raise ValueError("store dimensions and capacity must be positive integers")
        if not 1 <= prefetch_distance <= num_layers:
            raise ValueError("prefetch_distance must be in [1, num_layers]")
        self.capacity, self.num_layers = capacity, num_layers
        self.num_experts, self.embed_dim = num_experts, embed_dim
        self.prefetch_distance = prefetch_distance
        self.store_embed = torch.empty(capacity, embed_dim)
        self.store_traj = torch.empty(capacity, num_layers, num_experts)
        self._normalized_embed = torch.empty_like(self.store_embed)
        self._normalized_maps = torch.empty(capacity, num_layers * num_experts)
        self._prefix_norms = torch.empty(capacity, num_layers)
        self._layer_maps = torch.empty(num_layers, capacity, num_experts)
        self._version = 0
        self._row_versions = np.zeros(capacity, dtype=np.int64)
        self._snapshots = None
        self.data_size = 0
        self._lock = RLock()

    def _validate(self, tensor, shape, probability=False):
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device="cpu").detach()
        if tensor.ndim != len(shape) + 1 or tuple(tensor.shape[1:]) != tuple(shape):
            raise ValueError(f"expected [batch, {shape}], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("map data must be finite")
        if probability and (torch.any(tensor < 0) or not torch.allclose(
                tensor.sum(-1), torch.ones(tensor.shape[:-1]), atol=1e-5)):
            raise ValueError("expert maps must contain normalized nonnegative probabilities")
        return tensor

    @torch.inference_mode()
    def add(self, embeds, expert_maps):
        embeds = self._validate(embeds, (self.embed_dim,))
        expert_maps = self._validate(expert_maps, (self.num_layers, self.num_experts), True)
        if len(embeds) != len(expert_maps):
            raise ValueError("embeddings and maps must have the same batch size")
        with self._lock:
            # Consecutive inserts may replace the same row.
            for embed, expert_map in zip(embeds, expert_maps):
                if self.data_size < self.capacity:
                    index = self.data_size
                    self.data_size += 1
                else:
                    sem = self._normalized_embed @ F.normalize(embed, dim=0)
                    traj = self._normalized_maps @ F.normalize(
                        expert_map.flatten(), dim=0)
                    weight = self.prefetch_distance / self.num_layers
                    index = int((weight * sem + (1 - weight) * traj).argmax())
                self.store_embed[index].copy_(embed)
                self._normalized_embed[index].copy_(F.normalize(embed, dim=0))
                self.store_traj[index].copy_(expert_map)
                self._layer_maps[:, index].copy_(expert_map)
                self._normalized_maps[index].copy_(F.normalize(expert_map.flatten(), dim=0))
                self._prefix_norms[index].copy_(expert_map.square().sum(-1).cumsum(0).sqrt())
                self._row_versions[index] = self._version + 1
                if self._snapshots is not None:
                    self._snapshots[index] = self._freeze(index)
            self._version += 1

    def _freeze(self, index):
        probabilities = self.store_traj[index].clone()
        values = probabilities.numpy()
        order = np.argsort(-values, axis=-1, kind="stable")
        mass = np.take_along_axis(values, order, axis=-1).cumsum(-1, dtype=np.float64).astype(np.float32)
        return MapSnapshot(self._normalized_embed[index].clone(), probabilities, values,
                           self._prefix_norms[index].clone(), order, mass)

    def enable_snapshots(self):
        with self._lock:
            if self._snapshots is None:
                self._snapshots = [None] * self.capacity
                for index in range(self.data_size):
                    self._snapshots[index] = self._freeze(index)

    def snapshot_updates(self, version):
        """Return changed rows without waiting for a store writer."""
        if self._version == version or not self._lock.acquire(blocking=False):
            return None
        try:
            if self._snapshots is None:
                raise RuntimeError("enable_snapshots must precede snapshot_updates")
            indices = np.flatnonzero(self._row_versions[:self.data_size] > version)
            return self._version, self.data_size, [(int(i), self._snapshots[i]) for i in indices]
        finally:
            self._lock.release()

    @torch.inference_mode()
    def match_embed(self, embeds):
        embeds = self._validate(embeds, (self.embed_dim,))
        with self._lock:
            if not self.data_size:
                return None, None
            scores = F.normalize(embeds, dim=-1) @ self._normalized_embed[:self.data_size].T
            values, indices = scores.max(-1)
            return values.clamp(-1, 1), self.store_traj[indices].clone()

    @torch.inference_mode()
    def match_traj(self, trajs, state):
        """Match the observed trajectory prefix."""
        if not 1 <= trajs.shape[1] <= self.num_layers:
            raise ValueError("trajectory must contain 1..num_layers observed layers")
        data = torch.as_tensor(trajs, dtype=torch.float32, device="cpu").detach().numpy()
        if data.ndim != 3 or data.shape[2] != self.num_experts:
            raise ValueError("invalid trajectory shape")
        if not np.isfinite(data).all() or (data < 0).any() or not np.allclose(data.sum(-1), 1, atol=1e-5):
            raise ValueError("expert maps must contain finite normalized nonnegative probabilities")
        with self._lock:
            if not self.data_size:
                return None, None
            observed = data.shape[1]
            if (state.version != self._version or state.observed >= observed or
                    state.dots is None or state.dots.shape[0] != len(data)):
                state.version, state.observed = self._version, 0
                state.dots = np.zeros((len(data), self.data_size), dtype=np.float32)
                state.squared_norm = np.zeros(len(data), dtype=np.float32)
            maps = self._layer_maps.numpy()
            for layer in range(state.observed, observed):
                row = data[:, layer]
                state.dots += np.einsum("be,ce->bc", row, maps[layer, :self.data_size], optimize=False)
                state.squared_norm += (row * row).sum(-1)
            state.observed = observed
            scores = state.dots / np.maximum(np.sqrt(state.squared_norm)[:, None], 1e-12)
            scores /= np.maximum(self._prefix_norms.numpy()[:self.data_size, observed - 1], 1e-12)
            indices = scores.argmax(-1)
            values = np.clip(scores[np.arange(len(data)), indices], -1, 1)
            return torch.from_numpy(values), torch.from_numpy(self.store_traj.numpy()[indices])

    def export_store_data(self, state_path):
        path = Path(str(state_path) + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.savez(path, version=np.array(1),
                     embeds=self.store_embed[:self.data_size].numpy(),
                     maps=self.store_traj[:self.data_size].numpy())
        return path

    def import_store_data(self, state_path):
        with np.load(str(state_path) + ".npz", allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("unsupported expert-map store format")
            embeds = self._validate(data["embeds"], (self.embed_dim,))
            maps = self._validate(data["maps"], (self.num_layers, self.num_experts), True)
            if len(embeds) != len(maps):
                raise ValueError("store embeddings/maps have different lengths")
            if len(embeds) > self.capacity:
                raise ValueError("saved store exceeds configured capacity")
            with self._lock:
                self.data_size = 0
                self.add(embeds, maps)

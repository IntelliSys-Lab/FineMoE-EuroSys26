"""Graph-captured expert-map search."""

from dataclasses import dataclass
import struct

import numpy as np
import torch
import triton

from finemoe.kernels.predictor import _products, _metadata


@dataclass
class _DeviceMaps:
    first: int
    last: int
    bank: torch.Tensor
    norms: torch.Tensor
    dots: torch.Tensor
    query_norm: torch.Tensor
    scores: torch.Tensor
    active: torch.Tensor
    count: torch.Tensor


class CudaMapPredictor:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.expert_map_store
        self.capacity, self.experts = self.store.capacity, self.store.num_experts
        self.layers, self.width = self.store.num_layers, self.store.embed_dim
        self.distance, self.top_k = self.store.prefetch_distance, engine.top_k
        self.version, self.size = -1, 0
        self.active = False
        self.rows = [None] * self.capacity
        self.devices, self.metadata = {}, {}
        self.store.enable_snapshots()
        for device in engine.devices:
            layers = [i for i, target in enumerate(engine.layer_devices) if target == device]
            with torch.cuda.device(device):
                self.devices[device] = _DeviceMaps(
                    layers[0], layers[-1] + 1,
                    torch.empty(len(layers), self.capacity, self.experts, device=device, dtype=torch.float32),
                    torch.empty(len(layers), self.capacity, device=device, dtype=torch.float32),
                    torch.zeros(self.capacity, device=device, dtype=torch.float32),
                    torch.zeros(1, device=device, dtype=torch.float32),
                    torch.empty(self.capacity, device=device, dtype=torch.float32),
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.zeros(1, dtype=torch.int32, device=device))
                for layer in layers:
                    if layer == 0 or layer + self.distance < self.layers:
                        self.metadata[layer] = torch.empty(self.top_k + (4 if layer == 0 else 2),
                                                          dtype=torch.int64, device=device)
        with torch.cuda.device(engine.device):
            self.embeddings = torch.empty(self.capacity, self.width, device=engine.device, dtype=torch.float32)
            weight = engine.model.get_input_embeddings().weight
            self.input = weight.new_empty(1, self.width)
            self.semantic_norm = torch.zeros(1, device=engine.device, dtype=torch.float32)
            self.semantic_metadata = torch.empty(2, dtype=torch.int64, device=engine.device)
        self.refresh()
        from finemoe.models.attention_qwen import _capture
        with torch.cuda.device(engine.device), torch.inference_mode():
            stream = torch.cuda.Stream(device=engine.device)
            self.semantic_graph, _ = _capture(self._semantic, stream)

    @torch.inference_mode()
    def refresh(self):
        update = self.store.snapshot_updates(self.version)
        if update is None:
            return
        version, size, changed = update
        if changed:
            indices = torch.tensor([index for index, _ in changed], dtype=torch.int64).pin_memory()
            rows = [row for _, row in changed]
            maps = torch.stack([row.probabilities for row in rows]).transpose(0, 1).contiguous().pin_memory()
            norms = torch.stack([row.prefix_norms for row in rows]).T.contiguous().pin_memory()
            embeddings = torch.stack([row.embedding for row in rows]).pin_memory()
            for device, state in self.devices.items():
                with torch.cuda.device(device):
                    gpu_indices = indices.to(device, non_blocking=True)
                    state.bank.index_copy_(1, gpu_indices,
                        maps[state.first:state.last].to(device, non_blocking=True))
                    state.norms.index_copy_(1, gpu_indices,
                        norms[state.first:state.last].to(device, non_blocking=True))
                    if device == self.engine.device:
                        self.embeddings.index_copy_(0, gpu_indices, embeddings.to(device, non_blocking=True))
            for index, row in changed:
                self.rows[index] = row
        if self.size != size:
            for device, state in self.devices.items():
                with torch.cuda.device(device):
                    state.count.fill_(size)
        self.version, self.size = version, size

    def _semantic(self):
        state = self.devices[self.engine.device]
        _products[(triton.cdiv(self.capacity, 4),)](
            self.input, self.embeddings, state.norms, state.dots, self.semantic_norm,
            state.scores, state.active, state.count, self.capacity, self.width,
            True, True, 4, triton.next_power_of_2(self.width), enable_fp_fusion=False)
        _metadata[(1,)](state.scores, self.semantic_norm, state.scores, self.semantic_metadata,
            self.semantic_metadata, state.active, state.count, self.capacity, 0, True, False,
            triton.next_power_of_2(self.capacity), 1)

    def begin(self, embeddings):
        self.refresh()
        active = not self.engine._prefill and self.size > 0 and embeddings.shape[:2] == (1, 1)
        if self.active != active:
            for device, state in self.devices.items():
                with torch.cuda.device(device):
                    state.active.fill_(int(active))
        self.active = active
        self.engine.cache.expire_prefetches(-1, new_iteration=True)
        if active:
            self.input.copy_(embeddings.view(1, -1))
            self.semantic_graph.replay()

    def stage(self, layer, device):
        if not self.active or layer == 0 or layer + self.distance >= self.layers:
            return
        previous = self.engine.layer_devices[layer - 1]
        if previous != device:
            source, target = self.devices[previous], self.devices[device]
            target.dots.copy_(source.dots, non_blocking=True)
            target.query_norm.copy_(source.query_norm, non_blocking=True)

    def record(self, layer, probabilities, selected):
        """Capture search alongside routing, including when prediction is inactive."""
        if layer not in self.metadata:
            return
        state = self.devices[probabilities.device]
        query = layer + self.distance < self.layers
        if query:
            local = layer - state.first
            _products[(triton.cdiv(self.capacity, 16),)](
                probabilities, state.bank[local], state.norms[local], state.dots,
                state.query_norm, state.scores, state.active, state.count,
                self.capacity, self.experts, False, layer == 0, 16,
                triton.next_power_of_2(self.experts), enable_fp_fusion=False)
        _metadata[(1,)](state.scores, state.query_norm, selected, self.metadata[layer],
            self.semantic_metadata if layer == 0 else state.scores, state.active, state.count,
            self.capacity, self.top_k, query, layer == 0,
            triton.next_power_of_2(self.capacity), triton.next_power_of_2(self.top_k))

    def read_route(self, layer, selected):
        if not self.active or layer not in self.metadata:
            return selected[0].tolist(), None
        values = self.metadata[layer].tolist()
        return values[:self.top_k], values[self.top_k:]

    def consume(self, layer, values):
        if values is None:
            return
        if layer == 0:
            self._plan(values[2:], range(1, self.distance))
        if layer + self.distance < self.layers:
            self._plan(values[:2], [layer + self.distance])
        if self.distance == 1:
            # Submit distance-one predictions before their target layer.
            self.advance(layer)

    def _plan(self, values, layers):
        index, bits = values
        if index < 0:
            return
        score = struct.unpack("f", struct.pack("i", bits))[0]
        record = self.rows[index]
        threshold = min(1., max(0., 1. - score))
        for layer in layers:
            probabilities = record.array[layer]
            count = max(self.top_k, int(np.searchsorted(record.mass[layer], threshold, side="left")) + 1)
            experts = record.order[layer, :count].tolist()
            self.engine.cache.plan(layer, probabilities, experts)

    def advance(self, layer):
        for cache in self.engine.caches.values():
            cache.advance(layer)

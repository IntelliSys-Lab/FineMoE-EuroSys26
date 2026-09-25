"""Fixed-address expert slots with ordered CUDA copies and decode graphs."""

from contextlib import contextmanager
from dataclasses import dataclass
import gc

import torch
import torch.nn.functional as F
import numpy as np
import triton
from finemoe import _cache
from finemoe.kernels.experts import _accumulate


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


def pack_host_weights(weights):
    """Return adjacent pinned expert projections and their shared storage."""
    first = weights[0]
    offset = 0
    packed = first.is_pinned()
    for weight in weights:
        packed &= (weight.is_contiguous() and weight.dtype == first.dtype and
                   weight.untyped_storage().data_ptr() == first.untyped_storage().data_ptr() and
                   weight.data_ptr() == first.data_ptr() + offset * first.element_size())
        offset += weight.numel()
    if packed:
        storage = first.as_strided((offset,), (1,))
    else:
        storage = torch.empty(offset, dtype=first.dtype, device="cpu", pin_memory=True)
    if not packed:
        offset = 0
        for weight in weights:
            storage[offset:offset + weight.numel()].copy_(weight.reshape(-1))
            offset += weight.numel()
    views = tuple(part.view(weight.shape) for part, weight in
                  zip(storage.split([w.numel() for w in weights]), weights))
    return storage, views


class ExpertSlot:
    def __init__(self, shapes, storage):
        self.storage = storage
        self.weights = [part.view(shape) for part, shape in
                        zip(storage.split([shape.numel() for shape in shapes]), shapes)]
        self.graph = None

    def eager(self, hidden):
        gate, up = F.linear(hidden, self.weights[0]).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.weights[1])

    @torch.inference_mode()
    def capture(self, workspace):
        """Capture on the caller's prepared CUDA stream."""
        if self.graph is not None:
            return
        self.static_input, self.static_output = workspace.input, workspace.output
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        try:
            workspace.run(self.weights)
        finally:
            graph.capture_end()
        self.graph = graph

    def compute(self, hidden):
        if self.graph is not None and hidden.shape == self.static_input.shape:
            self.static_input.copy_(hidden)
            self.graph.replay()
            # Slot reuse must not overwrite a pending routing output.
            return self.static_output.clone()
        return self.eager(hidden)


class DecodeWorkspace:
    """Shared decode buffers; preserve BF16 rounding between operations."""

    def __init__(self, slot):
        first = slot.weights[0]
        hidden = first.shape[1]
        neurons = first.shape[0] // 2
        self.input = first.new_zeros(1, hidden)
        self.gate_up = first.new_empty(1, 2 * neurons)
        self.gate, self.up = self.gate_up.chunk(2, -1)
        self.output = first.new_empty(1, hidden)
        self.weight = first.new_ones(())
        self.accumulator = first.new_zeros(1, hidden)

    def run(self, weights):
        torch.mm(self.input, weights[0].t(), out=self.gate_up)
        F.silu(self.gate, inplace=True)
        self.gate.mul_(self.up)
        torch.mm(self.gate, weights[1].t(), out=self.output)
        hidden = self.output.numel()
        _accumulate[(triton.cdiv(hidden, 256),)](
            self.output, self.weight, self.accumulator, hidden, 256, enable_fp_fusion=False)


class ExpertCache:
    """Fixed GPU slots managed by the native cache and decode dispatcher."""

    def __init__(self, host_weights, capacity, device):
        if not host_weights or capacity < 1:
            raise ValueError("cache requires weights and at least one slot")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("cache requires an indexed CUDA device")
        self.host_weights, self.host_storage = {}, {}
        first = next(iter(host_weights.values()))
        shapes, dtype = [w.shape for w in first], first[0].dtype
        if len(shapes) != 2 or dtype != torch.bfloat16:
            raise ValueError("cache requires packed gate/up and down BF16 projections")
        for key, weights in sorted(host_weights.items()):
            if ([w.shape for w in weights] != shapes or
                    any(w.dtype != dtype or w.device.type != "cpu" for w in weights)):
                raise ValueError("experts must have identical shapes/dtypes and reside on CPU")
            self.host_storage[key], self.host_weights[key] = pack_host_weights(weights)
        self._indices = {key: i for i, key in enumerate(self.host_storage)}
        self._layers = {}
        for key, i in self._indices.items():
            self._layers.setdefault(key[0], []).append(i)
        self.capacity = min(capacity, len(host_weights))
        self.expert_bytes = sum(w.numel() * w.element_size() for w in first)
        self.weight_bytes = self.capacity * self.expert_bytes
        self.storage = torch.empty(self.capacity, sum(shape.numel() for shape in shapes),
                                   dtype=dtype, device=self.device)
        self.slots = [ExpertSlot(shapes, row) for row in self.storage]
        self.workspace = self._capture_stream = None
        self._graphs_ready = False
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._native = _cache.create(
            f"libcudart.so.{torch.version.cuda.split('.')[0]}", self.device.index,
            self.transfer_stream.cuda_stream, self.expert_bytes,
            [(key[0], storage.data_ptr()) for key, storage in self.host_storage.items()],
            [slot.storage.data_ptr() for slot in self.slots],
            (tuple(self.host_storage.values()), self.storage, self.transfer_stream))

    @property
    def stats(self):
        return CacheStats(*_cache.stats(self._native))

    def contains(self, key):
        return _cache.contains(self._native, self._indices[key])

    def update_probabilities(self, probabilities):
        if any(key not in self._indices for key in probabilities):
            raise ValueError("invalid cache probability")
        _cache.probabilities(self._native, [self._indices[key] for key in probabilities],
                             np.asarray(list(probabilities.values()), dtype=np.float64))

    def update_layer_probabilities(self, layer, probabilities):
        _cache.probabilities(self._native, self._layers[layer],
                             np.ascontiguousarray(probabilities, dtype=np.float64))

    def plan(self, layer, probabilities, order):
        _cache.plan(self._native, layer, self._layers[layer],
                    np.ascontiguousarray(probabilities, dtype=np.float64), order)

    def advance(self, layer):
        _cache.advance(self._native, layer)

    def reset_probabilities(self, value):
        _cache.reset_probabilities(self._native, value)

    def prefetch(self, key):
        return _cache.prefetch(self._native, self._indices[key])

    def expire_prefetches(self, layer, new_iteration=False):
        _cache.expire(self._native, layer, new_iteration)

    @contextmanager
    def acquire(self, key, next_key=None):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        index = _cache.acquire(self._native, self._indices[key],
                               self._indices[next_key] if next_key is not None else None, stream)
        try:
            yield self.slots[index]
        finally:
            _cache.release(self._native, index, torch.cuda.current_stream(self.device).cuda_stream)

    @torch.inference_mode()
    def compute(self, key, hidden, next_key=None):
        with self.acquire(key, next_key) as slot:
            return slot.compute(hidden)

    @torch.inference_mode()
    def compute_routed(self, ranked_keys, hidden, routing_weights, shared_graph, shared_output):
        _cache.dispatch(self._native, [(rank, self._indices[key]) for rank, key in ranked_keys],
                        hidden.data_ptr(), routing_weights.data_ptr(),
                        routing_weights.stride(0) * routing_weights.element_size(),
                        shared_graph.raw_cuda_graph_exec(),
                        torch.cuda.current_stream(self.device).cuda_stream)
        return self.workspace.accumulator.clone(), shared_output

    @torch.inference_mode()
    def warmup_graphs(self):
        if self._graphs_ready:
            return
        self.synchronize()
        with torch.cuda.device(self.device):
            if not any(self.contains(key) for key in self._indices):
                self.storage.zero_()
            if self.workspace is None:
                self.workspace = DecodeWorkspace(self.slots[0])
                self._capture_stream = torch.cuda.Stream(device=self.device)
            stream = self._capture_stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.workspace.run(self.slots[0].weights)
            stream.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            with torch.cuda.stream(stream):
                for slot in self.slots:
                    slot.capture(self.workspace)
            stream.synchronize()
            workspace = self.workspace
            _cache.configure(self._native, [slot.graph.raw_cuda_graph_exec() for slot in self.slots],
                             workspace.input.data_ptr(), workspace.weight.data_ptr(),
                             workspace.accumulator.data_ptr(),
                             workspace.input.numel() * workspace.input.element_size(),
                             (workspace, tuple(slot.graph for slot in self.slots)))
            self._graphs_ready = True

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def clear(self):
        _cache.clear(self._native)

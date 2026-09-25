"""Contiguous decoder placement with one expert cache per selected CUDA device."""

from dataclasses import fields
import torch

from .expert_cache import CacheStats


class DeviceContext:
    """Per-device cache and graph state."""
    def __init__(self, engine, device):
        self.parent = engine
        self.device = device
        self.cache = self.graph_positions = None
        self._staged = {}

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def begin_iteration(self):
        self._staged.clear()
        if self.graph_positions is not None:
            self.graph_positions.source = None

    def stage(self, value):
        """Stage shared inputs once per forward."""
        if isinstance(value, torch.Tensor):
            if value.device == self.device:
                return value
            key = id(value)
            if key not in self._staged:
                self._staged[key] = (value, value.to(self.device, non_blocking=True))
            return self._staged[key][1]
        if isinstance(value, tuple):
            return tuple(self.stage(item) for item in value)
        if isinstance(value, list):
            return [self.stage(item) for item in value]
        if isinstance(value, dict):
            return {key: self.stage(item) for key, item in value.items()}
        return value


class LayerCaches:
    """Expert access across device-local caches."""
    def __init__(self, caches, layer_devices):
        self.caches = caches
        self.device = next(iter(caches))
        self.by_layer = tuple(caches[device] for device in layer_devices)
        self.host_weights = {key: value for cache in caches.values() for key, value in cache.host_weights.items()}
        self.host_storage = {key: value for cache in caches.values() for key, value in cache.host_storage.items()}
        self.expert_bytes = next(iter(caches.values())).expert_bytes
        self.capacity = sum(cache.capacity for cache in caches.values())
        self.weight_bytes = sum(cache.weight_bytes for cache in caches.values())

    @property
    def stats(self):
        return CacheStats(**{f.name: sum(getattr(cache.stats, f.name) for cache in self.caches.values())
                             for f in fields(CacheStats)})

    @property
    def slots(self):
        return [slot for cache in self.caches.values() for slot in cache.slots]

    def update_probabilities(self, probabilities):
        if any(key not in self.host_weights or not 0 <= value <= 1 for key, value in probabilities.items()):
            raise ValueError("invalid cache probability")
        grouped = {}
        for key, value in probabilities.items():
            grouped.setdefault(self.by_layer[key[0]], {})[key] = value
        for cache, values in grouped.items():
            cache.update_probabilities(values)

    def update_layer_probabilities(self, layer, probabilities):
        self.by_layer[layer].update_layer_probabilities(layer, probabilities)

    def plan(self, layer, probabilities, order):
        self.by_layer[layer].plan(layer, probabilities, order)

    def prefetch(self, key):
        cache = self.by_layer[key[0]]
        with torch.cuda.device(cache.device):
            return cache.prefetch(key)

    def expire_prefetches(self, layer, new_iteration=False):
        for cache in self.caches.values():
            cache.expire_prefetches(layer, new_iteration)

    def clear(self):
        for cache in self.caches.values():
            cache.clear()

    def synchronize(self):
        for cache in self.caches.values():
            cache.synchronize()

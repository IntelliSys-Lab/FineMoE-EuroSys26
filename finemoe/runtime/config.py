"""FineMoE runtime configuration."""

from dataclasses import dataclass, fields
import json
from pathlib import Path
import torch


@dataclass(frozen=True)
class FineMoEConfig:
    devices: tuple[str, ...] = ("cuda:0",)  # decoder order
    cache_size: int | None = None  # expert slots per device
    device_memory_ratio: float = 0.8  # fraction of free memory after dense placement
    reserve_bytes: int = 2 * 1024**3  # KV cache, activations, graphs
    host_copy_workers: int = 4  # startup copies
    store_capacity: int = 1000
    prefetch_distance: int = 6
    collect_maps: bool = False  # collect maps and traces for the demo
    eval_mode: str = "offline"
    queue_capacity: int = 32
    trace_capacity: int = 64

    @classmethod
    def load(cls, value=None):
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            with open(value) as handle:
                value = json.load(handle)
        value = dict(value or {})
        unknown = value.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown FineMoE options: {sorted(unknown)}")
        return cls(**value)

    def __post_init__(self):
        if not isinstance(self.devices, (list, tuple)) or not self.devices:
            raise ValueError("devices must be a nonempty list")
        try:
            devices = tuple(torch.device(value) for value in self.devices)
        except (TypeError, RuntimeError) as error:
            raise ValueError("devices must contain valid devices") from error
        if (any(d.type != "cuda" or d.index is None for d in devices) or
                len(set(devices)) != len(devices)):
            raise ValueError("devices must contain distinct, indexed CUDA devices")
        object.__setattr__(self, "devices", tuple(str(d) for d in devices))
        for name in ("store_capacity", "prefetch_distance", "queue_capacity", "trace_capacity", "host_copy_workers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cache_size is not None and (not isinstance(self.cache_size, int)
                                           or isinstance(self.cache_size, bool)
                                           or self.cache_size < 1):
            raise ValueError("cache_size must be a positive integer")
        if not 0 < self.device_memory_ratio <= 1 or self.reserve_bytes < 0:
            raise ValueError("invalid memory ratio or reserve_bytes")
        if self.eval_mode not in ("online", "offline"):
            raise ValueError("eval_mode must be online or offline")
        if self.store_capacity > 4096:
            raise ValueError("store_capacity must not exceed 4096")
        if self.collect_maps and self.eval_mode != "online":
            raise ValueError("collect_maps requires eval_mode='online'")

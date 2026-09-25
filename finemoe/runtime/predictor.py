"""Asynchronous expert-map search and updates."""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread

import torch
import numpy as np

from finemoe.memory.expert_map import prefetch_rows, TrajectoryState


class Snapshot:
    """Retain the source until its host copy completes."""
    def __init__(self, tensor, stream=None, copy=True):
        self.event = None
        self.source = tensor.detach()
        if tensor.device.type == "cuda":
            self.tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(stream):
                self.tensor.copy_(self.source, non_blocking=True)
                self.source.record_stream(stream)
                self.event = torch.cuda.Event()
                self.event.record(stream)
        else:
            self.tensor = self.source.to("cpu", copy=copy)

    def get(self):
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        if self.source is not None:
            self.tensor = self.tensor.float()
            self.source = None
        return self.tensor


class SnapshotStack(Snapshot):
    """Join layer snapshots and apply the padding mask."""
    def __init__(self, snapshots, dim, mask=None):
        self.snapshots = tuple(snapshots)
        self.dim, self.mask = dim, mask

    def get(self):
        value = torch.stack([snapshot.get() for snapshot in self.snapshots], self.dim)
        return value if self.mask is None else value[self.mask.get().bool()]


@dataclass
class Context:
    iteration: int
    kind: str
    snapshot: Snapshot
    embeds: Snapshot | None = None


class MapPredictor:
    def __init__(self, store, cache, top_k, options):
        self.store, self.cache = store, cache
        self.top_k, self.options = top_k, options
        self.stream = torch.cuda.Stream(device=cache.device)
        self.streams = {cache.device: self.stream}
        for device in getattr(cache, "caches", {}):
            if device not in self.streams:
                self.streams[device] = torch.cuda.Stream(device=device)
        self._cv = Condition()
        self._queue = deque()
        self._pending = {}
        self._scheduled_iteration = -1
        self._scheduled_layer = -1
        self._iteration = -1
        self._layer = -1
        self._busy = False
        self._closed = False
        self._error = None
        self._match_iteration = -1
        self._match_state = None
        self._external_scheduler = False
        self.dropped_updates = 0
        self._thread = Thread(target=self._run, name="finemoe-map-search", daemon=True)
        self._thread.start()

    def advance(self, iteration, layer):
        with self._cv:
            self._iteration, self._layer = iteration, layer

    def set_external_scheduler(self, enabled):
        with self._cv:
            if enabled != self._external_scheduler:
                self._external_scheduler = enabled
                self._pending.clear()
                self._cv.notify_all()

    def publish(self, iteration, kind, tensor, embeds=None):
        self.check_error()
        with self._cv:
            if self._closed:
                raise RuntimeError("predictor is closed")
            if len(self._queue) >= self.options.queue_capacity:
                if kind == "update":
                    self.dropped_updates += 1
                return
            # Check queue capacity before allocating a snapshot.
            tensor = tensor() if callable(tensor) else tensor
            embeds = embeds() if callable(embeds) else embeds
            context = Context(iteration, kind, self.snapshot(tensor),
                              self.snapshot(embeds) if embeds is not None else None)
            self._queue.append(context)
            self._cv.notify_all()

    def snapshot(self, tensor):
        if isinstance(tensor, Snapshot):
            return tensor
        return Snapshot(tensor, self.streams.get(tensor.device))

    def _run(self):
        try:
            torch.init_num_threads()
            with torch.inference_mode():
                while True:
                    with self._cv:
                        self._cv.wait_for(lambda: self._closed or self._queue or self._pending)
                        if not self._queue and not self._pending and self._closed:
                            return
                        context = self._queue.popleft() if self._queue else None
                        self._busy = True
                    if context is not None:
                        self._process(context)
                    progressed = False if self._external_scheduler else self._dispatch()
                    with self._cv:
                        self._busy = False
                        self._cv.notify_all()
                        if self._pending and not self._queue and not progressed:
                            self._cv.wait(timeout=0.0005)
        except BaseException as error:
            with self._cv:
                self._error = error
                self._busy = False
                self._cv.notify_all()

    def _process(self, context):
        if context.kind != "update":
            with self._cv:
                if self._external_scheduler or context.iteration != self._iteration:
                    return
        data = context.snapshot.get()
        if context.kind == "update":
            self.store.add(context.embeds.get(), data)
            return
        distance = self.store.prefetch_distance
        layers = (range(distance) if context.kind == "semantic" else
                  [data.shape[1] + distance - 1])
        with self._cv:
            now = self._layer
            stale = context.iteration != self._iteration
        layers = [layer for layer in layers if now < layer < self.store.num_layers]
        if stale or not layers:
            return
        if context.kind == "semantic":
            scores, maps = self.store.match_embed(data)
        else:
            if self._match_iteration != context.iteration:
                self._match_iteration = context.iteration
                self._match_state = TrajectoryState()
            scores, maps = self.store.match_traj(data, self._match_state)
        if maps is None:
            return
        probabilities, selected = prefetch_rows(maps, scores, layers, self.top_k)
        with self._cv:
            if self._external_scheduler or context.iteration != self._iteration:
                return
        for layer, row, mask in zip(layers, probabilities, selected):
            candidates = np.flatnonzero(mask).tolist()
            candidate_probabilities = row[candidates].tolist()
            with self._cv:
                if (not self._external_scheduler and context.iteration == self._iteration
                        and layer > self._layer):
                    self.cache.update_layer_probabilities(layer, row)
                    self._pending.update({(layer, expert): (context.iteration, priority)
                                          for expert, priority in zip(candidates, candidate_probabilities)})

    def _dispatch(self):
        with self._cv:
            if self._external_scheduler:
                return False
            iteration, layer = self._iteration, self._layer
            by_layer = getattr(self.cache, "by_layer", None)
            self._pending = {key: value for key, value in self._pending.items()
                             if value[0] == iteration and key[0] > layer and not (
                                 by_layer[key[0]] if by_layer is not None else self.cache).contains(key)}
            if (iteration, layer) != (self._scheduled_iteration, self._scheduled_layer):
                self.cache.expire_prefetches(layer, iteration != self._scheduled_iteration)
                self._scheduled_iteration, self._scheduled_layer = iteration, layer
            if not self._pending:
                return False
            # Recompute priority using the remaining layer distance.
            candidates = sorted(self._pending, key=lambda k: (-self._pending[k][1] / (k[0] - layer), k))
        progressed, blocked, attempts = False, set(), 0
        for key in candidates:
            device = by_layer[key[0]].device if by_layer is not None else self.cache.device
            if device in blocked:
                continue
            if attempts >= 64:
                break
            with self._cv:
                if self._external_scheduler or iteration != self._iteration or key[0] <= self._layer:
                    self._pending.pop(key, None)
                    continue
            attempts += 1
            submitted = self.cache.prefetch(key)
            if submitted:
                self._pending.pop(key, None)
                progressed = True
            else:
                blocked.add(device)
        return progressed

    def check_error(self):
        if self._error is not None:
            raise RuntimeError("FineMoE prediction worker failed") from self._error

    def drain(self):
        with self._cv:
            self._cv.wait_for(lambda: self._error is not None or
                              (not self._queue and not self._pending and not self._busy))
        self.check_error()

    def close(self):
        with self._cv:
            self._closed = True
            self._iteration = -1
            self._layer = self.store.num_layers
            self._cv.notify_all()
        self._thread.join()
        # Finish pending copies before releasing their sources.
        for stream in self.streams.values():
            stream.synchronize()
        self._queue.clear()
        self.check_error()

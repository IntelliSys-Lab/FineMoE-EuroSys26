"""Qwen execution engine with device-local caches and contiguous decoder placement."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from threading import Lock
import torch

from finemoe.memory import ExpertMapStore, ExpertTracer
from finemoe.models.sparse_qwen import OffloadedQwenMoeBlock
from finemoe.models.attention_qwen import (
    GraphPositionBuffers, GraphedQwenDecoderLayer, GraphedQwenSdpaAttention, GraphedQwenDeltaNet,
    PlacedQwenDecoderLayer,
)
from .config import FineMoEConfig
from .expert_cache import ExpertCache
from .predictor import MapPredictor, Snapshot, SnapshotStack
from .placement import DeviceContext, LayerCaches


class OffloadEngine:
    def __init__(self, model, config=None):
        self.options = FineMoEConfig.load(config)
        self.model = model
        devices = tuple(torch.device(value) for value in self.options.devices)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if any(d.index >= torch.cuda.device_count() for d in devices):
            raise ValueError("requested CUDA device is not visible")
        self.devices, self.device = devices, devices[0]
        self._request_lock = Lock()
        self._closed = False
        self._graphs_ready = False
        self._active = False
        self._iteration = -1
        self._hooks = []
        config = model.config
        if config.model_type != "qwen3_5_moe_text":
            raise ValueError("Attach a Qwen3.5-MoE text model")
        if getattr(config, "quantization_config", None):
            raise ValueError("Quantized checkpoints require a separate expert backend")
        # Preserve the upstream expert accumulation order.
        config._experts_implementation = "eager"
        if config.hidden_act != "silu":
            raise ValueError("Expert slots currently support Qwen's SiLU activation")
        if any(p.device.type != "cpu" or p.is_meta for p in model.parameters()):
            raise ValueError("Attach a fully loaded CPU model")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("Attach a BF16 model")
        model.set_attn_implementation("sdpa")
        blocks = [(i, layer.mlp) for i, layer in enumerate(model.model.layers)
                  if hasattr(layer.mlp, "experts")]
        self.num_layers = len(blocks)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        if len(self.devices) > len(blocks):
            raise ValueError("each device must own at least one decoder layer")
        self.layer_devices = tuple(self.devices[i * len(self.devices) // len(model.model.layers)]
                                   for i in range(len(model.model.layers)))
        sparse_devices = tuple(self.layer_devices[index] for index, _ in blocks)
        self.contexts = {device: DeviceContext(self, device) for device in self.devices}
        if not blocks or self.options.prefetch_distance > self.num_layers:
            raise ValueError("prefetch_distance exceeds the number of MoE layers")
        self.expert_map_store = ExpertMapStore(
            self.options.store_capacity, self.num_layers, self.num_experts,
            config.hidden_size, self.options.prefetch_distance)
        self.expert_tracer = ExpertTracer(self.num_layers, self.num_experts,
                                         self.expert_map_store, self.options.trace_capacity)
        host_weights = {}
        first_experts = blocks[0][1].experts
        elements = first_experts.gate_up_proj[0].numel() + first_experts.down_proj[0].numel()
        host_storage = torch.empty(self.num_layers, self.num_experts, elements,
                                   dtype=first_experts.gate_up_proj.dtype, device="cpu",
                                   pin_memory=True)
        self.host_copy_workers = min(self.options.host_copy_workers,
                                     max(1, len(os.sched_getaffinity(0)) // torch.get_num_threads()))
        pool = (ThreadPoolExecutor(self.host_copy_workers, initializer=torch.init_num_threads)
                if self.host_copy_workers > 1 else nullcontext(None))
        @torch.inference_mode()
        def pack_layer(item):
            sparse_id, (_, block) = item
            experts = block.experts
            gate_up, down = experts.gate_up_proj.detach(), experts.down_proj.detach()
            gate_size, down_size = gate_up[0].numel(), down[0].numel()
            elements = gate_size + down_size
            layer_storage = host_storage[sparse_id]
            # Prepared weights are interleaved by expert.
            if (gate_up.stride(0) == elements and down.stride(0) == elements and
                    gate_up.untyped_storage().data_ptr() == down.untyped_storage().data_ptr() and
                    down.data_ptr() == gate_up.data_ptr() + gate_size * gate_up.element_size()):
                layer_storage.copy_(gate_up.as_strided(layer_storage.shape, layer_storage.stride()))
            else:
                layer_storage[:, :gate_size].copy_(gate_up.reshape(self.num_experts, gate_size))
                layer_storage[:, gate_size:].copy_(down.reshape(self.num_experts, down_size))
            shapes = [gate_up.shape[1:], down.shape[1:]]
            return {(sparse_id, expert_id): tuple(part.view(shape) for part, shape in
                    zip(row.split([shape.numel() for shape in shapes]), shapes))
                    for expert_id, row in enumerate(layer_storage)}

        with pool as executor:
            items = enumerate(blocks)
            layers = executor.map(pack_layer, items) if executor is not None else map(pack_layer, items)
            for rows in layers:
                host_weights.update(rows)
        for sparse_id, (index, block) in enumerate(blocks):
            context = self.contexts[self.layer_devices[index]] if len(self.devices) > 1 else self
            model.model.layers[index].mlp = OffloadedQwenMoeBlock(block, sparse_id, context)
        del blocks
        for index, layer in enumerate(model.model.layers):
            context = self.contexts[self.layer_devices[index]] if len(self.devices) > 1 else self
            if layer.block_type == "linear_attention":
                layer.linear_attn = GraphedQwenDeltaNet(layer.linear_attn, context)
            else:
                layer.self_attn = GraphedQwenSdpaAttention(layer.self_attn, context)
            if len(self.devices) > 1:
                model.model.layers[index] = PlacedQwenDecoderLayer(
                    layer, context, last=index == len(model.model.layers) - 1)
            else:
                model.model.layers[index] = GraphedQwenDecoderLayer(layer, self)
        model.requires_grad_(False)
        model.eval()
        if len(self.devices) == 1:
            model.to(self.device)
        else:
            for name, module in model.model.named_children():
                if name != "layers":
                    module.to(self.device)
            model.lm_head.to(self.device)
            for layer, device in zip(model.model.layers, self.layer_devices):
                layer.to(device)
        first = next(iter(host_weights.values()))
        expert_bytes = sum(w.numel() * w.element_size() for w in first)
        self.caches = {}
        for device in self.devices:
            weights = {key: value for key, value in host_weights.items() if sparse_devices[key[0]] == device}
            if self.options.cache_size is None:
                free, _ = torch.cuda.mem_get_info(device)
                budget = int(free * self.options.device_memory_ratio) - self.options.reserve_bytes
                capacity = min(budget // expert_bytes, len(weights))
                if capacity < 1:
                    raise ValueError(f"insufficient free memory on {device} after dense weights and runtime reserve")
            else:
                capacity = self.options.cache_size
            with torch.cuda.device(device):
                cache = ExpertCache(weights, capacity, device)
                self.caches[device] = self.contexts[device].cache = cache
                self.contexts[device].graph_positions = GraphPositionBuffers(cache.storage,
                    int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)))
        self.graph_positions = self.contexts[self.device].graph_positions
        self.cache = (next(iter(self.caches.values())) if len(self.devices) == 1 else
                      LayerCaches(self.caches, sparse_devices))
        self._request_done = {device: torch.cuda.Event() for device in self.devices}
        self._request_recorded = False
        self.predictor = MapPredictor(self.expert_map_store, self.cache, self.top_k, self.options)
        self.gpu_predictor = None
        self._gpu_iteration = False
        self._hooks.append(model.register_forward_pre_hook(self._before_forward, with_kwargs=True))
        self._hooks.append(model.get_input_embeddings().register_forward_hook(self._embedding_hook))

    def start_request(self, batch_size):
        if self._closed:
            raise RuntimeError("engine is closed")
        if not self._graphs_ready:
            raise RuntimeError("Call model.warmup() before serving")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("concurrent requests on one MoE instance are unsupported")
        try:
            if self._request_recorded:
                for device, event in self._request_done.items():
                    torch.cuda.current_stream(device).wait_event(event)
            if self.options.collect_maps:
                self.expert_tracer.start_request(batch_size)
            self._active = True
            self._first_iteration = True
        except BaseException:
            self._request_lock.release()
            raise

    def finish_request(self):
        if self._active:
            try:
                self.predictor.advance(-1, self.num_layers)
                self.gpu_predictor.advance(self.num_layers)
                self.predictor.drain()
            finally:
                try:
                    # Order graph scratch reuse across request streams.
                    for device, event in self._request_done.items():
                        event.record(torch.cuda.current_stream(device))
                    self._request_recorded = True
                finally:
                    self._active = False
                    self._request_lock.release()

    def _before_forward(self, module, args, kwargs):
        if not self._active:
            raise RuntimeError("Use the MoE wrapper, or start_request/finish_request around model calls")
        self._attention_mask = kwargs.get("attention_mask")
        self._output_router_logits = kwargs.get("output_router_logits", self.model.config.output_router_logits)
        if kwargs.get("inputs_embeds") is not None:
            self._begin_iteration(kwargs["inputs_embeds"])

    def _embedding_hook(self, module, args, output):
        self._begin_iteration(output)

    def _begin_iteration(self, embeddings):
        self._iteration += 1
        for context in self.contexts.values():
            # Restage positions even when their tensors are reused.
            context.begin_iteration()
        self._prefill = self._first_iteration
        self._first_iteration = False
        self._gpu_iteration = embeddings.shape[:2] == (1, 1)
        self.predictor.set_external_scheduler(self._gpu_iteration)
        self.predictor.advance(self._iteration, -1)
        if self._prefill:
            for cache in self.caches.values():
                cache.reset_probabilities(1 / self.num_experts)
        self._observe_prediction = (self.options.eval_mode == "online" or
                                    (not self._gpu_iteration and not self._prefill))
        self.gpu_predictor.begin(embeddings)
        if not self._observe_prediction and not self.options.collect_maps:
            return
        batch, length, _ = embeddings.shape
        mask = self._attention_mask
        mask = (torch.ones(batch, length, dtype=torch.bool, device=embeddings.device)
                if mask is None else mask[:, -length:].to(device=embeddings.device, dtype=torch.bool))
        if mask.shape != (batch, length):
            raise ValueError("expected a 2D attention mask")
        positions = torch.arange(length, device=embeddings.device).expand(batch, -1)
        self._last_token = positions.masked_fill(~mask, -1).amax(-1)
        self._batch_index = torch.arange(batch, device=embeddings.device)
        self._embeddings = embeddings[self._batch_index, self._last_token].detach()
        self._trajectory = []
        self._full_prefill = self._prefill and self.options.collect_maps
        if self._full_prefill:
            self._prompt_embeddings, self._prompt_mask = embeddings.detach(), mask
            self._prompt_trajectory = []
        if self.options.collect_maps:
            self.expert_tracer.begin_iteration(embeddings.detach(), mask, self._prefill)
        if not self._gpu_iteration and not self._prefill:
            self.predictor.publish(self._iteration, "semantic", self._embeddings)

    def advance(self, layer):
        self.predictor.advance(self._iteration, layer)
        if self._gpu_iteration:
            self.gpu_predictor.advance(layer)

    def observe(self, layer, probabilities, selected, cpu_probabilities=None):
        if self.options.collect_maps:
            self.expert_tracer.observe(layer, probabilities.detach(), selected.detach())
        if not self._observe_prediction:
            return
        if cpu_probabilities is not None:
            # The routing-ID read has completed this immutable snapshot.
            self._trajectory.append(Snapshot(cpu_probabilities, copy=False))
            if self._full_prefill:
                self._prompt_trajectory.append(Snapshot(cpu_probabilities[:, None], copy=False))
        elif len(self.devices) > 1:
            context = self.contexts[probabilities.device]
            last = probabilities[context.stage(self._batch_index), context.stage(self._last_token)].detach()
            self._trajectory.append(self.predictor.snapshot(last))
            if self._full_prefill:
                self._prompt_trajectory.append(self.predictor.snapshot(probabilities.detach()))
        else:
            self._trajectory.append(probabilities[self._batch_index, self._last_token].detach())
            if self._full_prefill:
                self._prompt_trajectory.append(probabilities.detach())
        if not self._gpu_iteration and not self._prefill and (
                layer + self.options.prefetch_distance < self.num_layers):
            self.predictor.publish(self._iteration, "trajectory", self._trajectory_snapshot)
        if layer == self.num_layers - 1 and self.options.eval_mode == "online":
            if self._full_prefill:
                self.predictor.publish(self._iteration, "update", self._prefill_snapshot,
                                       lambda: self._prompt_embeddings[self._prompt_mask])
                self._prompt_embeddings = self._prompt_mask = None
                self._prompt_trajectory = []
            else:
                self.predictor.publish(self._iteration, "update",
                                       self._trajectory_snapshot, self._embeddings)

    def _trajectory_snapshot(self):
        return (SnapshotStack(self._trajectory, 1) if isinstance(self._trajectory[0], Snapshot) else
                torch.stack(self._trajectory, 1))

    def _prefill_snapshot(self):
        return (SnapshotStack(self._prompt_trajectory, 2, self.predictor.snapshot(self._prompt_mask))
                if isinstance(self._prompt_trajectory[0], Snapshot) else
                torch.stack(self._prompt_trajectory, 2)[self._prompt_mask])

    def warmup(self):
        if self._active:
            raise RuntimeError("warmup must run outside a request")
        if self._closed:
            raise RuntimeError("engine is closed")
        self.predictor.drain()
        if self.gpu_predictor is None:
            from .cuda_predictor import CudaMapPredictor
            self.gpu_predictor = CudaMapPredictor(self)
        for device, cache in self.caches.items():
            layers = [layer for layer, target in zip(self.model.model.layers, self.layer_devices) if target == device]
            with torch.cuda.device(device):
                cache.warmup_graphs()
                for layer in layers:
                    layer.mlp.warmup_graphs(cache._capture_stream)
                for layer in layers:
                    attention = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
                    attention.warmup_graphs(cache._capture_stream)
                    layer.warmup_graphs(cache._capture_stream)
                torch.cuda.synchronize(device)
        self._graphs_ready = True

    def close(self):
        if self._closed:
            return
        if self._active:
            raise RuntimeError("cannot close an active request")
        try:
            self.predictor.close()
        finally:
            self.cache.synchronize()
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._closed = True

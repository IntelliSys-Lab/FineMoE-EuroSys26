"""Qwen routing with active-expert dispatch and a batch-one decode fast path."""

import torch
from torch import nn
import torch.nn.functional as F


class OffloadedQwenMoeBlock(nn.Module):
    def __init__(self, original, layer_id, engine):
        super().__init__()
        self.gate = original.gate
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.num_experts, self.top_k = original.gate.num_experts, original.gate.top_k
        self.layer_id = layer_id
        self.engine = engine
        self._router_graph = self._shared_graph = None

    def _route_eager(self, hidden):
        if getattr(self.engine, "_output_router_logits", False):
            # Use the gate hook when router logits are requested.
            logits, weights, selected = self.gate(hidden)
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            logits = F.linear(hidden, self.gate.weight)
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
            weights, selected = torch.topk(probabilities, self.top_k, dim=-1)
            weights /= weights.sum(dim=-1, keepdim=True)
            weights = weights.to(logits.dtype)
        if hidden.shape[0] == 1:
            self.engine.gpu_predictor.record(self.layer_id, probabilities, selected)
        return logits, probabilities, weights, selected

    def _shared_eager(self, hidden):
        shared = self.shared_expert(hidden)
        return torch.sigmoid(self.shared_expert_gate(hidden)) * shared

    @torch.inference_mode()
    def warmup_graphs(self, stream):
        if self._router_graph is not None and self._shared_graph is not None:
            return
        if self._router_graph is None:
            self._graph_input = self.gate.weight.new_zeros(1, self.gate.hidden_dim)
        stream.wait_stream(torch.cuda.current_stream(self.gate.weight.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._route_eager(self._graph_input)
                self._shared_eager(self._graph_input)
        stream.synchronize()
        with torch.cuda.stream(stream):
            if self._router_graph is None:
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin()
                try:
                    self._graph_route = self._route_eager(self._graph_input)
                finally:
                    graph.capture_end()
                self._router_graph = graph
            if self._shared_graph is None:
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin()
                try:
                    self._graph_shared = self._shared_eager(self._graph_input)
                finally:
                    graph.capture_end()
                self._shared_graph = graph
        stream.synchronize()

    def _route(self, hidden):
        if (self._router_graph is not None and
                hidden.shape[0] == 1 and not self.engine._output_router_logits):
            self._graph_input.copy_(hidden)
            self._router_graph.replay()
            return self._export_graph_route(self._graph_route)
        if self._shared_graph is not None and hidden.shape[0] == 1:
            self._graph_input.copy_(hidden)
        return self._route_eager(hidden)

    def _export_graph_route(self, route):
        logits, probabilities, weights, selected = route
        if self.engine.options.collect_maps:
            # Retained traces must survive the next graph replay.
            probabilities = probabilities.clone()
        return logits, probabilities, weights, selected

    def _shared(self, hidden):
        if self._shared_graph is not None and hidden.shape[0] == 1:
            self._shared_graph.replay()
            return self._graph_shared
        return self._shared_eager(hidden)

    def forward(self, hidden_states, *, graph_route=None):
        batch, length, hidden_dim = hidden_states.shape
        hidden = hidden_states.reshape(-1, hidden_dim)
        self.engine.advance(self.layer_id)
        logits, probabilities, weights, selected = (
            self._route(hidden) if graph_route is None else self._export_graph_route(graph_route))
        cpu_probabilities = expert_ids = forecast = None
        if hidden.shape[0] == 1:
            if self.engine._observe_prediction:
                # The routing-ID read below also completes this copy.
                cpu_probabilities = torch.empty_like(probabilities, device="cpu", pin_memory=True)
                cpu_probabilities.copy_(probabilities, non_blocking=True)
            expert_ids, forecast = self.engine.gpu_predictor.read_route(self.layer_id, selected)
        self.engine.observe(self.layer_id, probabilities.view(batch, length, -1),
                            selected.view(batch, length, -1), cpu_probabilities)
        if forecast is not None:
            self.engine.gpu_predictor.consume(self.layer_id, forecast)
        shared = None
        if hidden.shape[0] == 1:
            # Ascending expert IDs preserve the upstream summation order.
            ranks = sorted(range(self.top_k), key=expert_ids.__getitem__)
            result, shared = self.engine.cache.compute_routed(
                [(rank, (self.layer_id, expert_ids[rank])) for rank in ranks],
                hidden, weights[0], self._shared_graph, self._graph_shared)
        else:
            result = torch.zeros_like(hidden)
            flat = selected.flatten()
            order = torch.argsort(flat, stable=True)
            tokens = torch.div(order, self.top_k, rounding_mode="floor")
            routed_weights = weights.flatten().index_select(0, order).unsqueeze(-1)
            counts = torch.bincount(flat, minlength=self.num_experts).tolist()
            offset = 0
            active = [(expert, count) for expert, count in enumerate(counts) if count]
            for index, (expert, count) in enumerate(active):
                token = tokens[offset:offset + count]
                weight = routed_weights[offset:offset + count]
                offset += count
                next_key = ((self.layer_id, active[index + 1][0]) if
                            index + 1 < len(active) else None)
                output = self.engine.cache.compute((self.layer_id, expert), hidden.index_select(0, token), next_key)
                result.index_add_(0, token, output * weight)
        if shared is None:
            shared = self._shared(hidden)
        result.add_(shared)
        return result.view(batch, length, hidden_dim)

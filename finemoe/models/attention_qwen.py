# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for FineMoE.

"""Piecewise decode graphs for Qwen3.5's hybrid attention and recurrent cache."""

from types import SimpleNamespace
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention, Qwen3_5MoeDecoderLayer, Qwen3_5MoeGatedDeltaNet,
    ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
)

from .linear_qwen import DeltaNetKernels


def _adopt(target, source):
    nn.Module.__init__(target)
    for name, value in vars(source).items():
        if not name.startswith("_"):
            setattr(target, name, value)
    for name, value in source.named_children():
        target.add_module(name, value)
    for name, value in source.named_parameters(recurse=False):
        target.register_parameter(name, value)
    for name, value in source.named_buffers(recurse=False):
        target.register_buffer(name, value, persistent=name not in source._non_persistent_buffers_set)


def _capture(function, stream):
    stream.wait_stream(torch.cuda.current_stream(stream.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        result = function()
    stream.synchronize()
    return graph, result


class GraphPositionBuffers:
    def __init__(self, weight, head_dim):
        self.cos = weight.new_ones(1, 1, head_dim)
        self.sin = torch.zeros_like(self.cos)
        self.source = None

    def stage(self, cos, sin):
        if self.source is None or self.source[0] is not cos or self.source[1] is not sin:
            self.cos.copy_(cos)
            self.sin.copy_(sin)
            self.source = (cos, sin)


class _RecurrentScratch:
    """Fixed graph state, separate from request caches."""
    def __init__(self, module):
        weight = module.in_proj_qkv.weight
        self.state = SimpleNamespace(
            conv_states=[weight.new_zeros(1, module.conv_dim, module.conv_kernel_size)],
            recurrent_states=[torch.zeros(1, module.num_v_heads, module.head_k_dim, module.head_v_dim,
                                          device=weight.device, dtype=torch.float32)],
            record_past=False)
        self.layers = {module.layer_idx: self.state}

    def has_previous_state(self, *args, **kwargs):
        return True

    def update_recurrent_state(self, value, layer_idx, **kwargs):
        self.state.recurrent_states[0].copy_(value)

    def stage(self, state):
        self.state.conv_states[0].copy_(state.conv_states[0])
        self.state.recurrent_states[0].copy_(state.recurrent_states[0])

    def publish(self, state):
        state.conv_states[0].copy_(self.state.conv_states[0])
        state.recurrent_states[0].copy_(self.state.recurrent_states[0])


class GraphedQwenDeltaNet(Qwen3_5MoeGatedDeltaNet):
    def __init__(self, original, engine):
        _adopt(self, original)
        self.engine = engine
        self._graph = None
        self.kernels = DeltaNetKernels()

    @torch.inference_mode()
    def warmup_kernels(self):
        if getattr(self, "_kernels_ready", False):
            return
        cache = DynamicCache(config=self.engine.model.config)
        weight = self.in_proj_qkv.weight
        self.kernels.forward(self, weight.new_zeros(1, 64, self.hidden_size), cache=cache)
        self.kernels.forward(self, weight.new_zeros(1, 1, self.hidden_size), cache=cache)
        self._kernels_ready = True

    @torch.inference_mode()
    def warmup_graphs(self, stream):
        if self._graph is not None:
            return
        self.warmup_kernels()
        self._hidden = self.in_proj_qkv.weight.new_zeros(1, 1, self.hidden_size)
        self._scratch = _RecurrentScratch(self)
        self._graph, self._output = _capture(lambda: self._step(self._hidden), stream)

    def _step(self, hidden):
        return self.kernels.forward(self, hidden, cache=self._scratch)

    def eligible(self, hidden, cache, mask):
        return (self._graph is not None and
                hidden.shape == self._hidden.shape and mask is None and
                isinstance(cache, DynamicCache) and cache.has_previous_state(self.layer_idx, state_idx=0) and
                not cache.layers[self.layer_idx].record_past)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if not self.eligible(hidden_states, cache_params, attention_mask):
            return self.kernels.forward(self, hidden_states, cache=cache_params, mask=attention_mask)
        state = cache_params.layers[self.layer_idx]
        self._hidden.copy_(hidden_states)
        self._scratch.stage(state)
        self._graph.replay()
        self._scratch.publish(state)
        return self._output.clone()


class GraphedQwenSdpaAttention(Qwen3_5MoeAttention):
    def __init__(self, original, engine):
        _adopt(self, original)
        self.engine = engine
        self._projection_graph = None

    def _project(self, hidden, cos, sin):
        query, gate = self.q_proj(hidden).view(1, 1, -1, 2 * self.head_dim).chunk(2, dim=-1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden).view(1, 1, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden).view(1, 1, -1, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value, gate.reshape(1, 1, -1)

    @torch.inference_mode()
    def warmup_graphs(self, stream):
        if self._projection_graph is not None:
            return
        self._graph_hidden = self.q_proj.weight.new_zeros(1, 1, self.config.hidden_size)
        positions = self.engine.graph_positions
        self._projection_graph, self._graph_qkv = _capture(
            lambda: self._project(self._graph_hidden, positions.cos, positions.sin), stream)

    def eligible(self, hidden, positions, cache, kwargs):
        return (self._projection_graph is not None and
                hidden.shape == self._graph_hidden.shape and isinstance(cache, DynamicCache) and
                not kwargs.get("output_attentions", False) and positions is not None and
                all(t.shape == self.engine.graph_positions.cos.shape for t in positions))

    def attend(self, query, key, value, attention_mask, cache, **kwargs):
        # DynamicCache must not retain graph scratch.
        if not cache.get_seq_length(self.layer_idx):
            key, value = key.clone(), value.clone()
        key, value = cache.update(key, value, self.layer_idx)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                         eager_attention_forward)
        output, _ = interface(self, query, key, value, attention_mask,
                              dropout=0.0, scaling=self.scaling, **kwargs)
        return output.reshape(1, 1, -1).contiguous()

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if not self.eligible(hidden_states, position_embeddings, past_key_values, kwargs):
            return super().forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        self._graph_hidden.copy_(hidden_states)
        self.engine.graph_positions.stage(*position_embeddings)
        self._projection_graph.replay()
        query, key, value, gate = self._graph_qkv
        output = self.attend(query, key, value, attention_mask, past_key_values, **kwargs)
        return self.o_proj(output * torch.sigmoid(gate)), None


class GraphedQwenDecoderLayer(Qwen3_5MoeDecoderLayer):
    """Capture dense decode work on both sides of dynamic expert dispatch."""
    def __init__(self, original, engine):
        _adopt(self, original)
        self.engine = engine
        self._prefix_graph = self._suffix_router_graph = None

    def _prefix(self):
        positions = self.engine.graph_positions
        return self.self_attn._project(self.input_layernorm(self._hidden), positions.cos, positions.sin)

    def _suffix(self):
        if self.block_type == "linear_attention":
            attention = self.linear_attn._step(self.input_layernorm(self._hidden))
        else:
            attention = self.self_attn.o_proj(self._attention_output * torch.sigmoid(self._qkv[3]))
        residual = self._hidden + attention
        hidden = self.post_attention_layernorm(residual)
        self.mlp._graph_input.copy_(hidden.view(1, self.hidden_size))
        return residual, hidden, self.mlp._route_eager(self.mlp._graph_input)

    @torch.inference_mode()
    def warmup_graphs(self, stream):
        if self._suffix_router_graph is not None:
            return
        self._hidden = self.input_layernorm.weight.new_zeros(1, 1, self.hidden_size)
        if self.block_type == "full_attention":
            self._prefix_graph, self._qkv = _capture(self._prefix, stream)
            self._attention_output = self._hidden.new_zeros(1, 1,
                self.self_attn.config.num_attention_heads * self.self_attn.head_dim)
        self._suffix_router_graph, self._combined_outputs = _capture(self._suffix, stream)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        eligible = (self._suffix_router_graph is not None and
                    hidden_states.shape == self._hidden.shape and not self.engine._output_router_logits and
                    not kwargs.get("output_attentions", False))
        if eligible:
            eligible = (self.linear_attn.eligible(hidden_states, past_key_values, attention_mask)
                        if self.block_type == "linear_attention" else
                        self.self_attn.eligible(hidden_states, position_embeddings, past_key_values, kwargs))
        if not eligible:
            return super().forward(hidden_states, position_embeddings, attention_mask,
                                   position_ids, past_key_values, **kwargs)
        self._hidden.copy_(hidden_states)
        if self.block_type == "linear_attention":
            state = past_key_values.layers[self.linear_attn.layer_idx]
            self.linear_attn._scratch.stage(state)
        else:
            self.engine.graph_positions.stage(*position_embeddings)
            self._prefix_graph.replay()
            output = self.self_attn.attend(*self._qkv[:3], attention_mask, past_key_values, **kwargs)
            self._attention_output.copy_(output)
        self._suffix_router_graph.replay()
        residual, hidden, route = self._combined_outputs
        if self.block_type == "linear_attention":
            self.linear_attn._scratch.publish(state)
        return residual + self.mlp(hidden, graph_route=route)


class PlacedQwenDecoderLayer(GraphedQwenDecoderLayer):
    """Decoder layer with device placement."""
    def __init__(self, original, engine, last=False):
        super().__init__(original, engine)
        self._last_placed_layer = last

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        with torch.cuda.device(self.engine.device):
            self.engine.gpu_predictor.stage(self.mlp.layer_id, self.engine.device)
            hidden_states = hidden_states.to(self.engine.device, non_blocking=True)
            output = super().forward(
                hidden_states, self.engine.stage(position_embeddings), self.engine.stage(attention_mask),
                self.engine.stage(position_ids), past_key_values, **self.engine.stage(kwargs))
            # The final norm and LM head reside on the input device.
            if self._last_placed_layer:
                output = output.to(self.engine.parent.device, non_blocking=True)
            return output

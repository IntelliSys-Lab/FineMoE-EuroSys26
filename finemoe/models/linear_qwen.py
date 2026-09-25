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

"""Qwen3.5 DeltaNet execution using FLA."""

import inspect
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as upstream


class DeltaNetKernels:
    def __init__(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        self.chunk = chunk_gated_delta_rule
        self.recurrent = fused_recurrent_gated_delta_rule
        self.conv = inspect.unwrap(upstream.causal_conv1d_fn)
        self.conv_update = inspect.unwrap(upstream.causal_conv1d_update)

    def forward(self, module, hidden, cache=None, mask=None):
        hidden = upstream.apply_mask_to_padding_states(hidden, mask)
        batch, length, _ = hidden.shape
        previous = cache is not None and cache.has_previous_state(module.layer_idx, state_idx=0)
        qkv = module.in_proj_qkv(hidden).transpose(1, 2)
        z = module.in_proj_z(hidden).reshape(batch, length, -1, module.head_v_dim)
        beta = module.in_proj_b(hidden).sigmoid()
        a = module.in_proj_a(hidden)
        if previous and length == 1 and not cache.layers[module.layer_idx].record_past:
            qkv = self.conv_update(qkv, cache.layers[module.layer_idx].conv_states[0],
                                  module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation)
        else:
            if cache is not None:
                qkv = cache.update_conv_state(qkv, module.layer_idx, conv_kernel_size=module.conv_kernel_size)
            qkv = self.conv(qkv, module.conv1d.weight.squeeze(1), module.conv1d.bias,
                            activation=module.activation)
            qkv = qkv[:, :, -length:]
        query, key, value = qkv.transpose(1, 2).split((module.key_dim, module.key_dim, module.value_dim), -1)
        query = query.reshape(batch, length, -1, module.head_k_dim)
        key = key.reshape(batch, length, -1, module.head_k_dim)
        value = value.reshape(batch, length, -1, module.head_v_dim)
        decay = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
        ratio = module.num_v_heads // module.num_k_heads
        if ratio > 1:
            query, key = query.repeat_interleave(ratio, 2), key.repeat_interleave(ratio, 2)
        state = cache.layers[module.layer_idx].recurrent_states[0] if previous else None
        function = self.recurrent if previous and length == 1 else self.chunk
        output, state = function(query, key, value, g=decay, beta=beta, initial_state=state,
                                 output_final_state=cache is not None, use_qk_l2norm_in_kernel=True)
        if cache is not None:
            cache.update_recurrent_state(state, module.layer_idx)
        output = module.norm(output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
        return module.out_proj(output.reshape(batch, length, -1))

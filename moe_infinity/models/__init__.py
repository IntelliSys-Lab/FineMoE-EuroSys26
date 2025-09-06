# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from .switch_transformers import SyncSwitchTransformersSparseMLP
from .nllb_moe import SyncNllbMoeSparseMLP
from .modeling_mixtral import SyncMixtralSparseMoeBlock, MixtralBlockSparseTop2MLP
from .modeling_grok import SyncGrokMoeBlock
from .modeling_arctic import SyncArcticMoeBlock, ArcticConfig
from .modeling_qwen import SyncQwen2MoeSparseMoeBlock, Qwen2MoeMLP
from .modeling_phi import SyncPhiMoESparseMoeBlock, PhiMoEBlockSparseTop2MLP
from .model_utils import apply_rotary_pos_emb

from dataclasses import dataclass, field
import torch


@dataclass
class ExpertTraceEntry:
    seq_id: str
    matrix: torch.Tensor
    iters: list = field(default_factory=list)
    num_new_tokens: int = 0
    num_prefill_tokens: int = 0

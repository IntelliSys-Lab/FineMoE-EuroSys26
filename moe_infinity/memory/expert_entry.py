from dataclasses import dataclass
import numpy as np
import hashlib
import networkx as nx


@dataclass
class ExpertTraceEntry:
    seq_id: str = None
    matrix: np.ndarray = None
    iters: dict = None
    num_new_tokens: int = 0
    num_prefill_tokens: int = 0

    def __hash__(self):
        return hash(self.seq_id)

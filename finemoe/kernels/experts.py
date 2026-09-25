"""BF16 expert-output accumulation."""

import triton
import triton.language as tl


@triton.jit
def _accumulate(Output, Weight, Accumulator, H: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Output + index, index < H, 0).to(tl.float32)
    weight = tl.load(Weight).to(tl.float32)
    weighted = (value * weight).to(tl.bfloat16).to(tl.float32)
    previous = tl.load(Accumulator + index, index < H, 0).to(tl.float32)
    tl.store(Accumulator + index, previous + weighted, index < H)

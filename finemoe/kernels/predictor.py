"""Map products and nearest-neighbor reduction for GPU prediction."""

import triton
import triton.language as tl


@triton.jit
def _products(Query, Bank, Norms, Dots, QueryNorm, Scores, Active, Count,
              C: tl.constexpr, WIDTH: tl.constexpr, SEMANTIC: tl.constexpr,
              RESET: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    if tl.load(Active):
        c = tl.program_id(0) * BC + tl.arange(0, BC)
        w = tl.arange(0, BW)
        live = tl.load(Count)
        query = tl.load(Query + w, w < WIDTH, 0).to(tl.float32)
        bank = tl.load(Bank + c[:, None] * WIDTH + w[None, :],
                       (c[:, None] < live) & (w[None, :] < WIDTH), 0)
        products = tl.sum(bank * query[None, :], 1)
        if not SEMANTIC:
            if not RESET:
                products += tl.load(Dots + c, c < C, 0)
            tl.store(Dots + c, products, c < C)
            norms = tl.load(Norms + c, c < live, 1)
            products /= tl.maximum(norms, 1.e-12)
        tl.store(Scores + c, products, c < C)
        if tl.program_id(0) == 0:
            squared_norm = tl.sum(query * query, 0)
            if not SEMANTIC and not RESET:
                squared_norm += tl.load(QueryNorm)
            tl.store(QueryNorm, squared_norm)


@triton.jit
def _metadata(Scores, QueryNorm, Selected, Output, Semantic, Active, Count,
              C: tl.constexpr, K: tl.constexpr, QUERY: tl.constexpr,
              FIRST: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    k = tl.arange(0, BK)
    selected = tl.load(Selected + k, k < K, 0).to(tl.int64)
    tl.store(Output + k, selected, k < K)
    index = tl.full((), -1, tl.int32)
    score = tl.full((), 0., tl.float32)
    if QUERY:
        if tl.load(Active):
            c = tl.arange(0, BC)
            scores = tl.load(Scores + c, c < tl.load(Count), -float("inf"))
            best = tl.max(scores, 0)
            index = tl.min(tl.where(scores == best, c, 2147483647), 0)
            score = tl.minimum(1., tl.maximum(-1., best / tl.maximum(tl.sqrt(tl.load(QueryNorm)), 1.e-12)))
    tl.store(Output + K, index.to(tl.int64))
    tl.store(Output + K + 1, score.to(tl.int32, bitcast=True).to(tl.int64))
    if FIRST:
        tl.store(Output + K + 2, tl.load(Semantic))
        tl.store(Output + K + 3, tl.load(Semantic + 1))

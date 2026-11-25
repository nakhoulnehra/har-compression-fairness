"""Bit allocation under a storage budget: exact multiple-choice knapsack, shared by every method.

Every allocation method (ours and the baselines) gives a sensitivity Omega[t][b] >= 0 for each quantizable tensor t
and candidate bit width b, assumes the drops add up over tensors (as ZeroQ and HAWQ do), and picks one b per tensor
that minimises sum_t Omega[t][b_t] subject to size_bits <= budget. The methods differ only in Omega.
"""

from __future__ import annotations

from math import gcd
from typing import Mapping, Sequence

import numpy as np
import torch.nn as nn

from .quant import fixed_bits, tensor_cost_bits, tensor_info

BIT_OPTIONS = (2, 3, 4, 8)


def mckp_min(costs: Sequence[Sequence[int]], values: Sequence[Sequence[float]], budget: int) -> list[int] | None:
    """Exact multiple-choice knapsack (minimisation). costs[g][j] (non-negative ints) and values[g][j] of option j
    in group g; choose exactly one option per group with total cost <= budget and the smallest total value.
    Returns the chosen option index per group, or None if even the cheapest options exceed the budget.
    Ties: the smallest total cost, then, group by group from the last, the earliest option in its list."""
    if len(costs) != len(values) or any(len(c) != len(v) for c, v in zip(costs, values)):
        raise ValueError("costs and values must have the same shape")
    if budget < 0:
        return None
    g_all = 0
    for c in costs:
        for x in c:
            if x < 0:
                raise ValueError("costs must be non-negative")
            g_all = gcd(g_all, int(x))
    unit = g_all or 1
    cap = int(budget) // unit
    if sum(min(c) for c in costs) // unit > cap:
        return None
    INF = np.inf
    dp = np.full(cap + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((len(costs), cap + 1), dtype=np.int16)
    for gi, (cs, vs) in enumerate(zip(costs, values)):
        new = np.full(cap + 1, INF)
        arg = np.zeros(cap + 1, dtype=np.int16)
        for j, (c, v) in enumerate(zip(cs, vs)):
            cu = int(c) // unit
            if cu > cap:
                continue
            cand = np.full(cap + 1, INF)
            cand[cu:] = dp[:cap + 1 - cu] + float(v)
            better = cand < new          # strict: an earlier option keeps the tie
            new = np.where(better, cand, new)
            arg = np.where(better, j, arg)
        dp, choice[gi] = new, arg
    if not np.isfinite(dp).any():
        return None
    best = float(np.min(dp))
    c = int(np.flatnonzero(dp == best)[0])     # smallest total cost among the optima
    picks = [0] * len(costs)
    for gi in range(len(costs) - 1, -1, -1):
        j = int(choice[gi, c])
        picks[gi] = j
        c -= int(costs[gi][j]) // unit
    assert c == 0
    return picks


def allocate(model: nn.Module, omega: Mapping[str, Mapping[int, float]], budget_bits: int,
             granularity: str = "channel", bit_options: Sequence[int] = BIT_OPTIONS) -> dict[str, int] | None:
    """{tensor: bits} minimising sum Omega under size_bits(model, bits) <= budget_bits (None if infeasible)."""
    info = tensor_info(model)
    if set(omega) != set(info):
        raise KeyError(f"omega must cover exactly the quantizable tensors; missing {sorted(set(info) - set(omega))}, "
                       f"extra {sorted(set(omega) - set(info))}")
    names = list(info)
    costs = [[tensor_cost_bits(info[n]["numel"], info[n]["rows"], b, granularity) for b in bit_options] for n in names]
    values = []
    for n in names:
        row = []
        for b in bit_options:
            v = float(omega[n][b])
            if not np.isfinite(v):
                raise ValueError(f"omega[{n}][{b}] is not finite")
            row.append(v)
        values.append(row)
    picks = mckp_min(costs, values, int(budget_bits) - fixed_bits(model))
    if picks is None:
        return None
    return {n: int(bit_options[j]) for n, j in zip(names, picks)}


def objective(omega: Mapping[str, Mapping[int, float]], bits: Mapping[str, int]) -> float:
    return float(sum(omega[n][b] for n, b in bits.items()))

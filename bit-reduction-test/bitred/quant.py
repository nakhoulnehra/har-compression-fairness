"""Weight-only symmetric fake quantization with one bit width per tensor, and its exact storage size.

Quantized tensors: every Conv / Linear weight, LSTM weight_ih / weight_hh and attention in_proj_weight, plus the
classification head (har.compression.prunable_tensors(include_head=True), the tensors of the plan's PTQ). Biases,
normalization parameters, gamma and the input-normalization buffers stay float32. Activations are not quantized.

    granularity "channel": one scale per output row of W reshaped to (out, -1)
                           (conv filter, linear row, LSTM gate row, attention projection row)
    granularity "tensor":  one scale per tensor (as har.compression.ptq_symmetric)
    s = max|W_row| / (2^(b-1) - 1),  W_q = clamp(round(W / s), -(2^(b-1) - 1), 2^(b-1) - 1) * s

Storage (bits): quantized tensor = numel * b + (rows or 1) * 32 for its float32 scales; every other parameter
numel * 32. Buffers (input normalization statistics, BatchNorm running statistics) are not counted, as in
har.compression.size_record. KB = 1024 bytes.
"""

from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn as nn

from har.compression import prunable_tensors

SCALE_BITS = 32
FLOAT_BITS = 32
GRANULARITIES = ("channel", "tensor")


def quant_tensors(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """(name, owning module, parameter name) of every quantizable tensor, in model order."""
    return prunable_tensors(model, include_head=True)


def tensor_info(model: nn.Module) -> dict[str, dict]:
    """name -> {"numel", "rows"} for every quantizable tensor (rows = output channels = number of scales)."""
    return {n: {"numel": int(getattr(m, p).numel()), "rows": int(getattr(m, p).shape[0])}
            for n, m, p in quant_tensors(model)}


def quantize_weight(w: torch.Tensor, bits: int, granularity: str = "channel") -> torch.Tensor:
    """Symmetric round-to-nearest fake quantization of one weight tensor (returns a new tensor)."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    qmax = 2 ** (int(bits) - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits must be >= 2 for symmetric quantization, got {bits}")
    W = w.detach().reshape(w.shape[0], -1) if granularity == "channel" else w.detach().reshape(1, -1)
    s = W.abs().amax(dim=1, keepdim=True) / qmax
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.clamp(torch.round(W / s), -qmax, qmax) * s).reshape(w.shape)


def quantize_model(model: nn.Module, bits: Mapping[str, int], granularity: str = "channel") -> nn.Module:
    """A deepcopy of `model` (eval mode) with every tensor named in `bits` quantized to its bit width. A tensor
    absent from `bits` (or mapped to 32) stays float32."""
    q = copy.deepcopy(model)
    names = {n for n, _, _ in quant_tensors(q)}
    unknown = set(bits) - names
    if unknown:
        raise KeyError(f"not quantizable: {sorted(unknown)}")
    with torch.no_grad():
        for n, mod, p in quant_tensors(q):
            b = bits.get(n)
            if b is None or int(b) >= FLOAT_BITS:
                continue
            w = getattr(mod, p)
            w.copy_(quantize_weight(w, int(b), granularity))
    q.eval()
    return q


def uniform_bits(model: nn.Module, b: int) -> dict[str, int]:
    return {n: int(b) for n, _, _ in quant_tensors(model)}


def tensor_cost_bits(numel: int, rows: int, b: int, granularity: str = "channel") -> int:
    """Storage of one tensor at b bits (b >= 32 means float32, no scales)."""
    if int(b) >= FLOAT_BITS:
        return int(numel) * FLOAT_BITS
    n_scales = int(rows) if granularity == "channel" else 1
    return int(numel) * int(b) + n_scales * SCALE_BITS


def fixed_bits(model: nn.Module) -> int:
    """Storage of the parameters that are never quantized (float32)."""
    qnames = {n for n, _, _ in quant_tensors(model)}
    return sum(int(p.numel()) * FLOAT_BITS for n, p in model.named_parameters() if n not in qnames)


def size_bits(model: nn.Module, bits: Mapping[str, int], granularity: str = "channel") -> int:
    info = tensor_info(model)
    unknown = set(bits) - set(info)
    if unknown:
        raise KeyError(f"not quantizable: {sorted(unknown)}")
    return fixed_bits(model) + sum(tensor_cost_bits(v["numel"], v["rows"], bits.get(n, FLOAT_BITS), granularity)
                                   for n, v in info.items())


def size_kb(model: nn.Module, bits: Mapping[str, int], granularity: str = "channel") -> float:
    return size_bits(model, bits, granularity) / 8 / 1024


def mean_bits(model: nn.Module, bits: Mapping[str, int]) -> float:
    """Weight-count-weighted mean bit width over the quantizable tensors."""
    info = tensor_info(model)
    tot = sum(v["numel"] for v in info.values())
    return sum(v["numel"] * min(int(bits.get(n, FLOAT_BITS)), FLOAT_BITS) for n, v in info.items()) / tot

"""Tensor layout transforms independent of RTP runtime extensions."""

import torch


def split_kda_input(weight, heads, head_dim, tp_size, tp_rank):
    """Shard [Q,K,V,G,F_a,beta] by head while replicating F_a.

    Linear weights use RTP's [input, output] layout. F_a feeds every rank's
    independently column-sharded F_b, so its columns must not be TP-sharded.
    """
    if weight.ndim != 2 or tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid K3 fused input tensor or TP rank")
    if heads <= 0 or head_dim <= 0 or heads % tp_size:
        raise ValueError("K3 heads must divide TP size")
    width = heads * head_dim
    fa_width = weight.shape[1] - 4 * width - heads
    if fa_width <= 0:
        raise ValueError("K3 fused input must contain a positive F_a width")
    pieces = weight.split([width] * 4 + [fa_width, heads], dim=1)
    return torch.cat(
        [
            piece if index == 4 else piece.chunk(tp_size, dim=1)[tp_rank]
            for index, piece in enumerate(pieces)
        ],
        dim=1,
    ).contiguous()

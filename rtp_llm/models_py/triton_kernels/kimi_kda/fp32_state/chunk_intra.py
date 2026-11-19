# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, no backward.

import torch

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_intra import (
    chunk_kda_fwd_intra as chunk_kda_fwd_intra_64,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.wy_fast import recompute_w_u_fwd


def _solve_large_tril_in_place(matrices: torch.Tensor) -> None:
    """Invert batched 128/256 lower-triangular matrices in 64-token blocks."""
    size = matrices.shape[-1]
    block_size = 64
    block_count = size // block_size

    diagonal = torch.stack(
        [
            matrices[
                ...,
                index * block_size : (index + 1) * block_size,
                index * block_size : (index + 1) * block_size,
            ]
            for index in range(block_count)
        ],
        dim=1,
    )
    batch, _, heads = diagonal.shape[:3]
    diagonal = diagonal.reshape(batch * block_count, heads, block_size, block_size)
    diagonal = diagonal.permute(0, 2, 1, 3).contiguous()
    diagonal = solve_tril(diagonal, output_dtype=matrices.dtype)
    diagonal = diagonal.permute(0, 2, 1, 3).reshape(
        batch, block_count, heads, block_size, block_size
    )
    for index in range(block_count):
        start = index * block_size
        matrices[..., start : start + block_size, start : start + block_size].copy_(
            diagonal[:, index]
        )

    merge_size = 2 * block_size
    while merge_size <= size:
        half = merge_size // 2
        for start in range(0, size, merge_size):
            middle, stop = start + half, start + merge_size
            upper_inv = matrices[..., start:middle, start:middle]
            lower_inv = matrices[..., middle:stop, middle:stop]
            lower_left = matrices[..., middle:stop, start:middle]
            product = torch.matmul(lower_left, upper_inv)
            lower_left.copy_(-torch.matmul(lower_inv, product))
        merge_size *= 2


def _solve_large_tril(
    matrix: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> torch.Tensor:
    batch, token_count, heads, chunk_size = matrix.shape

    def solve_range(batch_index: int, start: int, length: int) -> None:
        full_chunks, remainder = divmod(length, chunk_size)
        full_tokens = full_chunks * chunk_size
        if full_chunks:
            chunks = matrix[batch_index, start : start + full_tokens].view(
                full_chunks, chunk_size, heads, chunk_size
            )
            _solve_large_tril_in_place(chunks.permute(0, 2, 1, 3))
        if remainder:
            tail = matrix.new_zeros(1, heads, chunk_size, chunk_size)
            tail[..., :remainder, :].copy_(
                matrix[batch_index, start + full_tokens : start + length]
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
            _solve_large_tril_in_place(tail)
            matrix[batch_index, start + full_tokens : start + length].copy_(
                tail[0, :, :remainder].permute(1, 0, 2)
            )

    if cu_seqlens is None:
        for batch_index in range(batch):
            solve_range(batch_index, 0, token_count)
    else:
        boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        for start, stop in zip(boundaries, boundaries[1:]):
            solve_range(0, start, stop - start)
    return matrix


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    if chunk_size == 64:
        return chunk_kda_fwd_intra_64(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            disable_recompute=disable_recompute,
        )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    shape = (*k.shape[:3], chunk_size)
    Aqk = k.new_empty(shape)
    Akk = k.new_zeros(shape)
    Aqk, Akk = chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        sub_chunk_size=chunk_size,
    )
    Akk = _solve_large_tril(Akk, cu_seqlens)
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk

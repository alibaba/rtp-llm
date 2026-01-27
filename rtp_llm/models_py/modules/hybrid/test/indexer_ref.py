"""
Reference implementation of Indexer from DeepSeek-V3.2-Exp/inference/model.py
Extracted from lines 434-487 with necessary dependencies.
"""

import math
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import deep_gemm
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from fastsafetensors.frameworks import K
from torch import nn

from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_transform_ragged_fused
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

block_size = 128

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    From model.py lines 405-425.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform for activation rotation.
    From model.py lines 428-432.
    """
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def precompute_freqs_cis(attn_config: AttentionConfigs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = attn_config.rope_head_dim
    seqlen = 16384
    beta_fast = 32
    beta_slow = 1
    base = 10000
    factor = 40
    original_seq_len = 4096

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    k_fp8, k_scale = kv
    q = q.float()
    k = k_fp8.float()

    # mask_lo = (
    #     torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    # )
    # mask_hi = (
    #     torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    # )
    # mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    # logits = logits.masked_fill(~mask, float("-inf"))
    if k_scale.dim() == 2:
        k_scale = k_scale.squeeze(-1)  # [N, 1] -> [N]
    logits = logits * k_scale.unsqueeze(0)  # [M, N] * [1, N]

    return logits


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def _ref_torch_impl(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))
        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)
    return torch.where(mask, indices + topk_indices_offset, indices)


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: Tuple, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def prepare_indexer_params(
    attention_inputs: PyAttentionInputs, blocksize: int
) -> SimpleNamespace:
    # Extract basic information
    is_prefill = attention_inputs.is_prefill
    input_lengths = attention_inputs.input_lengths  # [batch_size]
    sequence_lengths = attention_inputs.sequence_lengths  # [decode_batch_size]
    kv_cache_block_id = (
        attention_inputs.kv_cache_block_id_device
    )  # [batch_size, max_blocks]
    seq_size_per_block = blocksize  # Page size, typically 64

    ks = None
    ke = None
    page_table_1 = None
    topk_indices_offset = None
    expanded_seq_lens = None
    batch_size = input_lengths.size(0) if is_prefill else sequence_lengths.size(0)
    seq_lens = None
    cu_seqlens_q = None
    positions_d = None
    slot_mapping = None

    if is_prefill:
        # Prefill mode
        seq_lens = input_lengths.to(torch.int32)
        extend_seq_lens = input_lengths.cpu().tolist()

        # cu_seqlens_q: cumulative sequence lengths [0, len1, len1+len2, ...]
        cu_seqlens_q = attention_inputs.cu_seqlens.to(torch.int32).to("cuda")

        kv_lens = input_lengths + attention_inputs.prefix_lengths
        q_lens = input_lengths

        # expanded_seq_lens: repeat each seq_len for each token in that sequence
        # For example: if seq_lens = [3, 2], expanded = [3, 3, 3, 2, 2]
        expanded_seq_lens = torch.cat(
            [
                torch.arange(
                    kv_len - qo_len + 1,
                    kv_len + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                for qo_len, kv_len in zip(q_lens, kv_lens, strict=True)
            ]
        )

        # topk_indices_offset: cumulative offset for ragged KV layout
        # cu_seqlens_q[:-1] gives the starting position of each sequence
        topk_indices_offset = torch.repeat_interleave(
            cu_seqlens_q[:-1], seq_lens.to("cuda")
        ).to(torch.int32)

        # Calculate total tokens and allocate tensors
        total_tokens = seq_lens.sum().item()
        positions_d = torch.empty(total_tokens, dtype=torch.int32, device="cuda")
        batch_indice_d = torch.empty(total_tokens, dtype=torch.int32, device="cuda")

        # Reference: FlashInferMlaParams.cc fillParamsInternal (prefill branch)
        offset = 0
        ks_list = []
        ke_list = []
        k_offset = 0
        for i in range(batch_size):
            input_length = extend_seq_lens[i]
            prefix_length = attention_inputs.prefix_lengths[i]

            # Fill batch_indice_d and positions_d for this batch
            # Reference: FlashInferMlaParams.cc line 202-203
            for j in range(input_length):
                batch_indice_d[offset] = i
                positions_d[offset] = j + prefix_length
                offset += 1

            kv_len = int(kv_lens[i].item())
            ks_i = torch.full(
                (input_length,),
                k_offset,
                dtype=torch.int32,
                device="cuda",
            )
            seq_lens_expanded = torch.arange(
                kv_len - input_length + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke_i = ks_i + seq_lens_expanded

            ks_list.append(ks_i)
            ke_list.append(ke_i)
            k_offset += kv_len

        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

    else:
        # Decode mode
        seq_lens = sequence_lengths.to(torch.int32) + 1
        extend_seq_lens = [1] * batch_size

        # In decode mode, each request generates 1 token
        cu_seqlens_q = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device=sequence_lengths.device
        )

        # expanded_seq_lens: in decode mode, it's same as seq_lens
        expanded_seq_lens = seq_lens

        # Calculate batch_indice_d and positions_d for decode mode
        # Reference: FlashInferMlaParams.cc line 246-247
        batch_indice_d = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        positions_d = sequence_lengths.to(torch.int32).to("cuda")

        # page_table_1: converted global kv cache indices in sparse mla
        max_seq_len = seq_lens.max().item()
        page_table_1 = (
            torch.arange(max_seq_len, dtype=torch.int32, device=sequence_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    # Calculate slot_mapping for indexer KV cache
    # slot_mapping maps logical token positions to physical KV cache locations
    # Formula: slot_mapping = block_numbers * block_size + block_offsets
    #
    # Example:
    # - positions_d = [0, 1, 2, ..., 65, 66]  (token positions in sequence)
    # - block_size = 64
    # - block_indices = [0, 0, 0, ..., 1, 1]  (logical block index)
    # - block_table[batch_id] = [100, 200, ...]  (physical block IDs)
    # - block_numbers = [100, 100, ..., 200, 200]
    # - block_offsets = [0, 1, 2, ..., 1, 2]
    # - slot_mapping = [6400, 6401, ..., 12801, 12802]

    # Step 1: Calculate block indices (which logical block each token belongs to)
    block_indices = positions_d // seq_size_per_block  # [total_tokens]

    # Step 2: Calculate block offsets (position within each block)
    block_offsets = positions_d % seq_size_per_block  # [total_tokens]

    # Step 3: Get physical block numbers from block_table using gather
    # Create linear indices for gathering: batch_id * max_blocks + block_idx
    assert attention_inputs.kv_cache_block_id_device is not None
    if kv_cache_block_id is not None and kv_cache_block_id.numel() > 0:
        max_blocks = kv_cache_block_id.size(1)
        # Flatten block_table for gather operation
        flat_block_table = kv_cache_block_id.view(-1)  # [batch_size * max_blocks]

        # Calculate flat indices: batch_id * max_blocks + block_idx
        flat_indices = batch_indice_d.long() * max_blocks + block_indices.long()

        # Gather physical block numbers
        block_numbers = flat_block_table[flat_indices]  # [total_tokens]

        # Step 4: Calculate final slot mapping
        slot_mapping = block_numbers.long() * seq_size_per_block + block_offsets.long()

    seqlens_32 = seq_lens.to(torch.int32).to("cuda")
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_32, blocksize, deep_gemm.get_num_sms()
    )

    return SimpleNamespace(
        expanded_seq_lens=expanded_seq_lens,
        page_table_1=page_table_1,
        cu_seqlens_q=cu_seqlens_q,
        topk_indices_offset=topk_indices_offset,
        batch_size=batch_size,
        seq_lens=seqlens_32,
        extend_seq_lens=extend_seq_lens,
        positions_d=positions_d,
        slot_mapping=slot_mapping,
        block_table=attention_inputs.kv_cache_block_id_device,
        cu_seq_lens=attention_inputs.cu_seqlens,
        ks=ks,
        ke=ke,
        is_prefill=is_prefill,
        schedule_metadata=schedule_metadata,
    )


class IndexerRef(torch.nn.Module):
    """
    Indexer for DeepSeek-V3.2 sparse attention.
    From model.py lines 435-487.

    Note: This requires ModelArgs for initialization. For standalone use,
    you'll need to create a compatible args object or modify __init__.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get dimensions from config or use defaults
        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128
        self.scale_fmt = "ue8m0"  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.tokens_per_block
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2

        # Create linear layers
        # wq_b: projects q_lora to index_n_heads * index_head_dim
        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # wk: projects hidden_states to index_head_dim
        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # k_norm: LayerNorm for keys
        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        # weights_proj: projects hidden_states to index_n_heads (for attention weights)
        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.cos_sin_cache = weights[W.rope_cos_sin_cache]

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(attn_config), persistent=False
        )
        self.register_buffer(
            "k_cache",
            torch.zeros(128, 2048, self.index_head_dim, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(
                128, 2048, self.index_head_dim // block_size, dtype=torch.float32
            ),
            persistent=False,
        )

    def prepare(self, attention_inputs: PyAttentionInputs):
        """Prepare indexer parameters from attention inputs"""
        self.params = prepare_indexer_params(attention_inputs, self.blocksize)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, kv_cache: KVCache):
        q = self.wq_b(qr)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1
        )
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1
        )
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        apply_rope_with_cos_sin_cache_inplace(
            positions=self.params.positions_d,
            query=q_pe,
            key=k_pe,
            head_size=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=True,
        )
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)

        weights = self.weights_proj(x.float()) * self.index_n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

        if self.params.is_prefill:
            kv_fp8 = (k_fp8, k_scale)
            weights = weights.squeeze(-1)
            # clean_logits = False时，超出有效序列范围的 logits 值未清除
            # clean_logits has done in topk
            # ref_logits = _ref_fp8_mqa_logits(
            #     q=q_fp8.to(torch.float32),
            #     kv=kv_fp8,
            #     weights=weights,
            #     cu_seqlen_ks=ks,
            #     cu_seqlen_ke=ke,
            # )
            ref_logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                self.params.ks,
                self.params.ke,
                clean_logits=False,
            )

            # _ref_torch_transform_ragged_impl只支持expanded_seq_lens中所有len相同的情况
            # 单独测试fast_topk_transform_ragged_fused
            topk_indices = fast_topk_transform_ragged_fused(
                score=ref_logits,
                lengths=self.params.expanded_seq_lens.to("cuda"),
                topk_indices_offset=self.params.topk_indices_offset.to("cuda"),
                topk=self.index_topk,
                row_starts=self.params.ks,
            )
            # topk_indices = _ref_torch_transform_ragged_impl(
            #     score=ref_logits,
            #     seq_len=self.params.expanded_seq_lens.to("cuda")[-1].item(),
            #     topk_indices_offset=self.params.topk_indices_offset.to("cuda"),
            #     topk=self.index_topk,
            #     row_starts=self.params.ks,
            # )
        else:
            block_kv = self.blocksize
            num_heads_kv = 1
            head_dim_with_sf = 132
            kv_cache_fp8 = kv_cache.kv_scale_base.view(
                kv_cache.kv_scale_base.shape[0],
                block_kv,
                num_heads_kv,
                head_dim_with_sf,
            ).view(dtype=torch.uint8)
            # len of k
            max_seq_len = self.params.block_table.shape[1] * self.blocksize
            # clean_logits has done in topk
            ref_logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8.unsqueeze(1),
                kv_cache_fp8.view(dtype=torch.uint8),
                weights,
                self.params.seq_lens.to(torch.int32).to("cuda"),
                self.params.block_table,
                self.params.schedule_metadata,
                max_seq_len,
                clean_logits=False,
            )
            topk_indices = _ref_torch_transform_decode_impl(
                score=ref_logits,
                seq_len=self.params.expanded_seq_lens.to("cuda")[0].item(),
                src_page_table=self.params.page_table_1.to("cuda"),
                topk=self.index_topk,
                row_starts=None,
            )

        topk_indices_offset = (
            self.params.topk_indices_offset.to("cuda")
            if self.params.topk_indices_offset is not None
            else None
        )
        return topk_indices, ref_logits, topk_indices_offset

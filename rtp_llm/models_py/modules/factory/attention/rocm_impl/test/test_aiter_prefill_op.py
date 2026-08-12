"""Operator-level tests for the ROCm prefill attention path.

Exercises ``AiterPrefillAttnOp.forward`` end-to-end against a torch SDPA
reference. Both branches the C++ FusedRopeKVCachePrefillOp can feed in are
covered:

  * ``kv_cache is None`` — encoder-only path. Goes through ``split_raw_qkv``
    and ``aiter.flash_attn_varlen_func`` with the QKV pulled from a single
    packed buffer.
  * ``kv_cache is not None`` — the RoPE op delivers K/V already padded to
    ``[batch, H_kv, max_seqlen_k, D]``. Goes through ``unpad_kv_vectorized``
    and ``aiter.flash_attn_varlen_func``. This is the path the PR reviewer
    flagged: previously a Python for-loop unpad, now a single advanced-index
    gather. The numerical output must match the reference for both uniform
    and varied sequence lengths.

Skips automatically off ROCm or without ``aiter`` so the suite stays green
on the rest of the fleet.
"""

import math
import unittest
from typing import List, Optional, Sequence
from unittest.mock import patch

import torch
import torch.nn.functional as F

_IS_ROCM_BUILD = torch.version.hip is not None

try:
    import aiter  # noqa: F401

    _AITER_AVAILABLE = True
except ImportError:
    if _IS_ROCM_BUILD:
        raise
    _AITER_AVAILABLE = False

try:
    from rtp_llm.models_py.modules.factory.attention import attn_factory
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeImplTriton,
        AiterPrefillAttnOp,
        AiterPrefillAttnOpPaged,
        AiterPrefillAttnOpTriton,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
        AiterPrefillImplPaged,
        FMHAParams,
        _run_triton_paged_attention,
        validate_v_layout,
    )
    from rtp_llm.ops import (
        AttentionConfigs,
        FMHAConfig,
        KvCacheDataType,
        PyAttentionInputs,
        RopeConfig,
        RopeStyle,
    )
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCacheDecodeOpAsm,
        FusedRopeKVCacheDecodeOpNonAsm,
        FusedRopeKVCachePrefillOpAsm,
        FusedRopeKVCachePrefillOpNonAsm,
        LayerKVCache,
        get_typemeta,
    )

    _OPS_IMPORTABLE = True
except ImportError:
    if _IS_ROCM_BUILD:
        raise
    _OPS_IMPORTABLE = False


def _is_rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _make_attn_configs(
    head_num: int, head_num_kv: int, head_dim: int, tokens_per_block: int = 16
):
    """Build the minimal AttentionConfigs needed by AiterPrefillAttnOp."""
    cfg = AttentionConfigs()
    cfg.head_num = head_num
    cfg.kv_head_num = head_num_kv
    cfg.size_per_head = head_dim
    cfg.tokens_per_block = tokens_per_block
    cfg.kernel_tokens_per_block = tokens_per_block
    cfg.is_causal = True
    cfg.use_mla = False
    cfg.dtype = torch.float16
    cfg.kv_cache_dtype = KvCacheDataType.BASE
    return cfg


def _make_prefill_inputs(input_lengths: List[int], device: torch.device):
    """Build the minimal PyAttentionInputs that FMHAParams reads in prefill mode."""
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = True
    attn_inputs.input_lengths = torch.tensor(
        input_lengths, dtype=torch.int32, device=device
    )
    attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32, device=device)
    return attn_inputs


def _make_rope_attn_configs(
    head_num: int,
    head_num_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    tokens_per_block: int = 16,
):
    """AttentionConfigs configured for need_rope_kv_cache=True with base RoPE.

    Mirrors the embedding-model setup so AiterPrefillImplAsm/NonAsm wire up the
    real FusedRopeKVCachePrefillOp during __init__.
    """
    cfg = _make_attn_configs(head_num, head_num_kv, head_dim, tokens_per_block)
    cfg.dtype = dtype
    cfg.need_rope_kv_cache = True
    rope = RopeConfig()
    rope.dim = head_dim
    rope.base = 10000
    rope.scale = 1.0
    rope.style = RopeStyle.Base
    cfg.rope_config = rope
    return cfg


def _make_mrope_attn_configs(
    head_num: int,
    head_num_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_dim: int = 64,
    mrope_sections: Sequence[int] = (11, 11, 10),
    tokens_per_block: int = 16,
):
    cfg = _make_rope_attn_configs(
        head_num, head_num_kv, head_dim, dtype, tokens_per_block
    )
    cfg.rope_config.style = RopeStyle.Mrope
    cfg.rope_config.dim = rope_dim
    cfg.rope_config.index_factor = 3
    cfg.rope_config.mrope_dim1 = mrope_sections[0]
    cfg.rope_config.mrope_dim2 = mrope_sections[1]
    cfg.rope_config.mrope_dim3 = mrope_sections[2]
    cfg.rope_config.mrope_interleaved = True
    return cfg


def _make_rope_prefill_inputs(
    input_lengths: List[int], device: torch.device, dtype: torch.dtype
):
    """PyAttentionInputs populated with the cu_seqlens / padding_offset /
    dtype fields the C++ RoPE op reads during prepare()."""
    attn_inputs = _make_prefill_inputs(input_lengths, device)
    attn_inputs.dtype = get_typemeta(torch.empty(1, dtype=dtype))
    # The C++ op reads input_lengths from CPU pinned memory in production.
    attn_inputs.input_lengths = torch.tensor(
        input_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.sequence_lengths = torch.tensor(
        input_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.prefix_lengths = torch.zeros(
        len(input_lengths), dtype=torch.int32, device="cpu"
    )

    cu = [0]
    for seq_len in input_lengths:
        cu.append(cu[-1] + seq_len)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    attn_inputs.cu_seqlens_device = cu_seqlens
    attn_inputs.cu_kv_seqlens_device = cu_seqlens

    max_seq_len = max(input_lengths)
    padding_offset = []
    for batch_idx, seq_len in enumerate(input_lengths):
        offset = batch_idx * max_seq_len - cu[batch_idx]
        padding_offset.extend([offset] * seq_len)
    attn_inputs.padding_offset = torch.tensor(
        padding_offset, dtype=torch.int32, device=device
    )
    return attn_inputs


def _make_mrope_prefill_inputs(
    input_lengths: List[int], device: torch.device, dtype: torch.dtype
):
    attn_inputs = _make_rope_prefill_inputs(input_lengths, device, dtype)
    total_tokens = sum(input_lengths)
    position_ids = torch.zeros(total_tokens, 3, dtype=torch.int32, device=device)
    cursor = 0
    for seq_len in input_lengths:
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)
        position_ids[cursor : cursor + seq_len, 0] = positions
        position_ids[cursor : cursor + seq_len, 1] = positions // 2
        position_ids[cursor : cursor + seq_len, 2] = positions % 3
        cursor += seq_len
    attn_inputs.combo_position_ids = position_ids.contiguous()
    return attn_inputs


def _make_mrope_decode_inputs(
    sequence_lengths: List[int],
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_block: int = 16,
):
    """Build one-token-per-request decode inputs and an isolated block table."""
    batch_size = len(sequence_lengths)
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = False
    attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
    attn_inputs.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device=device
    )
    attn_inputs.input_lengths = torch.ones(batch_size, dtype=torch.int32, device=device)
    # Empty is the production no-prefix representation. A nonempty CUDA tensor
    # would make decode forward call .max().item() during graph capture.
    attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
    attn_inputs.padding_offset = torch.zeros(
        batch_size, dtype=torch.int32, device=device
    )
    attn_inputs.cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    )
    attn_inputs.cu_kv_seqlens_device = attn_inputs.cu_seqlens_device
    attn_inputs.combo_position_ids = position_ids.contiguous()

    block_counts = [
        (seq_len + 1 + tokens_per_block - 1) // tokens_per_block
        for seq_len in sequence_lengths
    ]
    max_blocks = max(block_counts)
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    per_batch_block_ids = []
    next_block = 0
    for batch_idx, block_count in enumerate(block_counts):
        block_ids = list(range(next_block, next_block + block_count))
        per_batch_block_ids.append(block_ids)
        block_table[batch_idx, :block_count] = torch.tensor(
            block_ids, dtype=torch.int32
        )
        next_block += block_count
    attn_inputs.kv_cache_kernel_block_id = block_table
    attn_inputs.kv_cache_kernel_block_id_device = block_table.to(device)
    return attn_inputs, per_batch_block_ids, next_block


def _alloc_decode_kv_cache(
    num_blocks: int,
    head_num_kv: int,
    tokens_per_block: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    pool = torch.zeros(
        [num_blocks, 2, head_num_kv, tokens_per_block, head_dim],
        dtype=dtype,
        device=device,
    )
    layer_cache = LayerKVCache()
    layer_cache.kv_cache_base = pool
    return layer_cache, pool


def _read_decode_k_from_pool(
    pool: torch.Tensor,
    per_batch_block_ids: List[List[int]],
    sequence_lengths: List[int],
    head_num_kv: int,
    head_dim: int,
    tokens_per_block: int,
):
    """Read the K written at each request's decode position."""
    vector_size = 16 // pool.element_size()
    result = torch.empty(
        len(sequence_lengths),
        head_num_kv,
        head_dim,
        dtype=pool.dtype,
        device=pool.device,
    )
    for batch_idx, sequence_length in enumerate(sequence_lengths):
        block_id = per_batch_block_ids[batch_idx][sequence_length // tokens_per_block]
        local_position = sequence_length % tokens_per_block
        k_kernel = (
            pool[block_id, 0]
            .contiguous()
            .view(
                head_num_kv,
                head_dim // vector_size,
                tokens_per_block,
                vector_size,
            )
        )
        result[batch_idx] = k_kernel.permute(0, 2, 1, 3).reshape(
            head_num_kv, tokens_per_block, head_dim
        )[:, local_position, :]
    return result


def _apply_base_rope(q: torch.Tensor, k: torch.Tensor, input_lengths: List[int]):
    """Torch reference for base RoPE (style=Base, base=10000) — matches the
    RopeConfig produced by _make_rope_attn_configs. Used to validate the C++
    FusedRopeKVCachePrefillOp output without depending on any HF model code."""
    head_dim = q.shape[-1]
    half = head_dim // 2
    positions = []
    for seq_len in input_lengths:
        positions.extend(range(seq_len))
    pos = torch.tensor(positions, dtype=torch.float32, device=q.device)
    inv_freq = 10000 ** (
        -2.0 * torch.arange(half, dtype=torch.float32, device=q.device) / head_dim
    )
    angle = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    def rot(x):
        lo, hi = x[..., :half], x[..., half:]
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        return torch.cat([lo * cos_b - hi * sin_b, hi * cos_b + lo * sin_b], dim=-1)

    return rot(q).to(q.dtype), rot(k).to(k.dtype)


def _apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_sections: Sequence[int],
    rope_dim: int,
):
    """Independent Qwen3/3.5 interleaved-MRoPE reference."""
    head_dim = q.shape[-1]
    assert 0 < rope_dim <= head_dim and rope_dim % 2 == 0
    rotary_pairs = rope_dim // 2
    assert len(mrope_sections) == 3
    assert sum(mrope_sections) == rotary_pairs

    # Match the model contract: start with temporal frequencies, then replace
    # H/W slots at offsets 1/2 in the THW interleaving.
    axes = torch.zeros(rotary_pairs, dtype=torch.long, device=q.device)
    axes[1 : 3 * mrope_sections[1] : 3] = 1
    axes[2 : 3 * mrope_sections[2] : 3] = 2
    assert int((axes == 1).sum()) == mrope_sections[1]
    assert int((axes == 2).sum()) == mrope_sections[2]
    positions = position_ids[:, axes].to(torch.float32)
    inv_freq = 10000 ** (
        -2.0
        * torch.arange(rotary_pairs, dtype=torch.float32, device=q.device)
        / rope_dim
    )
    angle = positions * inv_freq.unsqueeze(0)
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    def rot(x):
        x_float = x.float()
        lo = x_float[..., :rotary_pairs]
        hi = x_float[..., rotary_pairs:rope_dim]
        tail = x_float[..., rope_dim:]
        return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin, tail], dim=-1)

    return rot(q).to(q.dtype), rot(k).to(k.dtype)


def _sdpa_reference(
    q: torch.Tensor,  # [total_q, H_q, D]
    k: torch.Tensor,  # [total_kv, H_kv, D]
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Per-sequence torch SDPA reference, GQA aware. Returns ``[total_q, H_q*D]``."""
    head_num = q.shape[1]
    head_num_kv = k.shape[1]
    head_dim = q.shape[2]
    repeat = head_num // head_num_kv
    out_chunks = []
    for i in range(cu_seqlens_q.numel() - 1):
        q_lo, q_hi = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
        k_lo, k_hi = int(cu_seqlens_k[i].item()), int(cu_seqlens_k[i + 1].item())
        if q_hi == q_lo:
            continue
        q_seq = q[q_lo:q_hi].transpose(0, 1).unsqueeze(0)
        k_seq = k[k_lo:k_hi].transpose(0, 1).unsqueeze(0)
        v_seq = v[k_lo:k_hi].transpose(0, 1).unsqueeze(0)
        if repeat > 1:
            k_seq = k_seq.repeat_interleave(repeat, dim=1)
            v_seq = v_seq.repeat_interleave(repeat, dim=1)
        # Causal only makes sense when q_len == k_len; otherwise the SDPA
        # mask shape would not align (cross-attn case is non-causal).
        is_causal = causal and (q_hi - q_lo) == (k_hi - k_lo)
        out = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        out = out.squeeze(0).transpose(0, 1).reshape(q_hi - q_lo, head_num * head_dim)
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=0)


def _pack_qkv(q, k, v):
    """Concatenate per-token Q/K/V into the [token_num, (Hq + 2*Hkv)*D] layout
    that ``AiterPrefillAttnOp._forward_varlen`` consumes."""
    token_num = q.shape[0]
    return torch.cat(
        [
            q.reshape(token_num, -1),
            k.reshape(token_num, -1),
            v.reshape(token_num, -1),
        ],
        dim=-1,
    )


def _pad_kv(k, v, cu_seqlens_k):
    """Inverse of unpad: build [B, H_kv, max_seqlen_k, D] padded K/V tensors,
    matching the layout C++ FusedRopeKVCachePrefillOp emits."""
    batch_size = cu_seqlens_k.numel() - 1
    head_num_kv = k.shape[1]
    head_dim = k.shape[2]
    seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    max_seqlen_k = max(seq_lens) if seq_lens else 0
    k_padded = torch.zeros(
        batch_size, head_num_kv, max_seqlen_k, head_dim, dtype=k.dtype, device=k.device
    )
    v_padded = torch.zeros_like(k_padded)
    for i, seq_len in enumerate(seq_lens):
        lo = int(cu_seqlens_k[i].item())
        k_padded[i, :, :seq_len, :] = k[lo : lo + seq_len].transpose(0, 1)
        v_padded[i, :, :seq_len, :] = v[lo : lo + seq_len].transpose(0, 1)
    return k_padded, v_padded


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillAttnOp module")
class TestAiterPrefillAttnOp(unittest.TestCase):
    """Numerical regression for the prefill path that uses
    ``aiter.flash_attn_varlen_func``.

    We cannot mock the kernel — the reviewer specifically wants confidence the
    new vectorized unpad and split helpers feed the same numbers into the
    kernel as before. So tests run on the ROCm runner and compare against a
    torch SDPA reference. Tolerances follow the convention from other
    operator tests in the repo (atol=rtol=1e-2 for fp16).
    """

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    def _build_op_and_params(
        self,
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
    ):
        cfg = _make_attn_configs(head_num, head_num_kv, head_dim)
        op = AiterPrefillAttnOp(cfg)
        attn_inputs = _make_prefill_inputs(input_lengths, self.device)
        params = op.prepare(attn_inputs)
        # FMHAParams.__init__ only fills cu_seqlens_q/k from input_lengths;
        # for these tests there is no prefix so cu_seqlens_k == cu_seqlens_q.
        # The forward() path moves them to query.device internally.
        return op, params

    def _check_varlen_no_kv_cache(
        self,
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
    ):
        """kv_cache=None branch — single packed QKV in, attention out."""
        op, params = self._build_op_and_params(
            input_lengths, head_num, head_num_kv, head_dim
        )
        total_tokens = sum(input_lengths)
        q = torch.randn(
            total_tokens, head_num, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        qkv = _pack_qkv(q, k, v)

        # token_q_num == token_kv_num here (no prefix, no cross-attn).
        actual = op.forward(qkv, kv_cache=None, fmha_params=params)
        ref = _sdpa_reference(
            q, k, v, params.cu_seqlens_q, params.cu_seqlens_k, causal=op.is_causal
        )
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    def _check_varlen_with_padded_kv(
        self,
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
    ):
        """kv_cache!=None branch — Q packed, K/V padded as the RoPE op emits.

        The op's job here is to unpad K/V via ``unpad_kv_vectorized`` and call
        ``flash_attn_varlen_func``. We pass kv_cache=<sentinel> so the FP8 +
        ``kv_cache is None`` branches are skipped; the actual cache contents
        are not read on this code path."""
        op, params = self._build_op_and_params(
            input_lengths, head_num, head_num_kv, head_dim
        )
        total_tokens = sum(input_lengths)
        q = torch.randn(
            total_tokens, head_num, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        k_padded, v_padded = _pad_kv(k, v, params.cu_seqlens_k)

        sentinel_cache = object()  # Branch-only check; never dereferenced.
        actual = op.forward(
            (q, k_padded, v_padded), kv_cache=sentinel_cache, fmha_params=params
        )
        ref = _sdpa_reference(
            q, k, v, params.cu_seqlens_q, params.cu_seqlens_k, causal=op.is_causal
        )
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    # ---- kv_cache=None varlen path ----------------------------------------

    def test_varlen_no_kv_cache_uniform(self):
        self._check_varlen_no_kv_cache(
            [16, 16, 16], head_num=8, head_num_kv=8, head_dim=64
        )

    def test_varlen_no_kv_cache_varied(self):
        self._check_varlen_no_kv_cache(
            [7, 23, 1, 11], head_num=8, head_num_kv=8, head_dim=64
        )

    def test_varlen_no_kv_cache_gqa(self):
        # H_q > H_kv — exercises the repeat_interleave inside flash_attn_varlen_func.
        self._check_varlen_no_kv_cache(
            [12, 19], head_num=16, head_num_kv=4, head_dim=128
        )

    # ---- kv_cache!=None vectorized-unpad path -----------------------------

    def test_varlen_padded_kv_uniform(self):
        self._check_varlen_with_padded_kv(
            [16, 16, 16], head_num=8, head_num_kv=8, head_dim=64
        )

    def test_varlen_padded_kv_varied(self):
        # The interesting case: max_seqlen_k > min_seqlen_k means the padded
        # tensor has slots the kernel must NOT see. Catches indexing bugs in
        # the vectorized unpad (e.g., reading from padded zero region).
        self._check_varlen_with_padded_kv(
            [3, 17, 5, 11], head_num=8, head_num_kv=8, head_dim=64
        )

    def test_varlen_padded_kv_gqa(self):
        self._check_varlen_with_padded_kv(
            [9, 25, 13], head_num=32, head_num_kv=4, head_dim=128
        )

    def test_varlen_padded_kv_single_batch(self):
        # Batch=1 is a common shape in microbenchmarks; the vectorized unpad
        # must still produce contiguous output even with no concat work to do.
        self._check_varlen_with_padded_kv([24], head_num=8, head_num_kv=8, head_dim=64)


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillImplPaged module")
class TestAiterPrefillImplPagedSupport(unittest.TestCase):
    """Unit tests for AiterPrefillImplPaged.support() classmethod.

    Validates prefix_lengths boundary logic. MTP draft prefill capture inputs are
    pre-filled with non-zero prefix in cuda_graph_runner.cc, so support() needs
    only the prefix>0 check.
    """

    def _make_attn_inputs(self, prefix_lengths):
        from types import SimpleNamespace

        return SimpleNamespace(
            prefix_lengths=prefix_lengths,
            is_cuda_graph=False,
            is_prefill=True,
            input_lengths=torch.tensor([4], dtype=torch.int32),
        )

    def _make_attn_configs(self, *, mrope=False, interleaved=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            rope_config=SimpleNamespace(
                style=RopeStyle.Mrope if mrope else RopeStyle.Base,
                mrope_interleaved=interleaved,
            )
        )

    def test_support_true_for_real_prefix(self):
        """prefix_lengths.max() > 0 => support() returns True."""
        pl = torch.tensor([0, 128, 0, 64], dtype=torch.int32)
        self.assertTrue(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(), self._make_attn_inputs(pl)
            )
        )

    def test_support_true_for_capture_filled_prefix(self):
        """MTP draft capture pre-fills prefix_lengths with max_seq_len => support() True."""
        pl = torch.full((4,), 1024, dtype=torch.int32)
        self.assertTrue(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(), self._make_attn_inputs(pl)
            )
        )

    def test_support_false_for_zero_prefix(self):
        """All-zero prefix_lengths => support() returns False (ASM/NonAsm preferred)."""
        pl = torch.zeros(4, dtype=torch.int32)
        self.assertFalse(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(), self._make_attn_inputs(pl)
            )
        )

    def test_support_false_for_empty_prefix_lengths(self):
        """Empty prefix_lengths tensor => support() returns False."""
        pl = torch.empty(0, dtype=torch.int32)
        self.assertFalse(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(), self._make_attn_inputs(pl)
            )
        )

    def test_support_false_for_non_interleaved_mrope(self):
        pl = torch.tensor([128], dtype=torch.int32)
        self.assertFalse(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(mrope=True, interleaved=False),
                self._make_attn_inputs(pl),
            )
        )

    def test_support_true_for_interleaved_mrope(self):
        pl = torch.tensor([128], dtype=torch.int32)
        self.assertTrue(
            AiterPrefillImplPaged.support(
                self._make_attn_configs(mrope=True, interleaved=True),
                self._make_attn_inputs(pl),
            )
        )


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillImplPaged module")
class TestUpdatePrefillParamsForCudaGraph(unittest.TestCase):
    """Unit tests for prefill CUDA-graph FMHA metadata refresh.

    Uses a lightweight stub to bypass the heavy __init__ chain (aiter, RoPE, etc.).
    Only exercises the cu_seqlens/prefix/scalar reconstruction logic.
    """

    def _make_stub(self, batch_size):
        """Build a minimal object with fmha_params matching capture-time batch_size."""
        from types import SimpleNamespace

        fmha_params = SimpleNamespace(
            cu_seqlens_q=torch.zeros(batch_size + 1, dtype=torch.int32),
            cu_seqlens_k=torch.zeros(batch_size + 1, dtype=torch.int32),
            prefix_lengths=None,
            max_seq_len=0,
            max_seqlen_q=0,
            max_seqlen_k=0,
            token_q_num=0,
            token_kv_num=0,
            kv_cache_block_id_device=None,
            prefill_seqlen_k_int32=torch.zeros(batch_size, dtype=torch.int32),
        )
        stub = AiterPrefillImplPaged.__new__(AiterPrefillImplPaged)
        stub.fmha_params = fmha_params
        return stub

    def _make_attn_inputs(
        self,
        input_lengths,
        prefix_lengths=None,
        cu_seqlens=None,
        cu_kv_seqlens=None,
        kv_block_id=None,
    ):
        from types import SimpleNamespace

        batch_size = len(input_lengths)
        if kv_block_id is None:
            kv_block_id = torch.zeros(batch_size, 4, dtype=torch.int32)
        inputs = SimpleNamespace(
            input_lengths=torch.tensor(input_lengths, dtype=torch.int32),
            prefix_lengths=prefix_lengths,
            cu_seqlens=cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            kv_cache_kernel_block_id_device=kv_block_id,
        )
        return inputs

    def _call_update(self, stub, attn_inputs):
        stub._refresh_prefill_fmha_params_for_cuda_graph(stub.fmha_params, attn_inputs)

    def test_rebuild_from_input_lengths_no_prefix(self):
        """Rebuild cu_seqlens from input_lengths, no prefix."""
        stub = self._make_stub(batch_size=4)
        inputs = self._make_attn_inputs([5, 5, 5, 5])
        self._call_update(stub, inputs)

        p = stub.fmha_params
        self.assertEqual(p.cu_seqlens_q.tolist(), [0, 5, 10, 15, 20])
        self.assertEqual(p.cu_seqlens_k.tolist(), [0, 5, 10, 15, 20])
        self.assertEqual(p.max_seq_len, 5)
        self.assertEqual(p.max_seqlen_q, 5)
        self.assertEqual(p.max_seqlen_k, 5)
        self.assertEqual(p.token_q_num, 20)
        self.assertEqual(p.token_kv_num, 20)
        # prefill_seqlen_k_int32 must be synced from cu_seqlens_k
        self.assertEqual(p.prefill_seqlen_k_int32.tolist(), [5, 5, 5, 5])

    def test_rebuild_with_prefix(self):
        """Rebuild cu_seqlens from input_lengths + prefix_lengths."""
        stub = self._make_stub(batch_size=3)
        inputs = self._make_attn_inputs(
            [5, 3, 5],
            prefix_lengths=torch.tensor([100, 200, 0], dtype=torch.int32),
        )
        self._call_update(stub, inputs)

        p = stub.fmha_params
        self.assertEqual(p.cu_seqlens_q.tolist(), [0, 5, 8, 13])
        self.assertEqual(p.cu_seqlens_k.tolist(), [0, 105, 308, 313])
        self.assertEqual(p.max_seq_len, 5)
        self.assertEqual(p.max_seqlen_k, 203)
        self.assertEqual(p.token_q_num, 13)
        self.assertEqual(p.token_kv_num, 313)
        # prefill_seqlen_k_int32 must match per-batch kv lengths
        self.assertEqual(p.prefill_seqlen_k_int32.tolist(), [105, 203, 5])

    def test_active_and_inactive_batches(self):
        """MTP draft: active batches have tokens, inactive batches have 0."""
        stub = self._make_stub(batch_size=4)
        inputs = self._make_attn_inputs(
            [5, 5, 3, 0],
            prefix_lengths=torch.tensor([100, 100, 100, 100], dtype=torch.int32),
        )
        self._call_update(stub, inputs)

        p = stub.fmha_params
        self.assertEqual(p.cu_seqlens_q.tolist(), [0, 5, 10, 13, 13])
        self.assertEqual(p.max_seq_len, 5)
        self.assertEqual(p.token_q_num, 13)

    def test_rebuild_ignores_stale_live_cu_seqlens(self):
        """Replay metadata is rebuilt from input and prefix lengths."""
        stub = self._make_stub(batch_size=2)
        cu_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 2, 4], dtype=torch.int32)
        inputs = self._make_attn_inputs(
            [5, 5],
            prefix_lengths=torch.tensor([100, 100], dtype=torch.int32),
            cu_seqlens=cu_q,
            cu_kv_seqlens=cu_k,
        )
        self._call_update(stub, inputs)

        p = stub.fmha_params
        self.assertEqual(p.cu_seqlens_q.tolist(), [0, 5, 10])
        self.assertEqual(p.cu_seqlens_k.tolist(), [0, 105, 210])
        self.assertEqual(p.prefix_lengths.tolist(), [100, 100])
        self.assertEqual(p.max_seqlen_k, 105)
        # prefill_seqlen_k_int32 is rebuilt from input and prefix lengths.
        self.assertEqual(p.prefill_seqlen_k_int32.tolist(), [105, 105])

    def test_prefix_batch_size_mismatch_raises(self):
        """prefix_lengths batch size != expected_batch raises ValueError."""
        stub = self._make_stub(batch_size=4)
        inputs = self._make_attn_inputs(
            [5, 5, 5, 5],
            prefix_lengths=torch.tensor([10, 10], dtype=torch.int32),
        )
        with self.assertRaises(ValueError):
            self._call_update(stub, inputs)

    def test_missing_kv_block_id_raises(self):
        """Missing kv_cache block ids raises ValueError."""
        from types import SimpleNamespace

        stub = self._make_stub(batch_size=2)
        inputs = SimpleNamespace(
            input_lengths=torch.tensor([5, 5], dtype=torch.int32),
            prefix_lengths=None,
            cu_seqlens=None,
            cu_kv_seqlens=None,
            kv_cache_kernel_block_id_device=None,
            kv_cache_block_id_device=None,
        )
        with self.assertRaises(ValueError):
            self._call_update(stub, inputs)

    def test_replay_query_length_above_capture_capacity_raises(self):
        stub = self._make_stub(batch_size=2)
        stub.fmha_params.graph_query_length = 4
        stub.fmha_params.graph_token_q_capacity = 8
        inputs = self._make_attn_inputs(
            [5, 0], prefix_lengths=torch.tensor([100, 100], dtype=torch.int32)
        )

        with self.assertRaisesRegex(
            ValueError, "query length exceeds capture capacity"
        ):
            self._call_update(stub, inputs)


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillImplPaged module")
class TestAiterPrefillImplPagedCudaGraphDispatch(unittest.TestCase):
    """Unit tests for small-q dispatch under CUDA graph."""

    def _make_impl_with_mocked_prepare(
        self, input_lengths, is_cuda_graph, need_rope_kv_cache=False
    ):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        batch_params = SimpleNamespace(workspace_bytes=1024)
        triton_params = SimpleNamespace(workspace_bytes=2048)
        batch_impl = MagicMock()
        batch_impl.prepare.return_value = batch_params
        triton_impl = MagicMock()
        triton_impl.prepare.return_value = triton_params
        rope_impl = MagicMock()
        observed_pad_query = []

        def prepare_rope(_):
            observed_pad_query.append(rope_impl.pad_query)
            return object()

        rope_impl.prepare.side_effect = prepare_rope

        cfg = SimpleNamespace(
            need_rope_kv_cache=need_rope_kv_cache,
            kv_head_num=1,
            size_per_head=8,
            kernel_tokens_per_block=16,
        )
        attn_inputs = SimpleNamespace(
            is_cuda_graph=is_cuda_graph,
            input_lengths=torch.tensor(input_lengths, dtype=torch.int32),
        )
        module_path = "rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter"
        with patch(
            f"{module_path}.AiterPrefillAttnOpPaged", return_value=batch_impl
        ), patch(
            f"{module_path}.AiterPrefillAttnOpTriton", return_value=triton_impl
        ), patch(
            f"{module_path}.FusedRopeKVCachePrefillOpAsm", return_value=rope_impl
        ), patch(
            f"{module_path}.common.create_write_cache_store_impl"
        ):
            impl = AiterPrefillImplPaged(cfg, attn_inputs)

        return (
            impl,
            batch_impl,
            triton_impl,
            batch_params,
            triton_params,
            observed_pad_query,
        )

    def test_cuda_graph_prepares_only_selected_backend_workspace(self):
        cases = [([4, 1], "triton"), ([5, 1], "batch")]
        for input_lengths, expected_backend in cases:
            with self.subTest(expected_backend=expected_backend):
                (
                    impl,
                    batch_impl,
                    triton_impl,
                    batch_params,
                    triton_params,
                    _,
                ) = self._make_impl_with_mocked_prepare(input_lengths, True)

                self.assertEqual(impl.backend, expected_backend)
                if expected_backend == "triton":
                    triton_impl.prepare.assert_called_once_with(impl.attn_inputs)
                    batch_impl.prepare.assert_not_called()
                    self.assertIs(impl.triton_fmha_params, triton_params)
                    self.assertIsNone(impl.fmha_params)
                else:
                    batch_impl.prepare.assert_called_once_with(impl.attn_inputs)
                    triton_impl.prepare.assert_not_called()
                    self.assertIs(impl.fmha_params, batch_params)
                    self.assertIsNone(impl.triton_fmha_params)

    def test_eager_prepares_only_selected_backend_during_construction(self):
        impl, batch_impl, triton_impl, _, triton_params, _ = (
            self._make_impl_with_mocked_prepare([4, 1], False)
        )

        batch_impl.prepare.assert_not_called()
        triton_impl.prepare.assert_called_once_with(impl.attn_inputs)
        self.assertIs(impl.triton_fmha_params, triton_params)
        self.assertIsNone(impl.fmha_params)

    def test_capture_stride_is_requested_only_for_graph_triton_rope(self):
        cases = (
            ([4, 1], True, True, True),
            ([5, 1], True, True, False),
            ([4, 1], False, True, False),
            ([4, 1], True, False, False),
        )
        for input_lengths, is_cuda_graph, need_rope, expected in cases:
            with self.subTest(
                input_lengths=input_lengths,
                is_cuda_graph=is_cuda_graph,
                need_rope=need_rope,
            ):
                *_, observed_pad_query = self._make_impl_with_mocked_prepare(
                    input_lengths, is_cuda_graph, need_rope
                )
                self.assertEqual(observed_pad_query, [expected])

    def _make_stub(self, graph_prepared: bool):
        from types import SimpleNamespace

        calls = []
        fmha_params = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            max_seqlen_q=4,
            token_q_num=8,
            cuda_graph_prepared=graph_prepared,
        )
        batch_impl = SimpleNamespace(enable_cuda_graph=True)
        triton_impl = SimpleNamespace()
        triton_fmha_params = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            max_seqlen_q=4,
            token_q_num=8,
        )

        def batch_forward(qkv, kv_cache, params):
            calls.append("batch")
            return torch.full((8, 1), 1.0)

        def triton_forward(qkv, kv_cache, params):
            calls.append("triton")
            self.assertIs(params, triton_fmha_params)
            return torch.full((8, 1), 2.0)

        batch_impl.forward = batch_forward
        triton_impl.forward = triton_forward

        stub = AiterPrefillImplPaged.__new__(AiterPrefillImplPaged)
        stub.fmha_params = fmha_params
        stub.enable_cuda_graph = True
        stub.backend = "triton"
        stub.need_rope_kv_cache = False
        stub.write_cache_store_impl = None
        stub.attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
        stub.head_num_kv = 1
        stub.tokens_per_block = 16
        stub.head_dim = 1
        stub.batch_prefill_impl = batch_impl
        stub.triton_prefill_impl = triton_impl
        stub.triton_fmha_params = triton_fmha_params
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(1, 2, 1, 16, 1),
        )
        qkv = (torch.empty(8, 1, 1),)
        return stub, qkv, kv_cache, calls

    def test_cuda_graph_uses_triton_when_small_q_and_graph_prepared(self):
        stub, qkv, kv_cache, calls = self._make_stub(graph_prepared=True)

        out = AiterPrefillImplPaged.forward(stub, qkv, kv_cache)

        self.assertEqual(calls, ["triton"])
        self.assertTrue(torch.equal(out, torch.full((8, 1), 2.0)))

    def test_cuda_graph_uses_triton_when_small_q_even_without_batch_graph_workspace(
        self,
    ):
        stub, qkv, kv_cache, calls = self._make_stub(graph_prepared=False)

        out = AiterPrefillImplPaged.forward(stub, qkv, kv_cache)

        self.assertEqual(calls, ["triton"])
        self.assertTrue(torch.equal(out, torch.full((8, 1), 2.0)))

    def _run_cuda_graph_prepare(self, captured_backend):
        from types import SimpleNamespace

        calls = []
        fmha_params = object()
        triton_fmha_params = object()
        attn_inputs = object()

        def prepare_batch(params, inputs):
            calls.append(("prepare_batch", params, inputs))

        def prepare_triton(params, inputs):
            calls.append(("prepare_triton", params, inputs))

        def prepare_rope(inputs):
            calls.append(("prepare_rope", inputs))

        def refresh(params, inputs):
            calls.append(("refresh", params, inputs))

        stub = AiterPrefillImplPaged.__new__(AiterPrefillImplPaged)
        stub.backend = captured_backend
        stub.fmha_params = fmha_params
        stub.triton_fmha_params = triton_fmha_params
        stub.batch_prefill_impl = SimpleNamespace(prepare_cuda_graph=prepare_batch)
        stub.triton_prefill_impl = SimpleNamespace(prepare_cuda_graph=prepare_triton)
        stub.rope_params = SimpleNamespace(prepare_in_place=prepare_rope)
        stub._refresh_prefill_fmha_params_for_cuda_graph = refresh

        stub.prepare_cuda_graph(attn_inputs)
        return calls, fmha_params, triton_fmha_params, attn_inputs

    def test_cuda_graph_replay_refreshes_only_captured_backend(self):
        for captured_backend in ("triton", "batch"):
            with self.subTest(captured_backend=captured_backend):
                (
                    calls,
                    fmha_params,
                    triton_fmha_params,
                    attn_inputs,
                ) = self._run_cuda_graph_prepare(captured_backend)

                selected_params = (
                    triton_fmha_params if captured_backend == "triton" else fmha_params
                )
                self.assertEqual(
                    calls,
                    [
                        ("refresh", selected_params, attn_inputs),
                        (
                            f"prepare_{captured_backend}",
                            selected_params,
                            attn_inputs,
                        ),
                        ("prepare_rope", attn_inputs),
                    ],
                )


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillAttnOpTriton module")
class TestAiterPrefillAttnOpTritonCudaGraphWorkspace(unittest.TestCase):
    """Unit tests for CUDA graph workspace ownership."""

    def test_compact_indices_keep_capture_query_stride_during_replay(self):
        """Replay gathers from the fixed-width layout recorded at capture time."""
        from types import SimpleNamespace
        from unittest.mock import patch

        cfg = _make_attn_configs(head_num=4, head_num_kv=2, head_dim=8)
        op = AiterPrefillAttnOpTriton(cfg)
        fmha_params = SimpleNamespace(
            # Capture used four query rows per sequence. Replay accepted one
            # and three tokens, so its runtime max_seqlen_q shrank to three.
            graph_query_length=4,
            max_seqlen_q=3,
            token_q_num=4,
            cu_seqlens_q=torch.tensor([0, 1, 4, 4], dtype=torch.int32),
        )

        with patch.object(
            torch,
            "repeat_interleave",
            wraps=torch.repeat_interleave,
        ) as repeat_interleave:
            compact_indices = op._calc_compact_indices(fmha_params)

        # The padded query layout remains [batch, capture_query_length].
        # Valid rows are the right-aligned suffix of each four-row batch slot.
        self.assertEqual(compact_indices.tolist(), [3, 5, 6, 7])
        self.assertEqual(
            repeat_interleave.call_args.kwargs["output_size"],
            fmha_params.token_q_num,
        )

    def test_graph_workspace_buffers_are_allocated_on_fmha_params_without_scale_buffer(
        self,
    ):
        from types import SimpleNamespace

        cfg = _make_attn_configs(head_num=4, head_num_kv=2, head_dim=8)
        op = AiterPrefillAttnOpTriton(cfg)
        fmha_params = SimpleNamespace(token_q_num=8)

        op._allocate_graph_workspace(
            fmha_params,
            num_seqs=2,
            query_length=4,
            max_seq_len=16,
            attn_dtype=torch.float16,
            device=torch.device("cpu"),
        )

        self.assertEqual(fmha_params.attention_output.shape, (8, 4, 8))
        self.assertEqual(fmha_params.compact_output.shape, (8, 4, 8))
        self.assertEqual(fmha_params.exp_sums.shape, (2, 2, 1, 8))
        self.assertEqual(fmha_params.max_logits.shape, (2, 2, 1, 8))
        self.assertEqual(fmha_params.temporary_output.shape, (2, 2, 1, 8, 8))
        self.assertEqual(fmha_params.compact_indices.shape, (8,))
        self.assertEqual(fmha_params.graph_query_length, 4)
        self.assertEqual(fmha_params.graph_token_q_capacity, 8)
        self.assertFalse(hasattr(fmha_params, "kv_scale"))
        self.assertFalse(hasattr(op, "_graph_output"))

    def test_forward_uses_prepared_graph_workspace_without_allocation(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        cfg = _make_attn_configs(head_num=4, head_num_kv=2, head_dim=8)
        op = AiterPrefillAttnOpTriton(cfg)
        op.enable_cuda_graph = True
        query = torch.empty(8, 4, 8, dtype=torch.float16)
        workspace = {
            "output": torch.empty(8, 4, 8, dtype=torch.float16),
            "exp_sums": torch.empty(2, 2, 1, 8, dtype=torch.float32),
            "max_logits": torch.empty(2, 2, 1, 8, dtype=torch.float32),
            "temporary_output": torch.empty(2, 2, 1, 8, 8, dtype=torch.float16),
        }
        compact_output = torch.empty(8, 4, 8, dtype=torch.float16)
        fmha_params = SimpleNamespace(
            kv_cache_block_id_device=torch.zeros(2, 1, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
            prefill_seqlen_k_int32=torch.tensor([7, 6], dtype=torch.int32),
            max_seqlen_q=4,
            max_seqlen_k=7,
            token_q_num=8,
            kv_scale=None,
            attention_output=workspace["output"],
            exp_sums=workspace["exp_sums"],
            max_logits=workspace["max_logits"],
            temporary_output=workspace["temporary_output"],
            compact_indices=torch.arange(8, dtype=torch.int32),
            compact_output=compact_output,
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(1, 2, 2, 16, 8, dtype=torch.float16),
            kv_scale_base=None,
        )

        kernel_path = (
            "rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter."
            "_run_triton_paged_attention"
        )
        with patch.object(
            op,
            "_allocate_graph_workspace",
            side_effect=AssertionError("forward must not allocate graph workspace"),
        ), patch(kernel_path, return_value=workspace["output"]) as run_attention:
            out = op.forward((query,), kv_cache, fmha_params)

        self.assertEqual(out.data_ptr(), compact_output.data_ptr())
        call_workspace = run_attention.call_args.kwargs["workspace"]
        for name, tensor in workspace.items():
            self.assertIs(call_workspace[name], tensor)

    @unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
    def test_prepare_cuda_graph_refreshes_triton_lengths_in_place(self):
        from types import SimpleNamespace

        cfg = _make_attn_configs(head_num=4, head_num_kv=2, head_dim=8)
        op = AiterPrefillAttnOpTriton(cfg)
        device = torch.device("cuda")
        fmha_params = SimpleNamespace(
            cu_seqlens_q=torch.zeros(4, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.zeros(4, dtype=torch.int32, device=device),
            prefill_seqlen_k_int32=torch.full(
                (3,), -1, dtype=torch.int32, device=device
            ),
            compact_indices=torch.full((6,), -1, dtype=torch.int32, device=device),
            graph_device=device,
            token_q_num=6,
            max_seqlen_q=2,
            max_seqlen_k=9,
        )
        prefill_ptr = fmha_params.prefill_seqlen_k_int32.data_ptr()
        block_ids = torch.zeros(3, 2, dtype=torch.int32, device=device)
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([2, 1, 0], dtype=torch.int32),
            prefix_lengths=torch.tensor([5, 6, 7], dtype=torch.int32),
            kv_cache_kernel_block_id_device=block_ids,
        )

        # Replay refreshes metadata at the owning implementation layer before
        # the Triton operator derives its compact output indices.
        impl = AiterPrefillImplPaged.__new__(AiterPrefillImplPaged)
        impl._refresh_prefill_fmha_params_for_cuda_graph(fmha_params, attn_inputs)
        op.prepare_cuda_graph(fmha_params, attn_inputs)

        self.assertEqual(fmha_params.prefill_seqlen_k_int32.data_ptr(), prefill_ptr)
        self.assertEqual(fmha_params.cu_seqlens_q.cpu().tolist(), [0, 2, 3, 3])
        self.assertEqual(fmha_params.cu_seqlens_k.cpu().tolist(), [0, 7, 14, 21])
        self.assertEqual(fmha_params.prefill_seqlen_k_int32.cpu().tolist(), [7, 7, 7])
        self.assertEqual(fmha_params.max_seqlen_q, 2)
        self.assertEqual(fmha_params.max_seqlen_k, 7)
        self.assertEqual(fmha_params.token_q_num, 3)
        self.assertEqual(fmha_params.token_kv_num, 21)
        self.assertEqual(
            fmha_params.kv_cache_block_id_device.data_ptr(), block_ids.data_ptr()
        )
        self.assertEqual(fmha_params.compact_indices.cpu().tolist(), [0, 1, 3, 0, 0, 0])

    def test_triton_paged_attention_requires_caller_scale_when_kv_scale_base_exists(
        self,
    ):
        query = torch.empty(1, 1, 8, dtype=torch.float16)
        paged_kv_cache = torch.empty(1, 2, 1, 16, 8, dtype=torch.float16)
        kv_scale_base = torch.empty(1, dtype=torch.float32)
        seq_lens = torch.ones(1, dtype=torch.int32)
        block_tables = torch.zeros(1, 1, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "kv_scale_buf.*required"):
            _run_triton_paged_attention(
                query=query,
                paged_kv_cache=paged_kv_cache,
                kv_scale_base=kv_scale_base,
                num_seqs=1,
                query_length=1,
                seq_lens=seq_lens,
                block_tables_id_device=block_tables,
                max_seq_len=1,
                num_kv_heads=1,
                context_partition_size=256,
                linear_v=False,
                kv_scale_buf=None,
            )

    def test_triton_paged_attention_allows_empty_kv_scale_base_without_scale_buffer(
        self,
    ):
        from unittest.mock import patch

        query = torch.empty(1, 1, 8, dtype=torch.float16)
        paged_kv_cache = torch.empty(1, 2, 1, 16, 8, dtype=torch.float16)
        kv_scale_base = torch.empty(0, dtype=torch.float32)
        seq_lens = torch.ones(1, dtype=torch.int32)
        block_tables = torch.zeros(1, 1, dtype=torch.int32)

        with patch.object(torch.ops.aiter, "pa_decode_gluon") as kernel:
            _run_triton_paged_attention(
                query=query,
                paged_kv_cache=paged_kv_cache,
                kv_scale_base=kv_scale_base,
                num_seqs=1,
                query_length=1,
                seq_lens=seq_lens,
                block_tables_id_device=block_tables,
                max_seq_len=1,
                num_kv_heads=1,
                context_partition_size=256,
                linear_v=False,
                kv_scale_buf=None,
            )

        self.assertIsNone(kernel.call_args.args[11])
        self.assertIsNone(kernel.call_args.args[12])
        self.assertIsNone(kernel.call_args.args[13])


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterPrefillTritonCudaGraphNumerics(unittest.TestCase):
    """Real RoPE -> pa_decode_gluon variable-length graph replay regression."""

    CAPTURE_LENGTHS = [4, 4, 4]
    REPLAY_LENGTH_CASES = ([4, 4, 2], [2, 2, 2], [4, 2, 4])
    PREFIX_LENGTHS = [8, 8, 8]
    HEAD_NUM = 8
    HEAD_NUM_KV = 2
    HEAD_DIM = 128
    TOKENS_PER_BLOCK = 16

    def setUp(self):
        torch.manual_seed(11)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.block_table = torch.arange(
            len(self.CAPTURE_LENGTHS), dtype=torch.int32, device=self.device
        ).view(-1, 1)

    def _make_inputs(self, input_lengths, is_cuda_graph):
        inputs = _make_rope_prefill_inputs(input_lengths, self.device, self.dtype)
        inputs.is_cuda_graph = is_cuda_graph
        if is_cuda_graph:
            inputs.padding_offset = inputs.padding_offset.cpu().pin_memory()
        inputs.prefix_lengths = torch.tensor(
            self.PREFIX_LENGTHS, dtype=torch.int32, device="cpu"
        ).pin_memory()

        kv_lengths = [
            query_length + prefix_length
            for query_length, prefix_length in zip(input_lengths, self.PREFIX_LENGTHS)
        ]
        cu_kv = [0]
        for kv_length in kv_lengths:
            cu_kv.append(cu_kv[-1] + kv_length)
        inputs.cu_kv_seqlens_device = torch.tensor(
            cu_kv, dtype=torch.int32, device=self.device
        )

        inputs.kv_cache_kernel_block_id = self.block_table.cpu().pin_memory()
        inputs.kv_cache_kernel_block_id_device = self.block_table.clone()
        inputs.kv_cache_block_id_device = inputs.kv_cache_kernel_block_id_device
        return inputs

    def _make_qkv(self, token_num):
        query = torch.randn(
            token_num,
            self.HEAD_NUM,
            self.HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        key = torch.randn(
            token_num,
            self.HEAD_NUM_KV,
            self.HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        value = torch.randn_like(key)
        return _pack_qkv(query, key, value)

    def _cache_dtype(self, kv_cache_dtype):
        if kv_cache_dtype != KvCacheDataType.FP8:
            return self.dtype
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz

    def _make_cache(self, cache_snapshot, kv_cache_dtype):
        cache = LayerKVCache()
        cache.kv_cache_base = cache_snapshot.clone()
        cache.kv_scale_base = (
            torch.ones(
                cache_snapshot.shape[0],
                2 * self.HEAD_NUM_KV * self.TOKENS_PER_BLOCK,
                dtype=torch.float32,
                device=self.device,
            )
            if kv_cache_dtype == KvCacheDataType.FP8
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        return cache

    def _copy_replay_inputs_in_place(self, capture_inputs, replay_inputs):
        capture_inputs.input_lengths.copy_(replay_inputs.input_lengths)
        capture_inputs.prefix_lengths.copy_(replay_inputs.prefix_lengths)
        capture_inputs.cu_seqlens_device.copy_(replay_inputs.cu_seqlens_device)
        capture_inputs.cu_kv_seqlens_device.copy_(replay_inputs.cu_kv_seqlens_device)
        # Production graph capture owns this tensor in pinned host memory.
        # Leave its capture-time values stale: prepare_in_place must rebuild
        # padding_offset against the captured row stride.
        capture_inputs.kv_cache_kernel_block_id.copy_(
            replay_inputs.kv_cache_kernel_block_id
        )
        capture_inputs.kv_cache_kernel_block_id_device.copy_(
            replay_inputs.kv_cache_kernel_block_id_device
        )

    def _run_replay_case(self, kv_cache_dtype, replay_lengths):
        cfg = _make_rope_attn_configs(
            self.HEAD_NUM,
            self.HEAD_NUM_KV,
            self.HEAD_DIM,
            self.dtype,
            self.TOKENS_PER_BLOCK,
        )
        cfg.kv_cache_dtype = kv_cache_dtype
        cfg.max_seq_len = 64

        capture_inputs = self._make_inputs(self.CAPTURE_LENGTHS, True)
        self.assertFalse(capture_inputs.padding_offset.is_cuda)
        replay_inputs = self._make_inputs(replay_lengths, False)
        capture_tokens = sum(self.CAPTURE_LENGTHS)
        replay_tokens = sum(replay_lengths)
        static_qkv = self._make_qkv(capture_tokens)
        replay_qkv = self._make_qkv(replay_tokens)

        cache_dtype = self._cache_dtype(kv_cache_dtype)
        cache_snapshot = torch.randn(
            len(self.CAPTURE_LENGTHS),
            2,
            self.HEAD_NUM_KV,
            self.TOKENS_PER_BLOCK,
            self.HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        ).to(cache_dtype)

        eager_cache = self._make_cache(cache_snapshot, kv_cache_dtype)
        eager_impl = AiterPrefillImplPaged(cfg, replay_inputs)
        expected = eager_impl.forward(replay_qkv, eager_cache, layer_idx=0).clone()

        graph_cache = self._make_cache(cache_snapshot, kv_cache_dtype)
        graph_impl = AiterPrefillImplPaged(cfg, capture_inputs)
        self.assertEqual(graph_impl.backend, "triton")

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            graph_impl.forward(static_qkv, graph_cache, layer_idx=0)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph_cache.kv_cache_base.copy_(cache_snapshot)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = graph_impl.forward(static_qkv, graph_cache, layer_idx=0)

        static_qkv[:replay_tokens].copy_(replay_qkv)
        graph_cache.kv_cache_base.copy_(cache_snapshot)
        self._copy_replay_inputs_in_place(capture_inputs, replay_inputs)
        graph_impl.prepare_cuda_graph(capture_inputs)
        graph.replay()
        torch.cuda.synchronize()

        for block_idx in range(cache_snapshot.shape[0]):
            with self.subTest(kv_cache_dtype=kv_cache_dtype, cache_block_idx=block_idx):
                self.assertTrue(
                    torch.equal(
                        graph_cache.kv_cache_base[block_idx],
                        eager_cache.kv_cache_base[block_idx],
                    ),
                    "CUDA graph and eager RoPE wrote different KV cache values",
                )

        actual = graph_output[:replay_tokens].reshape(replay_tokens, -1)
        tolerance = 0.1 if kv_cache_dtype == KvCacheDataType.FP8 else 0.02
        for token_idx in range(replay_tokens):
            with self.subTest(kv_cache_dtype=kv_cache_dtype, token_idx=token_idx):
                torch.testing.assert_close(
                    actual[token_idx],
                    expected[token_idx],
                    atol=tolerance,
                    rtol=tolerance,
                )

    def test_variable_length_replays_match_eager_bf16_and_fp8(self):
        for replay_lengths in self.REPLAY_LENGTH_CASES:
            for kv_cache_dtype in (KvCacheDataType.BASE, KvCacheDataType.FP8):
                with self.subTest(
                    replay_lengths=replay_lengths, kv_cache_dtype=kv_cache_dtype
                ):
                    self._run_replay_case(kv_cache_dtype, replay_lengths)


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillAttnOp module")
class TestCompactGatherReshape(unittest.TestCase):
    """Regression tests for _gather_and_reshape_kv_compact and block_table sanitize/pad.

    Validates that the compact gather path produces the same K/V layout as
    _reshape_kv_cache_vectorized for the referenced blocks, and that the
    block_table sanitize/pad logic correctly fills padding columns.

    These tests run on CPU (no aiter kernel needed) — they only exercise the
    tensor reshape / gather / sanitize logic. End-to-end kernel coverage
    (including the actual mha_batch_prefill_func call with real kv_cache_base
    and block_table) is provided by ROCm smoke tests that exercise the full
    prefill-with-prefix path on GPU.
    """

    def _make_op(
        self,
        head_num_kv=4,
        head_dim=128,
        tokens_per_block=16,
        kv_cache_dtype=None,
    ):
        cfg = _make_attn_configs(
            head_num=8,
            head_num_kv=head_num_kv,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
        )
        if kv_cache_dtype is not None:
            cfg.kv_cache_dtype = kv_cache_dtype
        return AiterPrefillAttnOp(cfg, v1_kv_layout=True)

    def _make_kv_cache_5d(self, num_blocks, hk, ps, hd, dtype=torch.float16):
        """Create a 5D KV cache: [num_blocks, 2, hk, ps, hd]."""
        return torch.randn(num_blocks, 2, hk, ps, hd, dtype=dtype)

    def _make_kv_cache_2d(self, num_blocks, hk, ps, hd, dtype=torch.float16):
        """Create a 2D flat KV cache: [num_blocks, 2*hk*ps*hd]."""
        return torch.randn(num_blocks, 2 * hk * ps * hd, dtype=dtype)

    def _make_compact_bufs(self, block_table, hk, ps, hd, dtype=torch.float16):
        """Build block_indices, compact_block_table, and compact K/V buffers."""
        block_indices = block_table.reshape(-1).to(torch.int64)
        num_gathered = block_indices.numel()
        compact_block_table = torch.arange(
            num_gathered, dtype=torch.int32, device=block_table.device
        ).view_as(block_table)
        vs = 16 // torch.tensor([], dtype=dtype).element_size()
        n = num_gathered + 1
        k_buf = torch.zeros(
            (n, hk, hd // vs, ps, vs), dtype=dtype, device=block_table.device
        )
        v_buf = torch.zeros(
            (n, hk, ps // vs, hd, vs), dtype=dtype, device=block_table.device
        )
        return block_indices, compact_block_table, k_buf, v_buf

    def _assert_compact_equiv(self, op, kv_cache, block_table):
        """Assert compact gather plus remap equals full reshape indexed by block_table."""
        k_full, v_full = op._reshape_kv_cache_vectorized(kv_cache)
        block_indices, compact_bt, k_buf, v_buf = self._make_compact_bufs(
            block_table,
            op.head_num_kv,
            op.tokens_per_block,
            op.head_dim,
            kv_cache.dtype,
        )
        k_compact, v_compact = op._gather_and_reshape_kv_compact(
            kv_cache, block_indices, k_buf, v_buf
        )

        # Remapped compact K/V should produce the same per-table K/V as the
        # original full K/V indexed by the original block_table.
        flat_bt = compact_bt.reshape(-1).to(torch.int64)
        orig_indices = block_table.reshape(-1).to(torch.int64)
        torch.testing.assert_close(k_compact[flat_bt], k_full[orig_indices])
        torch.testing.assert_close(v_compact[flat_bt], v_full[orig_indices])
        # Compact buffer has all referenced blocks + 1 trailing dummy zero-block
        # for CK speculative prefetch safety (no dedup since torch.unique removed).
        self.assertEqual(k_compact.shape[0], orig_indices.numel() + 1)

    def test_kv_cache_dtype_controls_compact_mode(self):
        cases = [
            (KvCacheDataType.BASE, torch.float16, True),
            (KvCacheDataType.FP8, torch.float8_e4m3fn, False),
        ]
        for kv_cache_dtype, expected_torch_dtype, expected_use_compact in cases:
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                op = self._make_op(kv_cache_dtype=kv_cache_dtype)
                self.assertEqual(op.kv_cache_torch_dtype, expected_torch_dtype)
                self.assertEqual(op.use_compact, expected_use_compact)

    # ---- 5D cache path ----------------------------------------------------

    def test_5d_single_batch(self):
        op = self._make_op()
        kv = self._make_kv_cache_5d(32, 4, 16, 128)
        bt = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    def test_5d_multi_batch(self):
        op = self._make_op()
        kv = self._make_kv_cache_5d(64, 4, 16, 128)
        bt = torch.tensor([[0, 5, 10], [1, 6, 11]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    def test_5d_repeated_blocks(self):
        """Same block referenced multiple times (e.g. shared prefix)."""
        op = self._make_op()
        kv = self._make_kv_cache_5d(16, 4, 16, 128)
        bt = torch.tensor([[0, 0, 1], [0, 2, 2]], dtype=torch.int32)
        block_indices, compact_bt, k_buf, v_buf = self._make_compact_bufs(
            bt, 4, 16, 128
        )
        k_compact, v_compact = op._gather_and_reshape_kv_compact(
            kv, block_indices, k_buf, v_buf
        )
        k_full, _ = op._reshape_kv_cache_vectorized(kv)
        orig_indices = bt.reshape(-1).to(torch.int64)
        torch.testing.assert_close(
            k_compact[compact_bt.reshape(-1).to(torch.int64)], k_full[orig_indices]
        )
        self.assertEqual(k_compact.shape[0], bt.numel() + 1)

    def test_5d_non_contiguous_blocks(self):
        """Block indices are sparse across a large pool."""
        op = self._make_op()
        kv = self._make_kv_cache_5d(1024, 4, 16, 128)
        bt = torch.tensor([[3, 500, 1023]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    # ---- 2D flat cache path -----------------------------------------------

    def test_2d_single_batch(self):
        op = self._make_op()
        kv = self._make_kv_cache_2d(32, 4, 16, 128)
        bt = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    def test_2d_multi_batch(self):
        op = self._make_op()
        kv = self._make_kv_cache_2d(64, 4, 16, 128)
        bt = torch.tensor([[0, 5, 10], [1, 6, 11]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    def test_2d_repeated_blocks(self):
        op = self._make_op()
        kv = self._make_kv_cache_2d(16, 4, 16, 128)
        bt = torch.tensor([[0, 0, 1], [0, 2, 2]], dtype=torch.int32)
        block_indices, compact_bt, k_buf, v_buf = self._make_compact_bufs(
            bt, 4, 16, 128
        )
        k_compact, v_compact = op._gather_and_reshape_kv_compact(
            kv, block_indices, k_buf, v_buf
        )
        k_full, _ = op._reshape_kv_cache_vectorized(kv)
        orig_indices = bt.reshape(-1).to(torch.int64)
        torch.testing.assert_close(
            k_compact[compact_bt.reshape(-1).to(torch.int64)], k_full[orig_indices]
        )
        self.assertEqual(k_compact.shape[0], bt.numel() + 1)

    def test_2d_oversized_stride_truncates_to_prefix(self):
        op = self._make_op()
        hk, ps, hd = 4, 16, 128
        exact = self._make_kv_cache_2d(8, hk, ps, hd)
        padded = torch.cat(
            [exact, torch.full((exact.shape[0], 32), 999.0, dtype=exact.dtype)], dim=1
        )
        k_exact, v_exact = op._reshape_kv_cache_vectorized(exact)
        k_pad, v_pad = op._reshape_kv_cache_vectorized(padded)
        torch.testing.assert_close(k_pad, k_exact)
        torch.testing.assert_close(v_pad, v_exact)

    def test_2d_linear_v_permute_element_order(self):
        op = self._make_op(head_num_kv=1, head_dim=8, tokens_per_block=8)
        ps = hd = 8
        kv = torch.arange(2 * ps * hd, dtype=torch.bfloat16).reshape(1, -1)
        vs = 16 // kv.element_size()
        _, actual = op._reshape_kv_cache_vectorized(kv)
        v_linear = kv[0, ps * hd :].reshape(hd, ps)
        j, h, w = torch.meshgrid(
            torch.arange(ps // vs), torch.arange(hd), torch.arange(vs), indexing="ij"
        )
        expected = v_linear[h, j * vs + w].reshape(1, 1, ps // vs, hd, vs)
        torch.testing.assert_close(actual, expected)

    # ---- FP8 fallback: compact should NOT be used -------------------------

    def test_fp8_uses_full_reshape(self):
        """When kv_cache is FP8, _forward_paged should use the full reshape path."""
        from types import SimpleNamespace

        op = self._make_op(kv_cache_dtype=KvCacheDataType.FP8)
        fp8_dtype = torch.float8_e4m3fn
        kv = torch.randn(16, 2, 4, 16, 128, dtype=torch.float16).to(fp8_dtype)
        q = torch.randn(4, 8, 128, dtype=torch.float16)
        block_table = torch.tensor([[0]], dtype=torch.int32)
        fmha_params = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            prefill_seqlen_k_int32=torch.tensor([4], dtype=torch.int32),
            max_seqlen_q=4,
            max_seqlen_k=4,
            token_q_num=4,
            sanitized_block_table=block_table,
            compact_block_table=torch.tensor([[0]], dtype=torch.int32),
            block_indices=block_table.reshape(-1).to(torch.int64),
            k_compact_buf=None,
            v_compact_buf=None,
        )
        kv_cache = SimpleNamespace(kv_cache_base=kv)
        expected = torch.zeros(4, 8, 128, dtype=torch.float16)

        def fake_full_reshape(kv_cache_base):
            self.assertIs(kv_cache_base, kv)
            return expected, expected

        def fake_prefill(query, k_cache, v_cache, *args, **kwargs):
            self.assertIs(k_cache, expected)
            self.assertIs(v_cache, expected)
            self.assertIs(kwargs["block_table"], block_table)
            return torch.zeros(4, 8, 128, dtype=torch.float16)

        prefill_func = (
            "rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter."
            "aiter.mha_batch_prefill_func"
        )

        with patch.object(
            op, "_gather_and_reshape_kv_compact", side_effect=AssertionError
        ), patch.object(
            op, "_reshape_kv_cache_vectorized", side_effect=fake_full_reshape
        ) as full_reshape, patch(
            prefill_func, side_effect=fake_prefill
        ):
            op._forward_paged(q, kv_cache, fmha_params)

        self.assertFalse(op.use_compact)
        self.assertEqual(full_reshape.call_count, 1)

    # ---- block table sanitization ------------------------------------------

    def test_sanitize_block_table_fills_padding_with_last_valid(self):
        """Padding columns are filled with last-valid-block-id per row.

        Valid-mask entries are left untouched (fail-fast for truly invalid ids).
        The helper also pads columns for CK speculative prefetch.
        """
        op = self._make_op()  # tokens_per_block=16
        bt = torch.tensor([[3, -1, 99, 5], [7, 8, 9, 10]], dtype=torch.int32)
        seqlen_k = torch.tensor([16, 33], dtype=torch.int32)
        # Row 0: valid_blocks=ceil(16/16)=1 → only col0 is valid, rest filled with bt[0,0]=3
        # Row 1: valid_blocks=ceil(33/16)=3 → cols 0-2 valid, col3 filled with bt[1,2]=9
        sanitized = op._sanitize_block_table(bt, seqlen_k=seqlen_k, max_seqlen_k=33)
        # Check the first 4 columns (original width) for sanitize correctness.
        first_4 = sanitized[:, :4].tolist()
        self.assertEqual(first_4, [[3, 3, 3, 3], [7, 8, 9, 9]])
        # Additional pad columns should all be filled with last-valid-block-id.
        if sanitized.shape[1] > 4:
            for row_idx, expected_fill in enumerate([3, 9]):
                pad_vals = sanitized[row_idx, 4:].tolist()
                self.assertTrue(
                    all(v == expected_fill for v in pad_vals),
                    f"Row {row_idx} pad columns should all be {expected_fill}, got {pad_vals}",
                )

    # ---- different head_dim / tokens_per_block configs --------------------

    def test_5d_small_head_dim(self):
        op = self._make_op(head_num_kv=2, head_dim=64, tokens_per_block=8)
        kv = self._make_kv_cache_5d(32, 2, 8, 64)
        bt = torch.tensor([[0, 3, 7], [1, 4, 8]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)

    def test_2d_small_head_dim(self):
        op = self._make_op(head_num_kv=2, head_dim=64, tokens_per_block=8)
        kv = self._make_kv_cache_2d(32, 2, 8, 64)
        bt = torch.tensor([[0, 3, 7], [1, 4, 8]], dtype=torch.int32)
        self._assert_compact_equiv(op, kv, bt)


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillAttnOp module")
class TestPagedPrefillKernelE2E(unittest.TestCase):
    """End-to-end regression for AiterPrefillAttnOpPaged.forward.

    Constructs real kv_cache_base (5D paged layout) and block_table with
    padding columns, then calls mha_batch_prefill_func through the operator's
    forward() method. Verifies the output against a torch SDPA reference
    computed from the same K/V data unpacked from the paged cache.

    This covers the sanitize+pad block_table logic together with the actual
    CK batch prefill kernel execution on GPU.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    @staticmethod
    def _prefix_causal_sdpa_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_lengths: List[int],
        prefix_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Per-sequence SDPA reference with prefix-cache causal mask.

        For prefix-cache prefill, Q token i (0-indexed within the input chunk)
        sits at KV position (prefix_len + i). It can attend to all KV positions
        j where j <= prefix_len + i. This is equivalent to a causal mask with
        Q offset = prefix_len.
        """
        repeat = head_num // head_num_kv
        scale = 1.0 / math.sqrt(head_dim)
        out_chunks = []
        q_offset = 0
        k_offset = 0
        for seq_idx, (q_len, p_len) in enumerate(zip(input_lengths, prefix_lengths)):
            kv_len = q_len + p_len
            q_seq = q[q_offset : q_offset + q_len]  # [q_len, H_q, D]
            k_seq = k[k_offset : k_offset + kv_len]  # [kv_len, H_kv, D]
            v_seq = v[k_offset : k_offset + kv_len]  # [kv_len, H_kv, D]

            # Transpose to [H, seq_len, D]
            q_h = q_seq.transpose(0, 1)  # [H_q, q_len, D]
            k_h = k_seq.transpose(0, 1)  # [H_kv, kv_len, D]
            v_h = v_seq.transpose(0, 1)  # [H_kv, kv_len, D]

            if repeat > 1:
                k_h = k_h.repeat_interleave(repeat, dim=0)
                v_h = v_h.repeat_interleave(repeat, dim=0)

            # Build prefix-causal attention mask: Q[i] attends to K[j] where j <= p_len + i
            # i.e. for each Q position i, the valid KV range is [0, p_len + i].
            q_positions = (
                torch.arange(q_len, device=q.device).unsqueeze(1) + p_len
            )  # [q_len, 1]
            k_positions = torch.arange(kv_len, device=q.device).unsqueeze(
                0
            )  # [1, kv_len]
            # mask[i, j] = True means BLOCKED (will be set to -inf)
            causal_mask = k_positions > q_positions  # [q_len, kv_len]

            # Compute attention: [H_q, q_len, D] x [H_q, D, kv_len] -> [H_q, q_len, kv_len]
            attn_weights = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
            attn_weights.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            # [H_q, q_len, kv_len] x [H_q, kv_len, D] -> [H_q, q_len, D]
            attn_out = torch.matmul(attn_weights, v_h)
            # Transpose back to [q_len, H_q, D] -> [q_len, H_q*D]
            attn_out = attn_out.transpose(0, 1).reshape(q_len, head_num * head_dim)
            out_chunks.append(attn_out)

            q_offset += q_len
            k_offset += kv_len

        return torch.cat(out_chunks, dim=0)

    def _run_paged_prefill_e2e(
        self,
        batch_size: int,
        input_lengths: List[int],
        prefix_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
        tokens_per_block: int,
    ):
        """Build real paged KV cache, run AiterPrefillAttnOpPaged.forward, compare to SDPA ref.

        Strategy: randomly initialize kv_cache_base, then extract logical K/V
        from it using the same vectorized view the kernel uses. This guarantees
        the reference computes attention on exactly the same data the kernel sees.
        """
        device = self.device
        dtype = self.dtype

        # Compute derived lengths
        kv_lengths = [il + pl for il, pl in zip(input_lengths, prefix_lengths)]
        total_q_tokens = sum(input_lengths)
        max_kv_len = max(kv_lengths)
        blocks_per_seq = (max_kv_len + tokens_per_block - 1) // tokens_per_block

        # Vectorization factor
        x = 16 // torch.tensor(0, dtype=dtype).element_size()  # fp16: x=8

        # Allocate paged KV cache pool with random data
        num_pool_blocks = batch_size * blocks_per_seq + 8
        kv_cache_base = torch.randn(
            num_pool_blocks,
            2,
            head_num_kv,
            tokens_per_block,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # Build block_table with extra padding columns (-1) to exercise sanitize.
        bt_cols = blocks_per_seq + 2
        block_table = torch.full(
            (batch_size, bt_cols), -1, dtype=torch.int32, device=device
        )
        block_offset = 0
        for b in range(batch_size):
            num_valid_blocks = (
                kv_lengths[b] + tokens_per_block - 1
            ) // tokens_per_block
            for col in range(num_valid_blocks):
                block_table[b, col] = block_offset + col
            block_offset += num_valid_blocks

        # Extract logical K/V from paged cache using the kernel's vectorized view.
        #
        # forward() does:
        #   k_raw = kv_cache_base.select(1, 0)  → [N, hk, ps, hd] contiguous
        #   k_vec = k_raw.view(N, hk, hd//x, ps, x)
        # The kernel interprets k_vec[h, a, b, c] as K[head=h, token=b, dim=a*x+c].
        #
        #   v_raw = kv_cache_base.select(1, 1)  → [N, hk, ps, hd] contiguous
        #   v_vec = v_raw.view(N, hk, ps//x, hd, x)
        # The kernel interprets v_vec[h, a, b, c] as V[head=h, token=a*x+c, dim=b].
        all_k_flat = []
        all_v_flat = []
        for b in range(batch_size):
            kv_len = kv_lengths[b]
            num_valid_blocks = (kv_len + tokens_per_block - 1) // tokens_per_block
            k_tokens = []
            v_tokens = []
            for blk_idx in range(num_valid_blocks):
                block_id = block_table[b, blk_idx].item()
                tok_start = blk_idx * tokens_per_block
                tok_end = min(tok_start + tokens_per_block, kv_len)
                num_toks = tok_end - tok_start

                # K: k_vec[h, a, b, c] = K[h, token=b, dim=a*x+c]
                # Read logical K[h, t, d] = k_vec[h, d//x, t, d%x]
                k_raw = kv_cache_base[block_id, 0]  # [hk, ps, hd] contiguous
                k_vec = k_raw.view(head_num_kv, head_dim // x, tokens_per_block, x)
                # k_vec shape: [hk, hd//x, ps, x] → permute to [hk, ps, hd//x, x] → reshape [hk, ps, hd]
                k_logical = k_vec.permute(0, 2, 1, 3).reshape(
                    head_num_kv, tokens_per_block, head_dim
                )
                k_tokens.append(
                    k_logical[:, :num_toks, :].permute(1, 0, 2)
                )  # [num_toks, hk, hd]

                # V: v_vec[h, a, b, c] = V[h, token=a*x+c, dim=b]
                # Read logical V[h, t, d] = v_vec[h, t//x, d, t%x]
                v_raw = kv_cache_base[block_id, 1]  # [hk, ps, hd] contiguous
                v_vec = v_raw.view(head_num_kv, tokens_per_block // x, head_dim, x)
                # v_vec shape: [hk, ps//x, hd, x] → permute to [hk, ps//x, x, hd] → reshape [hk, ps, hd]
                v_logical = v_vec.permute(0, 1, 3, 2).reshape(
                    head_num_kv, tokens_per_block, head_dim
                )
                v_tokens.append(
                    v_logical[:, :num_toks, :].permute(1, 0, 2)
                )  # [num_toks, hk, hd]

            all_k_flat.append(torch.cat(k_tokens, dim=0))
            all_v_flat.append(torch.cat(v_tokens, dim=0))

        # Generate Q tokens (only input_lengths, not prefix)
        q_flat = torch.randn(
            total_q_tokens, head_num, head_dim, dtype=dtype, device=device
        )

        # Build operator
        cfg = _make_attn_configs(head_num, head_num_kv, head_dim, tokens_per_block)
        op = AiterPrefillAttnOpPaged(cfg)

        # Construct cu_seqlens
        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for b in range(batch_size):
            cu_seqlens_q[b + 1] = cu_seqlens_q[b] + input_lengths[b]
            cu_seqlens_k[b + 1] = cu_seqlens_k[b] + kv_lengths[b]

        # Build a minimal FMHAParams-like object
        class _FakeParams:
            pass

        params = _FakeParams()
        params.cu_seqlens_q = cu_seqlens_q
        params.cu_seqlens_k = cu_seqlens_k
        params.max_seqlen_q = max(input_lengths)
        params.max_seqlen_k = max_kv_len
        params.token_q_num = total_q_tokens
        params.kv_cache_block_id_device = block_table

        # Build a minimal kv_cache object
        class _FakeKVCache:
            pass

        kv_cache = _FakeKVCache()
        kv_cache.kv_cache_base = kv_cache_base

        # Run AiterPrefillAttnOpPaged.forward
        qkv = (q_flat,)
        actual = op.forward(qkv, kv_cache, params)

        # Compute prefix-causal SDPA reference from the original flat K/V.
        k_all = torch.cat(all_k_flat, dim=0)
        v_all = torch.cat(all_v_flat, dim=0)
        ref = self._prefix_causal_sdpa_reference(
            q_flat,
            k_all,
            v_all,
            input_lengths,
            prefix_lengths,
            head_num,
            head_num_kv,
            head_dim,
        )

        # Numerical regression: kernel output must match reference within fp16 tolerance.
        self.assertFalse(torch.isnan(actual).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(actual).any(), "Output contains Inf")
        self.assertEqual(actual.shape, (total_q_tokens, head_num * head_dim))
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    def test_single_batch_with_prefix(self):
        """Single sequence with prefix cache — simplest paged prefill case."""
        self._run_paged_prefill_e2e(
            batch_size=1,
            input_lengths=[16],
            prefix_lengths=[32],
            head_num=8,
            head_num_kv=4,
            head_dim=64,
            tokens_per_block=16,
        )

    def test_multi_batch_varied_lengths(self):
        """Multiple sequences with different prefix/input lengths."""
        self._run_paged_prefill_e2e(
            batch_size=3,
            input_lengths=[8, 24, 12],
            prefix_lengths=[16, 48, 0],
            head_num=8,
            head_num_kv=4,
            head_dim=128,
            tokens_per_block=16,
        )

    def test_unaligned_seq_triggers_padding(self):
        """Sequence length not aligned to tokens_per_block — exercises block_table padding."""
        self._run_paged_prefill_e2e(
            batch_size=2,
            input_lengths=[7, 13],
            prefix_lengths=[19, 5],
            head_num=8,
            head_num_kv=8,
            head_dim=64,
            tokens_per_block=16,
        )

    def test_large_prefix_many_blocks(self):
        """Long prefix spanning many blocks — stresses sanitize+pad column expansion."""
        self._run_paged_prefill_e2e(
            batch_size=1,
            input_lengths=[4],
            prefix_lengths=[128],
            head_num=8,
            head_num_kv=4,
            head_dim=64,
            tokens_per_block=16,
        )


# ============================================================================
# no-cache RoPE wrapper regression — kv_cache=None + need_rope_kv_cache=True
# ============================================================================


class _FakeRopeKvCachePrefillOp:
    def __init__(self, output):
        self.calls = []
        self.output = output

    def forward(self, qkv, kv_cache, rope_params):
        self.calls.append((qkv, kv_cache, rope_params))
        return self.output


class _FakeFmhaOp:
    def __init__(self):
        self.calls = []
        self.output = object()

    def forward(self, fmha_input, kv_cache, fmha_params):
        self.calls.append((fmha_input, kv_cache, fmha_params))
        return self.output


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterPrefillImplNoKvRopeWrapper(unittest.TestCase):
    """Without a KV cache, embedding-style prefill still needs RoPE applied to
    Q/K. Both ASM and NonASM wrappers must call rope_kvcache_impl before fmha_impl
    and pass the RoPE output straight through — bypassing the prior shortcut
    that fed raw QKV into FMHA.

    Real RoPE/FMHA kernels are covered by TestAiterPrefillImplNoKvRopeRealOp
    below; this class isolates the wrapper logic with fakes so it runs anywhere.
    """

    def _check_no_kv_rope_path(self, impl_cls):
        impl = object.__new__(impl_cls)
        impl.need_rope_kv_cache = True
        qkv = torch.randn(4, 16)
        rope_output = (
            torch.randn(4, 2, 4),
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
        )
        impl.rope_kvcache_impl = _FakeRopeKvCachePrefillOp(rope_output)
        impl.fmha_impl = _FakeFmhaOp()
        impl.rope_params = object()
        impl.fmha_params = object()

        actual = impl.forward(qkv, kv_cache=None, layer_idx=0)

        self.assertIs(actual, impl.fmha_impl.output)
        self.assertEqual(len(impl.rope_kvcache_impl.calls), 1)
        self.assertEqual(len(impl.fmha_impl.calls), 1)
        self.assertEqual(impl.rope_kvcache_impl.calls[0], (qkv, None, impl.rope_params))
        self.assertEqual(
            impl.fmha_impl.calls[0],
            (impl.rope_kvcache_impl.output, None, impl.fmha_params),
        )

    def test_asm_no_kv_cache_still_applies_rope(self):
        self._check_no_kv_rope_path(AiterPrefillImplAsm)

    def test_nonasm_no_kv_cache_still_applies_rope(self):
        self._check_no_kv_rope_path(AiterPrefillImplNonAsm)


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterPrefillImplNoKvRopeRealOp(unittest.TestCase):
    """End-to-end numerical regression for the wrapper path with the real C++
    FusedRopeKVCachePrefillOp + AiterPrefillAttnOp on ROCm. Exercises both ASM
    and NonASM wrappers with varied-length GQA batches and asserts the output
    matches RoPE(Q,K) → flash attention reference."""

    def setUp(self):
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _check_real_no_kv_rope_path(self, impl_cls):
        input_lengths = [5, 3]
        head_num = 4
        head_num_kv = 2
        head_dim = 64
        cfg = _make_rope_attn_configs(head_num, head_num_kv, head_dim, dtype=self.dtype)
        attn_inputs = _make_rope_prefill_inputs(input_lengths, self.device, self.dtype)
        impl = impl_cls(cfg, attn_inputs)

        total_tokens = sum(input_lengths)
        q = torch.randn(
            total_tokens, head_num, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        qkv = _pack_qkv(q, k, v)

        actual = impl.forward(qkv, kv_cache=None, layer_idx=0)
        q_rope, k_rope = _apply_base_rope(q, k, input_lengths)
        ref = _sdpa_reference(
            q_rope,
            k_rope,
            v,
            attn_inputs.cu_seqlens_device,
            attn_inputs.cu_kv_seqlens_device,
            causal=True,
        )
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    def test_asm_no_kv_rope_real_op_matches_reference(self):
        self._check_real_no_kv_rope_path(AiterPrefillImplAsm)

    def test_nonasm_no_kv_rope_real_op_matches_reference(self):
        self._check_real_no_kv_rope_path(AiterPrefillImplNonAsm)


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestFusedRopeKVCacheMropeContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.head_num = 4
        self.head_num_kv = 2
        self.head_dim = 256
        self.rope_dim = 64
        self.sections = (11, 11, 10)
        self.tokens_per_block = 16

    def _make_config(self):
        return _make_mrope_attn_configs(
            head_num=self.head_num,
            head_num_kv=self.head_num_kv,
            head_dim=self.head_dim,
            dtype=self.dtype,
            rope_dim=self.rope_dim,
            mrope_sections=self.sections,
            tokens_per_block=self.tokens_per_block,
        )

    def test_prefill_and_decode_reject_section_sum_mismatch(self):
        op_classes = (
            FusedRopeKVCachePrefillOpAsm,
            FusedRopeKVCachePrefillOpNonAsm,
            FusedRopeKVCacheDecodeOpAsm,
            FusedRopeKVCacheDecodeOpNonAsm,
        )
        for op_class in op_classes:
            cfg = self._make_config()
            cfg.rope_config.mrope_dim3 += 1
            with self.subTest(op_class=op_class.__name__):
                with self.assertRaisesRegex(RuntimeError, "section sum"):
                    op_class(cfg)

    def test_rejects_invalid_axis_count_and_interleaved_capacity(self):
        cfg = self._make_config()
        cfg.rope_config.index_factor = 4
        with self.assertRaisesRegex(RuntimeError, "index_factor=3"):
            FusedRopeKVCacheDecodeOpAsm(cfg)

        # Although 12+12+8 is 32 pairs, H=12 cannot fit the 11 H slots in a
        # 32-pair THW-interleaved region. This is why the review's proposed
        # contiguous 12/12/8 fix is not valid for mrope_interleaved=true.
        cfg = _make_mrope_attn_configs(
            self.head_num,
            self.head_num_kv,
            self.head_dim,
            self.dtype,
            rope_dim=self.rope_dim,
            mrope_sections=(12, 12, 8),
        )
        with self.assertRaisesRegex(RuntimeError, "interleaved H/W capacity"):
            FusedRopeKVCachePrefillOpAsm(cfg)

    def _build_decode_case(self, op_class):
        sequence_lengths = [5, 3]
        position_ids = torch.tensor(
            [[5, 2, 1], [3, 7, 2]], dtype=torch.int32, device=self.device
        )
        attn_inputs, per_batch_block_ids, num_blocks = _make_mrope_decode_inputs(
            sequence_lengths,
            position_ids,
            self.device,
            self.dtype,
            self.tokens_per_block,
        )
        cfg = self._make_config()
        op = op_class(cfg)
        params = op.prepare(attn_inputs)
        layer_cache, pool = _alloc_decode_kv_cache(
            num_blocks,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
            self.dtype,
            self.device,
        )
        q = torch.randn(
            len(sequence_lengths),
            self.head_num,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        k = torch.randn(
            len(sequence_lengths),
            self.head_num_kv,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.randn_like(k)
        return (
            op,
            params,
            attn_inputs,
            layer_cache,
            pool,
            per_batch_block_ids,
            sequence_lengths,
            q,
            k,
            v,
        )

    def _assert_decode_matches_reference(self, op_class):
        (
            op,
            params,
            attn_inputs,
            layer_cache,
            pool,
            per_batch_block_ids,
            sequence_lengths,
            q,
            k,
            v,
        ) = self._build_decode_case(op_class)

        actual_q = op.forward(_pack_qkv(q, k, v), layer_cache, params)
        expected_q, expected_k = _apply_mrope(
            q,
            k,
            attn_inputs.combo_position_ids,
            self.sections,
            self.rope_dim,
        )
        actual_k = _read_decode_k_from_pool(
            pool,
            per_batch_block_ids,
            sequence_lengths,
            self.head_num_kv,
            self.head_dim,
            self.tokens_per_block,
        )
        torch.testing.assert_close(actual_q, expected_q, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_k, expected_k, atol=1e-2, rtol=1e-2)

    def test_asm_decode_q_and_k_cache_match_reference(self):
        self._assert_decode_matches_reference(FusedRopeKVCacheDecodeOpAsm)

    def test_nonasm_decode_q_and_k_cache_match_reference(self):
        self._assert_decode_matches_reference(FusedRopeKVCacheDecodeOpNonAsm)

    def test_decode_validates_position_ids_against_actual_qkv_tokens(self):
        for op_class in (
            FusedRopeKVCacheDecodeOpAsm,
            FusedRopeKVCacheDecodeOpNonAsm,
        ):
            (
                op,
                params,
                _attn_inputs,
                layer_cache,
                _pool,
                _per_batch_block_ids,
                _sequence_lengths,
                q,
                k,
                v,
            ) = self._build_decode_case(op_class)
            multi_token_qkv = _pack_qkv(
                q.repeat_interleave(2, dim=0),
                k.repeat_interleave(2, dim=0),
                v.repeat_interleave(2, dim=0),
            )
            with self.subTest(op_class=op_class.__name__):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "expected 12 for 4 tokens and index_factor 3",
                ):
                    op.forward(multi_token_qkv, layer_cache, params)

    def _assert_cuda_graph_replay_uses_new_position_ids(self, op_class):
        (
            op,
            params,
            attn_inputs,
            layer_cache,
            pool,
            per_batch_block_ids,
            sequence_lengths,
            q,
            k,
            v,
        ) = self._build_decode_case(op_class)
        qkv = _pack_qkv(q, k, v)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            op.forward(qkv, layer_cache, params)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        pool.zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_q = op.forward(qkv, layer_cache, params)

        replay_position_ids = torch.tensor(
            [[9, 1, 6], [4, 8, 2]], dtype=torch.int32, device=self.device
        )
        attn_inputs.combo_position_ids.copy_(replay_position_ids)
        pool.zero_()
        graph.replay()
        torch.cuda.synchronize()

        expected_q, expected_k = _apply_mrope(
            q,
            k,
            replay_position_ids,
            self.sections,
            self.rope_dim,
        )
        actual_k = _read_decode_k_from_pool(
            pool,
            per_batch_block_ids,
            sequence_lengths,
            self.head_num_kv,
            self.head_dim,
            self.tokens_per_block,
        )
        torch.testing.assert_close(captured_q, expected_q, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_k, expected_k, atol=1e-2, rtol=1e-2)

    def test_asm_cuda_graph_replay_uses_new_position_ids(self):
        self._assert_cuda_graph_replay_uses_new_position_ids(
            FusedRopeKVCacheDecodeOpAsm
        )

    def test_nonasm_cuda_graph_replay_uses_new_position_ids(self):
        self._assert_cuda_graph_replay_uses_new_position_ids(
            FusedRopeKVCacheDecodeOpNonAsm
        )


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterDecodeTritonNumerics(unittest.TestCase):
    """Numerical regression for production cache write -> Gluon decode.

    Populate a shared paged KV cache through the same ASM RoPE/cache writer used
    by AiterDecodeImplTriton, then compare the final decode attention result with
    an independent fp32 torch reference. Context lengths straddle Gluon's
    256-token context-partition boundary.
    """

    HEAD_NUM = 8
    KV_HEAD_NUM = 1
    HEAD_DIM = 128
    ROPE_DIM = 64
    MROPE_SECTIONS = (11, 11, 10)
    TOKENS_PER_BLOCK = 16

    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.config = _make_mrope_attn_configs(
            head_num=self.HEAD_NUM,
            head_num_kv=self.KV_HEAD_NUM,
            head_dim=self.HEAD_DIM,
            dtype=self.dtype,
            rope_dim=self.ROPE_DIM,
            mrope_sections=self.MROPE_SECTIONS,
            tokens_per_block=self.TOKENS_PER_BLOCK,
        )

    def _position_ids(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (positions, torch.div(positions, 2, rounding_mode="floor"), positions % 3),
            dim=1,
        ).to(dtype=torch.int32)

    def _make_shared_decode_inputs(
        self, sequence_lengths: List[int], block_table: torch.Tensor
    ) -> PyAttentionInputs:
        positions = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=self.device
        )
        attn_inputs, _, _ = _make_mrope_decode_inputs(
            sequence_lengths,
            self._position_ids(positions),
            self.device,
            self.dtype,
            self.TOKENS_PER_BLOCK,
        )
        shared_table = block_table.expand(len(sequence_lengths), -1).contiguous()
        attn_inputs.kv_cache_kernel_block_id = shared_table.cpu()
        attn_inputs.kv_cache_kernel_block_id_device = shared_table
        attn_inputs.kv_cache_block_id_device = shared_table
        return attn_inputs

    def _reference(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        query_rot, key_rot = _apply_mrope(
            query,
            key,
            position_ids,
            self.MROPE_SECTIONS,
            self.ROPE_DIM,
        )
        repeat = self.HEAD_NUM // self.KV_HEAD_NUM
        key_heads = key_rot.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
        value_heads = value.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
        query_heads = query_rot[-1:].float().transpose(0, 1)
        scores = torch.matmul(query_heads, key_heads.transpose(-1, -2)) / math.sqrt(
            self.HEAD_DIM
        )
        probs = torch.softmax(scores, dim=-1)
        return (
            torch.matmul(probs, value_heads)
            .transpose(0, 1)
            .reshape(1, self.HEAD_NUM * self.HEAD_DIM)
        )

    def _run_case(self, context_length: int) -> None:
        prefix_length = context_length - 1
        num_blocks = math.ceil(context_length / self.TOKENS_PER_BLOCK)
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).view(1, -1)

        layer_cache, _ = _alloc_decode_kv_cache(
            num_blocks,
            self.KV_HEAD_NUM,
            self.TOKENS_PER_BLOCK,
            self.HEAD_DIM,
            self.dtype,
            self.device,
        )
        layer_cache.kv_scale_base = torch.empty(0, device=self.device)

        query = torch.randn(
            context_length,
            self.HEAD_NUM,
            self.HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        key = torch.randn(
            context_length,
            self.KV_HEAD_NUM,
            self.HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        value = torch.randn_like(key)

        # Fill every prefix position with the production ASM decode cache writer.
        # Each synthetic request shares the same page table but writes a distinct
        # sequence position, which materializes one ordinary logical sequence.
        prefix_inputs = self._make_shared_decode_inputs(
            list(range(prefix_length)), block_table
        )
        prefix_writer = FusedRopeKVCacheDecodeOpAsm(self.config)
        prefix_params = prefix_writer.prepare(prefix_inputs)
        prefix_writer.forward(
            _pack_qkv(
                query[:prefix_length],
                key[:prefix_length],
                value[:prefix_length],
            ),
            layer_cache,
            prefix_params,
        )

        # The implementation writes the current K/V through the same writer and
        # then dispatches the one-token query to pa_decode_gluon.
        decode_inputs = self._make_shared_decode_inputs([prefix_length], block_table)
        impl = AiterDecodeImplTriton(self.config, decode_inputs)
        actual = impl.forward(
            _pack_qkv(
                query[prefix_length:],
                key[prefix_length:],
                value[prefix_length:],
            ),
            layer_cache,
            layer_idx=0,
        )

        positions = torch.arange(context_length, device=self.device)
        reference = self._reference(
            query,
            key,
            value,
            self._position_ids(positions),
        )
        self.assertFalse(torch.isnan(actual).any())
        self.assertFalse(torch.isinf(actual).any())
        relative_l2 = (actual.float() - reference).norm() / reference.norm()
        self.assertLess(
            relative_l2.item(),
            0.02,
            f"context_length={context_length}, relative_l2={relative_l2.item():.6f}",
        )

    def test_matches_fp32_reference_at_partition_boundary(self):
        for context_length in (255, 256, 257):
            with self.subTest(context_length=context_length):
                self._run_case(context_length)


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterPrefillImplMropePositionIds(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _check_mrope_matches_reference(self, impl_cls):
        input_lengths = [5, 3]
        head_num = 4
        head_num_kv = 2
        head_dim = 256
        rope_dim = 64
        mrope_sections = (11, 11, 10)
        cfg = _make_mrope_attn_configs(
            head_num=head_num,
            head_num_kv=head_num_kv,
            head_dim=head_dim,
            dtype=self.dtype,
            rope_dim=rope_dim,
            mrope_sections=mrope_sections,
        )
        attn_inputs = _make_mrope_prefill_inputs(input_lengths, self.device, self.dtype)

        impl = impl_cls(cfg, attn_inputs)
        total_tokens = sum(input_lengths)
        q = torch.randn(
            total_tokens, head_num, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )

        actual = impl.forward(_pack_qkv(q, k, v), kv_cache=None, layer_idx=0)
        q_rope, k_rope = _apply_mrope(
            q,
            k,
            attn_inputs.combo_position_ids,
            mrope_sections,
            rope_dim,
        )
        ref = _sdpa_reference(
            q_rope,
            k_rope,
            v,
            attn_inputs.cu_seqlens_device,
            attn_inputs.cu_kv_seqlens_device,
            causal=True,
        )

        # CKAttn's pybind surface intentionally does not expose position_ids.
        # Comparing the output against an independently rotated reference is
        # the public, end-to-end check that all three MRoPE axes were consumed.
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    def test_asm_mrope_matches_reference(self):
        self._check_mrope_matches_reference(AiterPrefillImplAsm)

    def test_nonasm_mrope_matches_reference(self):
        self._check_mrope_matches_reference(AiterPrefillImplNonAsm)


@unittest.skipUnless(_is_rocm() and _OPS_IMPORTABLE, "Requires ROCm attention wrappers")
class TestVLayoutContract(unittest.TestCase):
    def _make_case(self, head_dim: int, page: int):
        config = _make_attn_configs(8, 2, head_dim, page)
        config.need_rope_kv_cache = True
        config.dtype = torch.bfloat16
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        return config, inputs

    def _decode_flags(self, aiter: bool, asm: bool, triton: bool):
        flags = FMHAConfig()
        flags.use_aiter_pa, flags.use_asm_pa, flags.use_triton_pa = aiter, asm, triton
        return flags

    def test_invalid_geometry(self):
        for head, page, dtype, error in (
            (100, 32, KvCacheDataType.BASE, "V geometry"),
            (128, 8, KvCacheDataType.FP8, "width=16"),
            (128, 12, KvCacheDataType.BASE, "V geometry"),
        ):
            with self.subTest(head=head, page=page, dtype=dtype):
                config, inputs = self._make_case(head, page)
                config.kv_cache_dtype = dtype
                with self.assertRaisesRegex(ValueError, error):
                    validate_v_layout(config, inputs, FMHAConfig())

    def test_factory_rejects_layout_mismatch(self):
        config, inputs = self._make_case(256, 16)
        inputs.is_prefill = False
        flags = self._decode_flags(aiter=True, asm=True, triton=False)
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            attn_factory.get_fmha_impl(config, None, inputs, fmha_config=flags)

    def test_layout_mismatch_accepted_when_page_equals_width(self):
        config, inputs = self._make_case(256, 8)
        inputs.is_prefill = False
        validate_v_layout(
            config, inputs, self._decode_flags(aiter=False, asm=True, triton=False)
        )

    def test_fp8_no_asm_requires_page_equals_width(self):
        config, inputs = self._make_case(128, 32)
        config.kv_cache_dtype, inputs.is_prefill = KvCacheDataType.FP8, False
        flags = self._decode_flags(aiter=True, asm=False, triton=False)
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            validate_v_layout(config, inputs, flags)
        config.kernel_tokens_per_block = 16
        validate_v_layout(config, inputs, flags)

    def test_constructor_fallback_is_strict_only_with_layout_validator(self):
        class BrokenImpl:
            accepts_fmha_config = False
            support = support_parallelism_config = staticmethod(lambda *_: True)

            def __init__(self, *_):
                raise RuntimeError("constructor failed")

        class WorkingImpl(BrokenImpl):
            def __init__(self, *_):
                pass

        _, inputs = self._make_case(128, 16)
        inputs.is_prefill = False
        with patch.object(attn_factory, "DECODE_MHA_IMPS", [BrokenImpl, WorkingImpl]):
            with patch.object(attn_factory, "VALIDATE_FMHA_CONFIG", None):
                with self.assertLogs(level="WARNING"):
                    impl = attn_factory.get_fmha_impl(AttentionConfigs(), None, inputs)
                self.assertIsInstance(impl, WorkingImpl)
            with patch.object(attn_factory, "VALIDATE_FMHA_CONFIG", lambda *_: True):
                with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                    attn_factory.get_fmha_impl(AttentionConfigs(), None, inputs)


if __name__ == "__main__":
    unittest.main()

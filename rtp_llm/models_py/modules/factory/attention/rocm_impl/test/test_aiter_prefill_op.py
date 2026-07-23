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
from typing import List, Optional

import torch
import torch.nn.functional as F

# Both imports may fail outside the ROCm runtime; we want a clean skip rather
# than a collection-time error so the rest of the test suite is unaffected.
try:
    import aiter  # noqa: F401

    _AITER_AVAILABLE = True
except ImportError:
    _AITER_AVAILABLE = False

try:
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterPrefillAttnOp,
        AiterPrefillAttnOpPaged,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
        AiterPrefillImplPaged,
        FMHAParams,
    )
    from rtp_llm.ops import (
        AttentionConfigs,
        KvCacheDataType,
        PyAttentionInputs,
        RopeConfig,
        RopeStyle,
    )
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCacheDecodeOpAsm,
        FusedRopeKVCacheDecodeOpNonAsm,
        LayerKVCache,
        get_typemeta,
    )

    _OPS_IMPORTABLE = True
except ImportError:
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
    mrope_section=(16, 24, 24),
):
    """Qwen3-VL-style MRoPE configuration for the real ROCm fused op."""
    cfg = _make_rope_attn_configs(head_num, head_num_kv, head_dim, dtype)
    rope = cfg.rope_config
    rope.style = RopeStyle.Mrope
    rope.index_factor = 3
    rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3 = mrope_section
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


def _make_mrope_decode_inputs(
    sequence_lengths: List[int],
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_block: int = 16,
):
    """Build one-token-per-sequence inputs for the real ROCm decode RoPE op."""
    batch_size = len(sequence_lengths)
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = False
    attn_inputs.dtype = get_typemeta(torch.empty(1, dtype=dtype))
    attn_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.prefix_lengths = torch.zeros(
        batch_size, dtype=torch.int32, device="cpu"
    )
    attn_inputs.padding_offset = torch.zeros(
        batch_size, dtype=torch.int32, device=device
    )
    attn_inputs.cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    )
    attn_inputs.cu_kv_seqlens_device = torch.tensor(
        [0] + [sum(sequence_lengths[:idx]) + idx for idx in range(1, batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )
    max_blocks = max(
        (sequence_length + 1 + tokens_per_block - 1) // tokens_per_block
        for sequence_length in sequence_lengths
    )
    block_table = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device="cpu")
    next_block = 0
    for batch_idx, sequence_length in enumerate(sequence_lengths):
        block_count = (sequence_length + 1 + tokens_per_block - 1) // tokens_per_block
        block_table[batch_idx, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count
    attn_inputs.kv_cache_kernel_block_id = block_table
    attn_inputs.kv_cache_kernel_block_id_device = block_table.to(device)
    attn_inputs.combo_position_ids = (
        position_ids.to(device=device, dtype=torch.int32).contiguous().view(-1)
    )
    return attn_inputs, next_block


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
    mrope_section=(16, 24, 24),
):
    """Torch reference for token-major three-axis Qwen3-VL MRoPE."""
    head_dim = q.shape[-1]
    half = head_dim // 2
    if sum(mrope_section) != half:
        raise ValueError("MRoPE sections must cover half of the head dimension")

    axis_by_pair = torch.repeat_interleave(
        torch.arange(3, device=q.device),
        torch.tensor(mrope_section, device=q.device),
    )
    positions = position_ids.to(device=q.device, dtype=torch.float32)
    pair_positions = positions[:, axis_by_pair]
    inv_freq = 10000 ** (
        -2.0 * torch.arange(half, dtype=torch.float32, device=q.device) / head_dim
    )
    angle = pair_positions * inv_freq.unsqueeze(0)
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    def rot(x):
        lo, hi = x[..., :half], x[..., half:]
        return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)

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

    def test_support_true_for_real_prefix(self):
        """prefix_lengths.max() > 0 => support() returns True."""
        pl = torch.tensor([0, 128, 0, 64], dtype=torch.int32)
        self.assertTrue(AiterPrefillImplPaged.support(None, self._make_attn_inputs(pl)))

    def test_support_true_for_capture_filled_prefix(self):
        """MTP draft capture pre-fills prefix_lengths with max_seq_len => support() True."""
        pl = torch.full((4,), 1024, dtype=torch.int32)
        self.assertTrue(AiterPrefillImplPaged.support(None, self._make_attn_inputs(pl)))

    def test_support_false_for_zero_prefix(self):
        """All-zero prefix_lengths => support() returns False (ASM/NonAsm preferred)."""
        pl = torch.zeros(4, dtype=torch.int32)
        self.assertFalse(
            AiterPrefillImplPaged.support(None, self._make_attn_inputs(pl))
        )

    def test_support_false_for_empty_prefix_lengths(self):
        """Empty prefix_lengths tensor => support() returns False."""
        pl = torch.empty(0, dtype=torch.int32)
        self.assertFalse(
            AiterPrefillImplPaged.support(None, self._make_attn_inputs(pl))
        )


@unittest.skipUnless(_OPS_IMPORTABLE, "Requires AiterPrefillImplPaged module")
class TestUpdatePrefillParamsForCudaGraph(unittest.TestCase):
    """Unit tests for AiterPrefillImplPaged._update_prefill_params_for_cuda_graph.

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
        )
        stub = SimpleNamespace(fmha_params=fmha_params)
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
        AiterPrefillImplPaged._update_prefill_params_for_cuda_graph(stub, attn_inputs)

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

    def test_live_cu_seqlens_path(self):
        """When live cu_seqlens are provided, use them directly."""
        stub = self._make_stub(batch_size=2)
        cu_q = torch.tensor([0, 5, 10], dtype=torch.int32)
        cu_k = torch.tensor([0, 105, 210], dtype=torch.int32)
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
        # prefill_seqlen_k_int32 must be derived from live cu_seqlens_k
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

    # ---- FP8 fallback: compact should NOT be used -------------------------

    def test_fp8_uses_full_reshape(self):
        """When kv_cache is FP8, _forward_paged should use the full reshape path."""
        from types import SimpleNamespace
        from unittest.mock import patch

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
@unittest.skipUnless(_AITER_AVAILABLE, "Requires aiter")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention wrapper module")
class TestAiterPrefillImplMropeRealOp(unittest.TestCase):
    """Numerical and validation coverage for Qwen3-VL MRoPE position IDs."""

    def setUp(self):
        torch.manual_seed(2)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _check_mrope_matches_reference(self, impl_cls):
        input_lengths = [3, 2]
        head_num = 4
        head_num_kv = 2
        head_dim = 128
        cfg = _make_mrope_attn_configs(
            head_num, head_num_kv, head_dim, dtype=self.dtype
        )
        attn_inputs = _make_rope_prefill_inputs(input_lengths, self.device, self.dtype)
        position_ids = torch.tensor(
            [
                [0, 0, 0],
                [1, 4, 7],
                [2, 5, 8],
                [0, 0, 0],
                [3, 6, 9],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs.combo_position_ids = position_ids.flatten()
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

        actual_q, actual_k, actual_v = impl.rope_kvcache_impl.forward(
            qkv, None, impl.rope_params
        )
        ref_q, ref_k = _apply_mrope(q, k, position_ids)
        ref_k_padded, ref_v_padded = _pad_kv(ref_k, v, attn_inputs.cu_kv_seqlens_device)

        torch.testing.assert_close(actual_q, ref_q, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_k, ref_k_padded, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_v, ref_v_padded, atol=1e-2, rtol=1e-2)

    def test_asm_mrope_uses_all_three_position_axes(self):
        self._check_mrope_matches_reference(AiterPrefillImplAsm)

    def test_nonasm_mrope_uses_all_three_position_axes(self):
        self._check_mrope_matches_reference(AiterPrefillImplNonAsm)

    def test_prepare_in_place_refreshes_mrope_position_ids(self):
        input_lengths = [3, 2]
        head_num = 4
        head_num_kv = 2
        head_dim = 128
        cfg = _make_mrope_attn_configs(
            head_num, head_num_kv, head_dim, dtype=self.dtype
        )
        attn_inputs = _make_rope_prefill_inputs(input_lengths, self.device, self.dtype)
        first_position_ids = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs.combo_position_ids = first_position_ids.flatten()
        impl = AiterPrefillImplAsm(cfg, attn_inputs)

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
        first_q, _, _ = impl.rope_kvcache_impl.forward(qkv, None, impl.rope_params)

        replay_inputs = _make_rope_prefill_inputs(
            input_lengths, self.device, self.dtype
        )
        replay_position_ids = torch.tensor(
            [
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8],
                [0, 2, 4],
                [1, 3, 5],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        replay_inputs.combo_position_ids = replay_position_ids.flatten()
        impl.rope_params.prepare_in_place(replay_inputs)
        replay_q, _, _ = impl.rope_kvcache_impl.forward(qkv, None, impl.rope_params)
        expected_q, _ = _apply_mrope(q, k, replay_position_ids)

        self.assertFalse(torch.equal(first_q, replay_q))
        torch.testing.assert_close(replay_q, expected_q, atol=1e-2, rtol=1e-2)

    def test_mrope_missing_position_ids_fails_during_prepare(self):
        cfg = _make_mrope_attn_configs(4, 2, 128, dtype=self.dtype)
        attn_inputs = _make_rope_prefill_inputs([2], self.device, self.dtype)
        with self.assertRaisesRegex(
            RuntimeError, "requires non-empty combo_position_ids"
        ):
            AiterPrefillImplAsm(cfg, attn_inputs)


@unittest.skipUnless(_is_rocm(), "Requires ROCm GPU")
@unittest.skipUnless(_OPS_IMPORTABLE, "Requires ROCm attention operators")
class TestRocmDecodeMropeRealOp(unittest.TestCase):
    """Numerical coverage for the distinct ROCm decode MRoPE kernels."""

    def setUp(self):
        torch.manual_seed(3)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _check_decode_matches_reference(
        self,
        op_cls,
        sequence_lengths: List[int],
        position_ids: torch.Tensor,
    ):
        head_num = 4
        head_num_kv = 2
        head_dim = 128
        tokens_per_block = 16
        cfg = _make_mrope_attn_configs(
            head_num, head_num_kv, head_dim, dtype=self.dtype
        )
        cfg.max_seq_len = 64
        attn_inputs, block_count = _make_mrope_decode_inputs(
            sequence_lengths,
            position_ids,
            self.device,
            self.dtype,
            tokens_per_block,
        )
        op = op_cls(cfg)
        params = op.prepare(attn_inputs)
        token_num = len(sequence_lengths)
        q = torch.randn(
            token_num, head_num, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            token_num, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            token_num, head_num_kv, head_dim, dtype=self.dtype, device=self.device
        )
        qkv = _pack_qkv(q, k, v)
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.zeros(
            block_count,
            2,
            head_num_kv,
            tokens_per_block,
            head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        actual_q = op.forward(qkv, kv_cache, params)
        expected_q, _ = _apply_mrope(q, k, position_ids)
        torch.testing.assert_close(actual_q, expected_q, atol=1e-2, rtol=1e-2)

    def test_asm_single_sequence_uses_all_three_position_axes(self):
        self._check_decode_matches_reference(
            FusedRopeKVCacheDecodeOpAsm,
            [3],
            torch.tensor([[2, 5, 7]], dtype=torch.int32),
        )

    def test_nonasm_single_sequence_uses_all_three_position_axes(self):
        self._check_decode_matches_reference(
            FusedRopeKVCacheDecodeOpNonAsm,
            [3],
            torch.tensor([[2, 5, 7]], dtype=torch.int32),
        )

    def test_asm_multi_sequence_uses_token_major_position_ids(self):
        self._check_decode_matches_reference(
            FusedRopeKVCacheDecodeOpAsm,
            [3, 5],
            torch.tensor([[2, 5, 7], [4, 1, 9]], dtype=torch.int32),
        )

    def test_nonasm_multi_sequence_uses_token_major_position_ids(self):
        self._check_decode_matches_reference(
            FusedRopeKVCacheDecodeOpNonAsm,
            [3, 5],
            torch.tensor([[2, 5, 7], [4, 1, 9]], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()

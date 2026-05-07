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
        AiterPrefillImplPaged,
        FMHAParams,
    )
    from rtp_llm.ops import AttentionConfigs, PyAttentionInputs

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

    Validates draft-prefill-cuda-graph flag and prefix_lengths boundary logic.
    Does NOT require GPU — only exercises the static support() decision.
    """

    def _make_attn_inputs(self, prefix_lengths, is_draft_capture=False):
        from types import SimpleNamespace
        inputs = SimpleNamespace(
            prefix_lengths=prefix_lengths,
            is_draft_prefill_cuda_graph_capture=is_draft_capture,
            is_cuda_graph=False,
            is_prefill=True,
            input_lengths=torch.tensor([4], dtype=torch.int32),
        )
        return inputs

    def test_support_true_for_draft_capture_flag_with_zero_prefix(self):
        """MTP draft flag set + prefix_lengths all zeros => support() returns True."""
        pl = torch.zeros(4, dtype=torch.int32)
        inputs = self._make_attn_inputs(pl, is_draft_capture=True)
        self.assertTrue(AiterPrefillImplPaged.support(None, inputs))

    def test_support_true_for_real_prefix(self):
        """prefix_lengths.max() > 0 => support() returns True (no draft flag needed)."""
        pl = torch.tensor([0, 128, 0, 64], dtype=torch.int32)
        inputs = self._make_attn_inputs(pl, is_draft_capture=False)
        self.assertTrue(AiterPrefillImplPaged.support(None, inputs))

    def test_support_false_without_prefix_or_flag(self):
        """No prefix and no draft flag => support() returns False."""
        pl = torch.zeros(4, dtype=torch.int32)
        inputs = self._make_attn_inputs(pl, is_draft_capture=False)
        self.assertFalse(AiterPrefillImplPaged.support(None, inputs))

    def test_support_false_for_empty_prefix_lengths(self):
        """Empty prefix_lengths tensor and no draft flag => support() returns False."""
        pl = torch.empty(0, dtype=torch.int32)
        inputs = self._make_attn_inputs(pl, is_draft_capture=False)
        self.assertFalse(AiterPrefillImplPaged.support(None, inputs))

    def test_support_true_for_empty_prefix_with_draft_flag(self):
        """Empty prefix_lengths but draft flag set => support() returns True."""
        pl = torch.empty(0, dtype=torch.int32)
        inputs = self._make_attn_inputs(pl, is_draft_capture=True)
        self.assertTrue(AiterPrefillImplPaged.support(None, inputs))


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

    def _make_attn_inputs(self, input_lengths, prefix_lengths=None,
                          cu_seqlens=None, cu_kv_seqlens=None,
                          kv_block_id=None):
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


if __name__ == "__main__":
    unittest.main()

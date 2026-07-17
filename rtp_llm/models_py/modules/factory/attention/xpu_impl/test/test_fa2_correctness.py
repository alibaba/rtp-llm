"""XPU FA2 (vllm-xpu-kernels) numerical correctness tests.

Regression coverage for the P1 review finding: the XPU attention path has no
real-device correctness gate comparing the FA2 kernel against a reference
implementation. These tests exercise the actual `flash_attn_varlen` dispatcher
(rtp_llm/models_py/modules/base/xpu/vllm_xpu_ops.py) on a real XPU device and
compare against an independent PyTorch SDPA reference for:

  - prefill: varlen causal attention across multiple requests of different
    lengths (multi-batch).
  - decode: paged attention via the real block_table/seqused_k path, with
    per-request KV history lengths chosen to land exactly on, one below, and
    one above a page (tokens_per_block) boundary, and to span multiple pages
    (multi-batch + cross-block-boundary).

Both are run across several head-count / head-dim / page-size configs (see
_CONFIGS): a small toy shape plus REALISTIC model shapes (head_dim=128 GQA)
and a non-power-of-2 head_dim. This matters because vendor XPU kernels have
been observed to match a reference at small/power-of-2 shapes yet diverge at
the shapes real models actually use, so a toy-only gate can pass while
production silently produces wrong output.

Skip-vs-fail policy (see _import_guard.py): on a real XPU test run
(TEST_USING_DEVICE=XPU) a missing/broken FA2 kernel or import is a genuine
regression and FAILS the test; everywhere else (CPU dev boxes, non-XPU CI)
it is SKIPPED, since the real kernel genuinely is not available there.
"""

import torch
import torch.nn.functional as F
from unittest import TestCase, main

from rtp_llm.models_py.modules.factory.attention.xpu_impl.test._import_guard import (
    skip_or_fail_on_missing_import,
)

try:
    from rtp_llm.ops.compute_ops import CacheGroupType, KVCache
    from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
        _write_to_paged_cache,
    )
    from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import (
        flash_attn_varlen,
        is_fa2_available,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - import-time environment guard
    _IMPORT_OK = False
    _IMPORT_ERR = _e

DTYPE = torch.float16

# (num_heads, num_kv_heads, head_dim, tokens_per_block). Covers a small toy
# shape, two realistic GQA model shapes with head_dim=128, and a non-power-of-2
# head_dim. num_heads/num_kv_heads exercise GQA repeat ratios 2:1, 4:1, 7:1, 8:1.
_CONFIGS = [
    (8, 4, 64, 16),    # small toy shape (fast smoke)
    (32, 8, 128, 16),  # realistic: head_dim=128, 4:1 GQA (llama-ish)
    (28, 4, 128, 64),  # realistic: 7:1 GQA, larger page (qwen-ish)
    (16, 2, 80, 16),   # non-power-of-2 head_dim, 8:1 GQA
]


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _reference_attn(q, k, v, causal, n_rep):
    """Independent SDPA reference. q: [Sq,H,D], k/v: [Skv,Hkv,D] -> [Sq,H,D]."""
    qi = q.unsqueeze(0).transpose(1, 2)
    ki = _repeat_kv(k.unsqueeze(0).transpose(1, 2), n_rep)
    vi = _repeat_kv(v.unsqueeze(0).transpose(1, 2), n_rep)
    scale = q.shape[-1] ** -0.5
    oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale)
    return oi.transpose(1, 2).squeeze(0)


class XpuFa2CorrectnessTest(TestCase):
    def setUp(self):
        skip_or_fail_on_missing_import(
            self, _IMPORT_OK, _IMPORT_ERR, "rtp_llm.ops / xpu FA2 consumer"
        )
        skip_or_fail_on_missing_import(
            self, is_fa2_available(), RuntimeError("FA2 (_vllm_fa2_C) not importable"),
            "vllm-xpu-kernels FA2 kernel",
        )
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            skip_or_fail_on_missing_import(
                self, False, RuntimeError("no XPU device visible"), "XPU device"
            )
        self.device = torch.device("xpu")

    def _check_prefill(self, num_heads, num_kv_heads, head_dim, _tpb):
        """Varlen causal prefill across multiple requests of different lengths."""
        n_rep = num_heads // num_kv_heads
        torch.manual_seed(7)
        seq_lens = [1, 8, 33, 50]
        cu = [0]
        for l in seq_lens:
            cu.append(cu[-1] + l)
        total = cu[-1]

        q = torch.randn(total, num_heads, head_dim, dtype=DTYPE, device=self.device)
        k = torch.randn(total, num_kv_heads, head_dim, dtype=DTYPE, device=self.device)
        v = torch.randn(total, num_kv_heads, head_dim, dtype=DTYPE, device=self.device)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=self.device)
        max_seqlen = max(seq_lens)

        out = flash_attn_varlen(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )

        outs = []
        for i in range(len(seq_lens)):
            s, e = cu[i], cu[i + 1]
            outs.append(_reference_attn(q[s:e], k[s:e], v[s:e], True, n_rep))
        ref_out = torch.cat(outs, dim=0)

        torch.testing.assert_close(
            out.float(), ref_out.float(), rtol=1e-2, atol=1e-2,
            msg="FA2 prefill output diverges from SDPA reference",
        )

    def _check_decode(self, num_heads, num_kv_heads, head_dim, tpb):
        """Paged decode (real block_table/seqused_k path) across requests whose
        KV history lengths land on, one below/above, and span multiple TPB page
        boundaries."""
        n_rep = num_heads // num_kv_heads
        torch.manual_seed(123)
        kv_lens = [1, tpb - 1, tpb, tpb + 1, 2 * tpb + 1]
        num_requests = len(kv_lens)

        ref_k = [
            torch.randn(l, num_kv_heads, head_dim, dtype=DTYPE, device=self.device)
            for l in kv_lens
        ]
        ref_v = [
            torch.randn(l, num_kv_heads, head_dim, dtype=DTYPE, device=self.device)
            for l in kv_lens
        ]
        q_new = torch.randn(num_requests, num_heads, head_dim, dtype=DTYPE, device=self.device)

        max_blocks_per_req = max((l + tpb - 1) // tpb for l in kv_lens)
        bids_list = []
        next_block = 0
        for l in kv_lens:
            nblocks = (l + tpb - 1) // tpb
            bids_list.append(list(range(next_block, next_block + nblocks)))
            next_block += nblocks
        total_blocks = next_block

        kv_cache = KVCache()
        kv_cache.seq_size_per_block = tpb
        kv_cache.kernel_seq_size_per_block = tpb
        kv_cache.num_kv_heads = num_kv_heads
        kv_cache.head_dim = head_dim
        kv_cache.use_mla = False
        base = torch.zeros(
            total_blocks, 2 * tpb * num_kv_heads * head_dim, dtype=DTYPE, device=self.device
        )
        kv_cache.kv_cache_base_by_layer = [base]
        kv_cache.layer_attn_types = [CacheGroupType.FULL]
        lc = kv_cache.get_layer_cache(0)

        for i, l in enumerate(kv_lens):
            bids_cpu = torch.tensor(bids_list[i], dtype=torch.long)
            _write_to_paged_cache(ref_k[i], ref_v[i], lc, bids_cpu, 0, num_kv_heads, head_dim)

        # Pad each request's block-table row with its own last block id: those
        # padding slots are never read (bounded by seqused_k) but must stay a
        # valid in-range index so a stray OOB read cannot occur.
        bt = torch.zeros(num_requests, max_blocks_per_req, dtype=torch.int32)
        for i, bids in enumerate(bids_list):
            padded = bids + [bids[-1]] * (max_blocks_per_req - len(bids))
            bt[i] = torch.tensor(padded, dtype=torch.int32)
        block_table = bt.to(device=self.device)

        cache = lc.kv_cache_base
        k_cache = cache[:, 0]
        v_cache = cache[:, 1]

        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=self.device)
        cu_q = torch.arange(num_requests + 1, dtype=torch.int32, device=self.device)
        max_kv_len = max(kv_lens)

        out = flash_attn_varlen(
            q_new.contiguous(), k_cache, v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=None,
            max_seqlen_q=1, max_seqlen_k=max_kv_len,
            causal=False, block_table=block_table, seqused_k=seqused_k,
        )
        out = out.reshape(num_requests, num_heads, head_dim)

        ref_out = torch.stack([
            _reference_attn(q_new[i:i + 1], ref_k[i], ref_v[i], False, n_rep)[0]
            for i in range(num_requests)
        ])

        torch.testing.assert_close(
            out.float(), ref_out.float(), rtol=1e-2, atol=1e-2,
            msg="FA2 paged decode output diverges from SDPA reference across "
                "cross-block-boundary KV lengths",
        )

    def test_prefill_causal_multi_batch(self):
        for cfg in _CONFIGS:
            with self.subTest(num_heads=cfg[0], num_kv_heads=cfg[1], head_dim=cfg[2], tpb=cfg[3]):
                self._check_prefill(*cfg)

    def test_decode_paged_cross_block_boundary_multi_batch(self):
        for cfg in _CONFIGS:
            with self.subTest(num_heads=cfg[0], num_kv_heads=cfg[1], head_dim=cfg[2], tpb=cfg[3]):
                self._check_decode(*cfg)


if __name__ == "__main__":
    main()

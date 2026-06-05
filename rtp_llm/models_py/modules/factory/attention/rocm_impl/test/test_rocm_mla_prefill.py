"""Correctness test for the fused ROCm MLA prefill impl.

``RocmMlaPrefillImpl`` decompresses latent KV then runs a single fused
``aiter.flash_attn_varlen_func`` over ragged ``[prefix ++ current]`` per stream
(replacing the old chunked FP32 padded einsum). This test checks that the fused
output matches a self-contained FP32 reference (``attention_ref``) for:

  - single stream, no reuse;
  - multiple streams of differing lengths, no reuse (exercises ragged assembly);
  - single stream with a reused prefix gathered from the paged cache.

Runs only on the ROCm CI runner because it calls aiter.
"""

import math
from typing import Dict, List
from unittest import SkipTest, TestCase, main

import torch

try:
    import aiter  # noqa: F401

    _HAS_AITER = True
except Exception:
    _HAS_AITER = False

from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter_mla import (
    RocmMlaPrefillImpl,
    _apply_neox_rope,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W


class _FakeKVCache:
    """Duck-typed stand-in for the pybind ``LayerKVCache``.

    The real type exposes ``seq_size_per_block`` as a read-only property (set
    from C++), so it cannot be configured from Python. The impl only reads
    ``kv_cache_base`` and ``seq_size_per_block``, so a plain object suffices.
    """

    def __init__(self, kv_cache_base, seq_size_per_block):
        self.kv_cache_base = kv_cache_base
        self.seq_size_per_block = seq_size_per_block


HEAD_NUM = 16
NOPE = 128
ROPE = 64
KV_LORA = 512
V_DIM = 128
QK_DIM = NOPE + ROPE  # 192
PAGE_SIZE = 64
MAX_SEQ = 4096


def attention_ref(batch_size, q, k, v, causal: bool, sm_scale: float):
    """Self-contained FP32 attention reference (bottom-right causal).

    Mirrors ``modules/hybrid/test/mla_attention_ref.py`` but inlined to keep this
    test's BUILD target free of cross-package source deps. Supports asymmetric
    qk/vo head dims (qk=192, vo=128).
    """
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )
    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )
    return o_ref, lse_ref * math.log2(math.e)


def _make_cos_sin_cache(device) -> torch.Tensor:
    """[MAX_SEQ, ROPE] cache: first half cosines, second half sines (engine layout)."""
    half = ROPE // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(MAX_SEQ, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [MAX_SEQ, half]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float32)


def _make_attn_configs() -> AttentionConfigs:
    cfg = AttentionConfigs()
    cfg.head_num = HEAD_NUM
    cfg.kv_head_num = HEAD_NUM
    cfg.size_per_head = QK_DIM
    cfg.nope_head_dim = NOPE
    cfg.rope_head_dim = ROPE
    cfg.kv_lora_rank = KV_LORA
    cfg.v_head_dim = V_DIM
    cfg.q_lora_rank = 0
    cfg.use_mla = True
    cfg.is_sparse = False
    cfg.kernel_tokens_per_block = PAGE_SIZE
    return cfg


def _build_impl(attn_inputs, weights, cos_sin_cache) -> RocmMlaPrefillImpl:
    return RocmMlaPrefillImpl(
        _make_attn_configs(),
        attn_inputs,
        [weights],
        cos_sin_cache,
    )


class RocmMlaPrefillTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("GPU not available")
        if not _HAS_AITER:
            raise SkipTest("aiter not available (ROCm runner only)")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.cos_sin_cache = _make_cos_sin_cache(self.device)
        self.kv_b_w = torch.randn(
            KV_LORA, HEAD_NUM * (NOPE + V_DIM), dtype=torch.bfloat16, device=self.device
        )
        self.weights: Dict[str, torch.Tensor] = {W.mla_kv_b_w: self.kv_b_w}
        self.scale = QK_DIM**-0.5

    # -- helpers ---------------------------------------------------------------

    def _decompress(self, ck: torch.Tensor):
        kv = ck.float() @ self.kv_b_w.float()
        kv = kv.view(-1, HEAD_NUM, NOPE + V_DIM)
        return kv[:, :, :NOPE], kv[:, :, NOPE:]

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rows = self.cos_sin_cache[positions]
        half = rows.shape[-1] // 2
        cos = rows[:, :half].to(x.dtype)
        sin = rows[:, half:].to(x.dtype)
        return _apply_neox_rope(x, cos, sin)

    def _ref_kv_full(self, ck, kpe_roped, q_slice, positions):
        """Build reference q/k (192) and v (128) for one stream's tokens."""
        k_nope, value = self._decompress(ck)  # [N,H,nope], [N,H,v]
        k = torch.cat(
            [k_nope, kpe_roped.float()[:, None, :].expand(-1, HEAD_NUM, -1)], dim=-1
        )
        q_nope = q_slice[:, :, :NOPE].float()
        q_pe = self._rope(q_slice[:, :, NOPE:], positions).float()
        q = torch.cat([q_nope, q_pe], dim=-1)
        return q, k, value

    def _assert_close(self, out, ref):
        out = out.float()
        ref = ref.float().reshape(out.shape)
        cos = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0), dim=1
        )
        self.assertGreater(cos.item(), 0.99, f"cosine sim too low: {cos.item()}")

    # -- tests -----------------------------------------------------------------

    def _run_no_reuse(self, input_lengths: List[int]):
        total = sum(input_lengths)
        n = len(input_lengths)
        # one block row per stream, enough blocks to hold the stream
        max_blocks = (max(input_lengths) + PAGE_SIZE - 1) // PAGE_SIZE
        block_ids = torch.arange(1, n * max_blocks + 1, dtype=torch.int32).view(
            n, max_blocks
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(n, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_host = block_ids

        q = torch.randn(
            total, HEAD_NUM, QK_DIM, dtype=torch.bfloat16, device=self.device
        )
        ck = torch.randn(total, KV_LORA, dtype=torch.bfloat16, device=self.device)
        kpe = torch.randn(total, ROPE, dtype=torch.bfloat16, device=self.device)

        cache = torch.zeros(
            n * max_blocks + 1,
            PAGE_SIZE,
            KV_LORA + ROPE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        kv_cache = _FakeKVCache(cache, PAGE_SIZE)

        impl = _build_impl(attn_inputs, self.weights, self.cos_sin_cache)
        out = impl.forward(q.clone(), ck.clone(), kpe.clone(), kv_cache, 0)

        # reference: per-stream causal attention over current tokens
        refs = []
        off = 0
        for L in input_lengths:
            sl = slice(off, off + L)
            positions = torch.arange(0, L, device=self.device)
            kpe_roped = self._rope(kpe[sl], positions)
            qf, kf, vf = self._ref_kv_full(ck[sl], kpe_roped, q[sl], positions)
            o_ref, _ = attention_ref(1, qf, kf, vf, causal=True, sm_scale=self.scale)
            refs.append(o_ref.reshape(L, HEAD_NUM * V_DIM))
            off += L
        self._assert_close(out, torch.cat(refs, dim=0))

    def test_single_stream_no_reuse(self):
        self._run_no_reuse([7])
        self._run_no_reuse([200])

    def test_multi_stream_no_reuse(self):
        self._run_no_reuse([7, 200, 33])

    def test_single_stream_with_reuse(self):
        reuse_len, L = 128, 100
        max_blocks = (reuse_len + L + PAGE_SIZE - 1) // PAGE_SIZE
        block_ids = torch.arange(1, max_blocks + 1, dtype=torch.int32).view(
            1, max_blocks
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([L], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([reuse_len], dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_host = block_ids

        q = torch.randn(L, HEAD_NUM, QK_DIM, dtype=torch.bfloat16, device=self.device)
        ck = torch.randn(L, KV_LORA, dtype=torch.bfloat16, device=self.device)
        kpe = torch.randn(L, ROPE, dtype=torch.bfloat16, device=self.device)

        cache = torch.zeros(
            max_blocks + 1,
            PAGE_SIZE,
            KV_LORA + ROPE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Pre-fill the reused prefix [0, reuse_len) with latent = [ck ++ roped k_pe].
        pre_ck = torch.randn(
            reuse_len, KV_LORA, dtype=torch.bfloat16, device=self.device
        )
        pre_kpe_raw = torch.randn(
            reuse_len, ROPE, dtype=torch.bfloat16, device=self.device
        )
        pre_positions = torch.arange(0, reuse_len, device=self.device)
        pre_kpe_roped = self._rope(pre_kpe_raw, pre_positions)
        pre_latent = torch.cat([pre_ck, pre_kpe_roped], dim=-1)
        row = block_ids[0]
        for pos in range(reuse_len):
            cache[row[pos // PAGE_SIZE], pos % PAGE_SIZE, :] = pre_latent[pos]

        kv_cache = _FakeKVCache(cache, PAGE_SIZE)

        impl = _build_impl(attn_inputs, self.weights, self.cos_sin_cache)
        out = impl.forward(q.clone(), ck.clone(), kpe.clone(), kv_cache, 0)

        # reference: prefix (as stored) ++ current, bottom-right causal.
        cur_positions = torch.arange(reuse_len, reuse_len + L, device=self.device)
        cur_kpe_roped = self._rope(kpe, cur_positions)
        q_cur, k_cur, v_cur = self._ref_kv_full(ck, cur_kpe_roped, q, cur_positions)
        k_pre = torch.cat(
            [
                self._decompress(pre_ck)[0],
                pre_kpe_roped.float()[:, None, :].expand(-1, HEAD_NUM, -1),
            ],
            dim=-1,
        )
        v_pre = self._decompress(pre_ck)[1]
        k_full = torch.cat([k_pre, k_cur], dim=0)
        v_full = torch.cat([v_pre, v_cur], dim=0)
        o_ref, _ = attention_ref(
            1, q_cur, k_full, v_full, causal=True, sm_scale=self.scale
        )
        self._assert_close(out, o_ref.reshape(L, HEAD_NUM * V_DIM))


if __name__ == "__main__":
    main()

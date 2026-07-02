"""Precision regression for the v3 fused QKV+bias+RoPE+packed-K/V kernel.

The kernel under test is ``add_fusedQKV_bias_transpose_prefill_v3_all_heads``
in ``fused_rope_kvcache_kernel.cu`` (5.77x kernel speedup). V3 is now the
default — there is no opt-in env var. Dispatch is gated entirely by the
production hot-path config below; anything outside it falls through to V1.

We assert against a torch fp32 reference for Q AND decode K/V back from the
paged cache for direct layout assertion. ``store_cache=true`` (i.e. a real
LayerKVCache) is mandatory for V3 to fire (see FusedRopeKVCacheOp.cc:219 +
the dispatch guard at fused_rope_kvcache_kernel.cu:686/1038); passing
``kv_cache=None`` silently falls through to V1 and the test would not
exercise V3 at all.

V3 activation requires the production hot-path config:
    bf16 + paged_fmha + prefix=0 + RopeStyle::Base + store_q + store_cache
    + no store_qkv + no store_kv + Tcache=BASE + no FP8 quant + no logn_attn
    + head_dim%8==0 + rot_dim%(VEC*2)==0 + rot_dim<=head_dim.

We build inputs that match the Qwen3.5-9B Full-Attn rank shape
(head_num=8, head_num_kv=2, head_dim=256), which is what V3 was tuned for.
Skips automatically off ROCm.
"""

import unittest
from typing import List

import torch

try:
    from rtp_llm.ops import AttentionConfigs, RopeConfig, RopeStyle
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCachePrefillOpAsm,
        FusedRopeKVCachePrefillOpNonAsm,
        LayerKVCache,
        PyAttentionInputs,
        get_typemeta,
    )

    _OPS_IMPORTABLE = True
except ImportError:
    _OPS_IMPORTABLE = False


def _is_rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _make_attn_configs(
    head_num: int,
    head_num_kv: int,
    head_dim: int,
    rope_dim: int,
    rope_base: int = 10000,
    tokens_per_block: int = 16,
):
    cfg = AttentionConfigs()
    cfg.head_num = head_num
    cfg.kv_head_num = head_num_kv
    cfg.size_per_head = head_dim
    cfg.tokens_per_block = tokens_per_block
    cfg.kernel_tokens_per_block = tokens_per_block
    cfg.is_causal = True
    cfg.use_mla = False
    cfg.dtype = torch.bfloat16

    rope = RopeConfig()
    rope.dim = rope_dim
    rope.base = rope_base
    rope.scale = 1.0
    rope.style = RopeStyle.Base
    cfg.rope_config = rope
    return cfg


def _build_block_table(
    input_lengths: List[int], tokens_per_block: int, device: torch.device
):
    """Assign each batch a contiguous range of unique block ids.

    Returns ``(block_id_table[B, max_blocks], num_blocks_used, per_batch_block_ids)``.
    The third return value lets the test decode K/V back from the pool by walking
    the same block assignment used by the kernel.
    """
    max_blocks = max((sl + tokens_per_block - 1) // tokens_per_block for sl in input_lengths)
    table = torch.zeros(len(input_lengths), max_blocks, dtype=torch.int32, device="cpu")
    next_block = 0
    per_batch: List[List[int]] = []
    for b, sl in enumerate(input_lengths):
        nb = (sl + tokens_per_block - 1) // tokens_per_block
        ids = list(range(next_block, next_block + nb))
        per_batch.append(ids)
        for j, bid in enumerate(ids):
            table[b, j] = bid
        next_block += nb
    return table.to(device), next_block, per_batch


def _make_prefill_inputs(
    input_lengths: List[int],
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_block: int = 16,
):
    """Minimal PyAttentionInputs for prefix==0 prefill with paged KV.

    Four less-obvious requirements that the C++ side reads in prepare/forward:
      - ``attn_inputs.dtype`` (not ``cfg.dtype``): drives torchDTypeToDataType
      - ``cu_seqlens`` (int32 on device): used by the kernel as a per-batch index
      - ``padding_offset`` (int32 on device, len == token_num): kernel reads
        ``padding_offset[token_idx]``; for tests without padding it is all-zero
      - ``kv_cache_kernel_block_id_{host,device}``: required when V3 fires —
        ``store_cache=true`` triggers ``getKBlockPtr(batch_idx, dst_kv_seq_idx)``
        which indexes into the converted offset array built from this table.

    Returns ``(attn_inputs, per_batch_block_ids)``. The block-id list lets the
    caller decode the paged pool back into per-token K/V for assertion.
    """
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = True
    attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
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
    for sl in input_lengths:
        cu.append(cu[-1] + sl)
    cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
    attn_inputs.cu_seqlens_device = cu_tensor
    attn_inputs.cu_kv_seqlens_device = cu_tensor

    # Kernel reads padding_offset[t] and computes:
    #   tgt = t + padding_offset[t]
    #   batch_idx = tgt / max_seq_len; seq_idx = tgt % max_seq_len
    # So padding_offset[t] for token t in batch b must be (b * max_seq_len - cu[b]).
    # All-zero only happens to work when input_lengths are uniform AND max_seq_len
    # divides global_token_idx the right way; varied seqlens break it.
    max_seq_len = max(input_lengths)
    padding_offset_list = []
    for b, sl in enumerate(input_lengths):
        offset = b * max_seq_len - cu[b]
        padding_offset_list.extend([offset] * sl)
    attn_inputs.padding_offset = torch.tensor(
        padding_offset_list, dtype=torch.int32, device=device
    )

    # Paged KV cache block table — required for V3 to actually fire (otherwise
    # store_cache=false and the dispatch guard rejects it). Each batch gets its
    # own contiguous block range so per-token decode below is unambiguous.
    block_table_dev, _, per_batch_block_ids = _build_block_table(
        input_lengths, tokens_per_block, device
    )
    attn_inputs.kv_cache_kernel_block_id_device = block_table_dev
    attn_inputs.kv_cache_kernel_block_id = block_table_dev.to(
        "cpu", non_blocking=False
    )
    return attn_inputs, per_batch_block_ids


def _alloc_paged_kv_cache(
    num_blocks: int,
    head_num_kv: int,
    tokens_per_block: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Production 5D MHA layout: [block_num, 2, hk, ps, hd] (see OpDefs.h:88)."""
    pool = torch.zeros(
        [num_blocks, 2, head_num_kv, tokens_per_block, head_dim],
        dtype=dtype,
        device=device,
    )
    layer_cache = LayerKVCache()
    layer_cache.kv_cache_base = pool
    return layer_cache, pool


def _decode_kv_from_pool(
    pool: torch.Tensor,
    per_batch_block_ids: List[List[int]],
    input_lengths: List[int],
    head_num_kv: int,
    head_dim: int,
    tokens_per_block: int,
    v_vec_layout: bool,
):
    """Read K/V back from the paged pool into ``[total_tokens, hk, hd]``.

    The kernel writes K/V via ``getKLocalIdx`` / ``getVLocalIdx`` (see
    ``kv_cache_utils.h:213-241``), which interpret each per-block byte buffer as:

        K (always vectorized):           [hk, hd/vs, ps, vs]
        V (NonAsm, v_vec_layout=False):  [hk, hd, ps]
        V (Asm,    v_vec_layout=True):   [hk, ps/vs, hd, vs]

    where ``vs = 16 / sizeof(elem)`` (8 for bf16). Pool storage is
    ``[block, 2, hk, ps, hd]`` torch-contig, so each ``pool[blk, k]`` block has
    ``hk*ps*hd`` contiguous elements that we re-view into the kernel's layout.
    """
    vs = 16 // pool.element_size()
    assert head_dim % vs == 0, "head_dim must be divisible by vs for K layout"
    if v_vec_layout:
        assert tokens_per_block % vs == 0, "ps must be divisible by vs for V_VEC"

    total = sum(input_lengths)
    K = torch.zeros(total, head_num_kv, head_dim, dtype=pool.dtype, device=pool.device)
    V = torch.zeros(total, head_num_kv, head_dim, dtype=pool.dtype, device=pool.device)

    pos = 0
    for b, sl in enumerate(input_lengths):
        for tok in range(sl):
            blk = per_batch_block_ids[b][tok // tokens_per_block]
            local = tok % tokens_per_block
            # K: re-view block storage as [hk, hd/vs, ps, vs] then slice token.
            k_kernel = pool[blk, 0].contiguous().view(
                head_num_kv, head_dim // vs, tokens_per_block, vs
            )
            # k_natural[h, p, d] = k_kernel[h, d//vs, p, d%vs]
            K[pos + tok] = (
                k_kernel.permute(0, 2, 1, 3).reshape(
                    head_num_kv, tokens_per_block, head_dim
                )[:, local, :]
            )
            v_block_flat = pool[blk, 1].contiguous().view(-1)
            if v_vec_layout:
                v_kernel = v_block_flat.view(
                    head_num_kv, tokens_per_block // vs, head_dim, vs
                )
                # v_natural[h, p, d] = v_kernel[h, p//vs, d, p%vs]
                V[pos + tok] = v_kernel[:, local // vs, :, local % vs]
            else:
                v_kernel = v_block_flat.view(head_num_kv, head_dim, tokens_per_block)
                V[pos + tok] = v_kernel[:, :, local]
        pos += sl
    return K, V


def _torch_reference(
    qkv: torch.Tensor,
    head_num: int,
    head_num_kv: int,
    head_dim: int,
    rope_dim: int,
    rope_base: float,
    rope_scale: float,
    cu_seqlens: torch.Tensor,
):
    """Bias-free QKV + split-halves (NeoX-style) RoPE + packed-K/V layout.

    Position id = within-batch local index (0..seq_len-1 per request). Both V1
    and V3 produce this:
      * V1 derives it as ``tgt_token_idx % seq_len`` (with padding_offset).
      * V3 self-computes the same way; its dispatch guard enforces prefix==0,
        so batch-local seq_idx == absolute RoPE position.

    inv_freq = base ** (-2d / rot_dim) for d ∈ [0, half), fp32 math then
    cast back to bf16 to mirror the kernel's ``__floats2bfloat162_rn``.
    """
    token_num, hidden = qkv.shape
    assert hidden == (head_num + 2 * head_num_kv) * head_dim
    q_size = head_num * head_dim
    kv_size = head_num_kv * head_dim
    Q = qkv[:, :q_size].reshape(token_num, head_num, head_dim).float()
    K = qkv[:, q_size : q_size + kv_size].reshape(
        token_num, head_num_kv, head_dim
    ).float()
    V = qkv[:, q_size + kv_size :].reshape(token_num, head_num_kv, head_dim)

    pos = torch.zeros(token_num, dtype=torch.float32, device=qkv.device)
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        seq_len = cu[i + 1] - cu[i]
        pos[cu[i] : cu[i + 1]] = torch.arange(
            seq_len, dtype=torch.float32, device=qkv.device
        )

    half = rope_dim // 2
    inv_freq = rope_base ** (
        -2.0 * torch.arange(half, dtype=torch.float32, device=qkv.device) / rope_dim
    )
    angle = pos.unsqueeze(1) * inv_freq.unsqueeze(0) * rope_scale  # [T, half]
    cos = torch.cos(angle)  # [T, half]
    sin = torch.sin(angle)

    def rope_split_halves(x: torch.Tensor) -> torch.Tensor:
        # x: [T, H, D] in fp32
        x_lo = x[..., :half]
        x_hi = x[..., half:rope_dim]
        out_lo = x_lo * cos.unsqueeze(1) - x_hi * sin.unsqueeze(1)
        out_hi = x_hi * cos.unsqueeze(1) + x_lo * sin.unsqueeze(1)
        out = x.clone()
        out[..., :half] = out_lo
        out[..., half:rope_dim] = out_hi
        return out

    Q_ref = rope_split_halves(Q).to(qkv.dtype)
    K_ref = rope_split_halves(K).to(qkv.dtype)
    return Q_ref, K_ref, V


@unittest.skipUnless(_is_rocm(), "ROCm not available")
@unittest.skipUnless(_OPS_IMPORTABLE, "rtp_llm rocm bindings not importable")
class FusedQKVTransposePrefillTest(unittest.TestCase):
    """Drives FusedRopeKVCachePrefillOpNonAsm and asserts Q + paged K/V == torch ref.

    V3 dispatch requires ``store_cache=true`` (FusedRopeKVCacheOp.cc:219), which
    is set from ``kv_cache.has_value()`` — so to actually trigger V3 the test
    MUST allocate a real LayerKVCache + block table. Earlier this suite passed
    ``kv_cache=None`` and silently exercised V1 only; we now wire up the paged
    pool and decode K/V back from it for assertion. NonAsm routes through
    ``invokeAddFusedQKVBiasTransposePrefillV1`` which calls V3 with
    ``v_vec_layout=false`` (V layout is flat ``[hk, hd, ps]`` per block).
    """

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.tokens_per_block = 16
        self.v_vec_layout = False  # NonAsm op → V cache is flat [hk, hd, ps]

    def _run(
        self,
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        head_dim: int,
        rope_dim: int = None,
    ):
        if rope_dim is None:
            rope_dim = head_dim
        cfg = _make_attn_configs(
            head_num, head_num_kv, head_dim,
            rope_dim=rope_dim, tokens_per_block=self.tokens_per_block,
        )
        op = FusedRopeKVCachePrefillOpNonAsm(cfg)
        op.use_paged_fmha = True  # required for v3 hot path
        attn_inputs, per_batch_block_ids = _make_prefill_inputs(
            input_lengths, self.device, self.dtype, self.tokens_per_block
        )
        params = op.prepare(attn_inputs)

        total_tokens = sum(input_lengths)
        hidden = (head_num + 2 * head_num_kv) * head_dim
        qkv = torch.randn(total_tokens, hidden, dtype=self.dtype, device=self.device)

        # Pool sized so every batch's contiguous block range fits without overlap.
        num_blocks = sum(
            (sl + self.tokens_per_block - 1) // self.tokens_per_block
            for sl in input_lengths
        )
        layer_cache, pool = _alloc_paged_kv_cache(
            num_blocks, head_num_kv, self.tokens_per_block, head_dim,
            self.dtype, self.device,
        )

        q_out, _, _ = op.forward(qkv, kv_cache=layer_cache, params=params)

        q_ref, k_ref, v_ref = _torch_reference(
            qkv,
            head_num=head_num,
            head_num_kv=head_num_kv,
            head_dim=head_dim,
            rope_dim=cfg.rope_config.dim,
            rope_base=float(cfg.rope_config.base),
            rope_scale=float(cfg.rope_config.scale),
            cu_seqlens=attn_inputs.cu_seqlens_device.cpu(),
        )

        # bf16 RoPE: ~2-bit ULP after 2 fp16 muls + 1 add; 1e-2 atol is the
        # convention used by the rest of the rocm kernel suite.
        torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)

        # Decode K/V from the paged pool using the kernel's documented layout
        # (kv_cache_utils.h:213-241). Mismatch here would catch any future
        # regression where V3 forgets to template on V_VEC_LAYOUT, or writes
        # K with the wrong vector stride.
        k_decoded, v_decoded = _decode_kv_from_pool(
            pool, per_batch_block_ids, input_lengths,
            head_num_kv, head_dim, self.tokens_per_block, self.v_vec_layout,
        )
        torch.testing.assert_close(k_decoded, k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(v_decoded, v_ref, atol=1e-2, rtol=1e-2)

    # ---- Qwen3.5-9B Full-Attn hot path (V3 specialization target) ----

    def test_qwen35_9b_uniform(self):
        # Uniform-length batch, multi-token; covers the inner k_tok loop.
        self._run([16, 16, 16], head_num=8, head_num_kv=2, head_dim=256)

    def test_qwen35_9b_varied(self):
        # Mixed seqlens — exercises padding_offset / batch_idx arithmetic.
        self._run([7, 23, 1, 11], head_num=8, head_num_kv=2, head_dim=256)

    def test_qwen35_9b_single_long(self):
        # Single long sequence — common for the 15k-token prefill bench.
        self._run([257], head_num=8, head_num_kv=2, head_dim=256)

    def test_token_count_not_multiple_of_kblock(self):
        # V3 launches one block per K_TOK=4 token chunk. When token_num % 4 != 0,
        # the tail block has tok_local lanes where token_idx >= token_num. They
        # MUST stay in the kernel and participate in the per-head __syncthreads
        # calls — early-returning would deadlock the surviving lanes (or trigger
        # UB). Tests every residue: 1, 2, 3 tail tokens.
        for total in (5, 6, 7, 13):  # %4 → 1, 2, 3, 1
            with self.subTest(total=total):
                self._run([total], head_num=8, head_num_kv=2, head_dim=256)

    # ---- Fallback shapes (V1 path; head_dim != 256 is outside V3 tuning) ----

    def test_multi_batch_packed_prefill_consistency(self):
        # Regression: prefill packing of N requests into one kernel launch must
        # give each request the same output it would get in a batch=1 prefill
        # (modulo bf16 ULP). V3 self-computes batch-local seq_idx from
        # padding_offset / seq_len, so this also locks in that behavior.
        # Q-only here (the per-token K/V layout assertion is covered by _run);
        # this test focuses on packing invariance, not paged-cache layout.
        seqs = [11, 11, 11]
        cfg = _make_attn_configs(8, 2, 256, rope_dim=256, tokens_per_block=self.tokens_per_block)
        op = FusedRopeKVCachePrefillOpNonAsm(cfg)
        op.use_paged_fmha = True
        head_num, head_num_kv, head_dim = 8, 2, 256
        hidden = (head_num + 2 * head_num_kv) * head_dim

        torch.manual_seed(42)
        per_request_qkv = [
            torch.randn(sl, hidden, dtype=self.dtype, device=self.device)
            for sl in seqs
        ]
        solo_q = []
        for x in per_request_qkv:
            ai, _ = _make_prefill_inputs(
                [x.shape[0]], self.device, self.dtype, self.tokens_per_block
            )
            lkv, _ = _alloc_paged_kv_cache(
                1, head_num_kv, self.tokens_per_block, head_dim,
                self.dtype, self.device,
            )
            params = op.prepare(ai)
            q, _, _ = op.forward(x, kv_cache=lkv, params=params)
            solo_q.append(q.clone())

        packed = torch.cat(per_request_qkv, dim=0)
        ai, _ = _make_prefill_inputs(seqs, self.device, self.dtype, self.tokens_per_block)
        lkv, _ = _alloc_paged_kv_cache(
            len(seqs), head_num_kv, self.tokens_per_block, head_dim,
            self.dtype, self.device,
        )
        params = op.prepare(ai)
        q_packed, _, _ = op.forward(packed, kv_cache=lkv, params=params)

        offset = 0
        for i, sl in enumerate(seqs):
            torch.testing.assert_close(
                q_packed[offset:offset + sl], solo_q[i], atol=1e-2, rtol=1e-2,
                msg=f"Q mismatch for request {i} in packed prefill",
            )
            offset += sl

    def test_smaller_head_dim_falls_back(self):
        # head_dim=128: V3 try_launch passes head_dim%8==0 check, but the
        # kernel was tuned for 256. Both V1 and V3 should still produce
        # the reference output. KV cache layout assertion covers both.
        self._run([12, 19], head_num=16, head_num_kv=4, head_dim=128)

    def test_partial_rotary(self):
        # Partial rotary: rope_dim < head_dim. V3's partial-rotary support
        # rotates only the first rot_dim/2 pairs; the rest pass through.
        # Reuse _run so K/V layout is also asserted.
        self._run([13, 17], head_num=8, head_num_kv=2, head_dim=256, rope_dim=64)


@unittest.skipUnless(_is_rocm(), "ROCm not available")
@unittest.skipUnless(_OPS_IMPORTABLE, "rtp_llm rocm bindings not importable")
class FusedQKVPrefixPrefillTest(unittest.TestCase):
    """NonAsm V1 prefix-prompt regression. V3 is unreachable here because the
    dispatch guard rejects ``max_prefix_prompt_length != 0``; this exercises
    the V1 fallback's built-in ``seqidx + prefix_prompt_length`` logic in
    context_rope, which would have been silently overridden if the C++ side
    still constructed an unprefixed position_ids tensor."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_prefix_prompt_rope_position(self):
        # Single batch, sl=5, prefix=8: RoPE positions for the new tokens
        # must be 8..12 (continuation), not 0..4. V3 dispatch rejects
        # max_prefix_prompt_length != 0, so this exercises the V1 fallback's
        # ``seqidx + prefix_prompt_length`` logic. Q-only (K layout assertion
        # is covered by the prefix==0 hot-path tests).
        head_num, head_num_kv, head_dim = 8, 2, 256
        sl, prefix = 5, 8
        cfg = _make_attn_configs(head_num, head_num_kv, head_dim, rope_dim=head_dim)
        op = FusedRopeKVCachePrefillOpNonAsm(cfg)
        op.use_paged_fmha = True

        attn_inputs, _ = _make_prefill_inputs([sl], self.device, self.dtype)
        attn_inputs.prefix_lengths = torch.tensor(
            [prefix], dtype=torch.int32, device="cpu"
        )
        params = op.prepare(attn_inputs)

        hidden = (head_num + 2 * head_num_kv) * head_dim
        qkv = torch.randn(sl, hidden, dtype=self.dtype, device=self.device)
        q_out, _, _ = op.forward(qkv, kv_cache=None, params=params)

        token_num = sl
        Q = qkv[:, : head_num * head_dim].reshape(token_num, head_num, head_dim).float()
        pos = torch.arange(prefix, prefix + sl, dtype=torch.float32, device=qkv.device)
        half = head_dim // 2
        inv_freq = float(cfg.rope_config.base) ** (
            -2.0 * torch.arange(half, dtype=torch.float32, device=qkv.device) / head_dim
        )
        angle = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos, sin = torch.cos(angle), torch.sin(angle)

        def rot(x):
            lo, hi = x[..., :half], x[..., half:]
            return torch.cat(
                [lo * cos.unsqueeze(1) - hi * sin.unsqueeze(1),
                 hi * cos.unsqueeze(1) + lo * sin.unsqueeze(1)], dim=-1)

        q_ref = rot(Q).to(self.dtype)
        torch.testing.assert_close(q_out.reshape(sl, head_num, head_dim),
                                   q_ref, atol=1e-2, rtol=1e-2)


@unittest.skipUnless(_is_rocm(), "ROCm not available")
@unittest.skipUnless(_OPS_IMPORTABLE, "rtp_llm rocm bindings not importable")
class FusedQKVTransposePrefillAsmTest(unittest.TestCase):
    """ASM-path mirror of FusedQKVTransposePrefillTest.

    Drives FusedRopeKVCachePrefillOpAsm (production hot path under
    USE_ASM_PA=1), which routes through invokeAddFusedQKVBiasTransposePrefill
    → V3 dispatch with V_VEC_LAYOUT=true (templated VLocalIdx).

    This class catches the ASM-only regression that hit production: V3 was
    initially templated only for the NonAsm V1 V layout
    ([numHeads, dimsPerHead, mTokensPerBlock]) and mismatched the ASM-side
    reader's templated <BASE> layout ([numHeads, mTokensPerBlock/8,
    dimsPerHead, 8]). Q precision is identical to the NonAsm path, but the
    test's value is asserting V3 fires here at all and produces correct Q
    under the same Qwen3.5-9B Full-Attn shape.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.tokens_per_block = 16
        # Asm op routes through invokeAddFusedQKVBiasTransposePrefill which
        # calls V3 with v_vec_layout=true (V layout = [hk, ps/vs, hd, vs]).
        self.v_vec_layout = True

    def _run(
        self,
        input_lengths,
        head_num,
        head_num_kv,
        head_dim,
        rope_dim=None,
    ):
        if rope_dim is None:
            rope_dim = head_dim
        cfg = _make_attn_configs(
            head_num, head_num_kv, head_dim,
            rope_dim=rope_dim, tokens_per_block=self.tokens_per_block,
        )
        op = FusedRopeKVCachePrefillOpAsm(cfg)
        op.use_paged_fmha = True
        attn_inputs, per_batch_block_ids = _make_prefill_inputs(
            input_lengths, self.device, self.dtype, self.tokens_per_block
        )
        params = op.prepare(attn_inputs)

        total_tokens = sum(input_lengths)
        hidden = (head_num + 2 * head_num_kv) * head_dim
        qkv = torch.randn(total_tokens, hidden, dtype=self.dtype, device=self.device)

        num_blocks = sum(
            (sl + self.tokens_per_block - 1) // self.tokens_per_block
            for sl in input_lengths
        )
        layer_cache, pool = _alloc_paged_kv_cache(
            num_blocks, head_num_kv, self.tokens_per_block, head_dim,
            self.dtype, self.device,
        )

        q_out, _, _ = op.forward(qkv, kv_cache=layer_cache, params=params)

        q_ref, k_ref, v_ref = _torch_reference(
            qkv,
            head_num=head_num,
            head_num_kv=head_num_kv,
            head_dim=head_dim,
            rope_dim=rope_dim,
            rope_base=float(cfg.rope_config.base),
            rope_scale=float(cfg.rope_config.scale),
            cu_seqlens=attn_inputs.cu_seqlens_device.cpu(),
        )
        torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)

        # Asserts V3<true> wrote V in the templated [hk, ps/vs, hd, vs] layout
        # and K in the standard vectorized layout. This is the regression that
        # hit production: V3 was initially templated only on the NonAsm flat V
        # layout and silently mismatched the ASM-side templated reader.
        k_decoded, v_decoded = _decode_kv_from_pool(
            pool, per_batch_block_ids, input_lengths,
            head_num_kv, head_dim, self.tokens_per_block, self.v_vec_layout,
        )
        torch.testing.assert_close(k_decoded, k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(v_decoded, v_ref, atol=1e-2, rtol=1e-2)

    def test_qwen35_9b_full_rotary(self):
        # Hot-path shape with full rotary — V3<true> dispatch.
        self._run([16, 16, 16], head_num=8, head_num_kv=2, head_dim=256)

    def test_qwen35_9b_partial_rotary(self):
        # Production Qwen3.5-9B shape: rope_dim=64 (partial_rotary_factor=0.25),
        # head_dim=256. This is THE shape that exposed the V layout bug under
        # USE_ASM_PA=1 — V3 must template on V_VEC_LAYOUT=true here.
        self._run([16, 16, 16], head_num=8, head_num_kv=2, head_dim=256, rope_dim=64)

    def test_qwen35_9b_partial_varied(self):
        # Mixed seqlens — exercises padding_offset / batch_idx arithmetic on
        # the ASM dispatch path under partial rotary.
        self._run([7, 23, 1, 11], head_num=8, head_num_kv=2, head_dim=256, rope_dim=64)

    def test_single_long_partial(self):
        self._run([257], head_num=8, head_num_kv=2, head_dim=256, rope_dim=64)


if __name__ == "__main__":
    unittest.main()

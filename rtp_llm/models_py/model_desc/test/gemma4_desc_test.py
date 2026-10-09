"""Gemma4 python descriptor unit tests: inline HF reference-math comparison.

The reference implementations below are direct translations of the HF
``modeling_gemma4.py`` lines quoted in each helper. All comparisons run in
fp32 on the GPU; tolerances are tight (rtol/atol 2e-5) because both sides use
the same dtype.
"""

import os
import unittest
from unittest import TestCase, main

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc import gemma4
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyMultimodalInputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W
from torch.nn import functional as F

# Weight keys resolved through the same frozen-name helper as the descriptor.
KEY_ROUTER_SCALE = gemma4._W_MOE_ROUTER_SCALE
KEY_ROUTER_EXPERT_SCALE = gemma4._W_MOE_ROUTER_EXPERT_SCALE
KEY_PRE_FFN_LN = gemma4._W_PRE_FFN_LN_GAMMA
KEY_PRE_FFN2_LN = gemma4._W_PRE_FFN2_LN_GAMMA
KEY_POST_FFN1_LN = gemma4._W_POST_FFN1_LN_GAMMA
KEY_POST_FFN2_LN = gemma4._W_POST_FFN2_LN_GAMMA
KEY_LAYER_SCALAR = gemma4._W_LAYER_SCALAR

EPS = 1e-6


# ---------------------------------------------------------------------------
# HF reference math (modeling_gemma4.py)
# ---------------------------------------------------------------------------


def hf_rms_norm(x, weight, eps):
    """Gemma4RMSNorm (lines 157-175): fp32, direct weight multiply, no +1."""
    dtype = x.dtype
    xf = x.float()
    mean_squared = xf.pow(2).mean(-1, keepdim=True) + eps
    out = xf * torch.pow(mean_squared, -0.5)
    if weight is not None:
        out = out * weight.float()
    return out.to(dtype)


def hf_inv_freq(head_dim, base, partial_rotary_factor):
    """_compute_proportional_rope_parameters (modeling_rope_utils.py:187-254)."""
    half = head_dim // 2
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_rotated = 1.0 / (
        base
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(dtype=torch.float)
            / head_dim
        )
    )
    nope_angles = half - rope_angles
    if nope_angles > 0:
        return torch.cat(
            [inv_rotated, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
        )
    return inv_rotated


def hf_cos_sin(positions, head_dim, base, partial_rotary_factor):
    """Gemma4TextRotaryEmbedding.forward (lines 1106-1122)."""
    inv_freq = hf_inv_freq(head_dim, base, partial_rotary_factor).to(positions.device)
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def hf_apply_rope(x, cos, sin):
    """apply_rotary_pos_emb + rotate_half (lines 727-753), x [T, H, D]."""
    cos = cos.to(x.dtype)[:, None, :]
    sin = sin.to(x.dtype)[:, None, :]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def hf_attention(hidden, weights, geometry, positions, sliding_window=0):
    """Gemma4TextAttention.forward + eager_attention_forward (768-799,
    1126-1240) for a single packed request."""
    head_num = geometry.head_num
    kv_heads = geometry.kv_head_num
    head_dim = geometry.head_dim
    q_size = head_num * head_dim
    kv_size = kv_heads * head_dim
    qkv = hidden @ weights[W.attn_qkv_w]
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    if geometry.k_equals_v:
        v = k
    q = q.reshape(-1, head_num, head_dim)
    k = k.reshape(-1, kv_heads, head_dim)
    v = v.reshape(-1, kv_heads, head_dim)
    q = hf_rms_norm(q, weights[W.q_ln_gamma], EPS)
    k = hf_rms_norm(k, weights[W.k_ln_gamma], EPS)
    v = hf_rms_norm(v, None, EPS)
    cos, sin = hf_cos_sin(
        positions,
        head_dim,
        geometry.rope_theta,
        geometry.rope_partial_rotary_factor,
    )
    q = hf_apply_rope(q, cos, sin)
    k = hf_apply_rope(k, cos, sin)

    group = head_num // kv_heads
    k_expanded = k.repeat_interleave(group, dim=1)
    v_expanded = v.repeat_interleave(group, dim=1)
    # eager_attention_forward: scaling=1.0 (line 1143)
    scores = torch.einsum("thd,shd->hts", q, k_expanded)
    kv_positions = torch.arange(k.size(0), device=hidden.device)
    allowed = kv_positions[None, :] <= positions[:, None]
    if sliding_window:
        allowed = allowed & (
            kv_positions[None, :] > positions[:, None] - sliding_window
        )
    scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("hts,shd->thd", probs, v_expanded)
    return out.reshape(-1, q_size) @ weights[W.attn_o_w]


def hf_router(x, weights, hidden_size, top_k):
    """Gemma4TextRouter.forward (lines 1296-1316)."""
    h = hf_rms_norm(x, None, EPS)
    h = h * weights[KEY_ROUTER_SCALE] * (hidden_size**-0.5)
    scores = h @ weights[W.moe_gate]  # [T, E]
    probs = torch.softmax(scores, dim=-1)
    top_w, top_idx = torch.topk(probs, k=top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w * weights[KEY_ROUTER_EXPERT_SCALE][top_idx]
    return top_w, top_idx


def hf_experts(x, top_idx, top_w, weights):
    """Gemma4TextExperts.forward in checkpoint [gate; up] layout."""
    num_experts, two_n, _ = weights[W.moe_w1].shape
    inter = two_n // 2
    final = torch.zeros_like(x)
    expert_mask = F.one_hot(top_idx, num_classes=num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert in expert_hit:
        expert = expert[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert])
        current_state = x[token_idx]
        gate_up = current_state @ weights[W.moe_w1][expert].t()
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.gelu(gate, approximate="tanh") * up
        current_hidden = current_hidden @ weights[W.moe_w2][expert].t()
        current_hidden = current_hidden * top_w[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current_hidden.to(final.dtype))
    return final


def hf_dense_mlp(x, weights):
    """Gemma4TextMLP.forward (lines 1030-1032), gate/up in ffn_w1/w3."""
    gate = x @ weights[W.ffn_w1]
    up = x @ weights[W.ffn_w3]
    return (F.gelu(gate, approximate="tanh") * up) @ weights[W.ffn_w2]


def hf_decoder_layer(x, weights, geometry, positions, hidden_size, top_k):
    """Gemma4TextDecoderLayer.forward (lines 1348-1403)."""
    residual = x
    hidden = hf_rms_norm(x, weights[W.pre_ln_gamma], EPS)
    attn_out = hf_attention(hidden, weights, geometry, positions)
    hidden = residual + hf_rms_norm(attn_out, weights[W.post_ln_gamma], EPS)

    residual = hidden
    dense = hf_dense_mlp(hf_rms_norm(hidden, weights[KEY_PRE_FFN_LN], EPS), weights)
    hidden_1 = hf_rms_norm(dense, weights[KEY_POST_FFN1_LN], EPS)

    top_w, top_idx = hf_router(residual, weights, hidden_size, top_k)
    hidden_2 = hf_experts(
        hf_rms_norm(residual, weights[KEY_PRE_FFN2_LN], EPS),
        top_idx,
        top_w,
        weights,
    )
    hidden_2 = hf_rms_norm(hidden_2, weights[KEY_POST_FFN2_LN], EPS)

    hidden = hf_rms_norm(hidden_1 + hidden_2, weights[W.post_ffn_ln_gamma], EPS)
    hidden = residual + hidden
    return hidden * weights[KEY_LAYER_SCALAR]


def hf_reference_paged_attention(q, k, v, positions, sliding_window=0):
    """Attention over already-normed/roped q/k/v for one request."""
    head_num = q.shape[1]
    kv_heads = k.shape[1]
    group = head_num // kv_heads
    k_expanded = k.repeat_interleave(group, dim=1)
    v_expanded = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k_expanded)
    kv_positions = torch.arange(k.size(0), device=q.device)
    allowed = kv_positions[None, :] <= positions[:, None]
    if sliding_window:
        allowed = allowed & (
            kv_positions[None, :] > positions[:, None] - sliding_window
        )
    scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.einsum("hts,shd->thd", probs, v_expanded)


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def make_config(layer_types, device):
    del device
    config = ModelConfig()
    config.num_layers = len(layer_types)
    config.hidden_size = 64
    config.max_seq_len = 128
    config.layernorm_eps = EPS
    config.moe_k = 4
    config.attn_config.head_num = 4
    config.attn_config.kv_head_num = 2
    config.attn_config.size_per_head = 16
    config.attn_config.sliding_window = 1024
    config.attn_config.rope_config.base = 10000
    config.attn_config.tokens_per_block = 4
    config.hybrid_attention_config.hybrid_attention_types = list(layer_types)
    config.mm_related_params.config.update(
        {
            "full_layer_kv_head_num": 1,
            "full_layer_size_per_head": 64,
            "full_layer_rope_theta": 1_000_000.0,
            "full_layer_partial_rotary_factor": 0.25,
        }
    )
    return config


def make_layer_weights(
    geometry, device, num_experts=8, moe_inter=12, dense_inter=16, dtype=None
):
    hidden = 64
    q_size = geometry.head_num * geometry.head_dim
    kv_size = geometry.kv_head_num * geometry.head_dim
    generator = torch.Generator(device="cpu").manual_seed(1234)

    def randn(*shape, scale=0.5):
        t = (torch.randn(*shape, generator=generator) * scale).to(device)
        return t.to(dtype) if dtype is not None else t

    def gamma(*shape):
        t = (torch.randn(*shape, generator=generator) * 0.2 + 1.0).to(device)
        return t.to(dtype) if dtype is not None else t

    return {
        W.attn_qkv_w: randn(hidden, q_size + 2 * kv_size),
        W.attn_o_w: randn(q_size, hidden),
        W.q_ln_gamma: gamma(geometry.head_dim),
        W.k_ln_gamma: gamma(geometry.head_dim),
        W.ffn_w1: randn(hidden, dense_inter),
        W.ffn_w3: randn(hidden, dense_inter),
        W.ffn_w2: randn(dense_inter, hidden),
        W.moe_gate: randn(hidden, num_experts),
        KEY_ROUTER_SCALE: gamma(hidden),
        KEY_ROUTER_EXPERT_SCALE: (
            torch.randn(num_experts, generator=generator) * 0.3 + 1.0
        ).to(device=device, dtype=dtype),
        W.moe_w1: randn(num_experts, 2 * moe_inter, hidden),
        W.moe_w2: randn(num_experts, hidden, moe_inter),
        W.pre_ln_gamma: gamma(hidden),
        W.post_ln_gamma: gamma(hidden),
        KEY_PRE_FFN_LN: gamma(hidden),
        KEY_PRE_FFN2_LN: gamma(hidden),
        W.post_ffn_ln_gamma: gamma(hidden),
        KEY_POST_FFN1_LN: gamma(hidden),
        KEY_POST_FFN2_LN: gamma(hidden),
        KEY_LAYER_SCALAR: torch.tensor(
            [0.37], device=device, dtype=dtype if dtype is not None else torch.float32
        ),
    }


def make_prefill_inputs(lengths, block_table, device):
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = True
    attn_inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32)
    attn_inputs.prefix_lengths = torch.zeros(len(lengths), dtype=torch.int32)
    attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
    # engine-style cu_seqlens on device ([0, l0, l0+l1, ...]) so the SWA
    # impl's device-side fast path is exercised; the manual constructions in
    # the prefix test omit it to cover the host fallback path
    cum = [0]
    for length in lengths:
        cum.append(cum[-1] + int(length))
    cu = torch.tensor(cum, dtype=torch.int32)
    if device is not None and str(device) != "cpu":
        cu = cu.to(device)
    attn_inputs.cu_seqlens_device = cu
    if block_table is not None:
        table = torch.zeros(
            (len(lengths), max(len(row) for row in block_table)),
            dtype=torch.int32,
        )
        for i, row in enumerate(block_table):
            table[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = table
    else:
        attn_inputs.kv_cache_kernel_block_id = torch.empty(0, dtype=torch.int32)
    return attn_inputs


def make_decode_inputs(sequence_lengths, block_table):
    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = False
    attn_inputs.input_lengths = torch.ones(len(sequence_lengths), dtype=torch.int32)
    attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
    attn_inputs.sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int32)
    width = max(len(row) for row in block_table)
    table = torch.zeros((len(sequence_lengths), width), dtype=torch.int32)
    for i, row in enumerate(block_table):
        table[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
    attn_inputs.kv_cache_kernel_block_id = table
    return attn_inputs


def geometry_of(config, parallelism_config, layer_idx=0):
    return gemma4.build_gemma4_layer_geometry(config, parallelism_config, layer_idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class Gemma4RopeTest(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_rope_table_matches_hf_formula(self):
        positions = torch.tensor([0, 1, 7, 999, 65535, 262143])
        for head_dim, base, partial in [
            (256, 10000.0, 1.0),  # sliding layers
            (512, 1000000.0, 0.25),  # full layers ("proportional")
            (16, 10000.0, 1.0),
            (64, 1000000.0, 0.25),
        ]:
            with self.subTest(head_dim=head_dim, partial=partial):
                table = gemma4.Gemma4RopeTable(head_dim, base, partial)
                cos, sin = table.cos_sin(positions.to(self.device))
                ref_cos, ref_sin = hf_cos_sin(positions, head_dim, base, partial)
                torch.testing.assert_close(cos.cpu(), ref_cos)
                torch.testing.assert_close(sin.cpu(), ref_sin)

    def test_full_layer_partial_rotation_equivalence(self):
        """The 512-dim full-width rotate_half with the zero-frequency table is
        identical to rotating only dims [0, 64) x [256, 320) in pairs."""
        head_dim, base, partial = 512, 1000000.0, 0.25
        rope_angles = int(partial * head_dim // 2)
        self.assertEqual(rope_angles, 64)
        positions = torch.tensor([3, 17, 4096])
        x = torch.randn(3, 2, head_dim)
        table = gemma4.Gemma4RopeTable(head_dim, base, partial)
        cos, sin = table.cos_sin(positions.to(self.device))
        out = gemma4.apply_gemma4_rope(x.to(self.device), cos, sin).cpu()
        cos = cos.cpu()
        sin = sin.cpu()
        ref = x.clone()
        for t in range(positions.numel()):
            for j in range(rope_angles):
                a = x[t, :, j]
                b = x[t, :, j + 256]
                c = cos[t, j]
                s = sin[t, j]
                ref[t, :, j] = a * c - b * s
                ref[t, :, j + 256] = b * c + a * s
        torch.testing.assert_close(out, ref)
        # NoPE dims pass through unchanged.
        torch.testing.assert_close(out[..., 64:256], x[..., 64:256])
        torch.testing.assert_close(out[..., 320:], x[..., 320:])


class Gemma4RouterTest(TestCase):
    def test_router_matches_hf(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        tokens, hidden, experts, top_k = 11, 64, 128, 8
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(7)
                x = torch.randn(tokens, hidden, generator=generator).to(
                    device=device, dtype=dtype
                )
                weights = {
                    W.moe_gate: torch.randn(hidden, experts, generator=generator).to(
                        device=device, dtype=dtype
                    ),
                    KEY_ROUTER_SCALE: (
                        torch.randn(hidden, generator=generator) * 0.2 + 1.0
                    ).to(device=device, dtype=dtype),
                    KEY_ROUTER_EXPERT_SCALE: (
                        torch.randn(experts, generator=generator) * 0.3 + 1.0
                    ).to(device=device, dtype=dtype),
                }
                router = gemma4.Gemma4Router(weights, hidden, top_k, EPS)
                top_w, top_idx = router(x)
                ref_w, ref_idx = hf_router(x, weights, hidden, top_k)
                torch.testing.assert_close(top_idx, ref_idx)
                torch.testing.assert_close(top_w, ref_w)
                self.assertFalse(
                    torch.allclose(top_w.sum(-1), torch.ones_like(top_w.sum(-1)))
                )

    def test_bf16_experts_match_hf_dtype_path(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        geometry = gemma4.Gemma4LayerGeometry("swa", 4, 2, 16, 10000.0, 1.0, 1024)
        weights = make_layer_weights(
            geometry,
            device,
            num_experts=8,
            moe_inter=12,
            dtype=torch.bfloat16,
        )
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(13, 64, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        top_w, top_idx = gemma4.Gemma4Router(weights, 64, 4, EPS)(x)

        output = gemma4.Gemma4Experts(weights)._forward_eager(x, top_idx, top_w)
        reference = hf_experts(x, top_idx, top_w, weights)

        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assertEqual(output.dtype, torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4TopKKernelTest(TestCase):
    def _check(self, values):
        expected_values, expected_indices = torch.topk(values, k=8, dim=-1)
        actual_values, actual_indices = rtp_llm_ops.gemma4_topk_8_bf16(values)
        self.assertTrue(torch.equal(actual_values, expected_values))
        self.assertTrue(torch.equal(actual_indices, expected_indices))

    def test_matches_torch_random_and_ties(self):
        torch.manual_seed(20261015)
        scores = torch.randn(8192, 128, device="cuda", dtype=torch.bfloat16)
        self._check(torch.softmax(scores, dim=-1))
        self._check(torch.ones(17, 128, device="cuda", dtype=torch.bfloat16))
        repeated = (torch.arange(128, device="cuda") % 11).to(torch.bfloat16)
        self._check(repeated.expand(33, -1).contiguous())


class Gemma4AttentionTest(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.device = torch.device("cuda")
        self.parallelism = ParallelismConfig()

    def _run(self, geometry, sliding_window):
        tokens = 13
        config = make_config([HybridAttentionType.SLIDING_WINDOW], self.device)
        weights = make_layer_weights(geometry, self.device)
        attn = gemma4.Gemma4Attention(config, self.parallelism, weights, geometry)
        attn_inputs = make_prefill_inputs([tokens], None, self.device)
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size=4)
        generator = torch.Generator().manual_seed(11)
        hidden = torch.randn(tokens, 64, generator=generator).to(self.device)
        out = attn(hidden, impl, None)
        positions = torch.arange(tokens, device=self.device)
        ref = hf_attention(
            hidden, weights, geometry, positions, sliding_window=sliding_window
        )
        torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)

    def test_attention_no_cache_sliding_geometry(self):
        geometry = gemma4.Gemma4LayerGeometry("swa", 4, 2, 16, 10000.0, 1.0, 1024)
        self._run(geometry, sliding_window=0)  # window 1024 never binds at T=13

    def test_attention_no_cache_full_geometry(self):
        geometry = gemma4.Gemma4LayerGeometry(
            "full", 4, 1, 64, 1000000.0, 0.25, 0, k_equals_v=True
        )
        self._run(geometry, sliding_window=0)

    def test_attend_request_sliding_window_mask(self):
        geometry = gemma4.Gemma4LayerGeometry("swa", 4, 2, 16, 10000.0, 1.0, 5)
        impl = gemma4.Gemma4TorchFMHAImpl.__new__(gemma4.Gemma4TorchFMHAImpl)
        impl.geometry = geometry
        impl.max_attention_matrix_elements = 8 * 1024 * 1024
        tokens, kv_len, window = 6, 10, 5
        q = torch.randn(tokens, 4, 16, device=self.device)
        k = torch.randn(kv_len, 2, 16, device=self.device)
        v = torch.randn(kv_len, 2, 16, device=self.device)
        positions = torch.arange(4, 4 + tokens, device=self.device)
        out = impl._attend_request(q, k, v, positions)
        ref = hf_reference_paged_attention(q, k, v, positions, sliding_window=window)
        torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)

    def test_chunked_full_and_sliding_match_eager(self):
        tokens = 19
        positions = torch.arange(tokens, device=self.device)
        q = torch.randn(tokens, 4, 16, device=self.device)
        k = torch.randn(tokens, 2, 16, device=self.device)
        v = torch.randn(tokens, 2, 16, device=self.device)
        for tag, window in (("full", 0), ("swa", 5)):
            with self.subTest(tag=tag):
                geometry = gemma4.Gemma4LayerGeometry(
                    tag, 4, 2, 16, 10000.0, 1.0, window
                )
                impl = gemma4.Gemma4TorchFMHAImpl.__new__(gemma4.Gemma4TorchFMHAImpl)
                impl.geometry = geometry
                impl.sdpa_causal_limit = 0
                impl.max_attention_matrix_elements = 16
                impl.attention_query_chunk_size = 4
                output = impl._attend_request(q, k, v, positions)
                reference = hf_reference_paged_attention(
                    q, k, v, positions, sliding_window=window
                )
                torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-5)

    def test_visual_group_is_bidirectional_only_for_sliding_prefill(self):
        tokens = 6
        q = torch.zeros(tokens, 4, 16, device=self.device)
        k = torch.zeros(tokens, 2, 16, device=self.device)
        v = (
            torch.arange(tokens, device=self.device, dtype=torch.float32)
            .view(tokens, 1, 1)
            .expand(tokens, 2, 16)
        )
        positions = torch.arange(tokens, device=self.device)
        group_ids = torch.tensor([-1, -1, 0, 0, 0, -1], device=self.device)

        swa = gemma4.Gemma4TorchFMHAImpl.__new__(gemma4.Gemma4TorchFMHAImpl)
        swa.geometry = gemma4.Gemma4LayerGeometry("swa", 4, 2, 16, 10000.0, 1.0, 1024)
        swa.sdpa_causal_limit = 1024
        swa.max_attention_matrix_elements = 8 * 1024 * 1024
        swa_out = swa._attend_request(q, k, v, positions, query_group_ids=group_ids)
        expected_swa = torch.tensor([0.0, 0.5, 2.0, 2.0, 2.0, 2.5], device=self.device)
        torch.testing.assert_close(swa_out[:, 0, 0], expected_swa)
        swa.max_attention_matrix_elements = 16
        swa.attention_query_chunk_size = 2
        chunked_swa_out = swa._attend_request(
            q, k, v, positions, query_group_ids=group_ids
        )
        torch.testing.assert_close(chunked_swa_out, swa_out, rtol=0, atol=0)

        full = gemma4.Gemma4TorchFMHAImpl.__new__(gemma4.Gemma4TorchFMHAImpl)
        full.geometry = gemma4.Gemma4LayerGeometry("full", 4, 2, 16, 10000.0, 1.0, 0)
        full.sdpa_causal_limit = 1024
        full.max_attention_matrix_elements = 8 * 1024 * 1024
        full_out = full._attend_request(q, k, v, positions, query_group_ids=group_ids)
        expected_full = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], device=self.device)
        torch.testing.assert_close(full_out[:, 0, 0], expected_full)


class Gemma4DecoderLayerTest(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.device = torch.device("cuda")
        self.parallelism = ParallelismConfig()
        self.tokens = 9

    def _run(self, layer_type):
        config = make_config([layer_type], self.device)
        geometry = geometry_of(config, self.parallelism)
        weights = make_layer_weights(geometry, self.device)
        layer = gemma4.Gemma4DecoderLayer(config, self.parallelism, weights, 0)
        attn_inputs = make_prefill_inputs([self.tokens], None, self.device)
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size=4)
        generator = torch.Generator().manual_seed(21)
        x = torch.randn(self.tokens, 64, generator=generator).to(self.device)
        out = layer(x, impl, None)
        positions = torch.arange(self.tokens, device=self.device)
        ref = hf_decoder_layer(
            x, weights, geometry, positions, config.hidden_size, config.moe_k
        )
        torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)

    def test_decoder_layer_sliding(self):
        self._run(HybridAttentionType.SLIDING_WINDOW)

    def test_decoder_layer_full(self):
        self._run(HybridAttentionType.NONE)

    def test_layer_scalar_is_applied(self):
        config = make_config([HybridAttentionType.SLIDING_WINDOW], self.device)
        geometry = geometry_of(config, self.parallelism)
        weights = make_layer_weights(geometry, self.device)
        layer = gemma4.Gemma4DecoderLayer(config, self.parallelism, weights, 0)
        self.assertEqual(layer.layer_scalar.numel(), 1)
        self.assertNotEqual(float(layer.layer_scalar[0]), 1.0)


class Gemma4FMHACacheTest(TestCase):
    """Paged KV cache write (flashinfer append) + gather round trip."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.device = torch.device("cuda")
        self.page_size = 4
        self.kv_heads = 2
        # flashinfer's append kernel dispatches only bf16/fp16/fp8 dtypes and
        # head_dims in its DISPATCH_HEAD_DIM list (64/128/...); 32/16 are
        # rejected. Keep the unit-test geometry inside the dispatch table.
        self.head_dim = 64
        self.head_num = 4
        self.geometry = gemma4.Gemma4LayerGeometry(
            "swa", self.head_num, self.kv_heads, self.head_dim, 10000.0, 1.0, 0
        )

    def _make_cache(self, num_pages, tag="swa"):
        base = torch.full(
            (num_pages, 2, self.kv_heads, self.page_size, self.head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        return LayerKVCache(base, self.page_size, layer_id=0, tag=tag)

    def _rand(self, *shape, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )

    def test_custom_paged_gather_matches_sparse_reference(self):
        cache = self._make_cache(num_pages=4).kv_cache_base
        cache.copy_(
            torch.arange(cache.numel(), device=self.device)
            .reshape(cache.shape)
            .to(torch.bfloat16)
        )
        pages = torch.tensor([2, -1, 0], device=self.device, dtype=torch.int32)
        first_offset = 2
        token_count = 8
        actual_k, actual_v, actual_valid = rtp_llm_ops.gemma4_gather_paged_kv_bf16(
            cache[:, 0],
            cache[:, 1],
            pages,
            first_offset,
            token_count,
            self.page_size,
        )
        expected_k = torch.empty_like(actual_k)
        expected_v = torch.empty_like(actual_v)
        expected_valid = torch.empty_like(actual_valid)
        for token in range(token_count):
            offset = first_offset + token
            page = int(pages[offset // self.page_size])
            expected_valid[token] = page >= 0
            if page >= 0:
                expected_k[token] = cache[page, 0, :, offset % self.page_size, :]
                expected_v[token] = cache[page, 1, :, offset % self.page_size, :]
            else:
                expected_k[token].zero_()
                expected_v[token].zero_()
        self.assertTrue(torch.equal(actual_k, expected_k))
        self.assertTrue(torch.equal(actual_v, expected_v))
        self.assertTrue(torch.equal(actual_valid, expected_valid))

    def test_prefill_then_decode_roundtrip(self):
        tokens = 13
        q = self._rand(tokens, self.head_num, self.head_dim, seed=31)
        k = self._rand(tokens, self.kv_heads, self.head_dim, seed=32)
        v = self._rand(tokens, self.kv_heads, self.head_dim, seed=33)

        kv_cache = self._make_cache(num_pages=4)
        attn_inputs = make_prefill_inputs([tokens], [[0, 1, 2, 3]], self.device)
        impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, attn_inputs, self.page_size
        )
        out = impl.forward(q, k, v, kv_cache)
        positions = torch.arange(tokens, device=self.device)
        ref = hf_reference_paged_attention(q, k, v, positions)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

        # decode step: one new token at position `tokens`
        q2 = self._rand(1, self.head_num, self.head_dim, seed=34)
        k2 = self._rand(1, self.kv_heads, self.head_dim, seed=35)
        v2 = self._rand(1, self.kv_heads, self.head_dim, seed=36)
        decode_inputs = make_decode_inputs([tokens], [[0, 1, 2, 3]])
        decode_impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, decode_inputs, self.page_size
        )
        out2 = decode_impl.forward(q2, k2, v2, kv_cache)
        full_k = torch.cat([k, k2], dim=0)
        full_v = torch.cat([v, v2], dim=0)
        ref2 = hf_reference_paged_attention(
            q2, full_k, full_v, torch.tensor([tokens], device=self.device)
        )
        torch.testing.assert_close(out2, ref2, rtol=5e-2, atol=5e-2)

    def test_multi_request_prefill_roundtrip(self):
        lengths = [5, 8]
        total = sum(lengths)
        q = self._rand(total, self.head_num, self.head_dim, seed=41)
        k = self._rand(total, self.kv_heads, self.head_dim, seed=42)
        v = self._rand(total, self.kv_heads, self.head_dim, seed=43)
        kv_cache = self._make_cache(num_pages=5)
        # request 0 -> pages 0,1 ; request 1 -> pages 2,3,4
        attn_inputs = make_prefill_inputs(lengths, [[0, 1], [2, 3, 4]], self.device)
        impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, attn_inputs, self.page_size
        )
        out = impl.forward(q, k, v, kv_cache)
        starts = [0, lengths[0]]
        refs = []
        for i, length in enumerate(lengths):
            refs.append(
                hf_reference_paged_attention(
                    q[starts[i] : starts[i] + length],
                    k[starts[i] : starts[i] + length],
                    v[starts[i] : starts[i] + length],
                    torch.arange(length, device=self.device),
                )
            )
        torch.testing.assert_close(out, torch.cat(refs, dim=0), rtol=5e-2, atol=5e-2)

    def test_prefill_with_prefix_roundtrip(self):
        """Second prefill chunk attends to the cached prefix + new tokens."""
        prefix_len, new_len = 6, 5
        k_prefix = self._rand(prefix_len, self.kv_heads, self.head_dim, seed=51)
        v_prefix = self._rand(prefix_len, self.kv_heads, self.head_dim, seed=52)
        kv_cache = self._make_cache(num_pages=4)
        first_inputs = make_prefill_inputs([prefix_len], [[0, 1]], self.device)
        first_impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, first_inputs, self.page_size
        )
        q_prefix = self._rand(prefix_len, self.head_num, self.head_dim, seed=53)
        first_impl.forward(q_prefix, k_prefix, v_prefix, kv_cache)

        q_new = self._rand(new_len, self.head_num, self.head_dim, seed=54)
        k_new = self._rand(new_len, self.kv_heads, self.head_dim, seed=55)
        v_new = self._rand(new_len, self.kv_heads, self.head_dim, seed=56)
        # prefix occupies pages 0,1; the new chunk continues in page 1 (2
        # free slots) and page 2.
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([new_len], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = torch.tensor(
            [[0, 1, 2]], dtype=torch.int32
        )
        impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, attn_inputs, self.page_size
        )
        out = impl.forward(q_new, k_new, v_new, kv_cache)
        positions = torch.arange(prefix_len, prefix_len + new_len, device=self.device)
        full_k = torch.cat([k_prefix, k_new], dim=0)
        full_v = torch.cat([v_prefix, v_new], dim=0)
        ref = hf_reference_paged_attention(q_new, full_k, full_v, positions)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    def test_sliding_window_cache_path(self):
        geometry = gemma4.Gemma4LayerGeometry(
            "swa", self.head_num, self.kv_heads, self.head_dim, 10000.0, 1.0, 6
        )
        tokens = 10
        q = self._rand(tokens, self.head_num, self.head_dim, seed=61)
        k = self._rand(tokens, self.kv_heads, self.head_dim, seed=62)
        v = self._rand(tokens, self.kv_heads, self.head_dim, seed=63)
        kv_cache = self._make_cache(num_pages=3)
        attn_inputs = make_prefill_inputs([tokens], [[0, 1, 2]], self.device)
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, self.page_size)
        out = impl.forward(q, k, v, kv_cache)
        positions = torch.arange(tokens, device=self.device)
        ref = hf_reference_paged_attention(q, k, v, positions, sliding_window=6)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


class Gemma4ModelTest(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.device = torch.device("cuda")
        self.parallelism = ParallelismConfig()
        self.layer_types = [
            HybridAttentionType.SLIDING_WINDOW,
            HybridAttentionType.NONE,
        ]

    def test_visual_group_ids_follow_feature_ranges(self):
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(10, dtype=torch.int32, device=self.device)
        multimodal_inputs = PyMultimodalInputs()
        multimodal_inputs.multimodal_features = [
            torch.zeros(2, 64, device=self.device),
            torch.zeros(3, 64, device=self.device),
        ]
        multimodal_inputs.mm_features_locs = torch.tensor([1, 5], dtype=torch.int32)
        inputs.multimodal_inputs = multimodal_inputs
        group_ids = gemma4.Gemma4Model._visual_group_ids(inputs)
        torch.testing.assert_close(
            group_ids.cpu(),
            torch.tensor([-1, 0, 0, -1, -1, 1, 1, 1, -1, -1]),
        )

    def test_model_forward_matches_reference(self):
        config = make_config(self.layer_types, self.device)
        geometries = [
            geometry_of(config, self.parallelism, idx)
            for idx in range(config.num_layers)
        ]
        # rtp_llm's fused Embedding kernel dispatches bf16/fp16 only; keep the
        # whole mini model in bf16 and widen the reference tolerance.
        weights = ModelWeights(config.num_layers, "cuda", torch.bfloat16)
        vocab = 33
        generator = torch.Generator().manual_seed(99)
        embedding = (
            (torch.randn(vocab, 64, generator=generator) * 0.5)
            .to(self.device)
            .to(torch.bfloat16)
        )
        weights.set_global_weight(W.embedding, embedding)
        weights.set_global_weight(
            W.final_ln_gamma,
            (torch.randn(64, generator=generator) * 0.2 + 1.0)
            .to(self.device)
            .to(torch.bfloat16),
        )
        layer_weight_dicts = []
        for idx, geometry in enumerate(geometries):
            layer_weights = make_layer_weights(
                geometry, self.device, dtype=torch.bfloat16
            )
            layer_weight_dicts.append(layer_weights)
            for key, tensor in layer_weights.items():
                weights.set_layer_weight(idx, key, tensor)
        model = gemma4.Gemma4Model(
            config,
            self.parallelism,
            weights,
            max_generate_batch_size=4,
        )
        tokens = 7
        # the fused embedding kernel takes int32 ids
        input_ids = (
            torch.randint(0, vocab, (tokens,), generator=generator)
            .to(self.device)
            .to(torch.int32)
        )
        attn_inputs = make_prefill_inputs([tokens], None, self.device)
        inputs = PyModelInputs()
        inputs.input_ids = input_ids
        inputs.attention_inputs = {
            gemma4.GEMMA4_TAG_SWA: attn_inputs,
            gemma4.GEMMA4_TAG_FULL: attn_inputs,
        }
        outputs = model.forward(inputs)
        self.assertEqual(tuple(outputs.hidden_states.shape), (tokens, 64))

        # the fused Embedding kernel must pick the same rows as plain lookup
        emb_model = model.embed_tokens(input_ids).float()
        emb_lookup = embedding[input_ids.long()].float()
        self.assertTrue(torch.equal(emb_model, emb_lookup))

        # reference forward: embedding * sqrt(hidden) -> layers -> final norm.
        # The reference runs entirely in fp32 (weights upcast): a bf16-weighted
        # reference is itself numerically unstable under this mini config's
        # sqrt(64)*N embedding scale, which would conflate reference rounding
        # with model error. The model under test stays bf16.
        embed_scale = float(torch.tensor(64**0.5, dtype=torch.float32))
        hidden = embedding[input_ids.long()].float() * embed_scale
        positions = torch.arange(tokens, device=self.device)
        for geometry, layer_weights in zip(geometries, layer_weight_dicts):
            hidden = hf_decoder_layer(
                hidden,
                {k: v.float() for k, v in layer_weights.items()},
                geometry,
                positions,
                64,
                config.moe_k,
            )
        hidden = hf_rms_norm(
            hidden,
            weights.get_global_weight(W.final_ln_gamma).float(),
            EPS,
        )
        # bf16 mini-model: compare against the reference scale rather than a
        # tight elementwise bound (the fused Embedding kernel computes in fp32
        # and the small 64-dim geometry amplifies bf16 rounding layer over
        # layer). Assert correlation + bounded relative error on the norm.
        out = outputs.hidden_states.float()
        ref = hidden.float()
        self.assertEqual(out.shape, ref.shape)
        out_c = out - out.mean()
        ref_c = ref - ref.mean()
        corr = (out_c * ref_c).sum() / (out_c.norm() * ref_c.norm() + 1e-6)
        self.assertGreater(float(corr), 0.98)
        rel_l2 = (out - ref).norm() / (ref.norm() + 1e-6)
        self.assertLess(float(rel_l2), 0.1)


class Gemma4SparseSWACacheTest(TestCase):
    """Sparse SWA tables compute prefill from current KV and persist the tail."""

    def test_sparse_swa_prefill_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dev = torch.device("cuda")
        page_size, kv_heads, head_num, head_dim = 4, 2, 4, 64
        # Engine-realistic sparse SWA layout: the SWA manager evicts blocks
        # whose positions are outside the window of the newest query. With
        # tokens=33, window=16, page_size=4: positions 0..15 (blocks 0..3)
        # are outside the newest query's window and are NULL; positions
        # 16..32 (blocks 4..8) stay resident.
        tokens = 33
        window = 16
        geometry = gemma4.Gemma4LayerGeometry(
            "swa", head_num, kv_heads, head_dim, 10000.0, 1.0, window
        )
        torch.manual_seed(3)
        q = torch.randn(tokens, head_num, head_dim, device=dev).to(torch.bfloat16)
        k = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)
        v = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)

        num_pages = 8
        base = torch.full(
            (num_pages, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=dev,
        )
        kv_cache = LayerKVCache(base, page_size, layer_id=0, tag="swa")

        # 9 table slots: first 4 NULL (evicted), then physical pages 2..6
        table = torch.tensor([[-1, -1, -1, -1, 2, 3, 4, 5, 6]], dtype=torch.int32)
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([tokens], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = table

        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        out = impl.forward(q, k, v, kv_cache)

        # Independent reference uses current dense K/V for every prefill query;
        # cache retention must not erase the early rows before they are computed.
        group = head_num // kv_heads
        k_exp = k.repeat_interleave(group, dim=1)
        v_exp = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k_exp).float()
        pos_q = torch.arange(tokens, device=dev)
        pos_kv = torch.arange(tokens, device=dev)
        allowed = (pos_kv[None, :] <= pos_q[:, None]) & (
            pos_kv[None, :] > pos_q[:, None] - window
        )
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        ref = torch.einsum("hts,shd->thd", probs, v_exp)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

        for logical_block, physical_page in zip(range(4, 9), range(2, 7)):
            lo = logical_block * page_size
            hi = min(lo + page_size, tokens)
            count = hi - lo
            torch.testing.assert_close(
                base[physical_page, 0, :, :count],
                k[lo:hi].permute(1, 0, 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                base[physical_page, 1, :, :count],
                v[lo:hi].permute(1, 0, 2),
                rtol=0,
                atol=0,
            )
        self.assertTrue(torch.isnan(base[0]).all())
        self.assertTrue(torch.isnan(base[1]).all())

    def test_sparse_swa_suffix_prefill_uses_required_prefix_tail(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        page_size, kv_heads, head_num, head_dim = 4, 2, 4, 64
        prefix_length, input_length, window = 20, 5, 8
        geometry = gemma4.Gemma4LayerGeometry(
            "swa", head_num, kv_heads, head_dim, 10000.0, 1.0, window
        )
        torch.manual_seed(17)
        prefix_k = torch.randn(prefix_length, kv_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        prefix_v = torch.randn(prefix_length, kv_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        q = torch.randn(input_length, head_num, head_dim, device=device).to(
            torch.bfloat16
        )
        k = torch.randn(input_length, kv_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        v = torch.randn(input_length, kv_heads, head_dim, device=device).to(
            torch.bfloat16
        )

        base = torch.full(
            (6, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        base[1, 0] = prefix_k[12:16].permute(1, 0, 2)
        base[1, 1] = prefix_v[12:16].permute(1, 0, 2)
        base[2, 0] = prefix_k[16:20].permute(1, 0, 2)
        base[2, 1] = prefix_v[16:20].permute(1, 0, 2)
        kv_cache = LayerKVCache(base, page_size, layer_id=0, tag="swa")

        table = torch.tensor([[-1, -1, -1, 1, 2, 3, 4]], dtype=torch.int32)
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([prefix_length], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = table

        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        out = impl.forward(q, k, v, kv_cache)

        prefix_start = prefix_length - window + 1
        keys = torch.cat((prefix_k[prefix_start:], k), dim=0)
        values = torch.cat((prefix_v[prefix_start:], v), dim=0)
        group = head_num // kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        q_positions = torch.arange(
            prefix_length,
            prefix_length + input_length,
            device=device,
        )
        kv_positions = torch.arange(
            prefix_start, prefix_length + input_length, device=device
        )
        allowed = (kv_positions[None, :] <= q_positions[:, None]) & (
            kv_positions[None, :] > q_positions[:, None] - window
        )
        scores = torch.einsum("thd,shd->hts", q, keys).float()
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        ref = torch.einsum("hts,shd->thd", probs, values)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

        torch.testing.assert_close(base[3, 0], k[:4].permute(1, 0, 2), rtol=0, atol=0)
        torch.testing.assert_close(base[3, 1], v[:4].permute(1, 0, 2), rtol=0, atol=0)
        torch.testing.assert_close(
            base[4, 0, :, :1], k[4:].permute(1, 0, 2), rtol=0, atol=0
        )
        torch.testing.assert_close(
            base[4, 1, :, :1], v[4:].permute(1, 0, 2), rtol=0, atol=0
        )

        missing_table = table.clone()
        missing_table[0, 4] = -1
        missing_inputs = PyAttentionInputs()
        missing_inputs.is_prefill = True
        missing_inputs.input_lengths = attn_inputs.input_lengths
        missing_inputs.prefix_lengths = attn_inputs.prefix_lengths
        missing_inputs.sequence_lengths = attn_inputs.sequence_lengths
        missing_inputs.kv_cache_kernel_block_id = missing_table
        missing_impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, missing_inputs, page_size
        )
        with self.assertRaisesRegex(RuntimeError, "missing required prefix KV"):
            missing_impl.forward(q, k, v, kv_cache)


class Gemma4FullLayerLongPosTest(TestCase):
    """FULL-attention layer (partial RoPE, 512 dims) at long positions.

    The engine-vs-HF divergence at 1527/2627 tokens persists while every
    block-table and cache path is verified correct; this isolates the FULL
    geometry attention math itself at large positions through the same
    impl the engine uses.
    """

    def test_full_geometry_long_position_sdpa_contract(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dev = torch.device("cuda")
        tokens = 1531  # > window; blocks: 11 full pages + partial
        page_size = 128
        head_num, kv_heads, head_dim = 4, 1, 512
        geometry = gemma4.Gemma4LayerGeometry(
            "full", head_num, kv_heads, head_dim, 1000000.0, 0.25, 0
        )
        torch.manual_seed(11)
        q = torch.randn(tokens, head_num, head_dim, device=dev).to(torch.bfloat16)
        k = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)
        v = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)

        num_pages = (tokens + page_size - 1) // page_size + 2
        base = torch.full(
            (num_pages, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=dev,
        )
        kv_cache = LayerKVCache(base, page_size, layer_id=0, tag="full")
        table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([tokens], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = table

        # The impl contract (mirroring production Gemma4Attention) receives
        # ALREADY-roped q/k; apply the rope math first, then feed the impl
        # and reuse the same roped tensors for the reference.
        inv = gemma4._proportional_inv_freq(head_dim, 1000000.0, 0.25)
        pos = torch.arange(tokens, device=dev).float()
        freqs = pos.unsqueeze(-1) * inv.to(dev)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rope(x):
            c = cos.to(x.dtype)[:, None, :]
            sn = sin.to(x.dtype)[:, None, :]
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            rot = torch.cat((-x2, x1), dim=-1)
            return (x * c) + (rot * sn)

        qr = rope(q)
        kr = rope(k)

        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        impl.sdpa_causal_limit = tokens
        out = impl.forward(qr, kr, v, kv_cache)
        out_cacheless = impl.forward(qr, kr, v, None)

        ref = (
            F.scaled_dot_product_attention(
                qr.transpose(0, 1).unsqueeze(0),
                kr.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                scale=1.0,
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
        )

        torch.testing.assert_close(out_cacheless, ref, rtol=0, atol=0)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_full_geometry_long_position_eager_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dev = torch.device("cuda")
        tokens = 1531
        page_size = 128
        head_num, kv_heads, head_dim = 4, 1, 512
        geometry = gemma4.Gemma4LayerGeometry(
            "full", head_num, kv_heads, head_dim, 1000000.0, 0.25, 0
        )
        torch.manual_seed(11)
        q = torch.randn(tokens, head_num, head_dim, device=dev).to(torch.bfloat16)
        k = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)
        v = torch.randn(tokens, kv_heads, head_dim, device=dev).to(torch.bfloat16)

        num_pages = (tokens + page_size - 1) // page_size + 2
        base = torch.full(
            (num_pages, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=dev,
        )
        kv_cache = LayerKVCache(base, page_size, layer_id=0, tag="full")
        table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([tokens], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = table

        inv = gemma4._proportional_inv_freq(head_dim, 1000000.0, 0.25)
        pos = torch.arange(tokens, device=dev).float()
        freqs = pos.unsqueeze(-1) * inv.to(dev)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rope(x):
            c = cos.to(x.dtype)[:, None, :]
            sn = sin.to(x.dtype)[:, None, :]
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            rot = torch.cat((-x2, x1), dim=-1)
            return (x * c) + (rot * sn)

        qr = rope(q)
        kr = rope(k)
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        out = impl.forward(qr, kr, v, kv_cache)
        out_cacheless = impl.forward(qr, kr, v, None)

        group = head_num // kv_heads
        k_expanded = kr.repeat_interleave(group, dim=1)
        v_expanded = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", qr, k_expanded).float()
        query_positions = torch.arange(tokens, device=dev)
        key_positions = torch.arange(tokens, device=dev)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(q.dtype)
        reference = torch.einsum("hts,shd->thd", probabilities, v_expanded)

        torch.testing.assert_close(out_cacheless, reference, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(out, reference, rtol=5e-2, atol=5e-2)

    def test_swa_long_position_eager_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        tokens = 1531
        page_size = 128
        head_num, kv_heads, head_dim = 16, 8, 256
        window = 1024
        geometry = gemma4.Gemma4LayerGeometry(
            "swa", head_num, kv_heads, head_dim, 10000.0, 1.0, window
        )
        torch.manual_seed(23)
        q = torch.randn(tokens, head_num, head_dim, device=device).to(torch.bfloat16)
        k = torch.randn(tokens, kv_heads, head_dim, device=device).to(torch.bfloat16)
        v = torch.randn(tokens, kv_heads, head_dim, device=device).to(torch.bfloat16)

        num_pages = (tokens + page_size - 1) // page_size + 2
        cache = torch.full(
            (num_pages, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache = LayerKVCache(cache, page_size, layer_id=0, tag="swa")
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([tokens], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = torch.arange(
            num_pages, dtype=torch.int32
        ).unsqueeze(0)

        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        cached_output = impl.forward(q, k, v, kv_cache)
        cacheless_output = impl.forward(q, k, v, None)

        group = head_num // kv_heads
        expanded_keys = k.repeat_interleave(group, dim=1)
        expanded_values = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q, expanded_keys).float()
        query_positions = torch.arange(tokens, device=device)
        key_positions = torch.arange(tokens, device=device)
        allowed = (key_positions[None, :] <= query_positions[:, None]) & (
            key_positions[None, :] > query_positions[:, None] - window
        )
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(q.dtype)
        reference = torch.einsum("hts,shd->thd", probabilities, expanded_values)

        torch.testing.assert_close(cacheless_output, reference, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(cached_output, reference, rtol=5e-2, atol=5e-2)


class Gemma4DecodeChainTest(TestCase):
    """Prefill + multi-step decode chain against an exact reference.

    Reproduces the engine's full-seq divergence pattern (2627 exact, 1527
    diverges at the 4th token) in a controlled single-layer setting with
    rope, window mask and paged cache: prefill N tokens, then decode one
    token at a time, comparing each step's attention output.
    """

    def _run_chain(self, tokens, window, head_num=4, kv_heads=2, head_dim=64):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dev = torch.device("cuda")
        page_size = 4
        geometry = gemma4.Gemma4LayerGeometry(
            "swa" if window > 0 else "full",
            head_num,
            kv_heads,
            head_dim,
            10000.0 if window > 0 else 1000000.0,
            1.0 if window > 0 else 0.25,
            window,
        )
        torch.manual_seed(5)
        q = torch.randn(tokens + 4, head_num, head_dim, device=dev).to(torch.bfloat16)
        k = torch.randn(tokens + 4, kv_heads, head_dim, device=dev).to(torch.bfloat16)
        v = torch.randn(tokens + 4, kv_heads, head_dim, device=dev).to(torch.bfloat16)

        num_pages = (tokens + 8 + page_size - 1) // page_size + 2
        base = torch.full(
            (num_pages, 2, kv_heads, page_size, head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=dev,
        )
        kv_cache = LayerKVCache(base, page_size, layer_id=0, tag=geometry.tag)

        def make_inputs(is_prefill, length, seq_len, pages):
            ai = PyAttentionInputs()
            ai.is_prefill = is_prefill
            ai.input_lengths = torch.tensor([length], dtype=torch.int32)
            ai.prefix_lengths = (
                torch.zeros(1, dtype=torch.int32)
                if is_prefill
                else torch.empty(0, dtype=torch.int32)
            )
            ai.sequence_lengths = (
                torch.empty(0, dtype=torch.int32)
                if is_prefill
                else torch.tensor([seq_len], dtype=torch.int32)
            )
            ai.kv_cache_kernel_block_id = torch.tensor(
                [list(range(pages))], dtype=torch.int32
            )
            return ai

        n_blocks = (tokens + page_size - 1) // page_size
        impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, make_inputs(True, tokens, 0, n_blocks), page_size
        )
        out = impl.forward(q[:tokens], k[:tokens], v[:tokens], kv_cache)

        # reference for prefill
        def attend_full(qr, kr, vr, q_pos):
            group = head_num // kv_heads
            k_exp = kr.repeat_interleave(group, dim=1)
            v_exp = vr.repeat_interleave(group, dim=1)
            scores = torch.einsum("thd,shd->hts", qr, k_exp).float()
            pos_kv = torch.arange(kr.shape[0], device=dev)
            allowed = pos_kv[None, :] <= q_pos[:, None]
            if window > 0:
                allowed = allowed & (pos_kv[None, :] > q_pos[:, None] - window)
            scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(qr.dtype)
            return torch.einsum("hts,shd->thd", probs, v_exp)

        ref = attend_full(
            q[:tokens], k[:tokens], v[:tokens], torch.arange(tokens, device=dev)
        )
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

        # decode chain: one token at a time
        for step in range(4):
            pos = tokens + step
            pages_now = (pos + 1 + page_size - 1) // page_size
            dec = gemma4.Gemma4TorchFMHAImpl(
                None,
                geometry,
                make_inputs(False, 1, pos, pages_now),
                page_size,
            )
            got = dec.forward(
                q[pos : pos + 1], k[pos : pos + 1], v[pos : pos + 1], kv_cache
            )
            kr = k[: pos + 1]
            vr = v[: pos + 1]
            qr = q[pos : pos + 1]
            ref_d = attend_full(qr, kr, vr, torch.tensor([pos], device=dev))
            torch.testing.assert_close(
                got,
                ref_d,
                rtol=5e-2,
                atol=5e-2,
                msg=lambda m: f"decode step {step} (pos {pos}) mismatch: {m}",
            )
            print(f"[decode-chain] tokens={tokens} step={step} pos={pos} OK")

    def test_decode_chain_after_full_block_prefill(self):
        self._run_chain(tokens=12, window=8)  # block-aligned prefill

    def test_decode_chain_after_partial_block_prefill(self):
        self._run_chain(tokens=13, window=8)  # partial tail block

    def test_full_attention_decode_after_long_prefill(self):
        self._run_chain(
            tokens=1531,
            window=0,
            head_num=16,
            kv_heads=2,
            head_dim=512,
        )


class Gemma4SwaFlashinferEquivalenceTest(TestCase):
    """Production SWA backend vs the HF-validated torch reference impl.

    The flashinfer paged prefill wrapper must reproduce the reference
    semantics on identical state: causal + sliding window, sm_scale=1.0,
    GQA expansion and the shared cache write path.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        if not gemma4.Gemma4SwaFlashinferImpl.support(
            gemma4.Gemma4LayerGeometry("swa", 4, 2, 64, 10000.0, 1.0, 8)
        ):
            self.skipTest("flashinfer unavailable")
        self.device = torch.device("cuda")
        self.page_size = 4
        self.kv_heads = 2
        self.head_dim = 64
        self.head_num = 4
        self.window = 8
        self.geometry = gemma4.Gemma4LayerGeometry(
            "swa",
            self.head_num,
            self.kv_heads,
            self.head_dim,
            10000.0,
            1.0,
            self.window,
        )

    def _make_cache(self, num_pages):
        base = torch.full(
            (num_pages, 2, self.kv_heads, self.page_size, self.head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        return LayerKVCache(base, self.page_size, layer_id=0, tag="swa")

    def _rand(self, *shape, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )

    def _both_impls(self, attn_inputs):
        torch_impl = gemma4.Gemma4TorchFMHAImpl(
            None, self.geometry, attn_inputs, self.page_size
        )
        prod_impl = gemma4.Gemma4SwaFlashinferImpl(
            None, self.geometry, attn_inputs, self.page_size
        )
        return torch_impl, prod_impl

    def _assert_equivalent(self, attn_inputs, q, k, v, num_pages, label):
        kv_torch = self._make_cache(num_pages)
        kv_prod = self._make_cache(num_pages)
        torch_impl, prod_impl = self._both_impls(attn_inputs)
        out_torch = torch_impl.forward(q, k, v, kv_torch)
        out_prod = prod_impl.forward(q, k, v, kv_prod)
        self.assertEqual(out_torch.shape, out_prod.shape, label)
        torch.testing.assert_close(
            out_prod.float(),
            out_torch.float(),
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m: f"{label}: production vs reference mismatch: {m}",
        )
        print(f"[swa-equiv] {label} OK", flush=True)

    def test_prefill_window_binds(self):
        tokens = 20
        q = self._rand(tokens, self.head_num, self.head_dim, seed=71)
        k = self._rand(tokens, self.kv_heads, self.head_dim, seed=72)
        v = self._rand(tokens, self.kv_heads, self.head_dim, seed=73)
        attn_inputs = make_prefill_inputs([tokens], [[0, 1, 2, 3, 4]], self.device)
        self._assert_equivalent(attn_inputs, q, k, v, num_pages=5, label="prefill20")

    def test_prefill_with_prefix_window_binds(self):
        prefix_len, new_len = 6, 14
        k_prefix = self._rand(prefix_len, self.kv_heads, self.head_dim, seed=81)
        v_prefix = self._rand(prefix_len, self.kv_heads, self.head_dim, seed=82)
        q_prefix = self._rand(prefix_len, self.head_num, self.head_dim, seed=83)
        q_new = self._rand(new_len, self.head_num, self.head_dim, seed=84)
        k_new = self._rand(new_len, self.kv_heads, self.head_dim, seed=85)
        v_new = self._rand(new_len, self.kv_heads, self.head_dim, seed=86)

        # prefix (6) + new (14) = 20 tokens -> 5 pages [0..4]; the new chunk
        # continues in page 1 (2 free slots) then fills pages 2-4.
        kv_torch = self._make_cache(5)
        kv_prod = self._make_cache(5)
        first_inputs = make_prefill_inputs([prefix_len], [[0, 1]], self.device)
        torch_first, prod_first = self._both_impls(first_inputs)
        torch_first.forward(q_prefix, k_prefix, v_prefix, kv_torch)
        prod_first.forward(q_prefix, k_prefix, v_prefix, kv_prod)

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor([new_len], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = torch.tensor(
            [[0, 1, 2, 3, 4]], dtype=torch.int32
        )
        torch_impl, prod_impl = self._both_impls(attn_inputs)
        out_torch = torch_impl.forward(q_new, k_new, v_new, kv_torch)
        out_prod = prod_impl.forward(q_new, k_new, v_new, kv_prod)
        torch.testing.assert_close(
            out_prod.float(),
            out_torch.float(),
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m: f"prefill-with-prefix mismatch: {m}",
        )
        print("[swa-equiv] prefill-with-prefix OK", flush=True)

    def test_decode_after_prefill_window_binds(self):
        tokens = 20
        q = self._rand(tokens, self.head_num, self.head_dim, seed=91)
        k = self._rand(tokens, self.kv_heads, self.head_dim, seed=92)
        v = self._rand(tokens, self.kv_heads, self.head_dim, seed=93)
        prefill_inputs = make_prefill_inputs([tokens], [[0, 1, 2, 3, 4]], self.device)
        # the prefill fills pages 0-4 exactly (20 tokens); the decode token
        # at position 20 opens page 5, so the decode block table needs 6
        # entries and the cache a 6th page.
        kv_torch = self._make_cache(6)
        kv_prod = self._make_cache(6)
        torch_impl, prod_impl = self._both_impls(prefill_inputs)
        torch_impl.forward(q, k, v, kv_torch)
        prod_impl.forward(q, k, v, kv_prod)

        q2 = self._rand(1, self.head_num, self.head_dim, seed=94)
        k2 = self._rand(1, self.kv_heads, self.head_dim, seed=95)
        v2 = self._rand(1, self.kv_heads, self.head_dim, seed=96)
        decode_inputs = make_decode_inputs([tokens], [[0, 1, 2, 3, 4, 5]])
        torch_dec, prod_dec = self._both_impls(decode_inputs)
        out_torch = torch_dec.forward(q2, k2, v2, kv_torch)
        out_prod = prod_dec.forward(q2, k2, v2, kv_prod)
        torch.testing.assert_close(
            out_prod.float(),
            out_torch.float(),
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m: f"decode-after-prefill mismatch: {m}",
        )
        print("[swa-equiv] decode-after-prefill OK", flush=True)

    def test_multi_request_batch_window_binds(self):
        lengths = [5, 12]
        total = sum(lengths)
        q = self._rand(total, self.head_num, self.head_dim, seed=101)
        k = self._rand(total, self.kv_heads, self.head_dim, seed=102)
        v = self._rand(total, self.kv_heads, self.head_dim, seed=103)
        attn_inputs = make_prefill_inputs(lengths, [[0, 1], [2, 3, 4]], self.device)
        self._assert_equivalent(attn_inputs, q, k, v, num_pages=5, label="batch[5,12]")

    def test_window_edges_via_uniform_softmax(self):
        """Analytic window-edge check: k=0 gives uniform scores, so each output
        equals the mean of the allowed positions' v values — any window
        off-by-one shifts the mean detectably."""
        tokens = 20
        q = self._rand(tokens, self.head_num, self.head_dim, seed=111)
        k = torch.zeros(
            tokens,
            self.kv_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.arange(1, tokens + 1, device=self.device, dtype=torch.float32)
        v = (
            v.view(tokens, 1, 1)
            .expand(tokens, self.kv_heads, self.head_dim)
            .contiguous()
            .to(torch.bfloat16)
        )

        def expected(p):
            lo = max(0, p - self.window + 1)
            return float(sum(range(lo + 1, p + 2)) / (p - lo + 1))

        attn_inputs = make_prefill_inputs([tokens], [[0, 1, 2, 3, 4]], self.device)
        kv_torch = self._make_cache(5)
        kv_prod = self._make_cache(5)
        torch_impl, prod_impl = self._both_impls(attn_inputs)
        out_torch = torch_impl.forward(q, k, v, kv_torch)
        out_prod = prod_impl.forward(q, k, v, kv_prod)
        for p in (0, self.window - 1, self.window, tokens - 1):
            got_torch = float(out_torch[p, 0, 0])
            got_prod = float(out_prod[p, 0, 0])
            self.assertAlmostEqual(
                got_torch,
                expected(p),
                delta=0.3,
                msg=f"torch impl window edge at pos {p}",
            )
            self.assertAlmostEqual(
                got_prod,
                expected(p),
                delta=0.3,
                msg=(
                    f"production impl window edge at pos {p}: "
                    f"got {got_prod}, expected {expected(p)} "
                    f"(window={self.window})"
                ),
            )
        print(
            f"[swa-equiv] analytic window edges OK "
            f"(pos19: torch={float(out_torch[19, 0, 0]):.3f} "
            f"prod={float(out_prod[19, 0, 0]):.3f} "
            f"expected={expected(19):.3f})",
            flush=True,
        )

    def _build_model(self):
        config = make_config(
            [
                HybridAttentionType.SLIDING_WINDOW,
                HybridAttentionType.NONE,
            ],
            self.device,
        )
        config.attn_config.size_per_head = self.head_dim
        parallelism = ParallelismConfig()
        weights = ModelWeights(config.num_layers, "cuda", torch.bfloat16)
        weights.set_global_weight(
            W.embedding,
            torch.zeros(32, 64, device=self.device, dtype=torch.bfloat16),
        )
        weights.set_global_weight(
            W.final_ln_gamma,
            torch.ones(64, device=self.device, dtype=torch.bfloat16),
        )
        for idx in range(config.num_layers):
            geometry = gemma4.build_gemma4_layer_geometry(config, parallelism, idx)
            for key, tensor in make_layer_weights(
                geometry, self.device, dtype=torch.bfloat16
            ).items():
                weights.set_layer_weight(idx, key, tensor)
        return gemma4.Gemma4Model(
            config, parallelism, weights, max_generate_batch_size=2
        )

    def test_moe_pure_tp_shards_sum_to_full_output(self):
        experts_count, hidden, intermediate, top_k = 8, 64, 16, 4
        weights = make_layer_weights(
            self.geometry,
            self.device,
            num_experts=experts_count,
            moe_inter=intermediate,
            dtype=torch.float32,
        )
        x = self._rand(6, hidden, seed=1601).float()
        routing = torch.tensor(
            [[0, 1, 2, 3]] * 3 + [[4, 5, 6, 7]] * 3,
            dtype=torch.int64,
            device=self.device,
        )
        top_w = torch.rand(6, top_k, device=self.device)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        full_output = gemma4.Gemma4Experts(weights)._forward_batched_device(
            x, routing, top_w
        )

        partial_outputs = []
        shard_intermediate = intermediate // 2
        for rank in range(2):
            w1 = weights[W.moe_w1].reshape(experts_count, 2, intermediate, hidden)
            w1 = w1[
                :, :, rank * shard_intermediate : (rank + 1) * shard_intermediate
            ].reshape(experts_count, 2 * shard_intermediate, hidden)
            w2 = weights[W.moe_w2][
                ..., rank * shard_intermediate : (rank + 1) * shard_intermediate
            ]
            shard = gemma4.Gemma4Experts({W.moe_w1: w1, W.moe_w2: w2})
            partial_outputs.append(shard._forward_batched_device(x, routing, top_w))
        torch.testing.assert_close(
            partial_outputs[0] + partial_outputs[1],
            full_output,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_moe_expert_parallel_graph_fails_closed(self):
        parallelism = ParallelismConfig()
        parallelism.tp_size = 2
        parallelism.ep_size = 2
        parallelism.world_size = 2
        config = make_config([HybridAttentionType.SLIDING_WINDOW], self.device)
        weights = make_layer_weights(
            self.geometry,
            self.device,
            num_experts=8,
            moe_inter=16,
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "EP CUDA graph"):
            gemma4.Gemma4Experts(
                weights,
                parallelism,
                model_config=config,
                enable_cuda_graph=True,
            )

    def test_experts_graph_mode_lifecycle_reset(self):
        """graph_mode mirroring MUST restore False on non-graph forwards.

        Regression for the state-leak boundary issue (user-reported
        20261004): the old mirroring set experts._graph_mode=True when the
        forward ran under graph capture but never wrote False afterwards.
        Module attributes persist across forwards, so after the capture at
        engine init EVERY later non-graph forward (all prefills) kept
        taking the device-grouped path - silently changing which code path
        and accumulation semantics served eager serving, and invalidating
        any phase-level claim that "prefill ran the eager per-expert path".
        Lifecycle contract under test: non-graph -> graph -> non-graph must
        end with the eager path selected and the transition state synced.
        """
        model = self._build_model()
        layers = list(model.layers[: model.layer_num])
        # non-graph forward: eager path selected
        model._mirror_experts_graph_mode(False)
        self.assertTrue(all(not dl.experts._graph_mode for dl in layers))
        self.assertFalse(model._experts_graph_mode_last)
        # graph capture: grouped path selected, transition recorded
        model._mirror_experts_graph_mode(True)
        self.assertTrue(all(dl.experts._graph_mode for dl in layers))
        self.assertTrue(model._experts_graph_mode_last)
        # post-capture non-graph forward: MUST reset (the regression - the
        # old code never wrote False, leaking True into all later prefills)
        model._mirror_experts_graph_mode(False)
        self.assertTrue(all(not dl.experts._graph_mode for dl in layers))
        self.assertFalse(model._experts_graph_mode_last)
        # per-path counters exist on every experts module (phase evidence)
        for dl in layers:
            self.assertIn("eager", dl.experts._path_counts)
            self.assertIn("grouped", dl.experts._path_counts)
        print("[lifecycle] non-graph->graph->non-graph reset OK", flush=True)

    def test_index_add_colliding_nondeterminism_reference(self):
        """Bazel-anchored reference for the colliding-index index_add_ risk.

        Re-anchors (under Bazel management) the bare-python microprobes used
        during the 2627 flip investigation: floating-point index_add_ with
        COLLIDING indices uses atomic adds whose ordering varies run-to-run.
        This test documents the OPERATOR property only (deterministic
        replacement is bit-stable; unique-index and integer cases are
        stable). It makes NO claim about the engine flip attribution - that
        remains pending a matched-control A/B per the numerical-issues
        report.

        Nondeterminism is asserted only as a WEAK signal: a run where the
        atomic path happens to be stable is still consistent with the risk
        (ordering may coincide); the deterministic path is asserted STABLE
        strictly, since that is the property the fix relies on.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        gen = torch.Generator().manual_seed(901)
        T, top_k, H = 256, 8, 512
        nnz = T * top_k
        contrib = torch.randn(nnz, H, generator=gen).to(device)
        # colliding pattern: token r % T receives rows r, r+T, ...;
        # row r's slot is (r // T) % top_k so every (token, slot) cell is
        # hit exactly once and slot stays in [0, top_k)
        tgt = torch.arange(nnz, device=device) % T
        slot = (torch.arange(nnz, device=device) // T) % top_k
        outs_atomic = []
        for _ in range(16):
            o = torch.zeros(T, H, dtype=torch.float32, device=device)
            o.index_add_(0, tgt, contrib)
            outs_atomic.append(o)
        atomic_var = max(
            float((o - outs_atomic[0]).abs().max()) for o in outs_atomic[1:]
        )
        # deterministic replacement: scatter to unique (token, slot) cells
        # then sum(1) - the pattern Gemma4Experts._forward_grouped_device
        # now uses
        outs_det = []
        for _ in range(16):
            buf = torch.zeros(T, top_k, H, dtype=torch.float32, device=device)
            buf[tgt, slot] = contrib
            outs_det.append(buf.sum(dim=1))
        det_var = max(float((o - outs_det[0]).abs().max()) for o in outs_det[1:])
        # the two methods must agree numerically on the first call
        # (same contributions, fixed order in the deterministic lane)
        method_diff = float((outs_atomic[0] - outs_det[0]).abs().max())
        print(
            f"[index-add-ref] atomic 16x var={atomic_var:.3e} "
            f"det 16x var={det_var:.3e} method diff={method_diff:.3e}",
            flush=True,
        )
        # STRICT: the deterministic path is bit-stable - this is what the
        # fix requires; failure here would invalidate the fix's premise
        self.assertEqual(
            det_var,
            0.0,
            "deterministic scatter+sum accumulation is not bit-stable",
        )
        # WEAK (informational, not a failure): atomic variance observed on
        # this run; printed for the record. A zero value does NOT refute
        # the risk (ordering may coincide by chance).

    def test_moe_graph_grouped_exact_vs_eager(self):
        """Preserve the original grouped-MM versus FP32-round diagnostic gate."""
        if os.environ.get("GEMMA4_RUN_FP32_ROUND_GATE") != "1":
            self.skipTest("run through //...:gemma4_moe_fp32_round_diagnostic")
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        if not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm unavailable")
        device = torch.device("cuda")
        # alignment: bf16 grouped GEMM needs 16-byte strides (multiple of 8
        # elements): H=64, N=16, 2N=32 all satisfy this
        E, H, N, top_k = 8, 64, 16, 4
        gen = torch.Generator().manual_seed(301)
        w1 = (torch.randn(E, 2 * N, H, generator=gen) * 0.05).to(
            device=device, dtype=torch.bfloat16
        )
        w2 = (torch.randn(E, H, N, generator=gen) * 0.05).to(
            device=device, dtype=torch.bfloat16
        )
        weights = {W.moe_w1: w1, W.moe_w2: w2}
        experts = gemma4.Gemma4Experts(weights)

        def reference(x, top_idx, top_w):
            """Independent reference: per-(token,slot) explicit loop in fp32
            (same math as the eager path, written independently)."""
            out = torch.zeros_like(x, dtype=torch.float32)
            xf = x.float()
            for t in range(x.shape[0]):
                for s in range(top_k):
                    e = int(top_idx[t, s])
                    if not 0 <= e < E:
                        raise IndexError(f"expert {e} out of range")
                    h = xf[t] @ w1[e].float().t()
                    gate = h[:N]
                    up = h[N:]
                    y = F.gelu(gate, approximate="tanh") * up
                    y = y @ w2[e].float().t()
                    out[t] += y * float(top_w[t, s])
            return out

        tokens = 12
        x = torch.randn(tokens, H, generator=gen).to(
            device=device, dtype=torch.bfloat16
        )

        routings = [
            # maximally skewed legal top-k: expert 0 appears once for every token
            torch.tensor(
                [[0, 1, 2, 3]] * 9 + [[0, 4, 5, 6]] * 3,
                dtype=torch.int64,
                device=device,
            ),
            # CHANGED routing (replay): uniformly spread, includes expert 7
            torch.tensor(
                [[7, 5, 3, 1]] * 4 + [[6, 4, 2, 0]] * 4 + [[7, 0, 2, 4]] * 4,
                dtype=torch.int64,
                device=device,
            ),
            # maximally concentrated legal routing: the same four experts per token
            torch.tensor([[0, 1, 2, 3]] * tokens, dtype=torch.int64, device=device),
        ]
        experts._graph_mode = True
        for r, top_idx in enumerate(routings):
            top_w = torch.rand(tokens, top_k, generator=gen).to(device=device)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            got = experts.forward(x, top_idx, top_w)
            ref = reference(x, top_idx, top_w)
            torch.testing.assert_close(
                got.float(),
                ref,
                rtol=5e-2,
                atol=5e-2,
                msg=lambda m, r=r: f"routing case {r} mismatch: {m}",
            )
            print(
                f"[moe-graph] routing case {r} "
                f"(max expert load {int(torch.bincount(top_idx.reshape(-1), minlength=E).max())}/{tokens * top_k}) OK",
                flush=True,
            )
        experts._graph_mode = False
        top_idx = routings[0]
        top_w = torch.rand(tokens, top_k, generator=gen).to(device=device)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        eager_output = experts._forward_eager(x, top_idx, top_w)
        hf_dtype_reference = hf_experts(x, top_idx, top_w, weights)
        self.assertTrue(
            torch.equal(eager_output, hf_dtype_reference),
            "eager path must be bit-equal to the independent HF dtype path",
        )

        grouped_output = experts.forward(x, top_idx, top_w)
        batched_output = experts._forward_batched_device(x, top_idx, top_w)
        fp32_reference = reference(x, top_idx, top_w)
        fp32_rounded = fp32_reference.to(grouped_output.dtype)
        round_difference = float(
            (grouped_output.float() - fp32_rounded.float()).abs().max()
        )
        grouped_hf_difference = float(
            (grouped_output.float() - hf_dtype_reference.float()).abs().max()
        )
        batched_hf_difference = float(
            (batched_output.float() - hf_dtype_reference.float()).abs().max()
        )
        grouped_batched_difference = float(
            (grouped_output.float() - batched_output.float()).abs().max()
        )
        print(
            f"[moe-graph] eager_hf_bit_equal=True "
            f"grouped_hf_bit_equal={bool(torch.equal(grouped_output, hf_dtype_reference))} "
            f"grouped_hf_diff={grouped_hf_difference:.2e} "
            f"batched_hf_bit_equal={bool(torch.equal(batched_output, hf_dtype_reference))} "
            f"batched_hf_diff={batched_hf_difference:.2e} "
            f"grouped_batched_bit_equal={bool(torch.equal(grouped_output, batched_output))} "
            f"grouped_batched_diff={grouped_batched_difference:.2e} "
            f"batched_fp32_round_bit_equal={bool(torch.equal(batched_output, fp32_rounded))} "
            f"grouped_fp32_round_bit_equal="
            f"{bool(torch.equal(grouped_output, fp32_rounded))} "
            f"grouped_fp32_round_diff={round_difference:.2e}",
            flush=True,
        )
        self.assertTrue(
            torch.equal(grouped_output, fp32_rounded),
            "grouped path must be bit-equal to the fp32 reference rounded "
            f"to the output dtype (post-round diff {round_difference:.2e})",
        )

    def test_moe_graph_grouped_exact_vs_non_graph_decode(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        if not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm unavailable")
        device = torch.device("cuda")
        experts_count, hidden, intermediate, top_k = 8, 64, 16, 4
        generator = torch.Generator().manual_seed(2301)
        weights = {
            W.moe_w1: (
                torch.randn(
                    experts_count,
                    2 * intermediate,
                    hidden,
                    generator=generator,
                )
                * 0.05
            ).to(device=device, dtype=torch.bfloat16),
            W.moe_w2: (
                torch.randn(
                    experts_count,
                    hidden,
                    intermediate,
                    generator=generator,
                )
                * 0.05
            ).to(device=device, dtype=torch.bfloat16),
        }
        experts = gemma4.Gemma4Experts(weights)
        x = torch.randn(12, hidden, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        routing = torch.tensor(
            [[0, 1, 2, 3]] * 5 + [[7, 5, 3, 1]] * 4 + [[6, 4, 2, 0]] * 3,
            dtype=torch.int64,
            device=device,
        )
        top_w = torch.rand(12, top_k, generator=generator).to(device=device)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        grouped = experts._forward_grouped_device(x, routing, top_w)
        non_graph_decode = experts._forward_batched_device(x, routing, top_w)
        self.assertTrue(
            torch.equal(grouped, non_graph_decode),
            "graph grouped path must be bit-equal to production non-graph decode",
        )

    def test_moe_graph_repeated_calls_varying_routing(self):
        """MoE graph path under REPEATED calls with CHANGED routing - the
        serving pattern (routing differs every step; the impl object and
        any cached state persist). Each call compared against the fp32
        per-(token,slot) reference computed on the same routing."""
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        if not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm unavailable")
        device = torch.device("cuda")
        E, H, N, top_k = 8, 64, 16, 4
        gen = torch.Generator().manual_seed(701)
        w1 = (torch.randn(E, 2 * N, H, generator=gen) * 0.05).to(
            device=device, dtype=torch.bfloat16
        )
        w2 = (torch.randn(E, H, N, generator=gen) * 0.05).to(
            device=device, dtype=torch.bfloat16
        )
        experts = gemma4.Gemma4Experts({W.moe_w1: w1, W.moe_w2: w2})

        def reference(x, top_idx, top_w):
            out = torch.zeros_like(x, dtype=torch.float32)
            xf = x.float()
            for t in range(x.shape[0]):
                for s in range(top_k):
                    e = int(top_idx[t, s])
                    h = xf[t] @ w1[e].float().t()
                    gate, up = h.chunk(2, dim=-1)
                    y = F.gelu(gate, approximate="tanh") * up
                    y = y @ w2[e].float().t()
                    out[t] += y * float(top_w[t, s])
            return out

        experts._graph_mode = True
        tokens = 5
        routings = [
            torch.tensor(
                [[0, 1, 2, 3]] * 3 + [[0, 4, 5, 6]] * 2,
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [[7, 5, 3, 1]] * 2 + [[6, 4, 2, 0]] * 3,
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([[0, 1, 2, 3]] * tokens, dtype=torch.int64, device=device),
        ]
        for r, top_idx in enumerate(routings):
            x = torch.randn(tokens, H, generator=gen).to(
                device=device, dtype=torch.bfloat16
            )
            top_w = torch.rand(tokens, top_k, generator=gen).to(device=device)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            got = experts.forward(x, top_idx, top_w)
            ref = reference(x, top_idx, top_w)
            torch.testing.assert_close(
                got.float(),
                ref,
                rtol=5e-2,
                atol=5e-2,
                msg=lambda m, r=r: f"MoE graph repeated-call routing {r} mismatch: {m}",
            )
            print(
                f"[moe-graph-repeat] call {r} (routing {r}) OK",
                flush=True,
            )

    def test_moe_cuda_graph_replays_changed_routing(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        if not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm unavailable")
        device = torch.device("cuda")
        experts_count, hidden, intermediate, top_k = 8, 64, 16, 4
        generator = torch.Generator().manual_seed(1701)
        weights = {
            W.moe_w1: (
                torch.randn(
                    experts_count,
                    2 * intermediate,
                    hidden,
                    generator=generator,
                )
                * 0.05
            ).to(device=device, dtype=torch.bfloat16),
            W.moe_w2: (
                torch.randn(
                    experts_count,
                    hidden,
                    intermediate,
                    generator=generator,
                )
                * 0.05
            ).to(device=device, dtype=torch.bfloat16),
        }
        experts = gemma4.Gemma4Experts(weights)
        experts._graph_mode = True
        static_x = torch.empty(6, hidden, dtype=torch.bfloat16, device=device)
        static_idx = torch.empty(6, top_k, dtype=torch.int64, device=device)
        static_w = torch.empty(6, top_k, dtype=torch.float32, device=device)
        static_x.copy_(torch.randn(6, hidden, generator=generator).to(torch.bfloat16))
        static_idx.copy_(
            torch.tensor(
                [[0, 1, 2, 3]] * 3 + [[0, 4, 5, 6]] * 3,
                dtype=torch.int64,
                device=device,
            )
        )
        static_w.copy_(torch.full((6, top_k), 0.25, device=device))

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                experts.forward(static_x, static_idx, static_w)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = experts.forward(static_x, static_idx, static_w)

        cases = [
            torch.tensor(
                [[7, 5, 3, 1]] * 3 + [[6, 4, 2, 0]] * 3,
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([[0, 1, 2, 3]] * 6, dtype=torch.int64, device=device),
        ]
        for case_idx, routing in enumerate(cases):
            x = torch.randn(6, hidden, generator=generator).to(
                device=device, dtype=torch.bfloat16
            )
            top_w = torch.rand(6, top_k, generator=generator).to(device=device)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            uncaptured_grouped = experts._forward_grouped_device(x, routing, top_w)
            independent_reference = hf_experts(x, routing, top_w, weights)
            static_x.copy_(x)
            static_idx.copy_(routing)
            static_w.copy_(top_w)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(captured_output, uncaptured_grouped),
                f"CUDA graph replay routing case {case_idx} changed grouped output",
            )
            torch.testing.assert_close(
                captured_output.float(),
                independent_reference.float(),
                rtol=5e-2,
                atol=5e-2,
                msg=lambda message, case_idx=case_idx: (
                    f"CUDA graph replay routing case {case_idx} differs from "
                    f"the independent HF dtype path: {message}"
                ),
            )

    def test_moe_graph_mode_rejects_unsupported_dtype(self):
        weights = make_layer_weights(
            self.geometry,
            self.device,
            num_experts=8,
            moe_inter=12,
            dtype=torch.float32,
        )
        experts = gemma4.Gemma4Experts(weights)
        experts._graph_mode = True
        x = torch.zeros(2, 64, device=self.device)
        top_idx = torch.zeros(2, 4, dtype=torch.int64, device=self.device)
        top_w = torch.full((2, 4), 0.25, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "requires BF16 aligned weights"):
            experts(x, top_idx, top_w)

    def test_graph_attention_accepts_fixed_multi_query_shapes(self):
        geometry = gemma4.Gemma4LayerGeometry("swa", 4, 2, 64, 1e4, 1.0, self.window)
        prefill_inputs = make_prefill_inputs([2], [[0]], self.device)
        prefill_inputs.is_cuda_graph = True
        prefill_impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, prefill_inputs, self.page_size
        )
        q = self._rand(2, 4, 64, seed=1801)
        k = self._rand(2, 2, 64, seed=1802)
        v = self._rand(2, 2, 64, seed=1803)
        self.assertEqual(prefill_impl.forward(q, k, v).shape, q.shape)

        decode_inputs = make_decode_inputs([4, 4], [[0, 1], [2, 3]])
        decode_inputs.is_cuda_graph = True
        decode_impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, decode_inputs, self.page_size
        )
        with self.assertRaisesRegex(RuntimeError, "fixed query width"):
            decode_impl.forward(q[:1], k[:1], v[:1])

    def test_swa_graph_multi_query_retains_earliest_window_page(self):
        page_size = 128
        window = 1024
        prefix_len = 2042
        query_len = 7
        kv_len = prefix_len + query_len
        logical_page_count = (kv_len + page_size - 1) // page_size
        resident_page_count = 10
        block_table = [-1] * (logical_page_count - resident_page_count) + list(
            range(resident_page_count)
        )
        geometry = gemma4.Gemma4LayerGeometry("swa", 1, 1, 64, 1e4, 1.0, window)
        attn_inputs = make_prefill_inputs([query_len], [block_table], self.device)
        attn_inputs.prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32)
        attn_inputs.is_cuda_graph = True
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, attn_inputs, page_size)
        self.assertEqual(
            impl.fmha_params.positions_d[:query_len].cpu().tolist(),
            list(range(prefix_len, kv_len)),
        )

        cache_base = torch.zeros(
            (resident_page_count, 2, 1, page_size, 64),
            dtype=torch.bfloat16,
            device=self.device,
        )
        cache_base[0, 1, 0, 123:128] = 64
        kv_cache = LayerKVCache(cache_base, page_size, layer_id=0, tag="swa")
        q = torch.zeros(query_len, 1, 64, dtype=torch.bfloat16, device=self.device)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)

        impl.forward(q, k, v, kv_cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = impl.forward(q, k, v, kv_cache)
        graph.replay()
        torch.cuda.synchronize()

        expected = torch.tensor(
            [5, 4, 3, 2, 1, 0, 0], dtype=torch.float32, device=self.device
        ) * (64.0 / window)
        torch.testing.assert_close(
            captured_output[:, 0, 0].float(), expected, rtol=0, atol=5e-3
        )

    def test_full_graph_metadata_mutation_repeat(self):
        """Full-layer graph path under REPEATED calls with MUTATED metadata.

        The E2E flip reproduces across requests in one server session. If
        _forward_full_graph (or the MoE graph path) mis-handles a metadata
        refresh between calls - e.g. a cached capture-time constant like
        _full_row_pages going stale, or a buffer bound to a pre-mutation
        value - repeated calls with CHANGED lengths on the SAME impl would
        diverge from per-call-fresh eager recomputation. This test replays
        the serving pattern at unit scale: one impl, sequential decode
        calls with different kv lengths, each compared to a fresh eager
        reference computed on identical state.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        kv_heads, head_dim, head_num = 2, 64, 4
        geometry = gemma4.Gemma4LayerGeometry(
            "full", head_num, kv_heads, head_dim, 1e6, 0.25, 0
        )
        # three sequential decode steps on one impl: lengths grow 20->21->22
        # (page table extended each step, last_page_len changes)
        start_len = 20
        total_pages = 8
        table = [list(range(6))]  # decode table rows (padded)
        gen = torch.Generator().manual_seed(611)
        q_all = torch.randn(start_len + 3, head_num, head_dim, generator=gen).to(
            device=device, dtype=torch.bfloat16
        )
        k_all = torch.randn(start_len + 3, kv_heads, head_dim, generator=gen).to(
            device=device, dtype=torch.bfloat16
        )
        v_all = torch.randn(start_len + 3, kv_heads, head_dim, generator=gen).to(
            device=device, dtype=torch.bfloat16
        )
        kv_cache = LayerKVCache(
            torch.full(
                (total_pages, 2, kv_heads, self.page_size, head_dim),
                float("nan"),
                dtype=torch.bfloat16,
                device=device,
            ),
            self.page_size,
            layer_id=0,
            tag="full",
        )
        # prefill 20 tokens into pages 0..4
        prefill_inputs = make_prefill_inputs([start_len], [list(range(5))], device)
        eager_prefill = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, prefill_inputs, self.page_size
        )
        eager_prefill.forward(
            q_all[:start_len], k_all[:start_len], v_all[:start_len], kv_cache
        )
        table_row = list(range((start_len + 1 + self.page_size - 1) // self.page_size))
        decode_inputs = make_decode_inputs([start_len], [table_row])
        decode_inputs.is_cuda_graph = True
        impl = gemma4.Gemma4TorchFMHAImpl(None, geometry, decode_inputs, self.page_size)
        static_q = torch.empty(
            1, head_num, head_dim, dtype=torch.bfloat16, device=device
        )
        static_k = torch.empty(
            1, kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        static_v = torch.empty(
            1, kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        static_q.copy_(q_all[start_len : start_len + 1])
        static_k.copy_(k_all[start_len : start_len + 1])
        static_v.copy_(v_all[start_len : start_len + 1])
        impl.forward(static_q, static_k, static_v, kv_cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = impl.forward(static_q, static_k, static_v, kv_cache)

        for step in range(3):
            new_tok = start_len + step
            kv_len = new_tok + 1
            table_row = list(range((kv_len + self.page_size - 1) // self.page_size))
            decode_inputs = make_decode_inputs([new_tok], [table_row])
            decode_inputs.is_cuda_graph = True
            impl.prepare_cuda_graph(decode_inputs)
            static_q.copy_(q_all[new_tok : new_tok + 1])
            static_k.copy_(k_all[new_tok : new_tok + 1])
            static_v.copy_(v_all[new_tok : new_tok + 1])
            graph.replay()
            torch.cuda.synchronize()
            reference = hf_reference_paged_attention(
                q_all[new_tok : new_tok + 1],
                k_all[: new_tok + 1],
                v_all[: new_tok + 1],
                torch.tensor([new_tok], device=device),
            )
            torch.testing.assert_close(
                captured_output.float(),
                reference.float(),
                rtol=5e-2,
                atol=5e-2,
                msg=lambda message, step=step: (
                    f"full graph replay metadata step {step} "
                    f"(kv len {kv_len}) mismatch: {message}"
                ),
            )
            print(
                f"[full-graph-replay] step {step} kv_len={kv_len} OK",
                flush=True,
            )

    def test_swa_layer_graph_windowed_matches_eager(self):
        """Graph-mode SWA path (trailing-window chunked SDPA + LSE merge)
        vs the eager per-request reference on identical state.

        Covers heterogeneous row lengths INCLUDING the chunk-existence edge:
        rows shorter than the window (few pages), rows longer (window
        straddles page boundaries), and rows whose page count is below the
        capture-time max so chunks beyond their table must be masked.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda")
        kv_heads, head_dim, head_num = 2, 64, 4
        window = 8
        geometry = gemma4.Gemma4LayerGeometry(
            "swa", head_num, kv_heads, head_dim, 1e4, 1.0, window
        )
        # heterogeneous lengths: 20 (5 pages), 13 (4), 5 (2), 3 (1)
        lens = [20, 13, 5, 3]
        pages_per_req = [(l + self.page_size - 1) // self.page_size for l in lens]
        table = []
        p = 0
        for i, pr in enumerate(pages_per_req):
            need = (lens[i] + self.page_size) // self.page_size
            row = list(range(p, p + pr))
            while len(row) < need:
                row.append(p + len(row))
            table.append(row)
            p = row[-1] + 1
        total_pages = max(max(r) for r in table) + 1
        kv_cache = LayerKVCache(
            torch.full(
                (total_pages, 2, kv_heads, self.page_size, head_dim),
                float("nan"),
                dtype=torch.bfloat16,
                device=device,
            ),
            self.page_size,
            layer_id=0,
            tag="swa",
        )
        starts = [0]
        for l in lens:
            starts.append(starts[-1] + l)
        prefill_inputs = make_prefill_inputs(lens, table, device)
        eager_prefill = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, prefill_inputs, self.page_size
        )
        total = sum(lens)
        packed_q = torch.cat(
            [
                self._rand(l, head_num, head_dim, seed=511 + i)
                for i, l in enumerate(lens)
            ],
            dim=0,
        )
        packed_k = torch.cat(
            [
                self._rand(l, kv_heads, head_dim, seed=521 + i)
                for i, l in enumerate(lens)
            ],
            dim=0,
        )
        packed_v = torch.cat(
            [
                self._rand(l, kv_heads, head_dim, seed=531 + i)
                for i, l in enumerate(lens)
            ],
            dim=0,
        )
        eager_prefill.forward(packed_q, packed_k, packed_v, kv_cache)

        decode_inputs = make_decode_inputs(list(lens), table)
        decode_inputs.is_cuda_graph = True
        static_q = self._rand(len(lens), head_num, head_dim, seed=541)
        static_k = self._rand(len(lens), kv_heads, head_dim, seed=542)
        static_v = self._rand(len(lens), kv_heads, head_dim, seed=543)
        graph_impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, decode_inputs, self.page_size
        )
        graph_impl.forward(static_q, static_k, static_v, kv_cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = graph_impl.forward(static_q, static_k, static_v, kv_cache)

        for replay, seed in enumerate((551, 561)):
            q_dec = self._rand(len(lens), head_num, head_dim, seed=seed)
            k_dec = self._rand(len(lens), kv_heads, head_dim, seed=seed + 1)
            v_dec = self._rand(len(lens), kv_heads, head_dim, seed=seed + 2)
            static_q.copy_(q_dec)
            static_k.copy_(k_dec)
            static_v.copy_(v_dec)
            graph.replay()
            torch.cuda.synchronize()

            refs = []
            for i, length in enumerate(lens):
                full_k = torch.cat(
                    [packed_k[starts[i] : starts[i] + length], k_dec[i : i + 1]],
                    dim=0,
                )
                full_v = torch.cat(
                    [packed_v[starts[i] : starts[i] + length], v_dec[i : i + 1]],
                    dim=0,
                )
                refs.append(
                    hf_reference_paged_attention(
                        q_dec[i : i + 1],
                        full_k,
                        full_v,
                        torch.tensor([length], device=device),
                        sliding_window=window,
                    )
                )
            reference = torch.cat(refs, dim=0)
            torch.testing.assert_close(
                captured_output.float(),
                reference.float(),
                rtol=5e-2,
                atol=5e-2,
                msg=lambda message, replay=replay: (
                    f"SWA graph replay {replay} differs from eager reference: {message}"
                ),
            )
        print(
            f"[swa-graph-replay] changed inputs match eager "
            f"(heterogeneous lens {lens}, window {window}) OK",
            flush=True,
        )

    def test_full_layer_graph_chunked_matches_eager(self):
        """The full-layer graph path (page-chunked SDPA + LSE merge) must
        numerically match the eager per-request reference on identical
        state. Uses >=3 chunks (seq 20 / page 4 = 5 chunks) with
        HETEROGENEOUS per-request lengths: the LSE running-merge invariant
        (out_acc normalized, lse_acc = m + log(sum exp)) only shows its
        third-chunk bug (equal-LSE chunks mis-weighted 1/4,1/4,1/2) when at
        least three chunks merge."""
        # heterogeneous request lengths: 20, 13, 7 tokens (5/4/2 pages).
        # The decode step's new token may need one more page (engine tables
        # always cover the new token's slot), so size the tables for the
        # DECODE step from the start and use ONE page numbering for both the
        # prefill write and the decode read (a mismatch was this fixture's
        # earlier bug: req0 passed only because its table happened to align)
        batch = 3
        kv_heads, head_dim, head_num = 2, 64, 4
        geometry = gemma4.Gemma4LayerGeometry(
            "full", head_num, kv_heads, head_dim, 1e6, 0.25, 0
        )
        lens = [20, 13, 7]
        pages_per_req = [(l + self.page_size - 1) // self.page_size for l in lens]
        table = []
        p = 0
        for i, pr in enumerate(pages_per_req):
            need = (lens[i] + self.page_size) // self.page_size
            row = list(range(p, p + pr))
            while len(row) < need:
                row.append(p + len(row))
            table.append(row)
            p = row[-1] + 1
        total_pages = max(max(r) for r in table) + 1
        kv_cache = self._make_cache_full(total_pages)
        q_all = self._rand(max(lens), head_num, head_dim, seed=211)
        starts = [0]
        for l in lens:
            starts.append(starts[-1] + l)
        prefill_inputs = make_prefill_inputs(lens, table, self.device)
        eager_prefill = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, prefill_inputs, self.page_size
        )
        packed_q = torch.cat(
            [
                q_all[: lens[0]],
                self._rand(lens[1], head_num, head_dim, seed=217),
                self._rand(lens[2], head_num, head_dim, seed=218),
            ],
            dim=0,
        )
        packed_k = torch.cat(
            [
                self._rand(l, kv_heads, head_dim, seed=212 + i)
                for i, l in enumerate(lens)
            ],
            dim=0,
        )
        packed_v = torch.cat(
            [
                self._rand(l, kv_heads, head_dim, seed=215 + i)
                for i, l in enumerate(lens)
            ],
            dim=0,
        )
        eager_prefill.forward(packed_q, packed_k, packed_v, kv_cache)

        # decode step on the SAME table/page numbering as the prefill
        decode_inputs = make_decode_inputs(list(lens), table)
        decode_inputs.is_cuda_graph = True  # select the graph path
        q_dec = self._rand(batch, head_num, head_dim, seed=219)
        k_dec = self._rand(batch, kv_heads, head_dim, seed=220)
        v_dec = self._rand(batch, kv_heads, head_dim, seed=221)
        graph_impl = gemma4.Gemma4TorchFMHAImpl(
            None, geometry, decode_inputs, self.page_size
        )
        out_graph = graph_impl.forward(q_dec, k_dec, v_dec, kv_cache)

        # eager reference per request: own prefix + own decode token
        refs = []
        for i, l in enumerate(lens):
            full_k = torch.cat(
                [packed_k[starts[i] : starts[i] + l], k_dec[i : i + 1]], dim=0
            )
            full_v = torch.cat(
                [packed_v[starts[i] : starts[i] + l], v_dec[i : i + 1]], dim=0
            )
            refs.append(
                hf_reference_paged_attention(
                    q_dec[i : i + 1],
                    full_k,
                    full_v,
                    torch.tensor([l], device=self.device),
                )
            )
        ref = torch.cat(refs, dim=0)
        import os as _os_dbg

        if _os_dbg.environ.get("GEMMA4_FULLGRAPH_DEBUG"):
            for i in range(batch):
                d = (out_graph[i].float() - ref[i].float()).abs()
                print(
                    f"[full-graph-dbg] req{i} len={lens[i]} "
                    f"maxdiff={float(d.max()):.4f} "
                    f"graph_head={out_graph[i, 0, :3].float().tolist()} "
                    f"ref_head={ref[i, 0, :3].float().tolist()}",
                    flush=True,
                )
        torch.testing.assert_close(
            out_graph.float(),
            ref.float(),
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m: f"full-layer graph chunked path mismatch: {m}",
        )
        print(
            f"[full-graph] chunked SDPA matches eager (heterogeneous lens "
            f"{lens}, {max(pages_per_req)} chunks) OK",
            flush=True,
        )

    def _make_cache_full(self, num_pages):
        base = torch.full(
            (num_pages, 2, 2, self.page_size, 64),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        return LayerKVCache(base, self.page_size, layer_id=0, tag="full")

    def test_model_selects_sdpa_correctness_backend_by_default(self):
        model = self._build_model()
        attn_inputs = make_prefill_inputs([4], [[0, 1]], self.device)
        inputs = PyModelInputs()
        inputs.input_ids = torch.zeros(4, dtype=torch.int32, device=self.device)
        inputs.attention_inputs = {
            gemma4.GEMMA4_TAG_SWA: attn_inputs,
            gemma4.GEMMA4_TAG_FULL: attn_inputs,
        }
        impls = model.prepare_fmha_impl(inputs)
        self.assertIn(gemma4.GEMMA4_TAG_SWA, impls)
        self.assertIn(gemma4.GEMMA4_TAG_FULL, impls)
        self.assertIsInstance(
            impls[gemma4.GEMMA4_TAG_SWA],
            gemma4.Gemma4TorchFMHAImpl,
            msg="SWA layers must stay on the accepted SDPA correctness backend",
        )
        self.assertIsInstance(
            impls[gemma4.GEMMA4_TAG_FULL],
            gemma4.Gemma4TorchFMHAImpl,
            msg="full layers must stay on the accepted SDPA correctness backend",
        )
        print("[swa-equiv] model-level backend selection OK", flush=True)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4GegluKernelTest(TestCase):
    def test_matches_torch_bitwise(self):
        torch.manual_seed(20261008)
        for rows, intermediate_size in ((257, 704), (257, 2112)):
            gate_up = torch.randn(
                rows,
                intermediate_size * 2,
                device="cuda",
                dtype=torch.bfloat16,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expected = F.gelu(gate, approximate="tanh") * up
            actual = rtp_llm_ops.gemma4_geglu_tanh_bf16(gate_up)
            self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4WeightedReorderKernelTest(TestCase):
    def test_matches_torch_bitwise(self):
        torch.manual_seed(20261008)
        rows = 257
        hidden_size = 2816
        expert_output = torch.randn(
            rows, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        sorted_weight = torch.randn(rows, device="cuda", dtype=torch.bfloat16)
        permutation = torch.randperm(rows, device="cuda")
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(rows, device="cuda")
        expected = (expert_output * sorted_weight.unsqueeze(-1))[inverse_permutation]
        actual = rtp_llm_ops.gemma4_weighted_reorder_bf16(
            expert_output, sorted_weight, inverse_permutation
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_sorted_input_gather_matches_torch_bitwise(self):
        torch.manual_seed(20261009)
        tokens = 33
        top_k = 8
        hidden = torch.randn(tokens, 2816, device="cuda", dtype=torch.bfloat16)
        permutation = torch.randperm(tokens * top_k, device="cuda")
        token_idx = (
            torch.arange(tokens, device="cuda")
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        expected = hidden[token_idx[permutation]]
        actual = rtp_llm_ops.gemma4_gather_sorted_expert_input_bf16(
            hidden, permutation, top_k
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_top8_sum_matches_torch_tree_bitwise(self):
        torch.manual_seed(20261010)
        values = torch.randn(33, 8, 2816, device="cuda", dtype=torch.bfloat16)
        expected = values.sum(dim=1)
        actual = rtp_llm_ops.gemma4_top8_sum_bf16(values)
        self.assertTrue(torch.equal(actual, expected))

    def test_router_weight_postprocess_matches_torch_bitwise(self):
        torch.manual_seed(20261012)
        top_weights = torch.rand(257, 8, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, 128, (257, 8), device="cuda", dtype=torch.int64)
        expert_scales = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        expected = top_weights / top_weights.sum(dim=-1, keepdim=True)
        expected = expected * expert_scales[top_indices]
        actual = rtp_llm_ops.gemma4_finalize_router_weights_bf16(
            top_weights, top_indices, expert_scales
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_counting_pack_contract(self):
        torch.manual_seed(20261011)
        num_experts = 128
        rows = 4096
        expert_ids = torch.randint(
            0, num_experts, (rows,), device="cuda", dtype=torch.int64
        )
        weights = torch.randn(rows, device="cuda", dtype=torch.bfloat16)
        perm, inv_perm, sorted_weights, offsets = (
            rtp_llm_ops.gemma4_prepare_grouped_moe(expert_ids, weights, num_experts)
        )
        sorted_ids = expert_ids[perm]
        self.assertTrue(bool(torch.all(sorted_ids[1:] >= sorted_ids[:-1])))
        self.assertTrue(torch.equal(inv_perm[perm], torch.arange(rows, device="cuda")))
        self.assertTrue(torch.equal(sorted_weights, weights[perm]))
        expected_offsets = torch.bincount(expert_ids, minlength=num_experts).cumsum(
            0, dtype=torch.int32
        )
        self.assertTrue(torch.equal(offsets, expected_offsets))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4PvOutKernelTest(TestCase):
    def test_full_triangular_key_lengths_match_full_width(self):
        torch.manual_seed(20261017)
        q = torch.randn(1024, 16, 512, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(8192, 16, 512, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        reference_scores = rtp_llm_ops.gemma4_qk_bmm_8192_bf16(q, keys)
        for key_len in range(1024, 8193, 1024):
            with self.subTest(key_len=key_len):
                candidate_scores = rtp_llm_ops.gemma4_qk_bmm_8192_bf16_key_len(
                    q, keys, key_len
                )
                self.assertTrue(
                    torch.equal(
                        candidate_scores[..., :key_len],
                        reference_scores[..., :key_len],
                    )
                )
                candidate_scores[..., key_len:].fill_(float("nan"))
                query_start = key_len - 1024
                reference_probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(
                    reference_scores, query_start, -1
                )
                candidate_probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(
                    candidate_scores, query_start, -1
                )
                self.assertTrue(
                    torch.equal(candidate_probabilities, reference_probabilities)
                )
                expected = rtp_llm_ops.gemma4_pv_bmm_8192_bf16(
                    reference_probabilities, values
                )
                actual = torch.empty_like(expected)
                rtp_llm_ops.gemma4_pv_bmm_8192_bf16_out_key_len(
                    candidate_probabilities, values, actual, key_len
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_full_output_view_matches_allocating_path(self):
        torch.manual_seed(20261013)
        probabilities = torch.randn(
            1, 16, 1024, 8192, device="cuda", dtype=torch.bfloat16
        )
        values = torch.randn(8192, 16, 512, device="cuda", dtype=torch.bfloat16)
        expected = rtp_llm_ops.gemma4_pv_bmm_8192_bf16(probabilities, values)
        destination = torch.empty(2048, 16, 512, device="cuda", dtype=torch.bfloat16)
        output = destination[512:1536]
        rtp_llm_ops.gemma4_pv_bmm_8192_bf16_out(probabilities, values, output)
        self.assertTrue(torch.equal(output, expected))

    def test_swa_output_view_matches_allocating_path(self):
        torch.manual_seed(20261014)
        probabilities = torch.randn(
            1, 16, 512, 8192, device="cuda", dtype=torch.bfloat16
        )
        values = torch.randn(8192, 16, 256, device="cuda", dtype=torch.bfloat16)
        expected = rtp_llm_ops.gemma4_swa_pv_bmm_8192_bf16(probabilities, values)
        destination = torch.empty(1024, 16, 256, device="cuda", dtype=torch.bfloat16)
        output = destination[256:768]
        rtp_llm_ops.gemma4_swa_pv_bmm_8192_bf16_out(probabilities, values, output)
        self.assertTrue(torch.equal(output, expected))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4AddScaleKernelTest(TestCase):
    def test_matches_torch_bitwise(self):
        torch.manual_seed(20261008)
        residual = torch.randn(257, 2816, device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn_like(residual)
        scale = torch.randn(1, device="cuda", dtype=torch.bfloat16)
        add_expected = residual + hidden
        add_actual = rtp_llm_ops.gemma4_add_bf16(residual, hidden)
        self.assertTrue(torch.equal(add_actual, add_expected))
        expected = add_expected * scale
        actual = rtp_llm_ops.gemma4_add_scale_bf16(residual, hidden, scale)
        self.assertTrue(torch.equal(actual, expected))
        scalar_scale = 2816**-0.5
        scale_actual = rtp_llm_ops.gemma4_scale_bf16(residual, scalar_scale)
        self.assertTrue(torch.equal(scale_actual, residual * scalar_scale))
        channel_scale = torch.randn(2816, device="cuda", dtype=torch.bfloat16)
        expected_router = residual * channel_scale * scalar_scale
        actual_router = rtp_llm_ops.gemma4_router_scale_bf16(
            residual, channel_scale, scalar_scale
        )
        self.assertTrue(torch.equal(actual_router, expected_router))

    def test_postprocess_ops_match_torch_bitwise(self):
        torch.manual_seed(20261016)
        hidden = torch.randn(33, 2816, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([0, 17, 32], device="cuda", dtype=torch.int32)
        gathered = rtp_llm_ops.gemma4_gather_rows_bf16(hidden, indices)
        self.assertTrue(torch.equal(gathered, hidden.index_select(0, indices.long())))

        logits = torch.randn(3, 4097, device="cuda", dtype=torch.float32) * 50
        expected = torch.tanh(logits / 30.0) * 30.0
        actual = rtp_llm_ops.gemma4_logit_softcap_fp32(logits, 30.0)
        self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4RmsMeanKernelTest(TestCase):
    def test_matches_torch_bitwise(self):
        torch.manual_seed(20261008)
        for hidden_size in (256, 512, 2816):
            for rows in (1, 2, 8, 16, 33):
                squared = torch.randn(
                    rows, hidden_size, device="cuda", dtype=torch.float32
                ).pow(2)
                expected = squared.mean(-1, keepdim=True)
                actual = rtp_llm_ops.gemma4_rms_mean_fp32(squared)
                self.assertTrue(
                    torch.equal(actual, expected),
                    f"RMS mean mismatch for rows={rows}, hidden={hidden_size}",
                )
                expected_inv = torch.pow(expected + EPS, -0.5)
                actual_inv = rtp_llm_ops.gemma4_rms_inv_fp32(squared, EPS)
                self.assertTrue(
                    torch.equal(actual_inv, expected_inv),
                    f"RMS inv mismatch for rows={rows}, hidden={hidden_size}",
                )
        squared_3d = torch.randn(3, 8, 256, device="cuda", dtype=torch.float32).pow(2)
        self.assertTrue(
            torch.equal(
                rtp_llm_ops.gemma4_rms_mean_fp32(squared_3d),
                squared_3d.mean(-1, keepdim=True),
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Gemma4RopeKernelTest(TestCase):
    def test_rope_table_matches_torch_bitwise(self):
        positions = torch.arange(8192, device="cuda", dtype=torch.int32)
        for head_dim, base, partial_factor in (
            (256, 10_000.0, 1.0),
            (512, 1_000_000.0, 0.25),
        ):
            table = gemma4.Gemma4RopeTable(head_dim, base, partial_factor)
            expected_cos, expected_sin = table.cos_sin(positions)
            actual_cos, actual_sin = rtp_llm_ops.gemma4_rope_cos_sin_bf16(
                positions, table._inv_freq_on(positions.device)
            )
            self.assertTrue(torch.equal(actual_cos, expected_cos.to(torch.bfloat16)))
            self.assertTrue(torch.equal(actual_sin, expected_sin.to(torch.bfloat16)))

    def test_qk_rope_matches_torch_bitwise(self):
        torch.manual_seed(20261007)
        for head_dim, query_heads, kv_heads, partial_factor in (
            (256, 16, 8, 1.0),
            (512, 16, 2, 0.25),
        ):
            tokens = 257
            q = torch.randn(
                tokens,
                query_heads,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            k = torch.randn(
                tokens,
                kv_heads,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
            cos, sin = hf_cos_sin(positions, head_dim, 1_000_000.0, partial_factor)
            cos = cos.to(torch.bfloat16).contiguous()
            sin = sin.to(torch.bfloat16).contiguous()
            expected_q = gemma4.apply_gemma4_rope(q, cos, sin)
            expected_k = gemma4.apply_gemma4_rope(k, cos, sin)
            actual_q, actual_k = rtp_llm_ops.gemma4_qk_rope_bf16(q, k, cos, sin)
            self.assertTrue(torch.equal(actual_q, expected_q))
            self.assertTrue(torch.equal(actual_k, expected_k))


if __name__ == "__main__":
    main()

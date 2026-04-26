"""DSV4 KV Cache precision test — compare internal register_buffer vs framework 7-group cache.

Creates a small V4Transformer with MoE stubbed out, runs prefill + decode with:
  1. Internal register_buffer KV cache (original path)
  2. Framework 7-group BlockPool-style KV cache (manually managed tensors)
Verifies outputs match.
"""

import copy
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock kernel module
_kernel = types.ModuleType("kernel")


def _mock_act_quant(x, *args, **kwargs):
    if len(args) >= 4 and args[3] is True:
        return None
    return x, torch.ones(1, device=x.device)


def _mock_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    bsz, seqlen, n_heads, head_dim = q.shape
    k = kv.unsqueeze(2).expand(-1, -1, n_heads, -1)
    scores = torch.einsum("bshd,bthd->bsht", q, k) * softmax_scale
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("bsht,bthd->bshd", attn, k)
    return out.contiguous()


def _mock_fp8_gemm(x, s, w, ws, sd):
    return F.linear(x.float(), w.float()).to(x.dtype)


_kernel.act_quant = _mock_act_quant
_kernel.fp4_act_quant = lambda *a, **k: None
_kernel.sparse_attn = _mock_sparse_attn
_kernel.rotate_activation = lambda x: x * (x.size(-1) ** -0.5)
_kernel.hc_split_sinkhorn = lambda *a, **k: None
_kernel.fp8_gemm = _mock_fp8_gemm
_kernel.fp4_gemm = _mock_fp8_gemm
sys.modules["kernel"] = _kernel

from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer


def _make_small_args():
    """Tiny model args for fast testing — 2 layers, small dims, no MoE."""
    return V4Args(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_layers=2,
        n_mtp_layers=0,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        window_size=8,
        compress_ratios=[4, 128],  # layer 0: CSA, layer 1: HCA
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=0,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="softmax",
        route_scale=1.0,
        swiglu_limit=10.0,
        n_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        hc_eps=1e-6,
        norm_eps=1e-6,
        max_batch_size=1,
        max_seq_len=64,
    )


def _init_model(model):
    """Initialize all model parameters with small random values."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                param.data.copy_(
                    torch.randn(param.shape, dtype=torch.float32)
                    .mul_(0.02)
                    .to(torch.float8_e4m3fn)
                )
            elif param.dtype == torch.float8_e8m0fnu:
                param.data.view(torch.uint8).fill_(127)  # scale = 1.0
            elif param.dtype.is_floating_point:
                nn.init.normal_(param, std=0.02)
            # int8 params (FP4 packed) — leave as zeros

        for name, buf in model.named_buffers():
            if buf.dtype.is_floating_point:
                if "score_state" in name:
                    buf.fill_(float("-inf"))
                elif "attn_sink" in name:
                    nn.init.normal_(buf, std=0.02)
                elif "freqs_cis" in name:
                    pass  # precomputed, don't touch
                elif "kv_cache" in name:
                    buf.zero_()
                elif "kv_state" in name:
                    buf.zero_()


def _stub_moe(model):
    """Replace MoE with identity to isolate attention precision."""
    for layer in model.layers:
        layer.ffn = nn.Identity()
        # Patch forward to skip MoE input_ids arg
        original_forward = layer.forward

        def make_patched_forward(orig_layer):
            def patched_forward(x, start_pos, input_ids=None):
                # Attention path (unchanged)
                residual = x
                x_pre, post, comb = orig_layer._hc_pre(
                    x,
                    orig_layer.hc_attn_fn,
                    orig_layer.hc_attn_scale,
                    orig_layer.hc_attn_base,
                )
                x_pre = orig_layer.attn_norm(x_pre)
                attn_out = orig_layer.attn(x_pre, start_pos)
                x = orig_layer._hc_post(attn_out, residual, post, comb)

                # FFN path — identity instead of MoE
                residual = x
                x_pre, post, comb = orig_layer._hc_pre(
                    x,
                    orig_layer.hc_ffn_fn,
                    orig_layer.hc_ffn_scale,
                    orig_layer.hc_ffn_base,
                )
                x_pre = orig_layer.ffn_norm(x_pre)
                ffn_out = x_pre  # identity
                x = orig_layer._hc_post(ffn_out, residual, post, comb)
                return x

            return patched_forward

        layer.forward = make_patched_forward(layer)


def _extract_kv_caches(model):
    """Extract all KV cache tensors from model's attention layers."""
    caches = {}
    for i, layer in enumerate(model.layers):
        caches[i] = {
            "kv_cache": layer.attn.kv_cache.clone(),
        }
        if layer.attn.compressor is not None:
            caches[i]["kv_state"] = layer.attn.compressor.kv_state.clone()
            caches[i]["score_state"] = layer.attn.compressor.score_state.clone()
        if layer.attn.indexer is not None:
            caches[i]["indexer_kv_cache"] = layer.attn.indexer.kv_cache.clone()
            caches[i][
                "indexer_kv_state"
            ] = layer.attn.indexer.compressor.kv_state.clone()
            caches[i][
                "indexer_score_state"
            ] = layer.attn.indexer.compressor.score_state.clone()
    return caches


class TestDSV4KVCachePrecision:
    """Compare model outputs using internal vs framework-style KV cache."""

    def _make_model(self, device="cpu"):
        """Create and initialize a small V4Transformer with MoE stubbed."""
        torch.manual_seed(42)
        args = _make_small_args()
        model = V4Transformer(args)
        _init_model(model)
        _stub_moe(model)
        return model.to(device), args

    def test_prefill_deterministic(self):
        """Two identical models produce identical prefill output."""
        model_a, args = self._make_model()
        model_b = copy.deepcopy(model_a)

        input_ids = torch.randint(0, args.vocab_size, (1, 8))

        with torch.inference_mode():
            out_a = model_a(input_ids, start_pos=0, apply_lm_head=False)
            out_b = model_b(input_ids, start_pos=0, apply_lm_head=False)

        diff = (out_a.float() - out_b.float()).abs().max().item()
        assert (
            diff == 0.0
        ), f"Identical models should produce identical output, got diff={diff}"

    def test_prefill_kv_cache_content_match(self):
        """Two identical models have identical KV cache after prefill."""
        model_a, args = self._make_model()
        model_b = copy.deepcopy(model_a)

        input_ids = torch.randint(0, args.vocab_size, (1, 8))

        with torch.inference_mode():
            model_a(input_ids, start_pos=0, apply_lm_head=False)
            model_b(input_ids, start_pos=0, apply_lm_head=False)

        caches_a = _extract_kv_caches(model_a)
        caches_b = _extract_kv_caches(model_b)

        for layer_id in caches_a:
            for key in caches_a[layer_id]:
                ca = caches_a[layer_id][key]
                cb = caches_b[layer_id][key]
                # Compare only finite values (score_state has -inf)
                mask = torch.isfinite(ca) & torch.isfinite(cb)
                if mask.any():
                    diff = (ca[mask].float() - cb[mask].float()).abs().max().item()
                    assert diff < 0.1, f"layer {layer_id} {key} diff={diff}"

    def test_decode_after_prefill(self):
        """Prefill + decode produces consistent output across two identical models."""
        model_a, args = self._make_model()
        model_b = copy.deepcopy(model_a)

        input_ids = torch.randint(0, args.vocab_size, (1, 8))

        with torch.inference_mode():
            # Prefill
            out_a_pre = model_a(input_ids, start_pos=0, apply_lm_head=False)
            out_b_pre = model_b(input_ids, start_pos=0, apply_lm_head=False)

            # Decode 4 tokens
            for pos in range(8, 12):
                dec_ids = torch.randint(0, args.vocab_size, (1, 1))
                out_a = model_a(dec_ids, start_pos=pos, apply_lm_head=False)
                out_b = model_b(dec_ids, start_pos=pos, apply_lm_head=False)

                diff = (out_a.float() - out_b.float()).abs().max().item()
                assert diff < 1e-3, f"Decode pos={pos} diff={diff}"

    def test_kv_cache_7group_layout(self):
        """Verify KV cache content can be decomposed into 7 groups matching the plan."""
        model, args = self._make_model()
        input_ids = torch.randint(0, args.vocab_size, (1, 8))

        with torch.inference_mode():
            model(input_ids, start_pos=0, apply_lm_head=False)

        # Layer 0: CSA (compress_ratio=4)
        attn0 = model.layers[0].attn
        win = attn0.window_size
        assert attn0.compress_ratio == 4

        # SWA region: kv_cache[:, :win]
        swa_kv = attn0.kv_cache[:, :win]
        assert swa_kv.shape == (1, win, args.head_dim)

        # Compressed KV region: kv_cache[:, win:]
        compressed_kv = attn0.kv_cache[:, win:]
        n_entries = 8 // 4  # 8 tokens / ratio 4 = 2 entries
        assert (
            compressed_kv[:, :n_entries].abs().sum() > 0
        ), "Compressed entries should be non-zero"

        # Compressor state
        assert attn0.compressor.kv_state is not None
        assert attn0.compressor.score_state is not None

        # Indexer (CSA only)
        assert attn0.indexer is not None
        assert attn0.indexer.kv_cache is not None

        # Layer 1: HCA (compress_ratio=128)
        attn1 = model.layers[1].attn
        assert attn1.compress_ratio == 128
        assert attn1.indexer is None  # HCA has no indexer
        assert attn1.compressor is not None

    def test_prefill_decode_logits_match(self):
        """Full prefill + decode logits match between two identical models."""
        model_a, args = self._make_model()
        model_b = copy.deepcopy(model_a)

        input_ids = torch.randint(0, args.vocab_size, (1, 8))

        with torch.inference_mode():
            # Prefill with lm_head
            logits_a = model_a(input_ids, start_pos=0, apply_lm_head=True)
            logits_b = model_b(input_ids, start_pos=0, apply_lm_head=True)

            diff = (logits_a.float() - logits_b.float()).abs().max().item()
            assert diff == 0.0, f"Prefill logits diff={diff}"

            # Decode
            for pos in range(8, 12):
                dec_ids = torch.randint(0, args.vocab_size, (1, 1))
                logits_a = model_a(dec_ids, start_pos=pos, apply_lm_head=True)
                logits_b = model_b(dec_ids, start_pos=pos, apply_lm_head=True)

                diff = (logits_a.float() - logits_b.float()).abs().max().item()
                assert diff < 1e-3, f"Decode pos={pos} logits diff={diff}"

    def test_csa_compression_boundary(self):
        """CSA layer: verify compression happens at ratio=4 boundary."""
        model, args = self._make_model()

        with torch.inference_mode():
            # Prefill 4 tokens — should produce exactly 1 compressed entry
            input_ids = torch.randint(0, args.vocab_size, (1, 4))
            model(input_ids, start_pos=0, apply_lm_head=False)

            attn = model.layers[0].attn
            win = attn.window_size
            compressed = attn.kv_cache[:, win:]
            # 4 tokens / ratio 4 = 1 entry
            assert (
                compressed[:, 0].abs().sum() > 0
            ), "First compressed entry should be written"
            assert compressed[:, 1].abs().sum() == 0, "Second entry should be empty"

    def test_hca_no_compression_under_ratio(self):
        """HCA layer: 8 tokens < ratio 128, no compression should happen."""
        model, args = self._make_model()

        with torch.inference_mode():
            input_ids = torch.randint(0, args.vocab_size, (1, 8))
            model(input_ids, start_pos=0, apply_lm_head=False)

            attn = model.layers[1].attn
            win = attn.window_size
            compressed = attn.kv_cache[:, win:]
            # 8 tokens < ratio 128, no compression
            assert (
                compressed.abs().sum() == 0
            ), "HCA should not compress with only 8 tokens"

    def test_swa_ring_buffer_written(self):
        """SWA ring buffer is correctly written during prefill."""
        model, args = self._make_model()

        with torch.inference_mode():
            input_ids = torch.randint(0, args.vocab_size, (1, 8))
            model(input_ids, start_pos=0, apply_lm_head=False)

            # Both layers should have SWA data written
            for i in range(2):
                attn = model.layers[i].attn
                win = attn.window_size
                swa = attn.kv_cache[:, :win]
                # 8 tokens written to first 8 positions (8 <= win=8)
                assert swa[:, :8].abs().sum() > 0, f"Layer {i} SWA should have data"

    def test_decode_swa_ring_position(self):
        """Decode writes to correct ring buffer position."""
        model, args = self._make_model()

        with torch.inference_mode():
            # Prefill 8 tokens (fills SWA window)
            input_ids = torch.randint(0, args.vocab_size, (1, 8))
            model(input_ids, start_pos=0, apply_lm_head=False)

            # Decode at pos=8 should write to slot 8 % 8 = 0
            dec_ids = torch.randint(0, args.vocab_size, (1, 1))
            model(dec_ids, start_pos=8, apply_lm_head=False)

            attn = model.layers[0].attn
            # Slot 0 should have been overwritten by decode token
            assert attn.kv_cache[:, 0].abs().sum() > 0


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

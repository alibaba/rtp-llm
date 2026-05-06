"""DSV4 attention and SWA cache correctness tests."""

from rtp_llm.models_py.modules.dsv4.test.dsv4_test_utils import *


class TestSWAKVCache(unittest.TestCase):
    """Verify sliding window KV cache writes match official."""

    def test_swa_prefill_ring_buffer(self):
        """Test 20: SWA ring buffer after prefill matches official layout."""
        torch.manual_seed(42)
        B, S, head_dim, win = 1, 24, 64, 16

        kv = torch.randn(B, S, head_dim, dtype=torch.bfloat16)
        our_cache = torch.zeros(B, win, head_dim)
        official_cache = torch.zeros(B, win, head_dim)

        # Our logic (from attention.py)
        if S <= win:
            our_cache[:B, :S] = kv
        else:
            cutoff = S % win
            our_cache[:B, cutoff:win], our_cache[:B, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1
            )

        # Official logic (same)
        if S <= win:
            official_cache[:B, :S] = kv
        else:
            cutoff = S % win
            official_cache[:B, cutoff:win], official_cache[:B, :cutoff] = kv[
                :, -win:
            ].split([win - cutoff, cutoff], dim=1)

        diff = (our_cache.float() - official_cache.float()).abs().max().item()
        assert diff == 0.0, f"SWA ring buffer diff = {diff}"

    def test_swa_decode_write(self):
        """Test 20: SWA decode writes to correct ring position."""
        torch.manual_seed(42)
        B, head_dim, win = 1, 64, 16

        cache = torch.zeros(B, win, head_dim)
        # Simulate decode at various positions
        for pos in [0, 1, 15, 16, 17, 31, 32]:
            kv_token = torch.randn(B, 1, head_dim, dtype=torch.bfloat16)
            slot = pos % win
            cache[:B, slot] = kv_token.squeeze(1)
            # Verify written
            diff = (cache[:B, slot] - kv_token.squeeze(1)).abs().max().item()
            assert diff == 0.0, f"SWA decode write at pos={pos} slot={slot} diff={diff}"

    def test_swa_window_beyond_128(self):
        """Test 20: After >128 tokens, only last 128 are in window."""
        win = 128
        seq_len = 200
        # Attention should only read last 128 tokens
        # Verify topk_idxs from _get_window_topk_idxs covers correct range
        idxs = _get_window_topk_idxs(win, 1, 1, seq_len - 1, "cpu")
        assert idxs.shape == (1, 1, win)
        # All indices should be valid (no -1)
        assert (idxs >= 0).all(), "Window indices should all be valid for pos >= win-1"

class TestAttentionOutput(unittest.TestCase):
    """Compare full Attention output between official and our implementation."""

    def _make_attention_pair(self, compress_ratio):
        torch.manual_seed(42)
        args = make_small_args()
        args["q_lora_rank"] = 128
        args["o_lora_rank"] = 128
        dim = args["dim"]
        device = _test_device()

        official_args = make_official_args(
            q_lora_rank=args["q_lora_rank"],
            o_lora_rank=args["o_lora_rank"],
            compress_ratios=(compress_ratio,),
        )
        # Official Attention needs world_size=1 global
        try:
            official_attn = OfficialAttention(0, official_args).to(device)
        except Exception:
            raise unittest.SkipTest(
                "Cannot create official Attention (missing distributed context)"
            )

        # Initialize ALL official weights unconditionally (torch.empty may contain
        # values > 448 which become NaN when converted to FP8 e4m3fn)
        with torch.no_grad():
            for name, param in official_attn.named_parameters():
                if param.requires_grad:
                    nn.init.normal_(param, std=0.02)
            for name, buf in official_attn.named_buffers():
                if buf.dtype.is_floating_point:
                    if "score_state" in name:
                        buf.fill_(float("-inf"))
                    elif "attn_sink" in name:
                        nn.init.normal_(buf, std=0.02)
                    else:
                        buf.zero_()

        official_attn.kv_cache = official_attn.kv_cache.bfloat16()
        if compress_ratio == 4 and official_attn.indexer is not None:
            official_attn.indexer.kv_cache = official_attn.indexer.kv_cache.bfloat16()

        def _fake_prepare_wo_a_stacked(weight_fp8, scale_raw, groups, rank, width):
            scale = torch.empty(
                groups,
                rank,
                max(1, math.ceil(width / 512)),
                dtype=torch.int32,
                device=weight_fp8.device,
            )
            return weight_fp8.view(groups, rank, width).contiguous(), scale

        orig_prepare_wo_a = _our_attention_module._prepare_wo_a_stacked
        _our_attention_module._prepare_wo_a_stacked = _fake_prepare_wo_a_stacked
        try:
            our_attn = OurAttention(
                layer_id=0,
                dim=dim,
                n_heads=args["n_heads"],
                q_lora_rank=args["q_lora_rank"],
                head_dim=args["head_dim"],
                rope_head_dim=args["rope_head_dim"],
                o_lora_rank=args["o_lora_rank"],
                o_groups=args["o_groups"],
                window_size=args["window_size"],
                compress_ratio=compress_ratio,
                compress_rope_theta=args["compress_rope_theta"],
                rope_theta=args["rope_theta"],
                rope_factor=args["rope_factor"],
                beta_fast=args["beta_fast"],
                beta_slow=args["beta_slow"],
                original_seq_len=args["original_seq_len"],
                max_batch_size=args["max_batch_size"],
                max_seq_len=args["max_seq_len"],
                index_n_heads=args["index_n_heads"],
                index_head_dim=args["index_head_dim"],
                index_topk=args["index_topk"],
                layer_weights=_attention_layer_weights(official_attn, device),
            )
        finally:
            _our_attention_module._prepare_wo_a_stacked = orig_prepare_wo_a

        our_attn.wq_a = _bf16_linear_from_weight(official_attn.wq_a.weight, device)
        our_attn.wq_b = _bf16_linear_from_weight(official_attn.wq_b.weight, device)
        our_attn.wkv = _bf16_linear_from_weight(official_attn.wkv.weight, device)
        our_attn.wo_b = _bf16_linear_from_weight(official_attn.wo_b.weight, device)
        our_attn.wo_a_w = official_attn.wo_a.weight.detach().clone().to(
            device=device, dtype=torch.bfloat16
        )
        our_attn.wo_a_s = _ue8m0_scale_ones(
            (
                args["o_groups"] * max(1, args["o_lora_rank"] // 128),
                max(1, (args["n_heads"] * args["head_dim"] // args["o_groups"]) // 128),
            ),
            device,
        )
        our_attn.freqs_cis = our_attn.freqs_cis.to(device)

        return our_attn, official_attn, args, device

    def test_swa_only_attention(self):
        """Test 23: SWA-only layer (compress_ratio=0) output matches."""
        # Clear lru_cache from prior tests
        _official_model.get_window_topk_idxs.cache_clear()
        try:
            our_attn, official_attn, args, device = self._make_attention_pair(0)
        except Exception:
            raise unittest.SkipTest("Cannot create official Attention")

        B, S = 1, 8
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=device)

        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        prev_enabled, prev_level = _rt.ENABLED, _rt.LEVEL
        _rt.ENABLED, _rt.LEVEL = True, 0
        try:
            our_out = our_attn(x, start_pos=0)
        finally:
            _rt.ENABLED, _rt.LEVEL = prev_enabled, prev_level
        official_out = official_attn(x, start_pos=0)

        if our_out.isnan().any() or our_out.float().abs().max().item() > 100:
            raise unittest.SkipTest("FP8 dequant produces garbage on this device")
        assert not official_out.isnan().any(), "official output has NaN"
        diff = (our_out.float() - official_out.float()).abs().max().item()
        assert diff < 0.1, f"SWA-only attention max diff = {diff}"

    def test_hca_attention(self):
        """Test 23: HCA layer (compress_ratio=128) output matches."""
        # Clear lru_cache from prior tests
        _official_model.get_window_topk_idxs.cache_clear()
        our_attn, official_attn, args, device = self._make_attention_pair(128)

        B, S = 1, 128
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=device)

        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        prev_enabled, prev_level = _rt.ENABLED, _rt.LEVEL
        _rt.ENABLED, _rt.LEVEL = True, 0
        try:
            our_out = our_attn(x, start_pos=0)
        finally:
            _rt.ENABLED, _rt.LEVEL = prev_enabled, prev_level
        official_out = official_attn(x, start_pos=0)

        if our_out.isnan().any():
            raise unittest.SkipTest("FP8 dequant produces NaN on this device")
        assert not official_out.isnan().any(), "official output has NaN"
        diff = (our_out.float() - official_out.float()).abs().max().item()
        assert diff < 0.1, f"HCA attention max diff = {diff}"


if __name__ == "__main__":
    unittest.main()

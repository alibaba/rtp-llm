"""DSV4 compressor and compressed-KV cache correctness tests."""

from rtp_llm.models_py.modules.dsv4.test.dsv4_test_utils import *


class TestCompressorPrecision(unittest.TestCase):
    """Compare Compressor output between official and our implementation."""

    def _make_csa_pair(self):
        """Create a fresh CSA compressor pair."""
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args(compress_ratios=(CSA_COMPRESS_RATIO,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=CSA_COMPRESS_RATIO, head_dim=head_dim
        ).to(device)

        kv_cache_size = _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO)
        our_comp = _make_our_compressor(
            official_comp,
            dim,
            head_dim,
            rope_head_dim,
            CSA_COMPRESS_RATIO,
            2,
            kv_cache_size,
            device,
        )
        our_ctx = _bind_standalone_pools(our_comp, 1, kv_cache_size, device)

        official_comp.kv_cache = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        return our_comp, official_comp, args, our_ctx

    def test_csa_prefill_compression(self):
        """Test 18: CSA prefill - compressed entries match official."""
        our_comp, official_comp, args, our_ctx = self._make_csa_pair()
        B, S = 1, 12
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=our_ctx.device)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is not None
        assert official_result is not None
        assert our_result.shape == official_result.shape
        diff = (our_result.float() - official_result.float()).abs().max().item()
        assert diff < 1e-3, f"CSA prefill max diff = {diff}"

    def test_csa_decode_compression(self):
        """Test 19: CSA decode - state accumulation and compression match."""
        our_comp, official_comp, args, our_ctx = self._make_csa_pair()
        B = 1

        # Prefill 8 tokens first
        torch.manual_seed(100)
        x_prefill = torch.randn(
            B, 8, args["dim"], dtype=torch.bfloat16, device=our_ctx.device
        )
        our_comp(x_prefill, start_pos=0)
        official_comp(x_prefill, start_pos=0)

        # Decode 4 tokens one by one
        for pos in range(8, 12):
            x_decode = torch.randn(
                B, 1, args["dim"], dtype=torch.bfloat16, device=our_ctx.device
            )
            our_result = our_comp(x_decode, start_pos=pos)
            official_result = official_comp(x_decode, start_pos=pos)

            if our_result is not None:
                assert official_result is not None
                diff = (our_result.float() - official_result.float()).abs().max().item()
                assert diff < 1e-3, f"CSA decode pos={pos} max diff = {diff}"
            else:
                assert official_result is None

    def test_csa_state_content(self):
        """Test 19: After prefill 12 tokens, kv_state and score_state match."""
        our_comp, official_comp, args, our_ctx = self._make_csa_pair()
        B = 1
        x = torch.randn(B, 12, args["dim"], dtype=torch.bfloat16, device=our_ctx.device)

        our_comp(x, start_pos=0)
        official_comp(x, start_pos=0)
        our_kv_state, our_score_state = _pool_state(our_ctx, our_comp)

        kv_diff = (
            (our_kv_state[:B].float() - official_comp.kv_state[:B].float())
            .abs()
            .max()
            .item()
        )
        assert kv_diff < COMPRESSOR_STATE_ATOL, f"kv_state max diff = {kv_diff}"

        our_score = our_score_state[:B].float()
        off_score = official_comp.score_state[:B].float()
        finite_mask = torch.isfinite(our_score) & torch.isfinite(off_score)
        if finite_mask.any():
            score_diff = (
                (our_score[finite_mask] - off_score[finite_mask]).abs().max().item()
            )
            assert (
                score_diff < COMPRESSOR_STATE_ATOL
            ), f"score_state max diff = {score_diff}"
        assert (our_score.isinf() == off_score.isinf()).all()

    def _make_hca_pair(self):
        """Create a fresh HCA compressor pair."""
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args(compress_ratios=(HCA_COMPRESS_RATIO,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=HCA_COMPRESS_RATIO, head_dim=head_dim
        ).to(device)

        kv_cache_size = _kv_cache_size(args["max_seq_len"], HCA_COMPRESS_RATIO)
        our_comp = _make_our_compressor(
            official_comp,
            dim,
            head_dim,
            rope_head_dim,
            HCA_COMPRESS_RATIO,
            2,
            kv_cache_size,
            device,
        )
        our_ctx = _bind_standalone_pools(our_comp, 1, kv_cache_size, device)

        official_comp.kv_cache = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        return our_comp, official_comp, args, our_ctx

    def test_hca_prefill_compression(self):
        """Test 18: HCA prefill - compressed entries match official."""
        our_comp, official_comp, args, our_ctx = self._make_hca_pair()
        B, S = 1, 128
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=our_ctx.device)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is not None
        diff = (our_result.float() - official_result.float()).abs().max().item()
        assert diff < 1e-3, f"HCA prefill max diff = {diff}"

    def test_hca_state_content(self):
        """Test 19: HCA state after partial prefill (not enough to compress)."""
        our_comp, official_comp, args, our_ctx = self._make_hca_pair()
        B = 1
        x = torch.randn(B, 64, args["dim"], dtype=torch.bfloat16, device=our_ctx.device)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is None
        assert official_result is None

        our_kv_state, _ = _pool_state(our_ctx, our_comp)
        kv_diff = (
            (our_kv_state[:B].float() - official_comp.kv_state[:B].float())
            .abs()
            .max()
            .item()
        )
        assert kv_diff < COMPRESSOR_STATE_ATOL, f"HCA kv_state max diff = {kv_diff}"

class TestSequenceLengthBoundaries(unittest.TestCase):
    """Test compressor behavior at various sequence length boundaries."""

    def _make_csa_compressor(self):
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        official_args = make_official_args(compress_ratios=(CSA_COMPRESS_RATIO,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=CSA_COMPRESS_RATIO, head_dim=args["head_dim"]
        ).to(device)
        kv_cache_size = _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO)
        comp = _make_our_compressor(
            official_comp,
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
            CSA_COMPRESS_RATIO,
            2,
            kv_cache_size,
            device,
        )
        ctx = _bind_standalone_pools(comp, 1, kv_cache_size, device)
        freqs_cis = precompute_freqs_cis(
            args["rope_head_dim"], args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        comp.freqs_cis = freqs_cis
        return comp, args, ctx

    def test_exact_ratio_multiple(self):
        """Length = compress_ratio multiple (no remainder)."""
        comp, args, ctx = self._make_csa_compressor()
        x = torch.randn(
            1, 8, args["dim"], dtype=torch.bfloat16, device=ctx.device
        )  # 8 = 4*2
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 2  # 8/4 = 2 entries

    def test_with_remainder(self):
        """Length has remainder (state has uncommitted tokens)."""
        comp, args, ctx = self._make_csa_compressor()
        x = torch.randn(
            1, 10, args["dim"], dtype=torch.bfloat16, device=ctx.device
        )  # 10 = 4*2 + 2
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 2  # only 8/4 = 2 entries, 2 tokens in state

        # State should have 2 uncommitted tokens
        # For overlap=True (CSA), state layout: [overlap(4), current(4)]
        # After 10 tokens: overlap has last 4 of committed, current has 2 uncommitted
        kv_state, _ = _pool_state(ctx, comp)
        assert kv_state[0, 4:6].abs().sum() > 0  # positions 4,5 have data
        assert kv_state[0, 6:8].abs().sum() == 0  # positions 6,7 empty

    def test_less_than_ratio(self):
        """Length < compress_ratio (no compression at all)."""
        comp, args, ctx = self._make_csa_compressor()
        x = torch.randn(
            1, 3, args["dim"], dtype=torch.bfloat16, device=ctx.device
        )  # 3 < 4
        result = comp(x, start_pos=0)
        assert result is None  # not enough tokens to compress

    def test_single_token_decode(self):
        """Length = 1 (single token decode)."""
        comp, args, ctx = self._make_csa_compressor()
        # Prefill 4 tokens first
        x = torch.randn(1, 4, args["dim"], dtype=torch.bfloat16, device=ctx.device)
        comp(x, start_pos=0)

        # Decode single tokens
        for pos in range(4, 8):
            x_dec = torch.randn(
                1, 1, args["dim"], dtype=torch.bfloat16, device=ctx.device
            )
            result = comp(x_dec, start_pos=pos)
            if pos == 7:  # (7+1) % 4 == 0 -> should compress
                assert result is not None, f"Expected compression at pos={pos}"
            else:
                assert result is None, f"Unexpected compression at pos={pos}"

    def test_cross_block_boundary(self):
        """Sequence grows from 0 -> 256 -> 512 crossing block boundaries."""
        comp, args, ctx = self._make_csa_compressor()
        # Prefill 256 tokens (= 1 block of 256 tokens, 64 compressed entries)
        x = torch.randn(1, 256, args["dim"], dtype=torch.bfloat16, device=ctx.device)
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 64  # 256/4 = 64 entries

    def test_standalone_pool_tables_follow_sequence_lengths(self):
        """Standalone pool tables are sized from real token lengths per batch."""
        comp, args, ctx = self._make_csa_compressor()
        kv_cache_size = _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO)

        ctx_257 = _bind_standalone_pools(
            comp,
            1,
            kv_cache_size,
            ctx.device,
            sequence_lengths=[257],
        )
        assert ctx_257.kv_block_table.shape == (1, 2)
        assert (ctx_257.kv_block_table[0, :2] > 0).all()
        assert (ctx_257.state_block_table[0, :2] > 0).all()

        ctx_513 = _bind_standalone_pools(
            comp,
            1,
            kv_cache_size,
            ctx.device,
            sequence_lengths=[513],
        )
        assert ctx_513.kv_block_table.shape == (1, 3)
        assert (ctx_513.kv_block_table[0] > 0).all()
        assert ctx_513.state_block_table[0, 0].item() == -1
        assert (ctx_513.state_block_table[0, 1:] > 0).all()
        assert ctx_513.state_pool.shape[0] == 3 * ctx_513.state_eb

        ctx_mixed = _bind_standalone_pools(
            comp,
            2,
            kv_cache_size,
            ctx.device,
            sequence_lengths=[128, 513],
        )
        assert ctx_mixed.kv_block_table.shape == (2, 3)
        assert (ctx_mixed.kv_block_table[0, :1] > 0).all()
        assert (ctx_mixed.kv_block_table[0, 1:] == -1).all()
        assert (ctx_mixed.kv_block_table[1] > 0).all()
        assert (ctx_mixed.state_block_table[0, :1] > 0).all()
        assert (ctx_mixed.state_block_table[0, 1:] == -1).all()
        assert ctx_mixed.state_block_table[1, 0].item() == -1
        assert (ctx_mixed.state_block_table[1, 1:] > 0).all()

class TestCompressedKVCache(unittest.TestCase):
    """Verify compressed KV entries written to cache match official."""

    def test_kv_cache_content_after_prefill(self):
        """Test 21: After prefill, compressed entries in kv_cache match."""
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args()
        official_comp = OfficialCompressor(
            official_args, compress_ratio=CSA_COMPRESS_RATIO, head_dim=head_dim
        ).to(device)

        kv_cache_size = _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO)
        our_comp = _make_our_compressor(
            official_comp,
            dim,
            head_dim,
            rope_head_dim,
            CSA_COMPRESS_RATIO,
            2,
            kv_cache_size,
            device,
        )
        our_ctx = _bind_standalone_pools(our_comp, 1, kv_cache_size, device)

        official_kv = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        official_comp.kv_cache = official_kv

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        B, S = 1, 20  # 20 tokens -> 5 compressed entries
        x = torch.randn(B, S, dim, dtype=torch.bfloat16, device=device)

        our_comp(x, start_pos=0)
        official_comp(x, start_pos=0)

        # Compare kv_cache content (first 5 entries should be written)
        n_entries = S // 4
        our_entries = _pool_kv_cache(our_ctx, our_comp)[0, :n_entries]
        official_entries = official_kv[0, :n_entries]

        diff = (our_entries.float() - official_entries.float()).abs().max().item()
        assert diff < 1e-3, f"KV cache entries max diff = {diff}"

    def test_kv_cache_content_after_decode(self):
        """Test 21: After decode, new compressed entry matches."""
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args()
        official_comp = OfficialCompressor(
            official_args, compress_ratio=CSA_COMPRESS_RATIO, head_dim=head_dim
        ).to(device)

        kv_cache_size = _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO)
        our_comp = _make_our_compressor(
            official_comp,
            dim,
            head_dim,
            rope_head_dim,
            CSA_COMPRESS_RATIO,
            2,
            kv_cache_size,
            device,
        )
        our_ctx = _bind_standalone_pools(our_comp, 1, kv_cache_size, device)

        official_kv = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        official_comp.kv_cache = official_kv

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        # Prefill 8 tokens
        B = 1
        x_prefill = torch.randn(B, 8, dim, dtype=torch.bfloat16, device=device)
        our_comp(x_prefill, start_pos=0)
        official_comp(x_prefill, start_pos=0)

        # Decode tokens 8-11 (should produce entry at pos 11)
        for pos in range(8, 12):
            x_dec = torch.randn(B, 1, dim, dtype=torch.bfloat16, device=device)
            our_comp(x_dec, start_pos=pos)
            official_comp(x_dec, start_pos=pos)

        # Compare all written entries (8+4=12 tokens -> 3 entries)
        n_entries = 3
        diff = (
            (
                _pool_kv_cache(our_ctx, our_comp)[0, :n_entries].float()
                - official_kv[0, :n_entries].float()
            )
            .abs()
            .max()
            .item()
        )
        assert diff < 1e-3, f"KV cache after decode max diff = {diff}"


if __name__ == "__main__":
    unittest.main()

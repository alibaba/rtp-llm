"""DSV4 Indexer top-k correctness tests."""

from rtp_llm.models_py.modules.dsv4.test.dsv4_test_utils import *


class TestIndexerTopk(unittest.TestCase):
    """Compare Indexer top-k selection between official and our implementation."""

    def _make_indexer_pair(self):
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        args["q_lora_rank"] = 128
        dim = args["dim"]
        rope_head_dim = args["rope_head_dim"]
        index_n_heads = args["index_n_heads"]
        index_head_dim = args["index_head_dim"]
        index_topk = args["index_topk"]
        q_lora_rank = args["q_lora_rank"]

        official_args = make_official_args(q_lora_rank=q_lora_rank)
        official_idx = OfficialIndexer(official_args)
        _init_official_indexer_weights(official_idx)

        our_idx = OurIndexer(
            dim=dim,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            compress_ratio=CSA_COMPRESS_RATIO,
            max_batch_size=2,
            max_seq_len=256,
            norm_eps=1e-6,
            layer_weights=_indexer_layer_weights(official_idx, device),
        )
        # This unit test validates Indexer math, not DeepGEMM FP8 kernels.
        # Use the same BF16 linear path as the official test module to avoid
        # SM100 TMA scale-layout requirements in the tiny synthetic setup.
        our_idx.wq_b = _bf16_linear_from_weight(official_idx.wq_b.weight, device)

        # Make kv_cache bfloat16 to match q dtype from wq_b output
        official_idx.kv_cache = official_idx.kv_cache.bfloat16()

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, 256, 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_idx.freqs_cis = freqs_cis
        official_idx.freqs_cis = freqs_cis.cpu()
        _bind_standalone_indexer_pools(
            our_idx,
            1,
            _kv_cache_size(args["max_seq_len"], CSA_COMPRESS_RATIO),
            device,
        )

        return our_idx, official_idx, args

    def test_indexer_prefill_topk(self):
        """Test 22: Indexer top-k indices match official during prefill."""
        our_idx, official_idx, args = self._make_indexer_pair()
        B, S = 1, 12
        x = torch.randn(
            B, S, args["dim"], dtype=torch.bfloat16, device=our_idx.freqs_cis.device
        )
        qr = torch.randn(
            B,
            S,
            args["q_lora_rank"],
            dtype=torch.bfloat16,
            device=our_idx.freqs_cis.device,
        )
        offset = S

        our_topk = our_idx(x, qr, start_pos=0, offset=offset)
        official_topk = official_idx(
            x.cpu(), qr.cpu(), start_pos=0, offset=offset
        ).to(our_topk.device)

        assert our_topk.shape == official_topk.shape
        match_rate = (our_topk == official_topk).float().mean().item()
        assert match_rate > 0.7, f"Indexer top-k match rate = {match_rate}"

    def test_indexer_decode_topk(self):
        """Test 22: Indexer top-k indices match official during decode."""
        our_idx, official_idx, args = self._make_indexer_pair()
        B = 1
        win = args["window_size"]

        torch.manual_seed(100)
        device = our_idx.freqs_cis.device
        x_pre = torch.randn(B, 8, args["dim"], dtype=torch.bfloat16, device=device)
        qr_pre = torch.randn(
            B, 8, args["q_lora_rank"], dtype=torch.bfloat16, device=device
        )
        our_idx(x_pre, qr_pre, start_pos=0, offset=8)
        official_idx(x_pre.cpu(), qr_pre.cpu(), start_pos=0, offset=8)

        x_dec = torch.randn(B, 1, args["dim"], dtype=torch.bfloat16, device=device)
        qr_dec = torch.randn(
            B, 1, args["q_lora_rank"], dtype=torch.bfloat16, device=device
        )
        our_topk = our_idx(x_dec, qr_dec, start_pos=8, offset=win)
        official_topk = official_idx(
            x_dec.cpu(), qr_dec.cpu(), start_pos=8, offset=win
        ).to(our_topk.device)

        assert our_topk.shape == official_topk.shape
        match_rate = (our_topk == official_topk).float().mean().item()
        assert match_rate > 0.95, f"Indexer decode top-k match rate = {match_rate}"


if __name__ == "__main__":
    unittest.main()

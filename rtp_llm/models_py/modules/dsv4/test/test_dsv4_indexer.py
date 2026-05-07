"""DSV4 Indexer top-k correctness tests."""

from rtp_llm.models_py.modules.dsv4.indexer import Indexer as OurIndexer
from rtp_llm.models_py.modules.dsv4.test.dsv4_test_utils import *


def _init_official_indexer_weights(official: OfficialIndexer):
    with torch.no_grad():
        nn.init.normal_(official.wq_b.weight, std=0.02)
        nn.init.normal_(official.weights_proj.weight, std=0.02)
    _init_official_compressor_weights(official.compressor)


def _bind_standalone_indexer_pools(
    indexer: OurIndexer,
    bsz: int,
    kv_cache_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    sequence_lengths=None,
):
    comp = indexer.compressor
    if KV_TOKENS_PER_BLOCK % comp.compress_ratio != 0:
        raise ValueError(
            f"KV_TOKENS_PER_BLOCK={KV_TOKENS_PER_BLOCK} must be divisible by "
            f"compress_ratio={comp.compress_ratio}"
        )
    if sequence_lengths is None:
        seq_lens = [int(kv_cache_size) * int(comp.compress_ratio)] * bsz
    elif isinstance(sequence_lengths, torch.Tensor):
        seq_lens = [int(v) for v in sequence_lengths.detach().cpu().reshape(-1)]
    elif isinstance(sequence_lengths, int):
        seq_lens = [int(sequence_lengths)] * bsz
    else:
        seq_lens = [int(v) for v in sequence_lengths]
    if len(seq_lens) == 1 and bsz > 1:
        seq_lens = seq_lens * bsz
    if len(seq_lens) != bsz:
        raise ValueError(f"sequence_lengths must have 1 or {bsz} values")

    kv_eb = KV_TOKENS_PER_BLOCK // comp.compress_ratio
    blocks_per_batch = [
        max(1, int(math.ceil(max(seq_len, 0) / KV_TOKENS_PER_BLOCK)))
        for seq_len in seq_lens
    ]
    max_blocks = max(blocks_per_batch)
    kv_block_table = torch.full((bsz, max_blocks), -1, device=device, dtype=torch.long)
    next_kv_block_id = 1
    for b, block_count in enumerate(blocks_per_batch):
        block_ids = torch.arange(
            next_kv_block_id,
            next_kv_block_id + block_count,
            device=device,
            dtype=torch.long,
        )
        kv_block_table[b, :block_count] = block_ids
        next_kv_block_id += block_count
    kv_pool = torch.zeros(
        next_kv_block_id * kv_eb,
        indexer.head_dim,
        dtype=dtype,
        device=device,
    )

    state_eb = comp._state_rows
    state_block_table = torch.full(
        (bsz, max_blocks), -1, device=device, dtype=torch.long
    )
    next_state_block_id = 1
    for b, block_count in enumerate(blocks_per_batch):
        first_tail_block = max(0, block_count - 2)
        tail_count = block_count - first_tail_block
        block_ids = torch.arange(
            next_state_block_id,
            next_state_block_id + tail_count,
            device=device,
            dtype=torch.long,
        )
        state_block_table[b, first_tail_block:block_count] = block_ids
        next_state_block_id += tail_count
    state_pool = torch.zeros(
        next_state_block_id * state_eb,
        2 * comp._state_dim,
        dtype=torch.float32,
        device=device,
    )
    indexer.set_pool_context(
        kv_pool, kv_block_table, kv_eb, state_pool, state_block_table, state_eb
    )
    return types.SimpleNamespace(
        kv_pool=kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        state_pool=state_pool,
        state_block_table=state_block_table,
        state_eb=state_eb,
        bsz=bsz,
        device=device,
    )


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
        official_topk = official_idx(x.cpu(), qr.cpu(), start_pos=0, offset=offset).to(
            our_topk.device
        )

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

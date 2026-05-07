"""DSV4 compressor and compressed-KV cache correctness tests."""

from rtp_llm.models_py.modules.dsv4.compressor import Compressor as OurCompressor
from rtp_llm.models_py.modules.dsv4.test.dsv4_test_utils import *

COMPRESSOR_STATE_ATOL = 5e-3
HCA_COMPRESS_RATIO = 128


def _our_compressor_weights(official: OfficialCompressor, device: torch.device):
    """Build the production-style weight dict required by OurCompressor."""
    return {
        "ape": official.ape.detach().clone().to(device=device),
        "wkv": official.wkv.weight.detach().clone().to(device=device),
        "wgate": official.wgate.weight.detach().clone().to(device=device),
        # rtp_llm_ops.rmsnorm expects a bf16 weight tensor.
        "norm": official.norm.weight.detach()
        .clone()
        .to(device=device, dtype=torch.bfloat16),
    }


def _make_our_compressor(
    official: OfficialCompressor,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    max_batch_size: int,
    kv_cache_size: int,
    device: torch.device,
):
    _init_official_compressor_weights(official)
    comp = OurCompressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        max_batch_size=max_batch_size,
        norm_eps=1e-6,
        compressor_weights=_our_compressor_weights(official, device),
    )
    comp.configure_kv_cache_shape(kv_cache_size)
    return comp


def _create_standalone_pools(
    comp: OurCompressor,
    max_batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Create standalone KV/state pools from max capacity only.

    KV cache pool uses a fixed token-based block size: ``KV_TOKENS_PER_BLOCK``
    original tokens per block. This maps to
    ``kv_eb = KV_TOKENS_PER_BLOCK // compress_ratio`` compressed entries per block.
    Pool sizes are derived from ``max_seq_len`` and ``max_batch_size`` so
    actual test sequence lengths only affect block-table layout.
    """
    if KV_TOKENS_PER_BLOCK % comp.compress_ratio != 0:
        raise ValueError(
            f"KV_TOKENS_PER_BLOCK={KV_TOKENS_PER_BLOCK} must be divisible by "
            f"compress_ratio={comp.compress_ratio}"
        )
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if max_seq_len < 0:
        raise ValueError("max_seq_len must be non-negative")

    max_kv_blocks_per_batch = max(1, int(math.ceil(max_seq_len / KV_TOKENS_PER_BLOCK)))
    kv_eb = KV_TOKENS_PER_BLOCK // comp.compress_ratio
    kv_pool = torch.zeros(
        (1 + max_batch_size * max_kv_blocks_per_batch) * kv_eb,
        comp.head_dim,
        dtype=dtype,
        device=device,
    )

    state_eb = comp._state_rows
    max_state_blocks_per_batch = min(2, max_kv_blocks_per_batch)
    state_pool = torch.zeros(
        (1 + max_batch_size * max_state_blocks_per_batch) * state_eb,
        2 * comp._state_dim,
        dtype=torch.float32,
        device=device,
    )

    return types.SimpleNamespace(
        kv_pool=kv_pool,
        kv_eb=kv_eb,
        max_kv_blocks_per_batch=max_kv_blocks_per_batch,
        state_pool=state_pool,
        state_eb=state_eb,
        max_state_blocks_per_batch=max_state_blocks_per_batch,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=device,
    )


def _bind_standalone_pools(
    comp: OurCompressor,
    pools,
    bsz: int,
    *,
    sequence_lengths,
):
    """Create actual sequence block tables and bind pre-created pools."""
    if sequence_lengths is None:
        raise ValueError("sequence_lengths must be provided")
    if bsz <= 0:
        raise ValueError("bsz must be positive")
    if bsz > pools.max_batch_size:
        raise ValueError(f"bsz={bsz} exceeds max_batch_size={pools.max_batch_size}")
    if isinstance(sequence_lengths, torch.Tensor):
        seq_tensor = sequence_lengths.detach().cpu().reshape(-1)
        seq_lens = [int(v) for v in seq_tensor.tolist()]
    elif isinstance(sequence_lengths, int):
        seq_lens = [int(sequence_lengths)] * bsz
    else:
        seq_lens = [int(v) for v in sequence_lengths]
    if len(seq_lens) == 1 and bsz > 1:
        seq_lens = seq_lens * bsz
    if len(seq_lens) != bsz:
        raise ValueError(f"sequence_lengths must have 1 or {bsz} values")
    max_actual_seq_len = max(seq_lens) if seq_lens else 0
    if max_actual_seq_len > pools.max_seq_len:
        raise ValueError(
            f"sequence_lengths max {max_actual_seq_len} exceeds "
            f"max_seq_len={pools.max_seq_len}"
        )

    n_kv_blocks_per_batch = [
        max(1, int(math.ceil(max(seq_len, 0) / KV_TOKENS_PER_BLOCK)))
        for seq_len in seq_lens
    ]
    n_kv_blocks = max(n_kv_blocks_per_batch)
    kv_block_table = torch.full(
        (bsz, n_kv_blocks), -1, device=pools.device, dtype=torch.long
    )
    for b, block_count in enumerate(n_kv_blocks_per_batch):
        first_block_id = 1 + b * pools.max_kv_blocks_per_batch
        block_ids = torch.arange(
            first_block_id,
            first_block_id + block_count,
            device=pools.device,
            dtype=torch.long,
        )
        kv_block_table[b, :block_count] = block_ids

    state_block_table = torch.full(
        (bsz, n_kv_blocks), -1, device=pools.device, dtype=torch.long
    )
    for b, block_count in enumerate(n_kv_blocks_per_batch):
        first_tail_block = max(0, block_count - 2)
        tail_count = block_count - first_tail_block
        first_block_id = 1 + b * pools.max_state_blocks_per_batch
        block_ids = torch.arange(
            first_block_id,
            first_block_id + tail_count,
            device=pools.device,
            dtype=torch.long,
        )
        state_block_table[b, first_tail_block:block_count] = block_ids
    comp.set_pool_context(
        pools.kv_pool,
        kv_block_table,
        pools.kv_eb,
        pools.state_pool,
        state_block_table,
        pools.state_eb,
    )
    return types.SimpleNamespace(
        kv_pool=pools.kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=pools.kv_eb,
        n_kv_blocks=n_kv_blocks,
        n_kv_blocks_per_batch=n_kv_blocks_per_batch,
        max_kv_blocks_per_batch=pools.max_kv_blocks_per_batch,
        state_pool=pools.state_pool,
        state_block_table=state_block_table,
        state_eb=pools.state_eb,
        max_state_blocks_per_batch=pools.max_state_blocks_per_batch,
        max_batch_size=pools.max_batch_size,
        max_seq_len=pools.max_seq_len,
        sequence_lengths=seq_lens,
        bsz=bsz,
        device=pools.device,
    )


def _pool_kv_cache(ctx, comp: OurCompressor, batch: int = None):
    if batch is None:
        return torch.stack(
            [_pool_kv_cache(ctx, comp, b) for b in range(ctx.bsz)], dim=0
        )
    block_ids = ctx.kv_block_table[batch]
    blocks = []
    for bid in block_ids.tolist():
        if bid <= 0:
            continue
        start = bid * ctx.kv_eb
        blocks.append(ctx.kv_pool[start : start + ctx.kv_eb])
    if not blocks:
        return torch.zeros(
            comp._kv_cache_t,
            comp.head_dim,
            dtype=ctx.kv_pool.dtype,
            device=ctx.kv_pool.device,
        )
    dense = torch.cat(blocks, dim=0)
    if dense.shape[0] < comp._kv_cache_t:
        pad = torch.zeros(
            comp._kv_cache_t - dense.shape[0],
            comp.head_dim,
            dtype=dense.dtype,
            device=dense.device,
        )
        dense = torch.cat([dense, pad], dim=0)
    return dense[: comp._kv_cache_t]


def _pool_state(ctx, comp: OurCompressor, batch: int = None):
    if batch is None:
        states = [_pool_state(ctx, comp, b) for b in range(ctx.bsz)]
        kv_states = torch.stack([s[0] for s in states], dim=0)
        score_states = torch.stack([s[1] for s in states], dim=0)
        return kv_states, score_states
    valid_block_ids = ctx.state_block_table[batch][ctx.state_block_table[batch] > 0]
    if valid_block_ids.numel() == 0:
        rows = torch.zeros(
            comp._state_rows,
            2 * comp._state_dim,
            dtype=ctx.state_pool.dtype,
            device=ctx.state_pool.device,
        )
        rows[:, comp._state_dim :] = float("-inf")
        return rows[:, : comp._state_dim], rows[:, comp._state_dim :]
    start = int(valid_block_ids[-1].item()) * ctx.state_eb
    rows = ctx.state_pool[start : start + comp._state_rows]
    return rows[:, : comp._state_dim], rows[:, comp._state_dim :]


class TestCompressorPrecision(unittest.TestCase):
    """Compare Compressor output between official and our implementation."""

    def _make_compressor_pair(self, compress_ratio: int):
        """Create matched official/RTP compressor instances for one ratio."""
        torch.manual_seed(42)
        device = _test_device()
        args = make_small_args()
        dim = args["dim"]
        head_dim = args["head_dim"]
        rope_head_dim = args["rope_head_dim"]

        official_args = make_official_args(compress_ratios=(compress_ratio,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=compress_ratio, head_dim=head_dim
        ).to(device)

        kv_cache_size = _kv_cache_size(args["max_seq_len"], compress_ratio)
        our_comp = _make_our_compressor(
            official_comp,
            dim,
            head_dim,
            rope_head_dim,
            compress_ratio,
            args["max_batch_size"],
            kv_cache_size,
            device,
        )
        pool_ctx = _create_standalone_pools(
            our_comp,
            args["max_batch_size"],
            args["max_seq_len"],
            device,
        )

        official_comp.kv_cache = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        return our_comp, official_comp, args, pool_ctx

    def _assert_prefill_compression_matches(
        self, compress_ratio: int, seqlen: int, label: str
    ):
        our_comp, official_comp, args, pool_ctx = self._make_compressor_pair(
            compress_ratio
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[seqlen],
        )
        x = torch.randn(
            1, seqlen, args["dim"], dtype=torch.bfloat16, device=our_ctx.device
        )

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is not None
        assert official_result is not None
        assert our_result.shape == official_result.shape
        diff = (our_result.float() - official_result.float()).abs().max().item()
        assert diff < 1e-3, f"{label} prefill max diff = {diff}"

    def _assert_decode_compression_matches(
        self,
        compress_ratio: int,
        prefill_len: int,
        decode_end: int,
        label: str,
    ):
        our_comp, official_comp, args, pool_ctx = self._make_compressor_pair(
            compress_ratio
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[decode_end],
        )
        torch.manual_seed(100)
        x_prefill = torch.randn(
            1,
            prefill_len,
            args["dim"],
            dtype=torch.bfloat16,
            device=our_ctx.device,
        )
        our_comp(x_prefill, start_pos=0)
        official_comp(x_prefill, start_pos=0)

        saw_compression = False
        for pos in range(prefill_len, decode_end):
            x_decode = torch.randn(
                1, 1, args["dim"], dtype=torch.bfloat16, device=our_ctx.device
            )
            our_result = our_comp(x_decode, start_pos=pos)
            official_result = official_comp(x_decode, start_pos=pos)

            if official_result is None:
                assert our_result is None, f"Unexpected {label} decode at pos={pos}"
                continue
            assert our_result is not None, f"Missing {label} decode at pos={pos}"
            diff = (our_result.float() - official_result.float()).abs().max().item()
            assert diff < 1e-3, f"{label} decode pos={pos} max diff = {diff}"
            saw_compression = True
        assert saw_compression, f"{label} decode never reached a compression boundary"

    def test_csa_prefill_compression(self):
        """Test 18: CSA prefill - compressed entries match official."""
        self._assert_prefill_compression_matches(CSA_COMPRESS_RATIO, 12, "CSA")

    def test_csa_decode_compression(self):
        """Test 19: CSA decode - state accumulation and compression match."""
        self._assert_decode_compression_matches(CSA_COMPRESS_RATIO, 8, 12, "CSA")

    def test_csa_state_content(self):
        """Test 19: After prefill 12 tokens, kv_state and score_state match."""
        our_comp, official_comp, args, pool_ctx = self._make_compressor_pair(
            CSA_COMPRESS_RATIO
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[12],
        )
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

    def test_hca_prefill_compression(self):
        """Test 18: HCA prefill - compressed entries match official."""
        self._assert_prefill_compression_matches(HCA_COMPRESS_RATIO, 128, "HCA")

    def test_hca_decode_compression(self):
        """Test 19: HCA decode reaches the ratio-128 compression boundary."""
        self._assert_decode_compression_matches(HCA_COMPRESS_RATIO, 64, 128, "HCA")

    def test_hca_state_content(self):
        """Test 19: HCA state after partial prefill (not enough to compress)."""
        our_comp, official_comp, args, pool_ctx = self._make_compressor_pair(
            HCA_COMPRESS_RATIO
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[64],
        )
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
            args["max_batch_size"],
            kv_cache_size,
            device,
        )
        pool_ctx = _create_standalone_pools(
            comp,
            args["max_batch_size"],
            args["max_seq_len"],
            device,
        )
        freqs_cis = precompute_freqs_cis(
            args["rope_head_dim"], args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        comp.freqs_cis = freqs_cis
        return comp, args, pool_ctx

    def test_exact_ratio_multiple(self):
        """Length = compress_ratio multiple (no remainder)."""
        comp, args, pool_ctx = self._make_csa_compressor()
        ctx = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[8],
        )
        x = torch.randn(
            1, 8, args["dim"], dtype=torch.bfloat16, device=ctx.device
        )  # 8 = 4*2
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 2  # 8/4 = 2 entries

    def test_with_remainder(self):
        """Length has remainder (state has uncommitted tokens)."""
        comp, args, pool_ctx = self._make_csa_compressor()
        ctx = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[10],
        )
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
        comp, args, pool_ctx = self._make_csa_compressor()
        ctx = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[3],
        )
        x = torch.randn(
            1, 3, args["dim"], dtype=torch.bfloat16, device=ctx.device
        )  # 3 < 4
        result = comp(x, start_pos=0)
        assert result is None  # not enough tokens to compress

    def test_single_token_decode(self):
        """Length = 1 (single token decode)."""
        comp, args, pool_ctx = self._make_csa_compressor()
        ctx = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[8],
        )
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
        comp, args, pool_ctx = self._make_csa_compressor()
        ctx = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[256],
        )
        # Prefill 256 tokens (= 1 block of 256 tokens, 64 compressed entries)
        x = torch.randn(1, 256, args["dim"], dtype=torch.bfloat16, device=ctx.device)
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 64  # 256/4 = 64 entries

    def test_standalone_pool_tables_follow_sequence_lengths(self):
        """Standalone pool tables are sized from real token lengths per batch."""
        comp, args, pool_ctx = self._make_csa_compressor()
        pool_ctx = _create_standalone_pools(
            comp,
            args["max_batch_size"],
            768,
            pool_ctx.device,
        )
        assert (
            pool_ctx.kv_pool.shape[0]
            == (1 + args["max_batch_size"] * 3) * pool_ctx.kv_eb
        )
        assert (
            pool_ctx.state_pool.shape[0]
            == (1 + args["max_batch_size"] * 2) * pool_ctx.state_eb
        )

        ctx_257 = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[257],
        )
        assert ctx_257.kv_block_table.shape == (1, 2)
        assert (ctx_257.kv_block_table[0, :2] > 0).all()
        assert (ctx_257.state_block_table[0, :2] > 0).all()

        ctx_513 = _bind_standalone_pools(
            comp,
            pool_ctx,
            1,
            sequence_lengths=[513],
        )
        assert ctx_513.kv_block_table.shape == (1, 3)
        assert (ctx_513.kv_block_table[0] > 0).all()
        assert ctx_513.state_block_table[0, 0].item() == -1
        assert (ctx_513.state_block_table[0, 1:] > 0).all()
        assert ctx_513.state_pool.shape[0] == pool_ctx.state_pool.shape[0]

        ctx_mixed = _bind_standalone_pools(
            comp,
            pool_ctx,
            2,
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

    def test_standalone_pool_requires_sequence_lengths(self):
        """Standalone pool binding must not infer sequence lengths implicitly."""
        comp, args, pool_ctx = self._make_csa_compressor()

        with self.assertRaisesRegex(ValueError, "sequence_lengths must be provided"):
            _bind_standalone_pools(
                comp,
                pool_ctx,
                1,
                sequence_lengths=None,
            )


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
            args["max_batch_size"],
            kv_cache_size,
            device,
        )
        B, S = 1, 20  # 20 tokens -> 5 compressed entries
        pool_ctx = _create_standalone_pools(
            our_comp,
            args["max_batch_size"],
            args["max_seq_len"],
            device,
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[S],
        )

        official_kv = torch.zeros(
            2, kv_cache_size, head_dim, dtype=torch.bfloat16, device=device
        )
        official_comp.kv_cache = official_kv

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        ).to(device)
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

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
            args["max_batch_size"],
            kv_cache_size,
            device,
        )
        pool_ctx = _create_standalone_pools(
            our_comp,
            args["max_batch_size"],
            args["max_seq_len"],
            device,
        )
        our_ctx = _bind_standalone_pools(
            our_comp,
            pool_ctx,
            1,
            sequence_lengths=[12],
        )

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

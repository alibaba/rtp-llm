"""CPU-only tests for the ENP8 host-sharded Engram lookup (ppu/engram.py).

Covers the ENP8 sharding math, the rtp-llm hash reference vs a scalar
oracle, FP8/E8M0 dequantization, the LocalEnpmGroup round protocol
(owner-only reads, -1 rows, query order, multi-round padding tails,
cross-rank delivery), the gated inject oracle and a full mini-pipeline on
a synthetic 2-rank table.
"""

import math
import unittest
import warnings

import torch

from rtp_llm.models_py.modules.dsv41.ppu import engram


def _layout(num_embeddings, enp):
    return engram.EngramShardLayout.from_table(num_embeddings, enp)


def _params(num_embeddings=300):
    return engram.EngramHashParams.build(
        layer_ids=(7,),
        max_ngram_size=4,
        n_heads=2,
        vocab_size=13,
        compressed_vocab_size=32,
        num_embeddings=num_embeddings,
        token_map=list(range(32)),
        pad_token_id=2,
    )


def _make_table(num_embeddings, dim=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    payload = torch.randn(num_embeddings, dim, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.randint(
        120, 136, (num_embeddings, dim // 32), generator=gen, dtype=torch.int64
    ).to(torch.uint8)
    return payload, scale


def _shards(payload, scale, layout):
    return [
        engram.HostShard.from_tensors(payload, scale, layout, rank)
        for rank in range(layout.enp_size)
    ]


def _reference_rows(full_payload, full_scale, ids):
    """Independent dequant reference straight from the full synthetic table."""
    ids = ids.detach().reshape(-1).to(torch.int64)
    out = torch.zeros(ids.numel(), full_payload.shape[1], dtype=torch.bfloat16)
    valid = (ids >= 0) & (ids < full_payload.shape[0])
    if bool(valid.any()):
        rows = full_payload.view(torch.uint8)[ids[valid]].view(torch.float8_e4m3fn)
        out[valid] = engram.dequant_rows(rows, full_scale[ids[valid]])
    return out


class ShardingMathTest(unittest.TestCase):
    def test_enp8_lcm_alignment_and_split(self):
        layout = _layout(1000, 8)
        self.assertEqual(layout.alignment, 256)  # lcm(256, 8)
        self.assertEqual(layout.padded_rows, 1024)
        self.assertEqual(layout.num_per_part, 128)
        self.assertEqual(layout.row_start(3), 384)
        self.assertEqual(layout.valid_rows(0), 128)
        self.assertEqual(layout.valid_rows(7), 104)  # min(128, 1000 - 896)
        self.assertEqual(layout.valid_rows(8), 0)
        self.assertEqual(layout.owner(999), 7)
        self.assertEqual(layout.owner(100), 0)
        total = sum(layout.valid_rows(r) for r in range(8))
        self.assertEqual(total, 1000)

    def test_enp8_exact_multiple(self):
        layout = _layout(2048, 8)
        self.assertEqual(layout.padded_rows, 2048)
        self.assertEqual(layout.num_per_part, 256)
        self.assertTrue(all(layout.valid_rows(r) == 256 for r in range(8)))
        self.assertTrue(all(layout.row_start(r) == 256 * r for r in range(8)))

    def test_enp3_alignment_is_lcm_768(self):
        layout = _layout(600, 3)
        self.assertEqual(layout.alignment, 768)
        self.assertEqual(layout.padded_rows, 768)
        self.assertEqual(layout.num_per_part, 256)
        self.assertEqual([layout.valid_rows(r) for r in range(3)], [256, 256, 88])
        self.assertEqual(layout.owner(300), 1)

    def test_enp1_single_owner(self):
        layout = _layout(100, 1)
        self.assertEqual(layout.alignment, 256)
        self.assertEqual(layout.padded_rows, 256)
        self.assertEqual(layout.num_per_part, 256)
        self.assertEqual(layout.valid_rows(0), 100)
        self.assertEqual(layout.owner(42), 0)

    def test_valid_flags(self):
        layout = _layout(1000, 8)
        self.assertTrue(layout.row_is_global_valid(0))
        self.assertTrue(layout.row_is_global_valid(999))
        self.assertFalse(layout.row_is_global_valid(1000))
        self.assertFalse(layout.row_is_global_valid(1023))  # padding row
        self.assertFalse(layout.row_is_global_valid(-1))
        # padding rows are owned by construction but load as zeros
        self.assertEqual(layout.owner(1023), 7)
        self.assertFalse(layout.owns(7, 1023))

    def test_real_table_bytes_estimate(self):
        # 32M rows, dim 6144, ENP8 -> ~23.6 GiB of host RAM per rank.
        gib = engram.shard_bytes_estimate(4_000_000, 6144) / 2**30
        self.assertAlmostEqual(gib, 23.6, delta=0.05)


class HostShardTest(unittest.TestCase):
    def setUp(self):
        self.layout = _layout(300, 2)  # rank0: [0,256), rank1: [256,300)
        self.payload, self.scale = _make_table(300, dim=64, seed=3)

    def test_from_tensors_slices_only_owned_rows(self):
        shard0, shard1 = _shards(self.payload, self.scale, self.layout)
        self.assertEqual(tuple(shard0.payload.shape), (256, 64))
        self.assertEqual(tuple(shard0.scale.shape), (256, 2))
        self.assertEqual(tuple(shard1.payload.shape), (44, 64))
        self.assertEqual(tuple(shard1.scale.shape), (44, 2))
        self.assertEqual(shard0.dim, 64)
        # distinct storage: the shard never pins the caller's full table
        self.assertNotEqual(shard0.payload.data_ptr(), self.payload.data_ptr())
        torch.testing.assert_close(
            shard0.payload.view(torch.uint8),
            self.payload.view(torch.uint8)[0:256],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            shard1.payload.view(torch.uint8),
            self.payload.view(torch.uint8)[256:300],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(shard1.scale, self.scale[256:300], rtol=0, atol=0)

    def test_bytes_estimate(self):
        shard = engram.HostShard.from_tensors(self.payload, self.scale, self.layout, 0)
        self.assertEqual(shard.bytes_estimate, 256 * 64 + 256 * 2)
        self.assertEqual(
            shard.bytes_estimate,
            engram.shard_bytes_estimate(256, 64),
        )

    def test_cpu_host_keeps_plain_tensors(self):
        shard = engram.HostShard.from_tensors(self.payload, self.scale, self.layout, 1)
        self.assertFalse(shard.pinned)
        self.assertFalse(shard.payload.is_pinned())

    def test_load_owned_rows_accepts_lazy_checkpoint_slices(self):
        shard = engram.HostShard.load_owned_rows(
            lambda: self.payload.view(torch.uint8)[256:444].view(torch.float8_e4m3fn),
            lambda: self.scale[256:444],
            self.layout,
            1,
        )
        self.assertEqual(tuple(shard.payload.shape), (44, 64))
        torch.testing.assert_close(
            shard.payload.view(torch.uint8),
            self.payload.view(torch.uint8)[256:300],
            rtol=0,
            atol=0,
        )
        # an allocated padding tail (num_per_part rows) is trimmed to valid_rows
        padded = engram.HostShard.load_owned_rows(
            self.payload.view(torch.uint8)[256:512].view(torch.float8_e4m3fn),
            self.scale[256:512],
            self.layout,
            1,
        )
        self.assertEqual(tuple(padded.payload.shape), (44, 64))

    def test_padding_rows_are_zero_payload_unit_scale(self):
        payload, scale = engram.make_padding_rows(3, 64)
        self.assertEqual(payload.dtype, torch.float8_e4m3fn)
        self.assertEqual(int(payload.view(torch.uint8).to(torch.int64).sum()), 0)
        self.assertTrue(bool((scale == 0x7F).all()))
        rows = engram.dequant_rows(payload, scale)
        self.assertEqual(int(rows.to(torch.float32).abs().sum()), 0)


class HashTest(unittest.TestCase):
    def setUp(self):
        self.params = _params()
        self.windows = torch.tensor([[7, 6, 5, 4], [4, 3, -1, -1], [8, 7, 6, 5]])

    def _oracle(self, windows, dead):
        params = self.params
        layers, heads = params.num_layers, params.n_heads
        expected = torch.empty(
            windows.shape[0], layers, params.n_hash_cols, dtype=torch.int64
        )
        for row, tokens in enumerate(windows.tolist()):
            for layer in range(layers):
                rolling, blocked = 0, False
                for shift, token in enumerate(tokens):
                    blocked |= token < 0 or bool(dead[row, shift])
                    value = params.pad_id if blocked else int(params.token_map[token])
                    rolling ^= value * int(params.multipliers[layer, shift])
                    if shift:
                        for head in range(heads):
                            col = (shift - 1) * heads + head
                            expected[row, layer, col] = rolling % int(
                                params.primes[layer, shift - 1, head]
                            ) + int(params.offsets[layer, col])
        return expected

    def test_hash_matches_scalar_reference_with_boundary_and_dead_tokens(self):
        dead = torch.zeros_like(self.windows, dtype=torch.bool)
        dead[-1, 1] = True
        actual = engram.compute_engram_hash_ids(self.windows, self.params, dead)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(tuple(actual.shape), (3, 1, 6))
        torch.testing.assert_close(
            actual, self._oracle(self.windows, dead), rtol=0, atol=0
        )

    def test_hash_is_deterministic(self):
        first = engram.compute_engram_hash_ids(self.windows, self.params)
        second = engram.compute_engram_hash_ids(self.windows, self.params)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        other = engram.compute_engram_hash_ids(self.windows.flip(0), self.params)
        self.assertFalse(torch.equal(first, other))

    def test_minus_one_terminates_longer_ngrams(self):
        # once a lookback hits -1, every older shift is pad, whatever it holds
        left = engram.compute_engram_hash_ids(
            torch.tensor([[4, 3, -1, 17]]), self.params
        )
        right = engram.compute_engram_hash_ids(
            torch.tensor([[4, 3, -1, 29]]), self.params
        )
        truncated = engram.compute_engram_hash_ids(
            torch.tensor([[4, 3, -1, -1]]), self.params
        )
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(left, truncated, rtol=0, atol=0)

    def test_build_token_windows_matches_reference_layout(self):
        windows = engram.build_token_windows(
            torch.tensor([7, 8, 9, 13, 14, 15]),
            torch.tensor([0, 3, 6]),
            torch.tensor([[6, 5, 4], [12, 11, 10]]),
        )
        self.assertEqual(windows[0].tolist(), [7, 6, 5, 4])
        self.assertEqual(windows[3].tolist(), [13, 12, 11, 10])
        self.assertEqual(windows.shape, (6, 4))


class DequantTest(unittest.TestCase):
    def test_matches_manual_formula(self):
        weight = torch.arange(128).reshape(4, 32).float().to(torch.float8_e4m3fn)
        scales = torch.tensor([[127], [128], [126], [129]], dtype=torch.uint8)
        actual = engram.dequant_rows(weight, scales)
        self.assertEqual(actual.dtype, torch.bfloat16)
        scale_f = (scales.to(torch.int32) << 23).view(torch.float32)
        expected = (weight.float() * scale_f.repeat_interleave(32, dim=1)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_bad_geometry(self):
        weight = torch.zeros(4, 32).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            engram.dequant_rows(weight, torch.zeros(4, 3, dtype=torch.uint8))


class LocalGroupLookupTest(unittest.TestCase):
    def setUp(self):
        self.layout = _layout(300, 2)
        self.payload, self.scale = _make_table(300, dim=64, seed=11)

    def test_owner_only_reads(self):
        group = engram.LocalEnpmGroup(_shards(self.payload, self.scale, self.layout))
        rows = torch.tensor([0, 17, 255, 42, 199])
        results = group.lookup([rows, None])
        self.assertEqual(group.shards[1].reads, 0)
        self.assertEqual(group.shards[0].reads, 5)
        torch.testing.assert_close(
            results[0], _reference_rows(self.payload, self.scale, rows), rtol=0, atol=0
        )
        self.assertEqual(tuple(results[1].shape), (0, 64))

    def test_invalid_rows_are_zero_and_order_preserved(self):
        group = engram.LocalEnpmGroup(_shards(self.payload, self.scale, self.layout))
        rows = torch.tensor([5, -1, 300, 12, 511, 299, 5])
        results = group.lookup([rows, rows.flip(0)])
        expected = _reference_rows(self.payload, self.scale, rows)
        torch.testing.assert_close(results[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            results[1],
            _reference_rows(self.payload, self.scale, rows.flip(0)),
            rtol=0,
            atol=0,
        )
        self.assertEqual(int(results[0][1].to(torch.float32).abs().sum()), 0)
        self.assertEqual(int(results[0][2].to(torch.float32).abs().sum()), 0)
        self.assertEqual(int(results[0][4].to(torch.float32).abs().sum()), 0)
        # duplicated row ids return the same row twice, in request order
        torch.testing.assert_close(results[0][0], results[0][6], rtol=0, atol=0)

    def test_cross_rank_requests_return_to_the_right_rank(self):
        group = engram.LocalEnpmGroup(_shards(self.payload, self.scale, self.layout))
        rank0_rows = torch.arange(256, 300)  # owned by rank 1
        rank1_rows = torch.arange(0, 10)  # owned by rank 0
        results = group.lookup([rank0_rows, rank1_rows])
        torch.testing.assert_close(
            results[0],
            _reference_rows(self.payload, self.scale, rank0_rows),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            results[1],
            _reference_rows(self.payload, self.scale, rank1_rows),
            rtol=0,
            atol=0,
        )
        self.assertEqual(group.shards[0].reads, 10)
        self.assertEqual(group.shards[1].reads, 44)

    def test_2d_rows_keep_column_layout(self):
        group = engram.LocalEnpmGroup(_shards(self.payload, self.scale, self.layout))
        rows = torch.tensor([[0, 250], [299, -1], [128, 127]])
        results = group.lookup([rows, None])
        self.assertEqual(tuple(results[0].shape), (3, 2, 64))
        expected = _reference_rows(self.payload, self.scale, rows.reshape(-1))
        torch.testing.assert_close(results[0].reshape(6, 64), expected, rtol=0, atol=0)

    def test_multi_round_with_padding_tails_and_empty_rank(self):
        layout = _layout(600, 3)
        payload, scale = _make_table(600, dim=64, seed=5)
        group = engram.LocalEnpmGroup(_shards(payload, scale, layout))
        gen = torch.Generator().manual_seed(9)
        rank0_rows = torch.randint(-2, 700, (5000,), generator=gen)
        rank1_rows = torch.randint(0, 600, (10,), generator=gen)
        rank2_rows = torch.empty(0, dtype=torch.int64)
        results = group.lookup([rank0_rows, rank1_rows, rank2_rows])
        # group-wide max 5000 -> rounds of 4096 and 904, every rank joins both
        self.assertEqual(group.rounds_executed, 2)
        self.assertEqual(group.round_lengths, [4096, 904])
        self.assertEqual(group.gathered_rounds[0].shape, (3 * 4096,))
        self.assertEqual(group.gathered_rounds[1].shape, (3 * 904,))
        # rank 1 padded its round-0 tail with the invalid row id
        self.assertEqual(int(group.gathered_rounds[0][4096 + 10]), -1)
        self.assertTrue(bool((group.gathered_rounds[0][4096 + 10 :] == -1).all()))
        # rank 1 exhausted: round 1 chunk is all -1; rank 2 is all -1 both rounds
        self.assertTrue(bool((group.gathered_rounds[1][4096:8192] == -1).all()))
        self.assertTrue(bool((group.gathered_rounds[0][8192:] == -1).all()))
        self.assertTrue(bool((group.gathered_rounds[1][2720:] == -1).all()))
        # empty rank still joins and produces an empty result
        self.assertEqual(tuple(results[2].shape), (0, 64))
        torch.testing.assert_close(
            results[0], _reference_rows(payload, scale, rank0_rows), rtol=0, atol=0
        )
        torch.testing.assert_close(
            results[1], _reference_rows(payload, scale, rank1_rows), rtol=0, atol=0
        )
        # every valid requested row is read exactly once, by its owner only
        all_rows = torch.cat([rank0_rows, rank1_rows])
        for rank in range(3):
            start = layout.row_start(rank)
            owned = (all_rows >= start) & (all_rows < start + layout.valid_rows(rank))
            self.assertEqual(group.shards[rank].reads, int(owned.sum()))

    def test_small_round_limit_forces_many_rounds(self):
        group = engram.LocalEnpmGroup(_shards(self.payload, self.scale, self.layout))
        rows = torch.arange(20)
        results = group.lookup([rows, rows[:3]], max_rows_per_round=4)
        self.assertEqual(group.rounds_executed, 5)
        self.assertEqual(group.round_lengths, [4, 4, 4, 4, 4])
        torch.testing.assert_close(
            results[0], _reference_rows(self.payload, self.scale, rows), rtol=0, atol=0
        )
        torch.testing.assert_close(
            results[1],
            _reference_rows(self.payload, self.scale, rows[:3]),
            rtol=0,
            atol=0,
        )


class DirectLookupTest(unittest.TestCase):
    def test_single_process_direct_lookup_enp1(self):
        layout = _layout(100, 1)
        payload, scale = _make_table(100, dim=64, seed=2)
        shard = engram.HostShard.from_tensors(payload, scale, layout, 0)
        rows = torch.tensor([3, 99, -1, 100, 255, 7, 99])
        out = engram.engram_lookup(
            rows, layer_id=7, shard=shard, layout=layout, group=None
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (7, 64))
        torch.testing.assert_close(
            out, _reference_rows(payload, scale, rows), rtol=0, atol=0
        )

    def test_offline_lookup_without_group_degrades_to_local_shard(self):
        # Wave2 contract: no process group + ENP>1 no longer raises (offline
        # single-process real-weight drivers). It degrades to a local-shard
        # lookup with a RuntimeWarning: this rank's owned rows resolve, rows
        # owned by other ENP ranks come back zero-filled.
        layout = _layout(300, 2)
        payload, scale = _make_table(300, dim=64, seed=4)
        shard = engram.HostShard.from_tensors(payload, scale, layout, 0)
        # Rank 0 owns rows [0, 256); row 260 belongs to rank 1.
        rows = torch.tensor([2, 260])
        # assertWarns scans sys.modules and trips rtp_llm's lazy __getattr__,
        # so record warnings explicitly instead.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = engram.engram_lookup(rows, layer_id=7, shard=shard, layout=layout)
        self.assertTrue(any("offline" in str(item.message) for item in caught), caught)
        self.assertEqual(tuple(out.shape), (2, 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = _reference_rows(payload, scale, rows[0:1])
        torch.testing.assert_close(out[0:1], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(out[1], torch.zeros(64)))

    def test_rejects_floating_row_ids(self):
        layout = _layout(100, 1)
        payload, scale = _make_table(100, dim=64, seed=6)
        shard = engram.HostShard.from_tensors(payload, scale, layout, 0)
        with self.assertRaises(TypeError):
            engram.engram_lookup(
                torch.tensor([1.0, 2.0]),
                layer_id=7,
                shard=shard,
                layout=layout,
                group=None,
            )


def _manual_gate(h, kv, q_weight, k_weight, eps, clamp=1e-6):
    """rtp-llm engram.py:626-639 oracle, one (token, copy) at a time."""
    tokens, copies, dim = h.shape
    out = torch.empty_like(h)
    for token in range(tokens):
        for lane in range(copies):
            hidden = h[token, lane].float()
            key = kv[token, lane * dim : (lane + 1) * dim].float()
            value = kv[token, copies * dim :].float()
            dot = (hidden * q_weight[lane].float() * k_weight[lane].float() * key).sum(
                -1
            )
            dot = dot * torch.rsqrt(hidden.square().mean(-1) + eps)
            dot = dot * torch.rsqrt(key.square().mean(-1) + eps) / math.sqrt(dim)
            gate_input = dot.abs().clamp_min(clamp).sqrt()
            gate_input = torch.where(dot < 0, -gate_input, gate_input)
            gate = torch.sigmoid(gate_input)
            out[token, lane] = (hidden + gate * value).to(h.dtype)
    return out


class GateInjectTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.hidden = torch.randn(3, 4, 32).bfloat16()
        self.kv = torch.randn(3, 5 * 32).bfloat16()
        self.q = torch.randn(4, 32).bfloat16()
        self.k = torch.randn(4, 32).bfloat16()

    def test_matches_manual_oracle(self):
        actual = engram.engram_gate_inject(
            self.hidden, self.q, self.k, self.kv, eps=1e-20
        )
        expected = _manual_gate(self.hidden, self.kv, self.q, self.k, 1e-20)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_negative_and_positive_dots_both_exercised(self):
        # h aligned with key -> positive dot; h = -key -> negative dot; the
        # shared value is ones so out - h is exactly the gate.
        key = torch.randn(4, 32).bfloat16()
        hidden = torch.stack([key, -key, key]).bfloat16()
        kv = torch.cat(
            [key.reshape(1, -1).repeat(3, 1), torch.ones(3, 32)], dim=1
        ).bfloat16()
        ones = torch.ones(4, 32).bfloat16()
        actual = engram.engram_gate_inject(hidden, ones, ones, kv, eps=1e-20)
        expected = _manual_gate(hidden, kv, ones, ones, 1e-20)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gate_pos = (actual[0] - hidden[0]).float()
        gate_neg = (actual[1] - hidden[1]).float()
        self.assertGreater(float(gate_pos.mean()), 0.5)
        self.assertLess(float(gate_neg.mean()), 0.5)

    def test_zero_dot_uses_positive_clamped_sqrt(self):
        hidden = torch.zeros(1, 1, 32)
        kv = torch.cat((torch.zeros(1, 32), torch.ones(1, 32)), dim=1)
        output = engram.engram_gate_inject(
            hidden, torch.ones(1, 32), torch.ones(1, 32), kv, eps=1e-20
        )
        torch.testing.assert_close(
            output,
            torch.full_like(hidden, torch.sigmoid(torch.tensor(1e-6).sqrt())),
        )

    def test_token_mask_zeroes_gate(self):
        mask = torch.tensor([True, False, True])
        actual = engram.engram_gate_inject(
            self.hidden, self.q, self.k, self.kv, eps=1e-20, token_mask=mask
        )
        torch.testing.assert_close(actual[1], self.hidden[1], rtol=0, atol=0)
        expected = _manual_gate(self.hidden, self.kv, self.q, self.k, 1e-20)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


class WkvProjectTest(unittest.TestCase):
    def test_plain_bf16_linear(self):
        torch.manual_seed(3)
        wkv = torch.randn(96, 384).bfloat16()
        rows = torch.randn(5, 384).bfloat16()
        out = engram.wkv_project(rows, wkv)
        self.assertEqual(tuple(out.shape), (5, 96))
        torch.testing.assert_close(
            out, torch.nn.functional.linear(rows, wkv), rtol=0, atol=0
        )
        # non-bf16 rows are cast, real shape [25600, 6144] respected by shape math
        out2 = engram.wkv_project(rows.float(), wkv)
        torch.testing.assert_close(
            out2, torch.nn.functional.linear(rows, wkv), rtol=0, atol=0
        )

    def test_rejects_shape_mismatch(self):
        wkv = torch.randn(8, 16).bfloat16()
        with self.assertRaises(ValueError):
            engram.wkv_project(torch.randn(4, 10).bfloat16(), wkv)


class MiniPipelineTest(unittest.TestCase):
    """window -> hash -> ENP lookup -> WKV -> gated inject on a 2-rank table."""

    def test_full_pipeline_matches_reference(self):
        torch.manual_seed(17)
        layout = _layout(300, 2)
        payload, scale = _make_table(300, dim=64, seed=23)
        params = _params(300)
        dim, hc, hidden = 24, 3, 24
        cols = params.n_hash_cols  # 6
        wkv = torch.randn((hc + 1) * hidden, cols * 64).bfloat16()
        q = torch.randn(hc, hidden).bfloat16()
        k = torch.randn(hc, hidden).bfloat16()

        ids = torch.arange(3, 27)
        windows = engram.build_token_windows(
            ids, torch.tensor([0, 24]), torch.tensor([[-1, -1, -1]])
        )
        hashes = engram.compute_engram_hash_ids(windows, params)[:, 0]
        self.assertTrue(bool((hashes >= 0).all()))
        self.assertTrue(bool((hashes < 300).all()))

        group = engram.LocalEnpmGroup(_shards(payload, scale, layout))
        rank1_rows = hashes[:5]
        results = group.lookup([hashes, rank1_rows])
        self.assertEqual(tuple(results[0].shape), (24, cols, 64))
        self.assertEqual(tuple(results[1].shape), (5, cols, 64))

        flat = results[0].reshape(24, cols * 64)
        kv = engram.wkv_project(flat, wkv)
        hidden_states = torch.randn(24, hc, hidden).bfloat16()
        output = engram.engram_gate_inject(hidden_states, q, k, kv, eps=1e-20)

        # independent reference from the full synthetic table
        expected_rows = _reference_rows(payload, scale, hashes.reshape(-1))
        expected_kv = torch.nn.functional.linear(
            expected_rows.reshape(24, cols * 64), wkv
        )
        expected_out = _manual_gate(hidden_states, expected_kv, q, k, 1e-20)
        torch.testing.assert_close(
            flat, expected_rows.reshape(24, cols * 64), rtol=0, atol=0
        )
        torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
        torch.testing.assert_close(output, expected_out, rtol=0, atol=0)
        # rank 1 got its own (owner-sharded) copy of the same rows
        torch.testing.assert_close(
            results[1],
            _reference_rows(payload, scale, rank1_rows.reshape(-1)).reshape(
                5, cols, 64
            ),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()

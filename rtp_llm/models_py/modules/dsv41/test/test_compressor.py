"""Owner compressor GPU probes against the actual frozen official definitions."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from official_compressor import load_official_compressor
from test_compact_writer import _load_official_kernel, _official_gpu, _pages

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    CacheIdentity,
    CacheLayout,
    CacheRegion,
)
from rtp_llm.models_py.modules.dsv41.compressor import (
    CompressorRoPE,
    OwnerCompressor,
    OwnerPageBinding,
    PairCarry,
    prepare_owner_hidden,
    prepare_owner_kv,
)


class OwnerCompressorGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError(
                "required owner compressor probe needs an actual Blackwell GPU"
            )
        cls.prior_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.official = load_official_compressor()
        cls.official_kernel = _load_official_kernel()

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.prior_tf32

    def setUp(self):
        self.layout = CacheLayout()
        self.identity = CacheIdentity(
            "2bc89ac599031fa673cab993f1df02fc4a98c673",
            self.layout.fingerprint,
            "bounded-v1",
        )
        self.generator = torch.Generator().manual_seed(1941)

    def values(self, rows):
        return (
            torch.randn((rows, 5120), generator=self.generator)
            .mul_(0.3)
            .bfloat16()
            .cuda()
        )

    def models(self, owner, sparse=False):
        ratio = 1 if owner == 20 else 2
        args = self.official.ModelArgs(
            dim=5120,
            head_dim=512,
            rope_head_dim=64,
            max_batch_size=1,
            n_layers=40,
            norm_eps=1e-20,
            compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20,
            kv_source_layers=(2, 8, 14, 20),
        )
        reference = self.official.Compressor(args, owner).cuda()
        dtype = torch.float32 if ratio == 2 else torch.bfloat16
        weight = (
            torch.randn((512, 5120), generator=self.generator)
            .mul_(0.01)
            .bfloat16()
            .to(dtype)
            .cuda()
        )
        gate = (
            torch.randn((512, 5120), generator=self.generator)
            .mul_(0.02)
            .bfloat16()
            .float()
            .cuda()
        )
        if sparse:
            # Powers-of-two projections make chunk-vs-token GEMM accumulation
            # exact; random dense prefill is separately compared below.
            weight.zero_()
            gate.zero_()
            indices = torch.arange(512, device="cuda")
            weight[indices, indices] = 0.5
            weight[indices, indices + 512] = -0.25
            gate[indices, indices + 1024] = 0.25
        norm = (
            torch.randn((512,), generator=self.generator)
            .mul_(0.1)
            .add_(1)
            .bfloat16()
            .cuda()
        )
        with torch.no_grad():
            reference.wkv.weight.copy_(weight)
            reference.norm.weight.copy_(norm)
            if ratio == 2:
                reference.wgate.weight.copy_(gate)
        candidate = OwnerCompressor(
            owner, weight, norm, gate if ratio == 2 else None, layout=self.layout
        )
        return candidate, reference

    def run_candidate(
        self, model, hidden, start=0, pair=None, request="request-a", identity=None
    ):
        return model(
            hidden,
            start_pos=start,
            request_id=request,
            identity=identity or self.identity,
            pair=pair,
        )

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_official_carry(self, result, reference):
        pair = result.next_pair
        self.assertEqual(pair.next_position, result.query_end)
        if result.query_end % 2:
            self.assert_exact(pair.partial_kv, reference.kv_state[0, 0])
            self.assert_exact(pair.partial_score, reference.score_state[0, 0])
        else:
            self.assertIsNone(pair.partial_kv)
            self.assertIsNone(pair.partial_score)

    def test_all_four_owners_match_official_dense_prefill(self):
        with torch.no_grad():
            for owner in (2, 8, 14, 20):
                candidate, reference = self.models(owner)
                for rows in (1, 2, 3, 127, 128, 129):
                    with self.subTest(owner=owner, rows=rows):
                        hidden = self.values(rows)
                        expected = reference(hidden.unsqueeze(0), 0)
                        result = self.run_candidate(candidate, hidden)
                        self.assert_exact(
                            result.unrotated,
                            (
                                expected[0]
                                if expected is not None
                                else hidden.new_empty((0, 512))
                            ),
                        )
                        self.assertEqual(
                            result.group_positions.tolist(),
                            list(
                                range(0, rows - rows % candidate.ratio, candidate.ratio)
                            ),
                        )
                        self.assertEqual(
                            result.visible_lengths.tolist(),
                            [(i + 1) // candidate.ratio for i in range(rows)],
                        )
                        if candidate.ratio == 2:
                            self.assert_official_carry(result, reference)

    def test_odd_chunk_carry_matches_actual_official_decode(self):
        for owner in (2, 8, 14):
            for partitions in ((1, 1, 3, 5, 1, 2), (3, 4, 2, 1, 3), (0, 3, 0, 7, 3)):
                with self.subTest(owner=owner, partitions=partitions), torch.no_grad():
                    candidate, reference = self.models(owner, sparse=True)
                    hidden = self.values(sum(partitions))
                    position, carry = 0, None
                    for count in partitions:
                        old_kv = (
                            None
                            if carry is None or carry.partial_kv is None
                            else carry.partial_kv.clone()
                        )
                        old_score = (
                            None
                            if carry is None or carry.partial_score is None
                            else carry.partial_score.clone()
                        )
                        result = self.run_candidate(
                            candidate,
                            hidden[position : position + count],
                            position,
                            carry,
                        )
                        expected_rows = []
                        for index in range(position, position + count):
                            expected = reference(
                                hidden[index : index + 1].unsqueeze(0), index
                            )
                            if expected is not None:
                                expected_rows.append(expected[0])
                        expected = (
                            torch.cat(expected_rows)
                            if expected_rows
                            else hidden.new_empty((0, 512))
                        )
                        self.assert_exact(result.unrotated, expected)
                        self.assertEqual(
                            result.visible_lengths.tolist(),
                            [(i + 1) // 2 for i in range(position, position + count)],
                        )
                        self.assertEqual(
                            result.group_positions.tolist(),
                            list(
                                range(position // 2 * 2, (position + count) // 2 * 2, 2)
                            ),
                        )
                        self.assert_official_carry(result, reference)
                        if old_kv is not None:
                            self.assert_exact(carry.partial_kv, old_kv)
                            self.assert_exact(carry.partial_score, old_score)
                        position += count
                        carry = result.next_pair

    def test_only_source_owners_and_official_weight_dtypes_are_accepted(self):
        candidate, _ = self.models(2)
        for layer in (0, 1, 3, 19, 21, 24, 36, 40, 41, 42):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                OwnerCompressor(
                    layer, candidate.wkv, candidate.norm_weight, candidate.wgate
                )
        with self.assertRaises(ValueError):
            OwnerCompressor(
                2, candidate.wkv.bfloat16(), candidate.norm_weight, candidate.wgate
            )
        with self.assertRaises(ValueError):
            OwnerCompressor(
                20, candidate.wkv.bfloat16(), candidate.norm_weight, candidate.wgate
            )
        with self.assertRaises(ValueError):
            OwnerCompressor(2, candidate.wkv, candidate.norm_weight)

    def test_carry_cannot_cross_owner_request_mode_or_position(self):
        candidate, _ = self.models(2)
        hidden = self.values(1)
        pair = self.run_candidate(candidate, hidden).next_pair
        other_owner, _ = self.models(8)
        with self.assertRaises(ValueError):
            self.run_candidate(other_owner, hidden, 1, pair)
        with self.assertRaises(ValueError):
            self.run_candidate(candidate, hidden, 1, pair, request="request-b")
        with self.assertRaises(ValueError):
            self.run_candidate(
                candidate,
                hidden,
                1,
                pair,
                identity=replace(self.identity, replay_fingerprint="full"),
            )
        with self.assertRaises(ValueError):
            self.run_candidate(candidate, hidden, 3, pair)
        with self.assertRaises(ValueError):
            self.run_candidate(candidate, hidden, 1, replace(pair, partial_score=None))

    def test_restored_even_boundary_requires_explicit_empty_pair(self):
        candidate, _ = self.models(2)
        hidden = self.values(3)
        with self.assertRaises(ValueError):
            self.run_candidate(candidate, hidden, 1024)
        with self.assertRaises(ValueError):
            PairCarry.empty(2, "request-a", self.identity, 1025)
        restored = PairCarry.empty(2, "request-a", self.identity, 1024)
        result = self.run_candidate(candidate, hidden, 1024, restored)
        self.assertEqual(result.group_positions.tolist(), [1024])
        self.assertEqual(result.visible_lengths.tolist(), [512, 513, 513])
        self.assertEqual(result.next_pair.next_position, 1027)

    def test_l20_input_uses_incoming_pre_mix_then_official_attention_norm(self):
        hidden = self.values(12).reshape(3, 4, 5120)
        pre = torch.tensor(
            [[1, 0, 0, 0], [0.5, 0.25, 0, 0.125], [0, 0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        )
        norm = torch.linspace(0.5, 1.5, 5120, device="cuda").bfloat16()
        reference_norm = self.official.RMSNorm(5120, 1e-20).cuda()
        with torch.no_grad():
            reference_norm.weight.copy_(norm)
            expected = reference_norm(
                self.official.hc_pre(None, hidden.unsqueeze(0), pre.unsqueeze(0))
            )[0]
        actual = prepare_owner_hidden(20, hidden, pre, norm)
        self.assert_exact(actual, expected)
        self.assertFalse(
            torch.equal(actual, reference_norm(hidden.float().mean(1).bfloat16()))
        )

    def official_frequencies(self, positions):
        rope = CompressorRoPE()
        self.official.precompute_freqs_cis.cache_clear()
        with torch.device("cuda"):
            frequencies = self.official.precompute_freqs_cis(
                rope.dimension,
                max(1, int(positions.max()) + 1) if positions.numel() else 1,
                rope.original_context,
                rope.theta,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )
        return frequencies.index_select(0, positions)

    def test_rope_index_consumption_and_native_bytes_match_official_gpu(self):
        for owner in (2, 8, 14, 20):
            with self.subTest(owner=owner), torch.no_grad():
                candidate, reference = self.models(owner)
                hidden = self.values(129)
                source = self.run_candidate(candidate, hidden)
                expected_latent = reference(hidden.unsqueeze(0), 0)[0]
                original = source.unrotated.clone()
                index_linear = self.official.Linear(
                    512, 128, dtype=torch.bfloat16
                ).cuda()
                index_norm = self.official.RMSNorm(128, 1e-20).cuda()
                index_linear.weight.copy_(
                    torch.randn((128, 512), generator=self.generator)
                    .mul_(0.02)
                    .bfloat16()
                    .cuda()
                )
                index_norm.weight.copy_(
                    torch.linspace(0.5, 1.5, 128, device="cuda").bfloat16()
                )
                frequencies = self.official_frequencies(source.group_positions)
                expected_index = index_norm(index_linear(expected_latent)).unsqueeze(0)
                self.official.apply_rotary_emb(expected_index[..., -64:], frequencies)
                expected_global = expected_latent.clone().unsqueeze(0)
                self.official.apply_rotary_emb(expected_global[..., -64:], frequencies)
                prepared = prepare_owner_kv(
                    source, index_linear.weight, index_norm.weight
                )
                self.assert_exact(source.unrotated, original)
                self.assert_exact(prepared.index_values, expected_index[0])
                self.assert_exact(prepared.global_values, expected_global[0])
                global_bytes, index_bytes = prepared.encode()
                for encoded, values, region in (
                    (global_bytes, expected_global[0], CacheRegion.GLOBAL),
                    (index_bytes, expected_index[0], CacheRegion.INDEX_K),
                ):
                    encoded.check()
                    self.assert_exact(
                        encoded.output,
                        _official_gpu(
                            self.official_kernel, values.contiguous(), region
                        ),
                    )

    def test_owner_store_preserves_joint_region_identity_and_exact_bytes(self):
        candidate, _ = self.models(2, sparse=True)
        source = self.run_candidate(candidate, self.values(7))
        index_weight = (
            torch.ones((128, 512), dtype=torch.bfloat16, device="cuda") * 0.125
        )
        index_norm = torch.ones((128,), dtype=torch.bfloat16, device="cuda")
        prepared = prepare_owner_kv(source, index_weight, index_norm)
        rows = source.unrotated.shape[0]
        entries = self.layout.token_block_size // 2
        global_pages, global_storage = _pages(CacheRegion.GLOBAL, rows, entries)
        index_pages, index_storage = _pages(CacheRegion.INDEX_K, rows, entries)
        global_binding = OwnerPageBinding(2, self.identity, global_pages)
        index_binding = OwnerPageBinding(2, self.identity, index_pages)
        slots = torch.tensor(
            [entries, -1, entries + 2], dtype=torch.int64, device="cuda"
        )
        index_slots = torch.tensor(
            [2 * entries, -1, 2 * entries + 2], dtype=torch.int64, device="cuda"
        )
        global_expected, index_expected = prepared.encode()
        for encoded in (global_expected, index_expected):
            encoded.check()
        for result in prepared.store(global_binding, index_binding, slots, index_slots):
            result.check()
        for storage, packed, width, page in (
            (global_storage, global_expected.output, 288, 1),
            (index_storage, index_expected.output, 68, 2),
        ):
            expected = torch.full_like(storage, 0x5A)
            expected[page, :width] = packed[0]
            expected[page, 2 * width : 3 * width] = packed[2]
            self.assert_exact(storage, expected)
        before = global_storage.clone(), index_storage.clone()
        for wrong in (
            replace(index_binding, owner_layer=8),
            replace(
                index_binding,
                identity=replace(self.identity, replay_fingerprint="full"),
            ),
        ):
            with self.assertRaises(ValueError):
                prepared.store(global_binding, wrong, slots, index_slots)
            self.assert_exact(global_storage, before[0])
            self.assert_exact(index_storage, before[1])

    def test_extreme_finite_gates_keep_fp32_stable_softmax(self):
        candidate, reference = self.models(2, sparse=True)
        hidden = self.values(3)
        hidden[0, 1024:1536] = 320
        hidden[1, 1024:1536] = -320
        with torch.no_grad():
            expected = reference(hidden.unsqueeze(0), 0)[0]
        result = self.run_candidate(candidate, hidden)
        self.assert_exact(result.unrotated, expected)
        self.assertTrue(torch.isfinite(result.unrotated).all().item())

    def test_tiny_latents_keep_the_official_text_norm_epsilon(self):
        for owner in (2, 20):
            with self.subTest(owner=owner), torch.no_grad():
                candidate, reference = self.models(owner, sparse=True)
                hidden = torch.full(
                    (3, 5120), 2**-30, dtype=torch.bfloat16, device="cuda"
                )
                expected = reference(hidden.unsqueeze(0), 0)[0]
                result = self.run_candidate(candidate, hidden)
                self.assert_exact(result.unrotated, expected)
                self.assertGreater(float(result.unrotated.abs().min()), 0.25)

    def test_disabled_or_autocast_execution_does_not_silently_fall_back(self):
        candidate, _ = self.models(2, sparse=True)
        hidden = self.values(2)
        with patch.dict(os.environ, {"DSV41_OWNER_COMPRESSOR": "0"}):
            with self.assertRaisesRegex(RuntimeError, "DSV41_OWNER_COMPRESSOR"):
                self.run_candidate(candidate, hidden)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "autocast"):
                self.run_candidate(candidate, hidden)

    def test_rotary_positions_at_real_context_boundary(self):
        # Boundary arithmetic/RoPE probe only; this is not a 1M-generation test.
        positions = torch.tensor(
            [0, 2, 65536, 1048574, 1048575], dtype=torch.int64, device="cuda"
        )
        self.assert_exact(
            CompressorRoPE().frequencies(positions),
            self.official_frequencies(positions),
        )
        candidate, _ = self.models(2, sparse=True)
        carry = PairCarry.empty(2, "request-a", self.identity, 1048574)
        result = self.run_candidate(candidate, self.values(2), 1048574, carry)
        self.assertEqual(result.query_end, 1048576)
        self.assertEqual(result.group_positions.tolist(), [1048574])
        with self.assertRaises(ValueError):
            self.run_candidate(candidate, self.values(3), 1048574, carry)


if __name__ == "__main__":
    unittest.main()

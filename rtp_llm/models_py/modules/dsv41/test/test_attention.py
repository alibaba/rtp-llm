"""Real local attention composition; no distributed or model-quality claim."""

import hashlib
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from official_compressor import load_official_compressor
from test_compact_writer import _load_official_kernel, _official_gpu
from torch import nn

from rtp_llm.models_py.modules.dsv41.attention import (
    V41Attention,
    V41AttentionCache,
    attention_rope,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheLayout, CacheRegion
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compressor import OwnerCompressor


class AttentionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getuid() == 0 or not torch.cuda.is_available():
            raise RuntimeError(
                "attention probes require non-root CUDA13 Blackwell execution"
            )
        if torch.cuda.get_device_capability()[
            0
        ] != 10 or not torch.version.cuda.startswith("13."):
            raise RuntimeError("attention probes require CUDA13 Blackwell")
        cls.prior_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.official = load_official_compressor()
        cls.quantizer = _load_official_kernel()
        cls.records = []

        def linear(inputs, outputs):
            module = nn.Linear(
                inputs, outputs, bias=False, dtype=torch.bfloat16, device="cuda"
            )
            module.weight.data.zero_()
            return module

        cls.wq_a = linear(5120, 1280)
        cls.wq_b = linear(1280, 64 * 512)
        cls.wkv = linear(5120, 512)
        cls.wo_b = linear(8 * 1024, 5120)
        cls.index_wq_b = linear(1280, 32 * 128)
        channel = torch.arange(512, device="cuda")
        cls.wkv.weight.data[channel, channel] = 1
        cls.wq_a.weight.data[torch.arange(1280), torch.arange(1280)] = 1
        cls.wo_b.weight.data[torch.arange(5120), (torch.arange(5120) % 8) * 1024] = 1
        cls.wo_a = torch.zeros((8, 1024, 4096), dtype=torch.bfloat16, device="cuda")
        cls.wo_a[:, 0, 0] = 1

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.prior_tf32
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            (Path(output) / "attention_components.json").write_text(
                json.dumps(
                    {
                        "scope": "initialized local attention exact-value/state components; not CP8, 1M or numerical acceptance",
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "torch": str(torch.__version__),
                        "observations": cls.records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def setUp(self):
        self.layout = CacheLayout()
        self.identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity(
            "2bc89ac599031fa673cab993f1df02fc4a98c673", self.layout
        )

    def cache(self, length=384, request="request-a"):
        return V41AttentionCache.allocate_local(
            request, self.identity, self.layout, length, device="cuda"
        )

    def hidden(self, rows):
        values = torch.zeros((rows, 5120), dtype=torch.bfloat16, device="cuda")
        values[:, :448] = 1
        return values

    def model(self, layer):
        owner = layer in (2, 8, 14, 20)
        scorer = layer in (2, 8, 14, 20, 24, 28, 32, 36)
        dtype = torch.bfloat16 if layer == 20 else torch.float32
        compressor = (
            OwnerCompressor(
                layer,
                self.wkv.weight.detach().to(dtype),
                torch.ones(512, dtype=torch.bfloat16, device="cuda"),
                (
                    torch.zeros((512, 5120), dtype=torch.float32, device="cuda")
                    if layer != 20
                    else None
                ),
                layout=self.layout,
            )
            if owner
            else None
        )
        wk = (
            torch.zeros((128, 512), dtype=torch.bfloat16, device="cuda")
            if owner
            else None
        )
        if wk is not None:
            indices = torch.arange(128, device="cuda")
            wk[indices, indices] = 1
        return V41Attention(
            layer,
            wq_a=self.wq_a,
            wq_b=self.wq_b,
            wkv=self.wkv,
            wo_b=self.wo_b,
            wo_a=self.wo_a,
            q_norm=torch.ones(1280, dtype=torch.bfloat16, device="cuda"),
            kv_norm=torch.ones(512, dtype=torch.bfloat16, device="cuda"),
            sinks=torch.zeros(64, dtype=torch.float32, device="cuda"),
            compressor=compressor,
            index_wq_b=self.index_wq_b if scorer else None,
            index_weights=(
                torch.zeros((32, 5120), dtype=torch.bfloat16, device="cuda")
                if scorer
                else None
            ),
            index_wk=wk,
            index_norm=(
                torch.ones(128, dtype=torch.bfloat16, device="cuda") if owner else None
            ),
        )

    def equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def digest(value):
            return hashlib.sha256(
                value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()

        self.records.append(
            {
                "test": self.id(),
                "shape": list(actual.shape),
                "actual_sha256": digest(actual),
                "expected_sha256": digest(expected),
            }
        )

    def expected(self, positions, ratio=0, floor=0):
        # Constant non-RoPE KV and zero Q give a uniform softmax with one sink.
        latent = torch.zeros((1, 512), dtype=torch.bfloat16, device="cuda")
        latent[:, :448] = (512 / 448) ** 0.5
        swa = _official_gpu(self.quantizer, latent, CacheRegion.SWA, inplace=True)[
            0, 0
        ].double()
        global_kv = (
            _official_gpu(self.quantizer, latent, CacheRegion.GLOBAL, inplace=True)[
                0, 0
            ].double()
            if ratio
            else 0
        )
        count = (positions + 1 - floor).clamp(max=128).double()
        global_count = (
            (positions + 1).div(ratio, rounding_mode="floor").double() if ratio else 0
        )
        result = (
            (swa * count + global_kv * global_count) / (count + global_count + 1)
        ).bfloat16()
        return result[:, None].expand(-1, 5120)

    def test_full_prefill_does_not_overwrite_early_swa_queries(self):
        model, cache = self.model(0), self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=257)
        actual = model(self.hidden(257), context)
        self.equal(actual, self.expected(torch.arange(257, device="cuda")))
        self.assertEqual(context.observations[0]["swa_write_tile"], 9)
        self.assertEqual(cache.swa_ends[0], 257)

    def test_swa_long_chunk_and_partitioned_continuation_match(self):
        model, cache = self.model(1), self.cache()
        expected = model(
            self.hidden(257), cache.begin_forward(epoch=0, start=0, end=257)
        )
        other = self.cache()
        outputs, start = [], 0
        for epoch, end in enumerate((127, 128, 129, 132, 257)):
            outputs.append(
                model(
                    self.hidden(end - start),
                    other.begin_forward(epoch=epoch, start=start, end=end),
                )
            )
            start = end
        self.equal(torch.cat(outputs), expected)

    def test_owner_pair_continuation_and_consumer_reuse(self):
        owner, consumer, cache = self.model(2), self.model(3), self.cache()
        start = 0
        for epoch, end in enumerate((3, 8, 129)):
            context = cache.begin_forward(epoch=epoch, start=start, end=end)
            hidden = self.hidden(end - start)
            self.equal(
                owner(hidden, context),
                self.expected(torch.arange(start, end, device="cuda"), ratio=2),
            )
            self.equal(
                consumer(hidden, context),
                self.expected(torch.arange(start, end, device="cuda"), ratio=2),
            )
            self.assertEqual(context.observations[-1]["source_rows"], 0)
            self.assertEqual(context.observations[-1]["index_rows"], 0)
            self.assertEqual(set(context.selections), {2})
            self.assertEqual(cache.owners[2].pair.next_position, end)
            start = end

    def test_l20_tail_owns_candidates_and_reindex_preserves_source(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=129)
        self.model(20)(self.hidden(129), context)
        late = context.tail(1, replay_floor=1)
        saved = late.selections[20].candidate_blocks.clone()
        with torch.inference_mode():
            context.selections[20].candidate_blocks.fill_(-1)
        self.equal(late.selections[20].candidate_blocks, saved)
        source_bytes = cache.owners[20].global_kv.pages.data.clone()
        output = self.model(24)(self.hidden(128), late)
        self.equal(
            output, self.expected(torch.arange(1, 129, device="cuda"), ratio=1, floor=1)
        )
        self.equal(cache.owners[20].global_kv.pages.data, source_bytes)
        self.assertEqual(late.observations[-1]["source_rows"], 0)
        self.assertEqual(late.observations[-1]["index_rows"], 128)

    def test_missing_owner_or_wrong_forward_cannot_consume_stale_indices(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=7, start=0, end=3)
        self.model(2)(self.hidden(3), context)
        cache.begin_forward(epoch=8, start=3, end=4)
        with self.assertRaisesRegex(ValueError, "stale"):
            context.indices_for(3)
        other = self.cache(request="request-b")
        context_b = other.begin_forward(epoch=0, start=0, end=3)
        with self.assertRaisesRegex(ValueError, "owner has not published"):
            context_b.indices_for(3)

    def test_failed_write_poison_prevents_partial_cache_reuse(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=3)
        cache.swa[0].page_ids.zero_()
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.model(0)(self.hidden(3), context)
        self.assertTrue(cache.poisoned)
        with self.assertRaisesRegex(RuntimeError, "discarded or restored"):
            cache.begin_forward(epoch=1, start=3, end=4)

    def test_empty_rank_projections_preserve_no_row_state(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=0)
        actual = self.model(20)(self.hidden(0), context)
        self.equal(actual, self.hidden(0))
        self.assertEqual(context.selections[20].topk.shape, (0, 512))
        self.assertEqual(context.observations[0]["source_rows"], 0)

    def test_rope_and_inverse_match_frozen_definitions_at_real_positions(self):
        generator = torch.Generator().manual_seed(314159)
        values = torch.randn((5, 64, 512), generator=generator).bfloat16().cuda()
        positions = torch.tensor([0, 127, 65535, 262143, 1048320], device="cuda")
        for branch in (False, True):
            with torch.device("cuda"):
                frequencies = self.official.precompute_freqs_cis(
                    64,
                    1048321,
                    65536 if branch else 0,
                    160000 if branch else 10000,
                    16,
                    32,
                    1,
                )
            selected = frequencies.index_select(0, positions)
            for inverse in (False, True):
                expected = values.clone()
                self.official.apply_rotary_emb(
                    expected[None, ..., -64:], selected, inverse
                )
                actual = attention_rope(
                    values, positions, global_branch=branch, inverse=inverse
                )
                self.equal(actual, expected)
        self.assertTrue(torch.isfinite(values).all())

    def test_query_projection_has_weighted_lora_norm_and_no_head_norm(self):
        model = self.model(0)
        # Give a private head projection a non-unit Q norm; the shared fixture stays zero-Q.
        projection = nn.Linear(
            1280, 64 * 512, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        projection.weight.data.zero_()
        projection.weight.data[:, 0] = 0.25
        model.wq_b = projection
        model.q_norm.fill_(2)
        hidden = self.hidden(3)
        positions = torch.arange(3, device="cuda")
        qr, actual, _ = model._project(hidden, positions)
        normalized = self.official.RMSNorm(1280, 1e-20).cuda()
        normalized.weight.data.copy_(model.q_norm)
        expected_qr = normalized(model.wq_a(hidden))
        self.equal(qr, expected_qr)
        expected = projection(expected_qr).reshape(1, 3, 64, 512)
        with torch.device("cuda"):
            frequencies = self.official.precompute_freqs_cis(64, 3, 0, 10000, 16, 32, 1)
        self.official.apply_rotary_emb(expected[..., -64:], frequencies)
        self.equal(actual, expected[0])

    def test_same_shape_indices_cannot_cross_requests_or_epochs(self):
        owner = self.model(20)
        cache = self.cache()
        first = cache.begin_forward(epoch=0, start=0, end=3)
        owner(self.hidden(3), first)
        retained = first.selection_for(20)
        other_cache = self.cache(request="request-b")
        second = other_cache.begin_forward(epoch=0, start=0, end=3)
        owner(self.hidden(3), second)
        second.selections[20] = retained
        with self.assertRaisesRegex(ValueError, "stale.*identity"):
            second.indices_for(21)
        with self.assertRaisesRegex(ValueError, "stale.*identity"):
            second.tail(1, replay_floor=1)
        next_chunk = cache.begin_forward(epoch=1, start=3, end=6)
        owner(self.hidden(3), next_chunk)
        next_chunk.selections[20] = retained
        with self.assertRaisesRegex(ValueError, "stale.*identity"):
            next_chunk.indices_for(23)

    def test_selection_binding_checks_source_owner_and_tensor_geometry(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=3)
        self.model(20)(self.hidden(3), context)
        selected = context.selection_for(20)
        for replacement in (
            replace(selected, query_owner=24),
            replace(selected, key_owner=14),
            replace(selected, query_identity=None),
        ):
            context.selections[20] = replacement
            with self.assertRaisesRegex(ValueError, "stale.*identity"):
                context.indices_for(21)
        context.selections[20] = replace(selected, topk=selected.topk[:, :8])
        with self.assertRaisesRegex(ValueError, "query range"):
            context.indices_for(21)
        context.selections[20] = selected
        with self.assertRaisesRegex(ValueError, "already published"):
            context.publish_selection(selected)

    def test_tail_rebinds_owned_query_results_to_its_actual_row_range(self):
        cache = self.cache()
        context = cache.begin_forward(epoch=0, start=0, end=129)
        self.model(20)(self.hidden(129), context)
        late = context.tail(1, replay_floor=1)
        selected = late.selection_for(21)
        self.assertEqual(selected.query_identity, late.query_identity)
        self.assertNotEqual(selected.query_identity, context.query_identity)
        self.equal(selected.topk, context.selections[20].topk[1:])
        self.assertNotEqual(
            selected.topk.untyped_storage().data_ptr(),
            context.selections[20].topk.untyped_storage().data_ptr(),
        )
        late.selections[20] = replace(selected, query_identity=context.query_identity)
        with self.assertRaisesRegex(ValueError, "stale.*identity"):
            late.indices_for(21)


if __name__ == "__main__":
    unittest.main()

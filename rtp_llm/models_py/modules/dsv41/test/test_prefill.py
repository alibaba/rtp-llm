"""GPU stage/state probes using initialized attention and mHC, not model quality."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import test_attention as attention_fixture
import test_transformer as target_fixture
from fixture import flash_config
import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.attention import V41AttentionCache
from rtp_llm.models_py.modules.dsv41.block import V41Block
from rtp_llm.models_py.modules.dsv41.ced import (
    ReplayConfig,
    ReplayMode,
    RowRange,
    build_prefill_plan,
)
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.prefill import V41L20Tail, V41PrefillExecutor
from rtp_llm.models_py.modules.dsv41.transformer import V41ImageFeatures, V41TargetModel


class ZeroMoe(nn.Module):
    def forward(self, hidden, image_mask):
        return torch.zeros_like(hidden)


class PrefillGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        attention_fixture.AttentionGpuTest.setUpClass()
        cls.helper = attention_fixture.AttentionGpuTest()
        cls.helper.setUp()
        cls.config = SimpleNamespace(pad_token_id=2, text=flash_config()["text_config"])
        cls.layout = cls.helper.layout
        weights = {
            "attn_norm.weight": torch.ones(5120, device="cuda", dtype=torch.bfloat16),
            "ffn_norm.weight": torch.ones(5120, device="cuda", dtype=torch.bfloat16),
        }
        for part in ("attn", "ffn"):
            weights[f"hc_{part}_fn"] = torch.zeros(24, 20480, device="cuda")
            weights[f"hc_{part}_scale"] = torch.ones(3, device="cuda")
            weights[f"hc_{part}_base"] = torch.zeros(24, device="cuda")
        blocks = [
            V41Block(cls.helper.model(layer), ZeroMoe(), weights) for layer in range(40)
        ]
        gate = torch.ones(4, 5120, dtype=torch.bfloat16, device="cuda")
        engrams = {
            layer: Engram(
                layer,
                target_fixture.RecordLookup(),
                target_fixture.FixedProjection(5120),
                gate,
                gate,
            )
            for layer in (1, 14)
        }
        cls.target = V41TargetModel(
            cls.config,
            torch.ones(129280, 5120, dtype=torch.bfloat16, device="cuda"),
            weights["attn_norm.weight"],
            torch.empty(129280, 5120, device="cuda"),
            blocks,
            engrams,
            target_fixture.RecordHasher(),
        )

        def projection(inputs, outputs):
            layer = nn.Linear(
                inputs, outputs, bias=False, dtype=torch.bfloat16, device="cuda"
            )
            layer.weight.data.zero_()
            channel = torch.arange(outputs, device="cuda")
            layer.weight.data[channel, channel] = 1
            return layer

        cls.draft = V41PrefillDraftCommit(
            cls.config,
            projection(15360, 5120),
            weights["attn_norm.weight"],
            [projection(5120, 512) for _ in range(3)],
            [torch.ones(512, dtype=torch.bfloat16, device="cuda") for _ in range(3)],
        )
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            (Path(output) / "prefill_components.json").write_text(
                json.dumps(
                    {
                        "scope": "real CUDA initialized attention/mHC/draft stage and exact state checks; MoE and Engram are orchestration fixtures; no full-model, CP8, PD, 1M or release acceptance",
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "torch": str(torch.__version__),
                        "observations": cls.records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        attention_fixture.AttentionGpuTest.tearDownClass()

    def cache(self, tokens, mode=ReplayMode.BOUNDED, request="prefill-a"):
        identity = ReplayConfig(mode).cache_identity(
            "2bc89ac599031fa673cab993f1df02fc4a98c673", self.layout
        )
        return V41AttentionCache.allocate_local(
            request, identity, self.layout, tokens, device="cuda"
        )

    def plan(self, cache, tokens, chunk, *, restored=None, images=(), **kwargs):
        mode = (
            ReplayMode.FULL
            if cache.identity.replay_fingerprint
            == ReplayConfig(ReplayMode.FULL).fingerprint
            else ReplayMode.BOUNDED
        )
        return build_prefill_plan(
            total_tokens=tokens,
            chunk_tokens=chunk,
            layout=cache.layout,
            model_revision=cache.identity.model_revision,
            config=ReplayConfig(mode),
            restored_checkpoint=None if restored is None else restored.checkpoint,
            images=images,
            **kwargs,
        )

    def rows(self, start, end, image=None):
        count = end - start
        ids = torch.full((count,), 7, device="cuda", dtype=torch.int32)
        types = torch.full_like(ids, -1)
        if image is not None:
            lo, hi = max(start, image.start), min(end, image.end)
            if lo < hi:
                ids[lo - start : hi - start] = 129264
                types[lo - start : hi - start] = 1
        rows = V41ModelRows(
            ids,
            types,
            torch.ones(count, device="cuda", dtype=torch.bool),
            torch.zeros(count, 3, device="cuda", dtype=torch.int32),
            torch.zeros(count, 3, device="cuda", dtype=torch.bool),
        )
        selected = rows.image_mask.nonzero().flatten()
        features = V41ImageFeatures(
            selected,
            types[selected].contiguous(),
            torch.full(
                (selected.numel(), 5120), 2, device="cuda", dtype=torch.bfloat16
            ),
        )
        return rows, features

    def execute(self, cache, plan, *, restored=None, image=None):
        runner = V41PrefillExecutor(
            self.target, cache, plan, draft_commit=self.draft, restored=restored
        )
        if restored is not None:
            hashes = {}
            for layer, (expected, first, last) in restored.swa.items():
                binding = cache.swa[layer]
                actual = binding.pages.data.index_select(0, binding.page_ids.long())
                self.equal(actual, expected)
                self.assertEqual(
                    (int(binding.valid_starts[0]), int(binding.valid_ends[0])),
                    (first, last),
                )
                hashes[f"swa_{layer}"] = hashlib.sha256(
                    actual.cpu().numpy().tobytes()
                ).hexdigest()
            for layer, tensors in restored.owners.items():
                owner = cache.owners[layer]
                for region, expected, pages, table in (
                    (
                        "global",
                        tensors[0],
                        owner.global_kv.pages,
                        owner.global_kv.page_table,
                    ),
                    ("index", tensors[1], owner.index_pages, owner.index_table),
                ):
                    actual = pages.data.index_select(
                        0, table[0, : expected.shape[0]].long()
                    )
                    self.equal(actual, expected)
                    hashes[f"{region}_{layer}"] = hashlib.sha256(
                        actual.cpu().numpy().tobytes()
                    ).hexdigest()
            self.records.append(
                {
                    "test": self.id(),
                    "restored_region_hashes": hashes,
                    "checkpoint_end": restored.checkpoint.materialized_end,
                }
            )
        results = []
        for epoch, extend in enumerate(plan.extends):
            rows, features = self.rows(
                extend.encoder_rows.start, extend.encoder_rows.end, image
            )
            results.append(
                runner.run_extend(rows, epoch=epoch, image_features=features)
            )
        self.records.append({"test": self.id(), "extends": runner.observations})
        return runner, results

    def equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def remap(self, cache):
        for layer, binding in cache.swa.items():
            data = torch.zeros(
                (3, binding.pages.data.shape[1]), device="cuda", dtype=torch.uint8
            )
            cache.swa[layer] = dataclasses.replace(
                binding,
                pages=dataclasses.replace(binding.pages, data=data),
                page_ids=torch.full_like(binding.page_ids, 2),
            )
        for owner in cache.owners.values():
            owner.global_kv.page_table.copy_(owner.global_kv.page_table.flip(1))
            owner.index_table.copy_(owner.index_table.flip(1))

    def compare_cache(self, actual, expected):
        self.assertEqual(actual.swa_ends, expected.swa_ends)
        hashes = {}
        for layer in actual.swa:
            left, right = actual.swa[layer], expected.swa[layer]
            self.equal(left.valid_starts, right.valid_starts)
            self.equal(left.valid_ends, right.valid_ends)
            actual_bytes = left.pages.data.index_select(0, left.page_ids.long())
            expected_bytes = right.pages.data.index_select(0, right.page_ids.long())
            self.equal(actual_bytes, expected_bytes)
            hashes[f"swa_{layer}"] = hashlib.sha256(
                actual_bytes.cpu().numpy().tobytes()
            ).hexdigest()
        for layer in actual.owners:
            left, right = actual.owners[layer], expected.owners[layer]
            self.assertEqual(left.materialized_end, right.materialized_end)
            for region, pages_a, table_a, pages_b, table_b in (
                (
                    "global",
                    left.global_kv.pages,
                    left.global_kv.page_table,
                    right.global_kv.pages,
                    right.global_kv.page_table,
                ),
                (
                    "index",
                    left.index_pages,
                    left.index_table,
                    right.index_pages,
                    right.index_table,
                ),
            ):
                actual_bytes = pages_a.data.index_select(0, table_a[0].long())
                expected_bytes = pages_b.data.index_select(0, table_b[0].long())
                self.equal(actual_bytes, expected_bytes)
                hashes[f"{region}_{layer}"] = hashlib.sha256(
                    actual_bytes.cpu().numpy().tobytes()
                ).hexdigest()
            if layer != 20:
                self.assertEqual(left.pair.next_position, right.pair.next_position)
                for name in ("partial_kv", "partial_score"):
                    a, b = getattr(left.pair, name), getattr(right.pair, name)
                    if b is None:
                        self.assertIsNone(a)
                    else:
                        self.equal(a.view(torch.uint8), b.view(torch.uint8))
        return hashes

    @torch.inference_mode()
    def test_varied_suffix_restore_matches_uninterrupted_same_mode(self):
        token_ids = torch.arange(10, 27, device="cuda")
        original = self.target.embedding[token_ids].clone()
        channels = torch.arange(5120, device="cuda")
        self.target.embedding[token_ids] = (
            ((token_ids[:, None] + channels[None, :] * 3) % 11 + 1) * 0.125
        ).bfloat16()

        def varied_rows(start, end):
            positions = torch.arange(start, end, device="cuda", dtype=torch.int32)
            history = (
                positions[:, None]
                + torch.arange(-3, 0, device="cuda", dtype=torch.int32)[None, :]
            )
            return V41ModelRows(
                (positions * 7) % 17 + 10,
                torch.full_like(positions, -1),
                torch.ones_like(positions, dtype=torch.bool),
                (history * 7) % 17 + 10,
                history >= 0,
            )

        try:
            for suffix in (1, 3, 127, 128, 129):
                with self.subTest(suffix=suffix):
                    live_cache = self.cache(1024 + suffix, request="live")
                    live = V41PrefillExecutor(
                        self.target,
                        live_cache,
                        self.plan(live_cache, 1024 + suffix, 192),
                        draft_commit=self.draft,
                    )
                    while live.progress.encoder_materialized_end < 1024:
                        extend = live.plan.extends[live.next_extend]
                        live.run_extend(
                            varied_rows(
                                extend.encoder_rows.start, extend.encoder_rows.end
                            ),
                            epoch=live.next_extend,
                        )
                    checkpoint = live.protected
                    restored_cache = self.cache(1024 + suffix, request="restored")
                    self.remap(restored_cache)
                    restored = V41PrefillExecutor(
                        self.target,
                        restored_cache,
                        self.plan(
                            restored_cache, 1024 + suffix, 192, restored=checkpoint
                        ),
                        draft_commit=self.draft,
                        restored=checkpoint,
                    )
                    for runner in (live, restored):
                        extend = runner.plan.extends[runner.next_extend]
                        result = runner.run_extend(
                            varied_rows(
                                extend.encoder_rows.start, extend.encoder_rows.end
                            ),
                            epoch=runner.next_extend,
                        )
                        if runner is live:
                            expected = result
                        else:
                            actual = result
                    self.equal(
                        actual.output.hidden_states, expected.output.hidden_states
                    )
                    self.equal(
                        actual.output.aux_hidden_states,
                        expected.output.aux_hidden_states,
                    )
                    self.equal(
                        actual.output.final_pre_mix.view(torch.uint8),
                        expected.output.final_pre_mix.view(torch.uint8),
                    )
                    self.assertEqual(
                        actual.aux_rows.positions, expected.aux_rows.positions
                    )
                    self.assertEqual(
                        actual.aux_rows.image_mask, expected.aux_rows.image_mask
                    )
                    hashes = self.compare_cache(restored_cache, live_cache)
                    self.records.append(
                        {
                            "test": self.id(),
                            "suffix": suffix,
                            "comparison": "uninterrupted versus remapped restored, same bounded policy and varying token/embedding input",
                            "final_region_hashes": hashes,
                            "output_sha256": hashlib.sha256(
                                actual.output.hidden_states.view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                            ).hexdigest(),
                        }
                    )
        finally:
            self.target.embedding[token_ids] = original

    @torch.inference_mode()
    def test_short_full_and_bounded_execute_same_complete_math(self):
        outputs = []
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            cache = self.cache(9, mode)
            runner, results = self.execute(cache, self.plan(cache, 9, 9))
            outputs.append(results[-1].output)
            self.assertEqual(set(cache.swa_ends), set(range(43)))
            self.assertEqual(set(cache.swa_ends.values()), {9})
            self.assertEqual(
                runner.observations[0]["draft_rows"]["stage_projection_rows"], (9, 9, 9)
            )
            layers = (
                runner.observations[0]["encoder_layers"]
                + runner.observations[0]["decoder_layers"]
            )
            self.assertEqual([entry["layer"] for entry in layers], list(range(40)))
            self.assertEqual(
                [entry["layer"] for entry in layers if entry["source_rows"]],
                [2, 8, 14, 20],
            )
            self.assertEqual(
                [entry["layer"] for entry in layers if entry["index_rows"]],
                [2, 8, 14, 20, 24, 28, 32, 36],
            )
        self.equal(outputs[0].hidden_states, outputs[1].hidden_states)
        self.equal(outputs[0].aux_hidden_states, outputs[1].aux_hidden_states)

    @torch.inference_mode()
    def test_cross_chunk_tail_stage_gating_protected_n_and_remapped_restore(self):
        image = RowRange(940, 944)
        cache = self.cache(1057)
        plan = self.plan(cache, 1057, 192, images=(image,))
        runner, results = self.execute(cache, plan, image=image)
        self.assertEqual(runner.progress.protected_checkpoint_end, 1024)
        self.assertEqual(runner.progress.decoder_checkpoint_end, 1057)
        for observation in runner.observations[:-2]:
            self.assertIsNone(observation["decoder_range"])
            self.assertEqual(observation["decoder_layers"], [])
            self.assertIsNone(observation["draft_rows"])
        self.assertEqual(
            [o["decoder_range"] for o in runner.observations[-2:]],
            [(896, 1024), (1024, 1057)],
        )
        self.assertEqual(
            runner.observations[-2]["draft_rows"]["main_projection_rows"], 128
        )
        self.assertEqual(
            runner.observations[-1]["draft_rows"]["main_projection_rows"], 33
        )
        self.assertEqual(results[-2].aux_rows.image_mask[44:48], (True,) * 4)
        protected = runner.protected
        self.assertEqual(protected.checkpoint.materialized_end, 1024)
        saved = protected.swa[42][0].clone()
        cache.swa[42].pages.data.zero_()
        self.equal(protected.swa[42][0], saved)
        for suffix in (3, 129):
            fresh = self.cache(1024 + suffix, request="prefix-reuser")
            self.remap(fresh)
            prefix_plan = self.plan(fresh, 1024 + suffix, 64, restored=protected)
            resumed, _ = self.execute(fresh, prefix_plan, restored=protected)
            expected_start = 1024 if suffix <= 128 else 1024 + suffix - 128
            self.assertEqual(
                resumed.observations[-1]["decoder_range"],
                (expected_start, 1024 + suffix),
            )
            for layer, binding in fresh.swa.items():
                self.assertEqual(int(binding.valid_ends[0]), 1024 + suffix)
                if suffix == 3:
                    self.assertLess(int(binding.valid_starts[0]), 1024)
            self.equal(protected.swa[42][0], saved)
        wrong = self.cache(1057, ReplayMode.FULL)
        with self.assertRaisesRegex(ValueError, "different policy"):
            protected.restore(wrong)

    @torch.inference_mode()
    def test_tail_has_bounded_owned_storage_and_rejects_stale_epoch(self):
        cache = self.cache(513)
        prior = None
        for epoch, (start, end) in enumerate(((0, 127), (127, 385))):
            rows, image = self.rows(start, end)
            context = cache.begin_forward(epoch=epoch, start=start, end=end)
            l20 = self.target.prefill_encoder(rows, context, image_features=image)
            prior = V41L20Tail.append(prior, l20, context)
            expected = l20.hidden_states[-min(128, end - start) :].clone()
            l20.hidden_states.zero_()
            context.selections[20].topk.fill_(-1)
            self.equal(prior.l20.hidden_states[-expected.shape[0] :], expected)
            self.assertEqual(
                prior.l20.hidden_states.untyped_storage().nbytes(),
                prior.l20.hidden_states.numel() * 2,
            )
            self.assertLess(prior.storage_bytes, 7 * 1024 * 1024)
        late, context = prior.select(cache, RowRange(257, 385), 257)
        self.assertEqual(late.rows.token_ids.numel(), 128)
        cache.begin_forward(epoch=2, start=385, end=386)
        with self.assertRaisesRegex(ValueError, "stale"):
            prior.select(cache, RowRange(257, 385), 257)
        with self.assertRaisesRegex(ValueError, "stale"):
            context.indices_for(21)

    @torch.inference_mode()
    def test_full_diagnostic_decode_still_runs_every_layer(self):
        cache = self.cache(8, ReplayMode.FULL)
        rows, features = self.rows(0, 5)
        context = cache.begin_forward(epoch=0, start=0, end=5)
        self.target(rows, context, execution_mode="full", image_features=features)
        rows, features = self.rows(5, 8)
        context = cache.begin_forward(epoch=1, start=5, end=8)
        result = self.target(
            rows, context, execution_mode="full", image_features=features
        )
        self.assertEqual(result.hidden_states.shape, (3, 5120))
        self.assertEqual(context.completed_layers, set(range(40)))
        self.assertTrue(all(o["query_rows"] == 3 for o in context.observations))
        self.records.append(
            {"test": self.id(), "target_decode_layers": context.observations}
        )

    @torch.inference_mode()
    def test_late_failure_poison_prevents_partial_handoff(self):
        class FailingDraft:
            def commit(self, *args, **kwargs):
                raise RuntimeError("injected draft write failure")

        cache = self.cache(3)
        runner = V41PrefillExecutor(
            self.target, cache, self.plan(cache, 3, 3), draft_commit=FailingDraft()
        )
        rows, features = self.rows(0, 3)
        with self.assertRaisesRegex(RuntimeError, "injected draft"):
            runner.run_extend(rows, epoch=0, image_features=features)
        self.assertTrue(cache.poisoned)
        self.assertEqual(runner.progress.decoder_checkpoint_end, 0)
        with self.assertRaisesRegex(RuntimeError, "discarded or restored"):
            cache.begin_forward(epoch=1, start=0, end=3)


if __name__ == "__main__":
    unittest.main()

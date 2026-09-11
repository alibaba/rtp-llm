"""GPU stage/state probes using initialized attention and mHC, not model quality."""

import dataclasses
import hashlib
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_attention as attention_fixture
import test_transformer as target_fixture
import torch
from fixture import flash_config
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    V41ImageInput,
    V41PreparedInputs,
    image_token_types,
)
from rtp_llm.models_py.modules.dsv41.attention import V41AttentionCache
from rtp_llm.models_py.modules.dsv41.block import V41Block
from rtp_llm.models_py.modules.dsv41.ced import (
    ReplayConfig,
    ReplayMode,
    RowRange,
    build_prefill_plan,
)
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.prefill import (
    V41L20Tail,
    V41LocalSnapshot,
    V41PrefillExecutor,
)
from rtp_llm.models_py.modules.dsv41.transformer import V41ImageFeatures, V41TargetModel
from torch import nn


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

    def prepared(self, tokens=1057, image_starts=(392, 1000)):
        ids = [7 + pos % 5 for pos in range(tokens)]
        types = [-1] * tokens
        images = []
        for start in image_starts:
            kinds = image_token_types(1, 1)
            ids[start : start + len(kinds)] = [129264] * len(kinds)
            types[start : start + len(kinds)] = kinds.tolist()
            images.append(
                V41ImageInput(
                    start,
                    torch.ones((9, 3, 14, 14), dtype=torch.bfloat16),
                    3,
                    3,
                    kinds,
                    "a" * 64,
                    "b" * 64,
                )
            )
        return V41PreparedInputs("", tuple(ids), tuple(types), tuple(images))

    @torch.inference_mode()
    def test_canonical_request_maps_later_images_and_history_before_ced_tail(self):
        class RecordingVision(nn.Module):
            def __init__(self):
                super().__init__()
                self.processor_config = SimpleNamespace(image_token_id=129264)
                self._device = torch.device("cuda")
                self.calls = []

            def encode_image(self, image):
                self.calls.append(image.start)
                return torch.full(
                    (image.length, 5120),
                    2 + len(self.calls),
                    dtype=torch.bfloat16,
                    device=self._device,
                )

        prepared = self.prepared()
        vision, previous = RecordingVision(), self.target.vision
        self.target.vision = vision
        try:
            runner = V41PrefillExecutor.from_prepared(
                self.target,
                self.cache(len(prepared.token_ids)),
                prepared,
                config=ReplayConfig(ReplayMode.BOUNDED),
                chunk_tokens=192,
                draft_commit=self.draft,
            )
            self.assertEqual(vision.calls, [392, 1000])
            for epoch, extend in enumerate(runner.plan.extends):
                start, end = extend.encoder_rows.start, extend.encoder_rows.end
                rows = runner.canonical.rows(start, end, device="cuda")
                expected_history, expected_valid = [], []
                for pos in range(start, end):
                    prior = list(range(pos - 3, pos))
                    expected_history.append(
                        [0 if p < 0 else prepared.token_ids[p] for p in prior]
                    )
                    expected_valid.append(
                        [p >= 0 and prepared.token_types[p] == -1 for p in prior]
                    )
                self.equal(
                    rows.history_ids,
                    torch.tensor(expected_history, dtype=torch.int32, device="cuda"),
                )
                self.equal(
                    rows.history_valid,
                    torch.tensor(expected_valid, dtype=torch.bool, device="cuda"),
                )
                features = runner.image_features.for_extend(start, end)
                features.validate(rows, 5120)
                hidden, _ = self.target._embed_rows(rows, features)
                for index, image in enumerate(prepared.images):
                    if start <= image.start < end:
                        offset = image.start - start
                        self.equal(
                            hidden[offset : offset + image.length],
                            torch.full_like(
                                hidden[offset : offset + image.length], 3 + index
                            ),
                        )
                runner.run_next(epoch=epoch)
            self.assertEqual(vision.calls, [392, 1000])
            self.assertEqual(runner.progress.decoder_checkpoint_end, 1057)
            self.assertEqual(runner.protected.checkpoint.materialized_end, 1024)
            self.assertEqual(
                runner.protected.canonical_prefix_sha256,
                runner.canonical.prefix_fingerprint(1024),
            )
            other_image = dataclasses.replace(
                prepared.images[0], content_sha256="c" * 64
            )
            changed_image = dataclasses.replace(
                prepared, images=(other_image, prepared.images[1])
            )
            with self.assertRaisesRegex(ValueError, "different canonical prefix"):
                V41PrefillExecutor.from_prepared(
                    self.target,
                    self.cache(1057),
                    changed_image,
                    config=ReplayConfig(ReplayMode.BOUNDED),
                    chunk_tokens=192,
                    draft_commit=self.draft,
                    restored=runner.protected,
                )
            tail_image = runner.tail.l20.rows.image_mask.nonzero().flatten()
            self.equal(
                tail_image + runner.tail.positions.start,
                torch.arange(1000, 1004, device="cuda"),
            )
            self.records.append(
                {
                    "test": self.id(),
                    "scope": "initialized LM execution with recorded vision fixture",
                    "image_encode_calls": vision.calls,
                    "checkpoint_prefix_sha256": runner.protected.canonical_prefix_sha256,
                    "extends": runner.observations,
                }
            )
        finally:
            self.target.vision = previous

    @torch.inference_mode()
    def test_canonical_checkpoint_rejects_changed_prefix_before_restore(self):
        prepared = self.prepared(image_starts=())
        runner = V41PrefillExecutor.from_prepared(
            self.target,
            self.cache(1057),
            prepared,
            config=ReplayConfig(ReplayMode.BOUNDED),
            chunk_tokens=192,
            draft_commit=self.draft,
        )
        first = runner.plan.extends[0].encoder_rows
        wrong_rows = runner.canonical.rows(first.start, first.end, device="cuda")
        wrong_rows.token_ids[0] += 1
        with self.assertRaisesRegex(ValueError, "disagree with the canonical"):
            runner.run_extend(wrong_rows, epoch=0)
        self.assertEqual(runner.next_extend, 0)
        self.assertEqual(runner.cache.active_epoch, -1)
        for epoch in range(len(runner.plan.extends)):
            final_result = runner.run_next(epoch=epoch)
        restored = runner.protected
        changed = list(prepared.token_ids)
        changed[10] += 1
        other = dataclasses.replace(prepared, token_ids=tuple(changed))
        destination = self.cache(1057, request="other")
        with self.assertRaisesRegex(ValueError, "different canonical prefix"):
            V41PrefillExecutor.from_prepared(
                self.target,
                destination,
                other,
                config=ReplayConfig(ReplayMode.BOUNDED),
                chunk_tokens=192,
                draft_commit=self.draft,
                restored=restored,
            )
        self.assertEqual(destination.swa_ends, {})
        self.assertFalse(destination.poisoned)
        resumed = V41PrefillExecutor.from_prepared(
            self.target,
            destination,
            prepared,
            config=ReplayConfig(ReplayMode.BOUNDED),
            chunk_tokens=192,
            draft_commit=self.draft,
            restored=restored,
        )
        output = resumed.run_next(epoch=0)
        self.assertEqual(output.context.start, 1024)
        self.assertEqual(output.context.end, 1057)
        self.assertEqual(output.context.replay_floor, 896)
        self.equal(output.output.hidden_states, final_result.output.hidden_states)
        self.equal(
            output.output.aux_hidden_states, final_result.output.aux_hidden_states
        )
        self.records.append(
            {"test": self.id(), "resumed_extends": resumed.observations}
        )

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

    def snapshot_payload_fixture(self, mode=ReplayMode.BOUNDED):
        end = self.layout.reuse_unit
        source = self.cache(end + 1, mode)
        floor = end - 128 if mode == ReplayMode.BOUNDED else 0
        for layer, binding in source.swa.items():
            binding.pages.data[1:].fill_(layer + 1)
            first = (
                end - source.layout.swa_entries
                if mode == ReplayMode.BOUNDED and layer <= 20
                else end - 128
            )
            binding.valid_starts.fill_(first)
            binding.valid_ends.fill_(end)
            source.swa_ends[layer] = end
        for layer, owner in source.owners.items():
            owner.global_kv.pages.data[1:].fill_(layer * 3 + 1)
            owner.index_pages.data[1:].fill_(layer * 3 + 2)
            owner.materialized_end = end
            if owner.global_kv.compress_ratio == 2:
                owner.pair = PairCarry.empty(
                    layer, source.request_id, source.identity, end
                )
        rows, _ = self.rows(end - 3, end)
        snapshot = V41LocalSnapshot.protect(
            source, end=end, replay_floor=floor, history_rows=rows
        )
        return source, snapshot

    def assert_unpublished_restore(self, cache):
        self.assertEqual(cache.swa_ends, {})
        self.assertEqual(
            {owner.materialized_end for owner in cache.owners.values()}, {0}
        )
        self.assertTrue(all(owner.pair is None for owner in cache.owners.values()))

    @torch.inference_mode()
    def test_snapshot_protect_rejects_unmapped_and_aliased_source_pages(self):
        rejected = []
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            source, snapshot = self.snapshot_payload_fixture(mode)
            for region in ("swa", "global", "index"):
                if region == "swa":
                    pages, ids = source.swa[42].pages, source.swa[42].page_ids
                elif region == "global":
                    owner = source.owners[20]
                    pages, ids = owner.global_kv.pages, owner.global_kv.page_table[0]
                else:
                    owner = source.owners[20]
                    pages, ids = owner.index_pages, owner.index_table[0]
                original = ids.clone()
                original_payload = pages.data.clone()
                for problem in ("missing", "negative", "out-of-range", "duplicate"):
                    if region == "swa" and problem == "duplicate":
                        continue
                    with self.subTest(mode=mode.value, region=region, problem=problem):
                        if problem == "missing":
                            ids[0] = 0
                        elif problem == "negative":
                            ids[0] = -1
                        elif problem == "out-of-range":
                            ids[0] = pages.data.shape[0]
                        else:
                            ids[1] = ids[0]
                        with mock.patch.object(
                            torch.Tensor,
                            "index_select",
                            side_effect=AssertionError(
                                "copied before complete validation"
                            ),
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "missing or aliased pages"
                            ):
                                V41LocalSnapshot.protect(
                                    source,
                                    end=snapshot.checkpoint.materialized_end,
                                    replay_floor=snapshot.checkpoint.replay_floor,
                                    history_rows=snapshot.history_rows,
                                )
                        self.equal(pages.data, original_payload)
                        self.assertFalse(source.poisoned)
                        rejected.append(f"{mode.value}/{region}/{problem}")
                        ids.copy_(original)
        self.records.append({"test": self.id(), "rejected_before_copy": rejected})

    @torch.inference_mode()
    def test_snapshot_requires_complete_nonpadding_canonical_tail_history(self):
        source, snapshot = self.snapshot_payload_fixture()
        for count in (0, 1, 2, 3):
            rows = V41ModelRows(
                *(
                    getattr(snapshot.history_rows, field.name)[:count].clone()
                    for field in dataclasses.fields(V41ModelRows)
                )
            )
            if count == 3:
                rows.valid[-1] = False
                rows.history_valid[-1].zero_()
            with self.subTest(history_rows=count):
                with self.assertRaisesRegex(ValueError, "canonical tail history"):
                    V41LocalSnapshot.protect(
                        source,
                        end=snapshot.checkpoint.materialized_end,
                        replay_floor=snapshot.checkpoint.replay_floor,
                        history_rows=rows,
                    )
                destination = self.cache(source.max_tokens, request="receiver")
                with self.assertRaisesRegex(ValueError, "canonical tail history"):
                    dataclasses.replace(snapshot, history_rows=rows).restore(
                        destination
                    )
                self.assert_unpublished_restore(destination)
                self.assertFalse(destination.poisoned)

    def shifted_swa_alias(self, cache):
        first, second = cache.swa[0], cache.swa[42]
        width = first.pages.data.shape[1]
        backing = torch.zeros((4, width), dtype=torch.uint8, device="cuda")
        backing[2].copy_(first.pages.data[1])
        cache.swa[0] = dataclasses.replace(
            first,
            pages=dataclasses.replace(first.pages, data=backing[:3]),
            page_ids=torch.full_like(first.page_ids, 2),
        )
        cache.swa[42] = dataclasses.replace(
            second, pages=dataclasses.replace(second.pages, data=backing[1:])
        )
        self.assertNotEqual(
            cache.swa[0].pages.data.data_ptr(), cache.swa[42].pages.data.data_ptr()
        )

    @torch.inference_mode()
    def test_snapshot_protect_rejects_shifted_cross_region_physical_aliases(self):
        source, snapshot = self.snapshot_payload_fixture()
        self.shifted_swa_alias(source)
        with self.assertRaisesRegex(ValueError, "aliases.*cache region"):
            V41LocalSnapshot.protect(
                source,
                end=snapshot.checkpoint.materialized_end,
                replay_floor=snapshot.checkpoint.replay_floor,
                history_rows=snapshot.history_rows,
            )
        self.assertFalse(source.poisoned)

    @torch.inference_mode()
    def test_snapshot_restore_rejects_cross_region_payload_and_metadata_aliases(self):
        source, snapshot = self.snapshot_payload_fixture()
        problems = (
            "target-draft-page",
            "shifted-target-draft-page",
            "mutable-bounds",
            "bound-page-map",
            "retained-swa",
            "retained-history",
        )
        for problem in problems:
            with self.subTest(problem=problem):
                destination = self.cache(source.max_tokens, request="receiver")
                first, last = destination.swa[0], destination.swa[42]
                candidate = snapshot
                if problem == "target-draft-page":
                    destination.swa[42] = dataclasses.replace(last, pages=first.pages)
                elif problem == "shifted-target-draft-page":
                    self.shifted_swa_alias(destination)
                elif problem == "mutable-bounds":
                    destination.swa[42] = dataclasses.replace(
                        last, valid_ends=first.valid_ends
                    )
                elif problem == "bound-page-map":
                    destination.swa[42] = dataclasses.replace(
                        last, valid_ends=first.page_ids
                    )
                elif problem == "retained-swa":
                    _, start, end = snapshot.swa[0]
                    candidate = dataclasses.replace(
                        snapshot,
                        swa={**snapshot.swa, 0: (last.pages.data[1:2], start, end)},
                    )
                else:
                    candidate = dataclasses.replace(
                        snapshot,
                        history_rows=dataclasses.replace(
                            snapshot.history_rows,
                            token_ids=last.pages.data[1, :12].view(torch.int32),
                        ),
                    )
                with mock.patch.object(
                    torch.Tensor,
                    "index_copy_",
                    side_effect=AssertionError("wrote before complete validation"),
                ):
                    with self.assertRaisesRegex(ValueError, "snapshot copy aliases"):
                        candidate.restore(destination)
                self.assert_unpublished_restore(destination)
                self.assertFalse(destination.poisoned)
                for binding in destination.swa.values():
                    self.assertEqual(int(torch.count_nonzero(binding.pages.data)), 0)
        self.records.append({"test": self.id(), "rejected_before_copy": list(problems)})

    @torch.inference_mode()
    def test_snapshot_accepts_disjoint_regions_in_one_allocation(self):
        source, snapshot = self.snapshot_payload_fixture()

        def pack(cache):
            first, last = cache.swa[0], cache.swa[42]
            joined = torch.cat((first.pages.data, last.pages.data), dim=0)
            cache.swa[0] = dataclasses.replace(
                first, pages=dataclasses.replace(first.pages, data=joined[:2])
            )
            cache.swa[42] = dataclasses.replace(
                last, pages=dataclasses.replace(last.pages, data=joined[2:])
            )
            self.assertEqual(
                cache.swa[0].pages.data.untyped_storage().data_ptr(),
                cache.swa[42].pages.data.untyped_storage().data_ptr(),
            )

        pack(source)
        protected = V41LocalSnapshot.protect(
            source,
            end=snapshot.checkpoint.materialized_end,
            replay_floor=snapshot.checkpoint.replay_floor,
            history_rows=snapshot.history_rows,
        )
        destination = self.cache(source.max_tokens, request="receiver")
        pack(destination)
        protected.restore(destination)
        for layer in (0, 42):
            self.equal(destination.swa[layer].pages.data[1:2], snapshot.swa[layer][0])
            source.swa[layer].pages.data.zero_()
            destination.swa[layer].pages.data.zero_()
            self.equal(protected.swa[layer][0], snapshot.swa[layer][0])
        self.records.append(
            {
                "test": self.id(),
                "same_allocation_disjoint_pages": True,
                "protected_snapshot_survives_source_and_restored_writes": True,
            }
        )

    @torch.inference_mode()
    def test_snapshot_rejects_missing_actual_payload_before_any_copy(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            source, snapshot = self.snapshot_payload_fixture(mode)
            corruptions = []
            for owner in (2, 20):
                for region in (0, 1):
                    for missing in (1, snapshot.owners[owner][region].shape[0]):
                        values = list(snapshot.owners[owner])
                        values[region] = values[region][:-missing]
                        owners = {**snapshot.owners, owner: tuple(values)}
                        corruptions.append(
                            (
                                f"owner{owner}-region{region}-missing{missing}",
                                dataclasses.replace(snapshot, owners=owners),
                            )
                        )
            for layer in (0, 42):
                data, first, end = snapshot.swa[layer]
                for name, payload in (
                    ("missing-ring", (data[:0], first, end)),
                    ("short-stride", (data[:, :-1], first, end)),
                    ("missing-history", (data, end - 127, end)),
                    ("old-end", (data, first, end - 1)),
                ):
                    corruptions.append(
                        (
                            f"swa{layer}-{name}",
                            dataclasses.replace(
                                snapshot, swa={**snapshot.swa, layer: payload}
                            ),
                        )
                    )
            for name in ("target_swa_start", "draft_swa_start"):
                checkpoint = dataclasses.replace(
                    snapshot.checkpoint,
                    replay_floor=0,
                    **{name: snapshot.checkpoint.materialized_end - 129},
                )
                self.assertTrue(checkpoint.is_complete(source.layout, source.identity))
                corruptions.append(
                    (name, dataclasses.replace(snapshot, checkpoint=checkpoint))
                )
            for name, corrupted in corruptions:
                with self.subTest(mode=mode.value, corruption=name):
                    destination = self.cache(
                        source.max_tokens, mode, request="receiver"
                    )
                    with self.assertRaisesRegex(
                        ValueError, "snapshot.*(payload|ranges)"
                    ):
                        corrupted.restore(destination)
                    self.assert_unpublished_restore(destination)
                    self.assertFalse(destination.poisoned)
                    for binding in destination.swa.values():
                        self.assertEqual(
                            int(torch.count_nonzero(binding.pages.data)), 0
                        )
                        self.assertEqual(int(binding.valid_starts[0]), 0)
                        self.assertEqual(int(binding.valid_ends[0]), 0)
                    for owner in destination.owners.values():
                        self.assertEqual(
                            int(torch.count_nonzero(owner.global_kv.pages.data)), 0
                        )
                        self.assertEqual(
                            int(torch.count_nonzero(owner.index_pages.data)), 0
                        )
            self.records.append(
                {
                    "test": self.id(),
                    "scope": "actual CUDA initialized payload copy validation, not model quality",
                    "mode": mode.value,
                    "rejected_corruptions": [name for name, _ in corruptions],
                }
            )

    @torch.inference_mode()
    def test_snapshot_rejects_unmapped_and_aliased_destination_pages(self):
        source, snapshot = self.snapshot_payload_fixture()
        for region in ("swa", "global", "index"):
            for problem in ("missing", "out-of-range", "duplicate"):
                if region == "swa" and problem == "duplicate":
                    continue
                with self.subTest(region=region, problem=problem):
                    destination = self.cache(source.max_tokens, request="receiver")
                    if region == "swa":
                        pages = destination.swa[42].pages
                        ids = destination.swa[42].page_ids
                    elif region == "global":
                        owner = destination.owners[20]
                        pages, ids = (
                            owner.global_kv.pages,
                            owner.global_kv.page_table[0],
                        )
                    else:
                        owner = destination.owners[20]
                        pages, ids = owner.index_pages, owner.index_table[0]
                    if problem == "missing":
                        ids[0] = 0
                    elif problem == "out-of-range":
                        ids[0] = pages.data.shape[0]
                    else:
                        ids[1] = ids[0]
                    with self.assertRaisesRegex(ValueError, "missing or aliased pages"):
                        snapshot.restore(destination)
                    self.assert_unpublished_restore(destination)
                    self.assertFalse(destination.poisoned)
                    for binding in destination.swa.values():
                        self.assertEqual(
                            int(torch.count_nonzero(binding.pages.data)), 0
                        )

    @torch.inference_mode()
    def test_snapshot_copy_failure_never_publishes_partial_owner_state(self):
        source, snapshot = self.snapshot_payload_fixture()
        destination = self.cache(source.max_tokens, request="receiver")
        copy = torch.Tensor.index_copy_
        copied = []

        def fail_after_first_owner(tensor, *args, **kwargs):
            if len(copied) == len(snapshot.swa) + 2:
                raise RuntimeError("injected local checkpoint copy failure")
            copied.append(tensor)
            return copy(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "index_copy_", fail_after_first_owner):
            with self.assertRaisesRegex(RuntimeError, "injected local checkpoint"):
                snapshot.restore(destination)
        self.assertEqual(len(copied), 45)
        self.assertTrue(destination.poisoned)
        self.assert_unpublished_restore(destination)
        self.assertGreater(int(torch.count_nonzero(destination.swa[0].pages.data)), 0)
        with self.assertRaisesRegex(RuntimeError, "discarded or restored"):
            destination.begin_forward(
                epoch=0, start=source.layout.reuse_unit, end=source.max_tokens
            )
        fresh = self.cache(source.max_tokens, request="retry")
        snapshot.restore(fresh)
        self.assertEqual(
            set(fresh.swa_ends.values()), {snapshot.checkpoint.materialized_end}
        )
        self.assertEqual(
            {owner.materialized_end for owner in fresh.owners.values()},
            {snapshot.checkpoint.materialized_end},
        )

    @torch.inference_mode()
    def test_snapshot_restore_on_copy_stream_is_complete_before_return(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            with self.subTest(mode=mode.value):
                source, snapshot = self.snapshot_payload_fixture(mode)
                if mode == ReplayMode.BOUNDED:
                    end = snapshot.checkpoint.materialized_end
                    self.assertEqual(snapshot.checkpoint.replay_floor, end - 128)
                    self.assertEqual(
                        snapshot.swa[0][1], end - source.layout.swa_entries
                    )
                    self.assertEqual(snapshot.swa[21][1], end - 128)
                    self.assertEqual(snapshot.swa[42][1], end - 128)
                    self.assertLess(
                        snapshot.swa[0][1], snapshot.checkpoint.replay_floor
                    )
                destination = self.cache(source.max_tokens, mode, request="receiver")
                self.remap(destination)
                ready = torch.cuda.Event()
                ready.record()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    stream.wait_event(ready)
                    snapshot.restore(destination)
                self.assertTrue(stream.query())
                hashes = {}
                for layer, (expected, first, end) in snapshot.swa.items():
                    binding = destination.swa[layer]
                    actual = binding.pages.data.index_select(0, binding.page_ids.long())
                    self.equal(actual, expected)
                    self.assertEqual(
                        (int(binding.valid_starts[0]), int(binding.valid_ends[0])),
                        (first, end),
                    )
                    hashes[f"swa_{layer}"] = hashlib.sha256(
                        actual.cpu().numpy().tobytes()
                    ).hexdigest()
                for layer, payloads in snapshot.owners.items():
                    owner = destination.owners[layer]
                    for name, expected, pages, table in (
                        (
                            "global",
                            payloads[0],
                            owner.global_kv.pages,
                            owner.global_kv.page_table,
                        ),
                        ("index", payloads[1], owner.index_pages, owner.index_table),
                    ):
                        actual = pages.data.index_select(
                            0, table[0, : expected.shape[0]].long()
                        )
                        self.equal(actual, expected)
                        hashes[f"{name}_{layer}"] = hashlib.sha256(
                            actual.cpu().numpy().tobytes()
                        ).hexdigest()
                self.records.append(
                    {
                        "test": self.id(),
                        "mode": mode.value,
                        "restored_region_hashes": hashes,
                    }
                )

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

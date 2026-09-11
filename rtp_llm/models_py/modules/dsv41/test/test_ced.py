import unittest
from dataclasses import replace

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import CacheLayout, MemoryCheckpoint
from rtp_llm.models_py.modules.dsv41.ced import (
    AuxRowMap,
    L20TailMetadata,
    L20TailRow,
    LateCompletion,
    PrefillProgress,
    ReplayConfig,
    ReplayMode,
    RowRange,
    build_prefill_plan,
    checkpoint_boundary,
    local_tail_indices,
    select_aux_hidden,
)


class PrefillPlanTest(unittest.TestCase):
    def setUp(self):
        self.layout = CacheLayout()
        self.bounded = ReplayConfig(ReplayMode.BOUNDED)

    def plan(self, total=8193, chunk=2048, **options):
        return build_prefill_plan(
            total_tokens=total,
            chunk_tokens=chunk,
            layout=self.layout,
            model_revision="frozen-model",
            config=options.pop("config", self.bounded),
            **options,
        )

    def complete_checkpoint(self, end, config=None):
        identity = (config or self.bounded).cache_identity("frozen-model", self.layout)
        return MemoryCheckpoint(
            end,
            identity,
            self.layout.required_slots,
            end - 128,
            end - 128,
            end - 128,
            history_ready=True,
            copy_complete=True,
            backing_protected=True,
        )

    def test_intermediate_chunks_skip_all_late_layers(self):
        plan = self.plan()
        self.assertEqual(len(plan.extends), 5)
        for extend in plan.extends[:3]:
            self.assertFalse(extend.needs_decoder_ep)
            self.assertIsNone(extend.query_rows(21))
            self.assertIsNone(extend.query_rows(39))
            self.assertEqual(len(extend.query_rows(20)), 2048)
        self.assertEqual(plan.extends[3].decoder_rows, RowRange(8064, 8192))
        self.assertEqual(plan.extends[4].decoder_rows, RowRange(8192, 8193))

    def test_exactly_four_sources_and_eight_scorers_receive_work(self):
        checkpoint = self.plan().extends[-2]
        sources = {
            layer: len(checkpoint.global_projection_rows(layer))
            for layer in range(40)
            if checkpoint.global_projection_rows(layer) is not None
        }
        queries = {
            layer: len(checkpoint.index_query_rows(layer))
            for layer in range(40)
            if checkpoint.index_query_rows(layer) is not None
        }
        self.assertEqual(sources, {2: 2048, 8: 2048, 14: 2048, 20: 2048})
        self.assertEqual(
            queries,
            {2: 2048, 8: 2048, 14: 2048, 20: 2048, 24: 128, 28: 128, 32: 128, 36: 128},
        )

    def test_full_keeps_all_forty_layers_on_every_extend(self):
        plan = self.plan(config=ReplayConfig())
        for extend in plan.extends:
            for layer in range(40):
                self.assertEqual(extend.query_rows(layer), extend.encoder_rows)
        self.assertIsNone(plan.fallback_reason)

    def test_real_one_million_boundary_has_finite_tail_work(self):
        plan = self.plan(total=1048320, chunk=8192)
        self.assertEqual(plan.extends[-1].encoder_rows.end + 256, 1048576)
        self.assertEqual(sum(len(x.encoder_rows) for x in plan.extends), 1048320)
        late = [x for x in plan.extends if x.decoder_rows is not None]
        self.assertEqual(len(late), 2)
        self.assertEqual([len(x.decoder_rows) for x in late], [128, 128])
        self.assertEqual(plan.protected_checkpoint_end, 1047552)
        self.assertTrue(all(len(x.retain_l20_rows) <= 128 for x in plan.extends))

    def test_protected_boundary_is_an_actual_extend_before_suffix(self):
        plan = self.plan(total=4500, chunk=3000)
        self.assertEqual(
            [x.encoder_rows for x in plan.extends],
            [RowRange(0, 3000), RowRange(3000, 4096), RowRange(4096, 4500)],
        )
        self.assertEqual(plan.extends[1].checkpoint_end, 4096)
        self.assertEqual(plan.extends[-1].required_protected_end, 4096)
        self.assertFalse(plan.extends[-1].uses_checkpoint_history)

    def test_restored_short_suffix_keeps_old_swa_history(self):
        plan = self.plan(
            total=4224, chunk=64, restored_checkpoint=self.complete_checkpoint(4096)
        )
        self.assertIsNone(plan.extends[0].decoder_rows)
        final = plan.extends[-1]
        self.assertEqual(final.decoder_rows, RowRange(4096, 4224))
        self.assertEqual(final.replay_floor, 3968)
        self.assertTrue(final.uses_checkpoint_history)

    def test_long_suffix_starts_a_new_bounded_replay(self):
        plan = self.plan(total=4225, restored_checkpoint=self.complete_checkpoint(4096))
        final = plan.extends[-1]
        self.assertEqual(final.decoder_rows, RowRange(4097, 4225))
        self.assertEqual(final.replay_floor, 4097)
        self.assertFalse(final.uses_checkpoint_history)

    def test_checkpoint_and_chunks_never_split_images(self):
        images = (RowRange(1000, 1800), RowRange(3900, 4300))
        plan = self.plan(total=4500, chunk=1024, images=images)
        self.assertEqual(plan.protected_checkpoint_end, 3072)
        for extend in plan.extends:
            for image in images:
                self.assertFalse(image.start < extend.encoder_rows.end < image.end)
        self.assertEqual(plan.extends[0].encoder_rows.end, 1000)
        self.assertEqual(checkpoint_boundary(4500, 1024, images), 3072)

    def test_full_output_consumers_record_explicit_fallback_and_identity(self):
        bounded = self.plan()
        for option in ("all_prompt_logprobs", "all_hidden_states"):
            full = self.plan(**{option: True})
            self.assertEqual(full.config.mode, ReplayMode.FULL)
            self.assertEqual(full.fallback_reason, option)
            self.assertNotEqual(full.identity, bounded.identity)
            self.assertTrue(all(x.decoder_rows == x.encoder_rows for x in full.extends))

    def test_incompatible_or_incomplete_checkpoints_cannot_resume(self):
        for checkpoint in (
            self.complete_checkpoint(4096, ReplayConfig()),
            replace(self.complete_checkpoint(4096), copy_complete=False),
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                self.plan(total=4224, restored_checkpoint=checkpoint)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.plan(
                total=4224,
                restored_checkpoint=self.complete_checkpoint(4096),
                all_hidden_states=True,
            )

    def test_invalid_image_budget_and_context_fail_before_execution(self):
        with self.assertRaisesRegex(ValueError, "image"):
            self.plan(total=2048, chunk=512, images=(RowRange(0, 1024),))
        with self.assertRaisesRegex(ValueError, "1048576"):
            self.plan(total=1048577)
        with self.assertRaises(ValueError):
            ReplayConfig("unknown")
        with self.assertRaises(ValueError):
            ReplayConfig(ReplayMode.BOUNDED, window=64)

    def test_empty_cp_rank_still_participates_in_late_ep(self):
        extend = self.plan().extends[-2]
        self.assertEqual(local_tail_indices(tuple(range(256)), extend.decoder_rows), ())
        self.assertTrue(extend.needs_decoder_ep)
        self.assertEqual(
            local_tail_indices((8063, 8064, 8191, 8192), extend.decoder_rows), (1, 2)
        )

    def test_progress_does_not_publish_aux_when_only_encoder_completed(self):
        plan = self.plan(total=2048, chunk=1024)
        progress = PrefillProgress()
        progress = progress.encoder_completed(plan.extends[0])
        self.assertEqual(progress.encoder_materialized_end, 1024)
        self.assertEqual(progress.decoder_checkpoint_end, 0)
        with self.assertRaisesRegex(ValueError, "decoder/aux"):
            progress.require_handoff(1024, 0)
        with self.assertRaises(ValueError):
            progress.decoder_completed(
                plan.extends[0], LateCompletion(RowRange(0, 1024))
            )

    def test_n_snapshot_protection_precedes_any_suffix_materialization(self):
        plan = self.plan(total=1152, chunk=1024)
        first, suffix = plan.extends
        progress = (
            PrefillProgress()
            .encoder_completed(first)
            .decoder_completed(first, LateCompletion(first.decoder_rows, True, True))
        )
        with self.assertRaisesRegex(ValueError, "protect checkpoint N"):
            progress.encoder_completed(suffix)
        checkpoint = self.complete_checkpoint(1024)
        with self.assertRaisesRegex(ValueError, "held reference"):
            progress.checkpoint_protected(
                replace(checkpoint, backing_protected=False), self.layout, plan.identity
            )
        progress = progress.checkpoint_protected(checkpoint, self.layout, plan.identity)
        progress = progress.encoder_completed(suffix).decoder_completed(
            suffix, LateCompletion(suffix.decoder_rows, True, True)
        )
        progress.require_handoff(1152, 1024)

    def test_late_completion_requires_valid_aux_and_all_draft_swa(self):
        final = self.plan(total=128).extends[-1]
        progress = PrefillProgress().encoder_completed(final)
        for completion in (
            LateCompletion(final.decoder_rows, True, False),
            LateCompletion(RowRange(1, 128), True, True),
            LateCompletion(final.decoder_rows, False, True),
        ):
            with self.assertRaisesRegex(ValueError, "valid aux"):
                progress.decoder_completed(final, completion)


class TailAndAuxTest(unittest.TestCase):
    def setUp(self):
        self.replay = ReplayConfig(ReplayMode.BOUNDED).fingerprint

    def row(self, position):
        return L20TailRow(
            position,
            position + 10000,
            100 <= position < 120,
            tuple(range(max(0, position + 1 - 512), position + 1)),
            tuple(range(max(0, (position + 8) // 8 - 2048), (position + 8) // 8)),
        )

    def test_l20_ring_retains_query_candidates_across_chunk_boundaries(self):
        tail = L20TailMetadata("request-a", 7, self.replay)
        tail = tail.append(tuple(self.row(p) for p in range(100)), forward_epoch=7)
        tail = tail.append(tuple(self.row(p) for p in range(100, 200)), forward_epoch=8)
        self.assertEqual([row.position for row in tail.rows], list(range(72, 200)))
        self.assertEqual(
            tail.selection(
                RowRange(100, 200),
                request_id="request-a",
                forward_epoch=8,
                replay_fingerprint=self.replay,
            ),
            tuple(range(28, 128)),
        )
        self.assertEqual(tail.rows[28].logical_row, 10100)
        self.assertTrue(tail.rows[28].image_mask)
        self.assertEqual(tail.rows[28].candidate_blocks[-1], 12)

    def test_expired_tail_and_wrong_request_are_rejected(self):
        tail = L20TailMetadata(
            "request-a", 7, self.replay, tuple(self.row(p) for p in range(72, 200))
        )
        for rows, request in (
            (RowRange(71, 200), "request-a"),
            (RowRange(72, 200), "request-b"),
        ):
            with self.assertRaises(ValueError):
                tail.selection(
                    rows,
                    request_id=request,
                    forward_epoch=7,
                    replay_fingerprint=self.replay,
                )
        with self.assertRaisesRegex(ValueError, "global token order"):
            tail.append([self.row(201)], forward_epoch=8)
        with self.assertRaisesRegex(ValueError, "older forward epoch"):
            tail.append([self.row(200)], forward_epoch=6)

    def test_tail_rejects_invalid_topk_and_missing_partial_candidate(self):
        row = self.row(16384)
        with self.assertRaisesRegex(ValueError, "partial block"):
            replace(row, candidate_blocks=tuple(range(2048)))
        with self.assertRaisesRegex(ValueError, "causal"):
            replace(row, topk=tuple(range(15874, 16386)))
        with self.assertRaisesRegex(ValueError, "valid count"):
            replace(row, topk=(0,))

    def test_aux_selection_precedes_projection_and_retains_only_valid_rows(self):
        rows = AuxRowMap(
            "request-a",
            7,
            self.replay,
            (120, 121, 122, 123),
            (12, 13, 14, 15),
            (False,) * 4,
        )
        hidden = torch.arange(4 * 15360, dtype=torch.float32).reshape(4, 15360)
        selected = select_aux_hidden(
            hidden,
            rows,
            (121, 123),
            request_id="request-a",
            forward_epoch=7,
            replay_fingerprint=self.replay,
        )
        self.assertEqual(tuple(selected.shape), (2, 15360))
        torch.testing.assert_close(
            selected, torch.stack((hidden[1], hidden[3])), rtol=0, atol=0
        )

    def test_aux_rejects_uncomputed_positions_stale_epoch_and_wrong_mode(self):
        rows = AuxRowMap(
            "request-a", 7, self.replay, (120, 121), (12, 13), (False, True)
        )
        for positions, epoch, replay in (
            ((119,), 7, self.replay),
            ((120,), 8, self.replay),
            ((120,), 7, ReplayConfig().fingerprint),
        ):
            with self.assertRaises(ValueError):
                rows.selection(
                    positions,
                    request_id="request-a",
                    forward_epoch=epoch,
                    replay_fingerprint=replay,
                )

    def test_empty_cp_aux_selection_is_a_valid_zero_row_tensor(self):
        rows = AuxRowMap("empty-rank", 7, self.replay, (), (), ())
        result = select_aux_hidden(
            torch.empty((0, 15360)),
            rows,
            (),
            request_id="empty-rank",
            forward_epoch=7,
            replay_fingerprint=self.replay,
        )
        self.assertEqual(tuple(result.shape), (0, 15360))


if __name__ == "__main__":
    unittest.main()

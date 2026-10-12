"""Source-aware publication, relocated pages and forward-local selection reuse."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_block_test import attention, block, fixture, indexer
from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
)
from rtp_llm.models.test.dsv41_compressor_test import compressor

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_attention": attention,
        "rtp_llm.models_py.modules.dsv4.v41_cache": codec,
        "rtp_llm.models_py.modules.dsv4.v41_compressor": compressor,
        "rtp_llm.models_py.modules.dsv4.v41_indexer": indexer,
    },
):
    execution = load_codec_source(
        "_dsv41_stage_attention_test", "v41_stage_attention.py"
    )


def rows(invocation="forward-1"):
    return execution.V41ForwardRows(
        invocation,
        torch.tensor([4, 4, -1]),
        torch.tensor([0, 1, -1]),
        torch.tensor([True, True, False]),
    ).snapshot()


def plan(sources, batch, *, memory=None):
    swa, _ = make_cache("swa_kv")
    writes = torch.tensor([0, 1, -1])
    values = dict(
        swa=swa, swa_writes=writes, swa_reads=torch.tensor([[0, -1], [0, 1], [-1, -1]])
    )
    if sources.compress_ratio:
        ratio = sources.compress_ratio
        global_cache, index_cache = memory or (
            make_cache("csa_kv" if ratio == 2 else "hca_kv")[0],
            make_cache("indexer_kv")[0],
        )
        values.update(
            kv_source_layer_id=sources.kv_source_layer_id,
            global_cache=global_cache,
            index_cache=index_cache,
            global_pages=torch.full((3, 1), 2, dtype=torch.int32),
            index_pages=torch.ones(3, 1, dtype=torch.int32),
        )
        if sources.owns_kv:
            values.update(
                global_writes=torch.tensor([0, 1, -1]) // ratio
                + 2 * global_cache.layout.entries,
                index_writes=torch.tensor([0, 1, -1]) // ratio
                + index_cache.layout.entries,
            )
            if ratio == 2:
                values.update(
                    pair_state=compressor.V41PairState(torch.zeros(2, 4096)),
                    pair_metadata=compressor.V41PairMetadata(
                        batch.request_ids,
                        batch.positions,
                        batch.valid_rows,
                        torch.tensor([-1, 0, -1]),
                        torch.full((3,), -1),
                        torch.tensor([0, 1, -1]),
                    ),
                )
    return execution.V41AttentionPlan(**values)


def prepare(op, batch, cache_plan):
    residual = torch.randn(3, 4, op.dim).bfloat16()
    residual[-1] = float("nan")
    state = block.V41ResidualState(residual, torch.rand(3, 4), op.layer_id)
    return op.prepare(
        state,
        batch.positions,
        batch.valid_rows,
        pair_state=cache_plan.pair_state,
        pair_metadata=cache_plan.pair_metadata,
    )


class V41StageAttentionTest(unittest.TestCase):
    def test_both_ratios_use_separate_global_and_index_allocations(self):
        for layer in (2, 20):
            op, _ = fixture(layer)
            batch = rows()
            cache_plan = plan(op.attention.sources, batch)
            cache_plan.validate(op.attention.sources, batch, 256)
            frame = prepare(op, batch, cache_plan)
            step = execution.V41StageAttention(
                op.attention.sources, max_positions=256, backend="eager"
            )
            actual, selected = step(frame, op.attention.sinks, cache_plan, batch, None)
            selected.validate()
            slots = torch.where(
                selected.result.topk >= 0,
                selected.result.topk.long()
                + 2 * cache_plan.global_cache.layout.entries,
                -1,
            )
            expected = attention.V41SparseAttention(backend="eager")(
                frame.query.query,
                op.attention.sinks,
                cache_plan.swa,
                cache_plan.swa_reads,
                valid_rows=batch.valid_rows,
                global_cache=cache_plan.global_cache,
                global_slots=slots,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(selected.source_layer_id, layer)
            if layer == 2:
                self.assertTrue(bool((selected.result.topk[0] == -1).all()))
                self.assertEqual(selected.result.topk[1, 0].item(), 0)
            self.assertTrue(bool((actual[-1] == 0).all()))

    def test_consumers_reuse_l20_and_later_indexer_keeps_original_candidates(self):
        batch = rows()
        source, _ = fixture(20)
        source_plan = plan(source.attention.sources, batch)
        source_frame = prepare(source, batch, source_plan)
        _, selected = execution.V41StageAttention(
            source.attention.sources, max_positions=256, backend="eager"
        )(source_frame, source.attention.sinks, source_plan, batch, None)
        candidates = selected.result.candidate_blocks
        before = source_plan.global_cache.raw.clone()
        for layer in (21, 24):
            op, _ = fixture(layer)
            cache_plan = plan(
                op.attention.sources,
                batch,
                memory=(source_plan.global_cache, source_plan.index_cache),
            )
            cache_plan.validate(op.attention.sources, batch, 256)
            frame = prepare(op, batch, cache_plan)
            step = execution.V41StageAttention(
                op.attention.sources, max_positions=256, backend="eager"
            )
            prior = selected
            value, selected = step(
                frame, op.attention.sinks, cache_plan, batch, selected
            )
            if layer == 21:
                self.assertIs(prior, selected)
            else:
                self.assertEqual(selected.source_layer_id, 24)
                selected.validate()
            self.assertIs(selected.result.candidate_blocks, candidates)
            self.assertTrue(bool(torch.isfinite(value).all()))
            torch.testing.assert_close(
                source_plan.global_cache.raw, before, rtol=0, atol=0
            )

    def test_stale_or_wrong_source_selection_rejected_before_publication(self):
        batch = rows()
        source, _ = fixture(20)
        source_plan = plan(source.attention.sources, batch)
        _, selected = execution.V41StageAttention(
            source.attention.sources, max_positions=256, backend="eager"
        )(
            prepare(source, batch, source_plan),
            source.attention.sinks,
            source_plan,
            batch,
            None,
        )
        op, _ = fixture(21)
        cache_plan = plan(
            op.attention.sources,
            batch,
            memory=(source_plan.global_cache, source_plan.index_cache),
        )
        frame = prepare(op, batch, cache_plan)
        before = cache_plan.swa.raw.clone()
        step = execution.V41StageAttention(
            op.attention.sources, max_positions=256, backend="eager"
        )
        for bad in (
            None,
            dataclasses.replace(selected, source_layer_id=24),
            dataclasses.replace(selected, rows=rows("forward-2")),
        ):
            with self.assertRaises(ValueError):
                step(frame, op.attention.sinks, cache_plan, batch, bad)
            torch.testing.assert_close(cache_plan.swa.raw, before, rtol=0, atol=0)

    def test_invalid_page_source_write_and_pair_identity_fail_preflight(self):
        op, _ = fixture(2)
        batch = rows()
        cache_plan = plan(op.attention.sources, batch)
        for bad in (
            dataclasses.replace(cache_plan, kv_source_layer_id=8),
            dataclasses.replace(cache_plan, global_pages=torch.full((3, 1), 9)),
            dataclasses.replace(cache_plan, global_writes=torch.tensor([0, 1, -1])),
            dataclasses.replace(
                cache_plan,
                pair_metadata=dataclasses.replace(
                    cache_plan.pair_metadata, previous_rows=torch.tensor([-1, 1, -1])
                ),
            ),
            dataclasses.replace(
                cache_plan,
                pair_metadata=dataclasses.replace(
                    cache_plan.pair_metadata, request_ids=torch.tensor([4, 5, -1])
                ),
            ),
        ):
            with self.assertRaises(ValueError):
                bad.validate(op.attention.sources, batch, 256)

    def test_rows_snapshot_canonicalizes_padding_and_rejects_duplicate_tokens(self):
        batch = rows()
        saved = batch.snapshot()
        batch.positions[0] = 7
        self.assertEqual(saved.positions.tolist(), [0, 1, -1])
        with self.assertRaisesRegex(ValueError, "different forward"):
            batch.require_same(saved)
        with self.assertRaisesRegex(ValueError, "repeats"):
            dataclasses.replace(saved, positions=torch.tensor([0, 0, -1])).snapshot()

    def test_imported_selections_validate_causality_duplicates_counts_and_padding(self):
        batch = rows()
        topk = torch.full((3, 512), -1, dtype=torch.int32)
        topk[0, 0], topk[1, :2] = 0, torch.tensor([0, 1])
        candidates = torch.full((3, 2048), -1, dtype=torch.int32)
        candidates[:2, 0] = 0
        good = execution.V41Selections(
            batch,
            20,
            20,
            1,
            indexer.V41IndexResult(
                topk, candidates, torch.tensor([1, 1, 0], dtype=torch.int32)
            ),
        )
        good.validate()
        for bad in (
            dataclasses.replace(good.result, topk=torch.zeros_like(topk)),
            dataclasses.replace(good.result, topk=topk + 1),
            dataclasses.replace(
                good.result, candidate_counts=torch.tensor([1, 2, 0], dtype=torch.int32)
            ),
            dataclasses.replace(good.result, candidate_blocks=candidates.roll(1, -1)),
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(good, result=bad).validate()


if __name__ == "__main__":
    unittest.main()

"""SM120 source-aware cache mapping and full-width 19-layer stage-one execution."""

import copy
import dataclasses
import json
import unittest
from pathlib import Path

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8.test import test_v41_block_hardware as fixture
from rtp_llm.models_py.modules.dsv4.v41_block import (
    V41Block,
    V41ResidualState,
    readout_v41_residual,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.models_py.modules.dsv4.v41_compressor import V41PairMetadata, V41PairState
from rtp_llm.models_py.modules.dsv4.v41_engram_stage import V41EngramStage
from rtp_llm.models_py.modules.dsv4.v41_projection import V41KVProjection
from rtp_llm.models_py.modules.dsv4.v41_stage import V41Stage
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import (
    V41AttentionPlan,
    V41ForwardRows,
    V41StageAttention,
)


def cache(tag):
    return V41PagedCache(
        torch.zeros(
            3, dsv41_page_layouts(1)[tag].stride_bytes, device="cuda", dtype=torch.uint8
        ),
        tag,
    )


def rows(empty=False):
    return V41ForwardRows(
        "stage-hardware",
        torch.tensor([0, 0, -1] if not empty else [], device="cuda", dtype=torch.int64),
        torch.tensor([0, 1, -1] if not empty else [], device="cuda", dtype=torch.int64),
        torch.tensor(
            [True, True, False] if not empty else [], device="cuda", dtype=torch.bool
        ),
    ).snapshot()


def plan(sources, batch, *, memory=None):
    n = batch.positions.numel()
    ratio = sources.compress_ratio
    global_cache, index_cache = memory or (
        cache("csa_kv" if ratio == 2 else "hca_kv"),
        cache("indexer_kv"),
    )
    reads = torch.full((n, 2), -1, device="cuda", dtype=torch.int64)
    if n:
        reads[0, 0] = 0
        reads[1] = torch.tensor([0, 1], device="cuda")
    kwargs = dict(
        swa=cache("swa_kv"),
        swa_writes=batch.positions.clone(),
        swa_reads=reads,
        kv_source_layer_id=sources.kv_source_layer_id,
        global_cache=global_cache,
        index_cache=index_cache,
        global_pages=torch.full((n, 1), 2, device="cuda", dtype=torch.int32),
        index_pages=torch.ones(n, 1, device="cuda", dtype=torch.int32),
    )
    if sources.owns_kv:
        kwargs.update(
            global_writes=batch.positions // ratio + 2 * global_cache.layout.entries,
            index_writes=batch.positions // ratio + index_cache.layout.entries,
        )
        if ratio == 2:
            kwargs.update(
                pair_state=V41PairState(torch.zeros(2, 4096, device="cuda")),
                pair_metadata=V41PairMetadata(
                    batch.request_ids,
                    batch.positions,
                    batch.valid_rows,
                    torch.tensor(
                        [-1, 0, -1] if n else [], device="cuda", dtype=torch.int64
                    ),
                    torch.full((n,), -1, device="cuda", dtype=torch.int64),
                    batch.positions.clone(),
                ),
            )
    return V41AttentionPlan(**kwargs)


class V41StageHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        prior = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", prior)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_source_step_both_ratios_relocated_planes_and_empty_rows(self):
        for ratio in (1, 2):
            op = fixture.fixture(ratio)
            step = V41StageAttention(op.attention.sources, max_positions=256)
            for empty in (False, True):
                batch = rows(empty)
                n = batch.positions.numel()
                cache_plan = plan(op.attention.sources, batch)
                cache_plan.validate(op.attention.sources, batch, 256)
                state = V41ResidualState(
                    torch.randn(n, 4, 5120, device="cuda").bfloat16(),
                    torch.rand(n, 4, device="cuda"),
                    op.layer_id,
                )
                frame = op.prepare(
                    state,
                    batch.positions,
                    batch.valid_rows,
                    pair_state=cache_plan.pair_state,
                    pair_metadata=cache_plan.pair_metadata,
                )
                actual, selected = step(
                    frame, op.attention.sinks, cache_plan, batch, None
                )
                selected.validate()
                expected_slots = torch.where(
                    selected.result.topk >= 0,
                    selected.result.topk.long()
                    + 2 * cache_plan.global_cache.layout.entries,
                    -1,
                )
                expected = step.attention(
                    frame.query.query,
                    op.attention.sinks,
                    cache_plan.swa,
                    cache_plan.swa_reads,
                    valid_rows=batch.valid_rows,
                    global_cache=cache_plan.global_cache,
                    global_slots=expected_slots,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(bool((actual[~batch.valid_rows] == 0).all()))

    def test_full_width_stage_one_matches_explicit_loop_and_rejects_capture(self):
        source = fixture.fixture(1)
        path = (
            Path(__file__).resolve().parents[5]
            / "config/test/testdata/deepseek_v41_flash.json"
        )
        data = json.loads(path.read_text())
        data["text_config"]["n_routed_experts"] = 8
        config = V41Config.from_dict(data)
        blocks = []
        for sources in config.layer_sources()[21:]:
            attention = copy.copy(source.attention)
            attention.sources = sources
            if not sources.owns_index:
                attention.index_wq_b = attention.index_weights = None
            kv = V41KVProjection(
                sources, source.kv.wkv, source.kv.kv_norm, source.kv.frequencies
            )
            ffn = copy.copy(source.feed_forward)
            ffn.layer_id = sources.layer_id
            blocks.append(
                V41Block(
                    attention,
                    kv,
                    source.attn_hc,
                    source.ffn_hc,
                    source.attn_norm,
                    source.ffn_norm,
                    ffn,
                )
            )
        engram = V41EngramStage(
            {}, None, device="cuda", max_host_bytes=0, max_weight_bytes=0
        ).initialize()
        self.addCleanup(engram.close)
        norm = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        driver = V41Stage(
            blocks,
            config,
            role="prefill",
            pp_rank=1,
            engram=engram,
            final_norm=norm,
            max_positions=256,
        )
        batch = rows()
        source_plan = plan(source.attention.sources, batch)
        source_state = V41ResidualState(
            torch.randn(3, 4, 5120, device="cuda").bfloat16(),
            torch.rand(3, 4, device="cuda"),
            20,
        )
        frame = source.prepare(source_state, batch.positions, batch.valid_rows)
        attended, seed = V41StageAttention(source.attention.sources, max_positions=256)(
            frame, source.attention.sinks, source_plan, batch, None
        )
        initial = source.finish(frame, attended)
        memory = source_plan.global_cache, source_plan.index_cache
        plans = {
            op.layer_id: plan(op.attention.sources, batch, memory=memory)
            for op in blocks
        }
        actual = driver.run(initial, batch, plans, selections=seed)
        state, selected = initial, seed.result
        for op, step in zip(blocks, driver.attentions):
            current = plans[op.layer_id]
            frame = op.prepare(state, batch.positions, batch.valid_rows)
            frame.kv.publish(current.swa, current.swa_writes)
            topk = selected.topk
            if step.indexer is not None:
                topk = step.indexer(
                    frame.query.index_query,
                    frame.query.index_weights,
                    current.index_cache,
                    current.index_pages,
                    batch.positions,
                    valid_rows=batch.valid_rows,
                    candidates=seed.result.candidate_blocks,
                ).topk
            slots = torch.where(
                topk >= 0, topk.long() + 2 * current.global_cache.layout.entries, -1
            )
            attended = step.attention(
                frame.query.query,
                op.attention.sinks,
                current.swa,
                current.swa_reads,
                valid_rows=batch.valid_rows,
                global_cache=current.global_cache,
                global_slots=slots,
            )
            state = op.finish(frame, attended)
            selected = dataclasses.replace(selected, topk=topk)
        torch.testing.assert_close(
            actual.hidden,
            readout_v41_residual(state, norm, batch.valid_rows),
            rtol=0,
            atol=0,
        )
        self.assertEqual(actual.selections.source_layer_id, 36)
        self.assertIs(
            actual.selections.result.candidate_blocks, seed.result.candidate_blocks
        )
        self.assertTrue(bool(torch.isfinite(actual.hidden).all()))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with self.assertRaisesRegex(RuntimeError, "capture requires"):
                driver.run(initial, batch, plans, selections=seed)


if __name__ == "__main__":
    unittest.main()

"""Complete rank-local stage loops with real small Engram tables and operators."""

import copy
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.test.dsv41_block_test import (
    block,
    expert,
    ffn,
    moe,
    queries,
    routed,
    routed_weights,
    shared_weights,
    weights_fixture,
)
from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
)
from rtp_llm.models.test.dsv41_collective_test import collective
from rtp_llm.models.test.dsv41_engram_stage_test import (
    hashing,
    loading,
    stage_module,
    write_checkpoint,
)
from rtp_llm.models.test.dsv41_projection_test import compressor, projection
from rtp_llm.models.test.dsv41_stage_attention_test import execution, rows

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_block": block,
        "rtp_llm.models_py.modules.dsv4.v41_cache": codec,
        "rtp_llm.models_py.modules.dsv4.v41_collective": collective,
        "rtp_llm.models_py.modules.dsv4.v41_engram_stage": stage_module,
        "rtp_llm.models_py.modules.dsv4.v41_projection": projection,
        "rtp_llm.models_py.modules.dsv4.v41_stage_attention": execution,
    },
):
    stage = load_codec_source("_dsv41_stage_test", "v41_stage.py")


def config_fixture():
    path = Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
    data = json.loads(path.read_text())
    data["text_config"].update(
        hidden_size=32,
        moe_intermediate_size=128,
        n_routed_experts=3,
        num_experts_per_tok=2,
        q_lora_rank=128,
        o_lora_rank=32,
        engram_num_embeddings=[10, 31],
        engram_n_heads=1,
        engram_head_dim=64,
        engram_compressed_vocab_size=3,
        engram_vocab_size=2,
    )
    return V41Config.from_dict(data)


def block_fixture(layer):
    sources, weights, frequencies = weights_fixture(layer)
    for name in ("attn.wq_a", "attn.wkv"):
        weights[f"v4.{name}.weight"] = weights[f"v4.{name}.weight"][:, :32].contiguous()
        weights[f"v4.{name}.scale"] = weights[f"v4.{name}.scale"][:, :1].contiguous()
    weights["v4.attn.wo_b.weight"] = weights["v4.attn.wo_b.weight"][:32].contiguous()
    weights["v4.attn.wo_b.scale"] = weights["v4.attn.wo_b.scale"][:1].contiguous()
    for name in (
        "v4.compressor.wkv.weight",
        "v4.compressor.wgate.weight",
        "v4.indexer.weights_proj.weight",
    ):
        if name in weights:
            weights[name] = weights[name][:, :32].contiguous()
    for kind in ("attn", "ffn"):
        weights[f"v4.hc.{kind}_fn"] = torch.randn(24, 128) * 0.01
        weights[f"v4.hc.{kind}_base"] = torch.randn(24) * 0.1
        weights[f"v4.hc.{kind}_scale"] = torch.tensor([0.5, 0.6, 0.7])
        weights[f"v4.{kind}_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
    attention, kv = queries.build_v41_projections(
        sources, weights, frequencies, backend="eager"
    )
    feed_forward = ffn.V41FeedForward(
        layer,
        ffn.V41Router(torch.randn(3, 32).bfloat16() / 32**0.5, torch.randn(3), topk=2),
        expert.V41SharedExpert(shared_weights(dim=32, inter=128), backend="eager"),
        moe.V41LocalMoE(
            routed.V41RoutedExperts(routed_weights(dim=32, inter=128), backend="eager")
        ),
    )
    return block.build_v41_block(weights, attention, kv, feed_forward)


def blocks_fixture(config):
    prototypes = {layer: block_fixture(layer) for layer in (0, 2, 20, 21, 24)}
    blocks = []
    for source in config.layer_sources():
        kind = (
            0
            if not source.compress_ratio
            else (
                2
                if source.owns_kv and source.compress_ratio == 2
                else 20 if source.owns_kv else 24 if source.owns_index else 21
            )
        )
        prototype = prototypes[kind]
        attention, kv = copy.copy(prototype.attention), copy.copy(prototype.kv)
        attention.sources = kv.sources = source
        feed_forward = copy.copy(prototype.feed_forward)
        feed_forward.layer_id = source.layer_id
        blocks.append(
            block.V41Block(
                attention,
                kv,
                prototype.attn_hc,
                prototype.ffn_hc,
                prototype.attn_norm,
                prototype.ffn_norm,
                feed_forward,
            )
        )
    return blocks


def plans_fixture(blocks, batch, *, memory=None):
    memory = {} if memory is None else memory
    result = {}
    n = batch.valid_rows.numel()
    for op in blocks:
        sources = op.attention.sources
        swa, _ = make_cache("swa_kv")
        writes = batch.positions.clone()
        reads = torch.full((n, 2), -1, dtype=torch.int64)
        for row in range(n):
            if batch.valid_rows[row]:
                reads[row, : int(batch.positions[row]) + 1] = torch.arange(
                    int(batch.positions[row]) + 1
                )
        args = dict(swa=swa, swa_writes=writes, swa_reads=reads)
        ratio = sources.compress_ratio
        if ratio:
            source = sources.kv_source_layer_id
            if source not in memory:
                memory[source] = (
                    make_cache("csa_kv" if ratio == 2 else "hca_kv")[0],
                    make_cache("indexer_kv")[0],
                )
            global_cache, index_cache = memory[source]
            args.update(
                kv_source_layer_id=source,
                global_cache=global_cache,
                index_cache=index_cache,
                global_pages=torch.full((n, 1), 2, dtype=torch.int32),
                index_pages=torch.ones(n, 1, dtype=torch.int32),
            )
            if sources.owns_kv:
                args.update(
                    global_writes=batch.positions // ratio
                    + 2 * global_cache.layout.entries,
                    index_writes=batch.positions // ratio + index_cache.layout.entries,
                )
                if ratio == 2:
                    previous = torch.full((n,), -1)
                    for row in range(n):
                        if batch.valid_rows[row] and batch.positions[row] % 2:
                            previous[row] = (
                                (batch.positions == batch.positions[row] - 1)
                                & batch.valid_rows
                            ).nonzero()[0, 0]
                    args.update(
                        pair_state=compressor.V41PairState(torch.zeros(2, 4096)),
                        pair_metadata=compressor.V41PairMetadata(
                            batch.request_ids,
                            batch.positions,
                            batch.valid_rows,
                            previous,
                            torch.full((n,), -1),
                            writes.clone(),
                        ),
                    )
        result[op.layer_id] = execution.V41AttentionPlan(**args)
    return result, memory


class V41StageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        write_checkpoint(root, ((1, 10), (14, 31)))
        self.config = config_fixture()
        sources = loading.plan_stage_engram_sources(
            self.config, root, range(21), "a" * 64
        )
        hasher = hashing.V41EngramHasher(
            hashing.V41EngramLayout.from_config(self.config.text, 2),
            [0, 1, 2, 0, 1, 2, 1],
        )
        self.engram = stage_module.V41EngramStage(
            sources,
            hasher,
            device="cpu",
            max_host_bytes=sum(s.table.nbytes for s in sources.values()),
            max_weight_bytes=sum(s.weight_bytes for s in sources.values()),
            directory=root,
            backend="eager",
        ).initialize()
        self.addCleanup(self.engram.close)
        for module in self.engram._modules.values():
            module.k_weight.fill_(1)
            # Synthetic FFN weights grow the stream; keep Engram above its BF16 ULP.
            module.projection.scales.fill_(130)
        self.empty = stage_module.V41EngramStage(
            {},
            None,
            device="cpu",
            max_host_bytes=0,
            max_weight_bytes=0,
            backend="eager",
        ).initialize()
        self.addCleanup(self.empty.close)
        self.blocks = blocks_fixture(self.config)
        self.norm = torch.ones(32, dtype=torch.bfloat16)
        self.batch = rows()
        self.windows = torch.tensor([[0, -1, -1, -1], [4, 0, -1, -1], [-1, -1, -1, -1]])
        self.initial = block.start_v41_residual(
            torch.randn(3, 32).bfloat16(), self.batch.valid_rows
        )

    def driver(self, role="decode", rank=0):
        blocks = (
            self.blocks
            if role == "decode"
            else self.blocks[:21] if rank == 0 else self.blocks[21:]
        )
        return stage.V41Stage(
            blocks,
            self.config,
            role=role,
            pp_rank=rank,
            engram=self.empty if rank else self.engram,
            final_norm=self.norm if role == "decode" or rank else None,
            max_positions=256,
        )

    def test_complete_decode_equals_split_21_19_with_before_block_engram(self):
        full_plans, _ = plans_fixture(self.blocks, self.batch)
        events = []
        original_apply = self.engram.apply

        def apply(layer, hidden, frame):
            events.append(("engram", layer))
            value = original_apply(layer, hidden, frame)
            self.assertFalse(torch.equal(value[:2], hidden[:2]))
            return value

        def record(op):
            original = op.prepare

            def prepare(*args, **kwargs):
                events.append(("block", op.layer_id))
                return original(*args, **kwargs)

            return prepare

        with patch.object(self.engram, "apply", side_effect=apply), patch.object(
            self.blocks[1], "prepare", side_effect=record(self.blocks[1])
        ), patch.object(
            self.blocks[14], "prepare", side_effect=record(self.blocks[14])
        ):
            full = self.driver().run(
                self.initial, self.batch, full_plans, windows=self.windows
            )
        self.assertEqual(
            events, [("engram", 1), ("block", 1), ("engram", 14), ("block", 14)]
        )
        first_plans, memory = plans_fixture(self.blocks[:21], self.batch)
        first = self.driver("prefill", 0).run(
            self.initial, self.batch, first_plans, windows=self.windows
        )
        self.assertEqual(first.state.next_layer_id, 21)
        self.assertIsNone(first.hidden)
        second_plans, _ = plans_fixture(self.blocks[21:], self.batch, memory=memory)
        second = self.driver("prefill", 1).run(
            first.state, first.rows, second_plans, selections=first.selections
        )
        torch.testing.assert_close(second.hidden, full.hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            second.state.residual, full.state.residual, rtol=0, atol=0
        )
        self.assertEqual(second.state.next_layer_id, 40)
        self.assertEqual(second.selections.source_layer_id, 36)
        torch.testing.assert_close(
            second.selections.result.candidate_blocks,
            first.selections.result.candidate_blocks,
            rtol=0,
            atol=0,
        )

    def test_all_plans_and_engram_masks_fail_before_first_block(self):
        driver = self.driver()
        plans, _ = plans_fixture(self.blocks, self.batch)
        bad = dict(plans)
        bad[39] = dataclasses.replace(plans[39], kv_source_layer_id=14)
        with patch.object(self.blocks[0], "prepare") as begin:
            with self.assertRaisesRegex(ValueError, "source"):
                driver.run(self.initial, self.batch, bad, windows=self.windows)
            wrong_windows = self.windows.clone()
            wrong_windows[0, 0] = -1
            with self.assertRaisesRegex(ValueError, "mask"):
                driver.run(self.initial, self.batch, plans, windows=wrong_windows)
            begin.assert_not_called()

    def test_consumer_cache_rebinding_and_swa_alias_rejected_before_execution(self):
        driver = self.driver()
        plans, _ = plans_fixture(self.blocks, self.batch)
        aliased_reads = plans[0].swa.raw.view(torch.int64).flatten()[:6].view(3, 2)
        aliased_reads.zero_()
        for replacement in (
            dataclasses.replace(plans[39], global_cache=make_cache("hca_kv")[0]),
            dataclasses.replace(plans[39], swa=plans[0].swa),
            dataclasses.replace(plans[39], swa_reads=aliased_reads),
        ):
            bad = dict(plans)
            bad[39] = replacement
            with patch.object(self.blocks[0], "prepare") as begin:
                with self.assertRaises(ValueError):
                    driver.run(self.initial, self.batch, bad, windows=self.windows)
                begin.assert_not_called()

    def test_empty_forward_and_fresh_invocation_after_failure(self):
        empty_rows = execution.V41ForwardRows(
            "empty",
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.bool),
        )
        empty_state = block.start_v41_residual(
            torch.empty(0, 32, dtype=torch.bfloat16), empty_rows.valid_rows
        )
        plans, _ = plans_fixture(self.blocks, empty_rows)
        driver = self.driver()
        result = driver.run(
            empty_state, empty_rows, plans, windows=torch.empty(0, 4, dtype=torch.int64)
        )
        self.assertEqual(result.hidden.shape, (0, 32))
        plans, _ = plans_fixture(self.blocks, self.batch)
        with patch.object(
            self.blocks[5], "finish", side_effect=RuntimeError("worker failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failure"):
                driver.run(self.initial, self.batch, plans, windows=self.windows)
        self.assertFalse(
            {"hashes", "rows", "state", "selections"}.intersection(driver.__dict__)
        )
        new_rows = dataclasses.replace(
            self.batch, invocation_id="replacement-worker-forward"
        )
        plans, _ = plans_fixture(self.blocks, new_rows)
        self.assertTrue(
            bool(
                torch.isfinite(
                    driver.run(
                        self.initial, new_rows, plans, windows=self.windows
                    ).hidden
                ).all()
            )
        )

    def test_stage_ownership_layer_range_and_missing_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordered global"):
            stage.V41Stage(
                self.blocks[:20],
                self.config,
                role="prefill",
                pp_rank=0,
                engram=self.engram,
            )
        with self.assertRaisesRegex(ValueError, "Engram ownership"):
            stage.V41Stage(
                self.blocks[21:],
                self.config,
                role="prefill",
                pp_rank=1,
                engram=self.engram,
                final_norm=self.norm,
                max_positions=256,
            )
        driver = self.driver("prefill", 1)
        plans, _ = plans_fixture(self.blocks[21:], self.batch)
        state = dataclasses.replace(self.initial, next_layer_id=21)
        with self.assertRaisesRegex(ValueError, "L20 selections"):
            driver.run(state, self.batch, plans)
        self.engram.close()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            plans, _ = plans_fixture(self.blocks, self.batch)
            self.driver().run(self.initial, self.batch, plans, windows=self.windows)


if __name__ == "__main__":
    unittest.main()

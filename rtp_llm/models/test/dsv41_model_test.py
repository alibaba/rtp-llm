"""Loader assembly, stage ownership and EP admission before cache publication."""

import copy
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.test import dsv41_stage_test as fixture
from rtp_llm.models.test.dsv41_feed_forward_test import dispatch
from rtp_llm.models.test.dsv41_mxfp8_test import linear
from rtp_llm.models.test.dsv41_projection_test import rope
from rtp_llm.utils.model_weight import W

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.rope": rope,
        "rtp_llm.models_py.modules.dsv4.v41_attention_projection": fixture.queries,
        "rtp_llm.models_py.modules.dsv4.v41_block": fixture.block,
        "rtp_llm.models_py.modules.dsv4.v41_collective": fixture.collective,
        "rtp_llm.models_py.modules.dsv4.v41_dispatch": dispatch,
        "rtp_llm.models_py.modules.dsv4.v41_engram_stage": fixture.stage_module,
        "rtp_llm.models_py.modules.dsv4.v41_feed_forward": fixture.ffn,
        "rtp_llm.models_py.modules.dsv4.v41_moe": fixture.moe,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
        "rtp_llm.models_py.modules.dsv4.v41_projection": fixture.projection,
        "rtp_llm.models_py.modules.dsv4.v41_routed": fixture.routed,
        "rtp_llm.models_py.modules.dsv4.v41_stage": fixture.stage,
    },
):
    model = fixture.load_codec_source("_dsv41_model_test", "v41_model.py")


def loader_prototype(layer):
    _, weights, _ = fixture.weights_fixture(layer)
    for name in ("attn.wq_a", "attn.wkv"):
        weights[f"v4.{name}.weight"] = weights[f"v4.{name}.weight"][:, :32].contiguous()
        weights[f"v4.{name}.scale"] = weights[f"v4.{name}.scale"][:, :1].contiguous()
    weights[W.v4_attn_wo_b_w] = weights[W.v4_attn_wo_b_w][:32].contiguous()
    weights[W.v4_attn_wo_b_s] = weights[W.v4_attn_wo_b_s][:1].contiguous()
    for name in (
        W.v4_compressor_wkv,
        W.v4_compressor_wgate,
        W.v4_indexer_weights_proj_w,
    ):
        if name in weights:
            weights[name] = weights[name][:, :32].contiguous()
    for kind in ("attn", "ffn"):
        weights[getattr(W, "v4_hc_" + kind + "_fn")] = torch.randn(24, 128) * 0.01
        weights[getattr(W, "v4_hc_" + kind + "_base")] = torch.randn(24) * 0.1
        weights[getattr(W, "v4_hc_" + kind + "_scale")] = torch.tensor([0.5, 0.6, 0.7])
        weights[getattr(W, "v4_" + kind + "_norm")] = torch.ones(
            32, dtype=torch.bfloat16
        )
    weights[W.v4_router_w] = (torch.randn(8, 32) / 32**0.5).bfloat16()
    weights[W.v4_router_bias] = torch.randn(8)
    for name, tensor in fixture.shared_weights(dim=32, inter=128).items():
        weights[getattr(W, "v4_shared_" + name)] = tensor
    experts = fixture.routed_weights(experts=8, dim=32, inter=128)
    for kind in ("w", "s"):
        gate, up = experts["w13_" + kind].chunk(2, dim=1)
        weights[getattr(W, "v4_routed_w1_" + kind)] = gate.contiguous()
        weights[getattr(W, "v4_routed_w3_" + kind)] = up.contiguous()
        weights[getattr(W, "v4_routed_w2_" + kind)] = experts["w2_" + kind]
    return weights


class V41ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = fixture.config_fixture()
        cls.config.text.update(n_routed_experts=8, vocab_size=7)
        cls.config = dataclasses.replace(cls.config, image_token_id=6)
        cls.prototypes = {i: loader_prototype(i) for i in (0, 2, 20, 21, 24)}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        fixture.write_checkpoint(root, ((1, 10), (14, 31)))
        sources = fixture.loading.plan_stage_engram_sources(
            self.config, root, range(21), "a" * 64
        )
        hasher = fixture.hashing.V41EngramHasher(
            fixture.hashing.V41EngramLayout.from_config(self.config.text, 2),
            [0, 1, 2, 0, 1, 2, 1],
        )
        self.engram = fixture.stage_module.V41EngramStage(
            sources,
            hasher,
            device="cpu",
            backend="eager",
            directory=root,
            max_host_bytes=sum(s.table.nbytes for s in sources.values()),
            max_weight_bytes=sum(s.weight_bytes for s in sources.values()),
        ).initialize()
        self.addCleanup(self.engram.close)
        self.empty = fixture.stage_module.V41EngramStage(
            {},
            None,
            device="cpu",
            backend="eager",
            max_host_bytes=0,
            max_weight_bytes=0,
        ).initialize()
        self.addCleanup(self.empty.close)

    def weights(self, role, pp_rank, ep_rank):
        begin, end = (
            (0, 40) if role == "decode" else (0, 21) if pp_rank == 0 else (21, 40)
        )
        ids = self.config.local_expert_ids(8 if role == "decode" else 4, ep_rank)
        layers = [{} for _ in range(40)]
        for source in self.config.layer_sources()[begin:end]:
            kind = (
                0
                if not source.compress_ratio
                else (
                    2
                    if source.owns_kv and source.compress_ratio == 2
                    else 20 if source.owns_kv else 24 if source.owns_index else 21
                )
            )
            weights = dict(self.prototypes[kind])
            for part in ("w1", "w3", "w2"):
                for kind in ("w", "s"):
                    name = getattr(W, "v4_routed_" + part + "_" + kind)
                    weights[name] = weights[name][ids.start : ids.stop]
            layers[source.layer_id] = weights
        globals_ = {}
        if begin == 0:
            globals_[W.embedding] = torch.randn(7, 32).bfloat16()
        if end == 40:
            globals_[W.final_ln_gamma] = torch.ones(32, dtype=torch.bfloat16)
            globals_[W.lm_head] = torch.randn(7, 32).bfloat16()
        return SimpleNamespace(weights=layers, global_weights=globals_)

    def build(self, role="prefill", pp_rank=1, ep_rank=3, weights=None, **kwargs):
        weights = self.weights(role, pp_rank, ep_rank) if weights is None else weights
        ranks = (
            tuple(range(8))
            if role == "decode"
            else tuple(range(4 * pp_rank, 4 * pp_rank + 4))
        )
        args = dict(
            role=role,
            pp_rank=pp_rank,
            ep_rank=ep_rank,
            world_ranks=ranks,
            generation="generation-1",
            process_group=object(),
            engram=self.engram if pp_rank == 0 else self.empty,
            rows_per_rank=4,
            max_positions=256,
            backend="eager",
        )
        args.update(kwargs)
        return model.build_v41_model_stage(self.config, weights, **args)

    def test_all_stage_ranges_bind_global_layer_ids_and_exact_ep_shards(self):
        for role, rank, ep_rank, bounds in (
            ("prefill", 0, 1, (0, 21)),
            ("prefill", 1, 3, (21, 40)),
            ("decode", 0, 7, (0, 40)),
        ):
            with self.subTest(role=role, rank=rank):
                weights = self.weights(role, rank, ep_rank)
                before = [dict(layer) for layer in weights.weights]
                loaded = self.build(role, rank, ep_rank, weights)
                stage = loaded.stage
                self.assertEqual((stage.begin, stage.end), bounds)
                self.assertEqual(
                    [b.layer_id for b in stage.blocks], list(range(*bounds))
                )
                self.assertIs(loaded.embedding, weights.global_weights.get(W.embedding))
                self.assertIs(
                    stage.final_norm, weights.global_weights.get(W.final_ln_gamma)
                )
                for op in stage.blocks:
                    tensors = weights.weights[op.layer_id]
                    self.assertIs(op.attention.wq_a.weight, tensors[W.v4_attn_wq_a_w])
                    self.assertIs(op.attention.frequencies, op.kv.frequencies)
                    self.assertEqual(op.attention.max_rows, 4)
                    self.assertEqual(op.max_rows, 4)
                    self.assertEqual(op.kv.max_rows, 16 if role == "prefill" else 4)
                    self.assertEqual(
                        op.attention.sources, self.config.layer_sources()[op.layer_id]
                    )
                    routed = op.feed_forward.routed
                    self.assertEqual(routed.plan.layer_id, op.layer_id)
                    self.assertEqual(
                        routed.local.expert_start_id,
                        (1 if role == "decode" else 2) * ep_rank,
                    )
                    self.assertFalse(routed._admitted)
                    experts = routed.local.experts
                    torch.testing.assert_close(
                        experts.w13,
                        torch.cat(
                            (tensors[W.v4_routed_w1_w], tensors[W.v4_routed_w3_w]),
                            dim=1,
                        ),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        experts.w2.data_ptr(), tensors[W.v4_routed_w2_w].data_ptr()
                    )
                    if op.layer_id >= 21:
                        self.assertEqual(op.kv.sources.kv_source_layer_id, 20)
                for actual, prior in zip(weights.weights, before):
                    self.assertEqual(set(actual), set(prior))
                    self.assertTrue(all(actual[k] is v for k, v in prior.items()))

    def test_rope_uses_checkpoint_bases_and_retains_yarn_below_training_length(self):
        loaded = self.build(pp_rank=0, ep_rank=0)
        plain, compressed = (
            loaded.stage.blocks[0].kv.frequencies,
            loaded.stage.blocks[2].kv.frequencies,
        )
        self.assertIs(plain, loaded.stage.blocks[1].kv.frequencies)
        self.assertIs(compressed, loaded.stage.blocks[20].kv.frequencies)
        self.assertFalse(torch.equal(plain, compressed))
        # Independent checkpoint equation; seqlen=256 must not disable YaRN.
        import math

        frequency = 1 / 160000 ** (torch.arange(0, 64, 2).float() / 64)
        corrected = (
            lambda rotations: 64
            * math.log(65536 / (rotations * 2 * math.pi))
            / (2 * math.log(160000))
        )
        low, high = max(math.floor(corrected(32)), 0), min(math.ceil(corrected(1)), 63)
        smooth = 1 - ((torch.arange(32).float() - low) / max(high - low, 1e-3)).clamp(
            0, 1
        )
        frequency = frequency / 16 * (1 - smooth) + frequency * smooth
        angles = torch.outer(torch.arange(256), frequency)
        torch.testing.assert_close(
            compressed, torch.polar(torch.ones_like(angles), angles), rtol=0, atol=0
        )

    def test_cp_kv_finish_accepts_full_rows_while_queries_remain_local(self):
        op = self.build(pp_rank=0, ep_rank=0).stage.blocks[0]
        projected = op.kv.project(
            torch.zeros(1, 32, dtype=torch.bfloat16),
            torch.tensor([0]),
            torch.tensor([True]),
        )
        full = dataclasses.replace(
            projected,
            swa=projected.swa.repeat(16, 1),
            positions=torch.arange(16),
            valid_rows=torch.ones(16, dtype=torch.bool),
        )
        self.assertEqual(op.kv.finish(full).swa.shape, (16, 512))
        with self.assertRaises(ValueError):
            op.attention.prepare(
                torch.zeros(16, 32, dtype=torch.bfloat16),
                full.positions,
                full.valid_rows,
            )

    def test_embedding_masks_padding_and_stage_one_requires_pp(self):
        loaded = self.build(pp_rank=0, ep_rank=0)
        valid = torch.tensor([True, False, True])
        initial = loaded.start(torch.tensor([6, -999, 1], dtype=torch.int32), valid)
        self.assertEqual(initial.next_layer_id, 0)
        torch.testing.assert_close(
            initial.residual[0], loaded.embedding[6].expand(4, 32), rtol=0, atol=0
        )
        self.assertEqual(initial.residual[1].count_nonzero(), 0)
        torch.testing.assert_close(
            initial.previous_pre_mix,
            torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 0], [1.0, 0, 0, 0]]),
            rtol=0,
            atol=0,
        )
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            loaded.start(torch.tensor([7]), torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "bounded"):
            loaded.start(
                torch.zeros(5, dtype=torch.int64), torch.ones(5, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "PP frame"):
            self.build().start(
                torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.bool)
            )

    def test_all_loader_layers_are_validated_before_conversion_and_failure_keeps_weights(
        self,
    ):
        for mutation in (
            lambda w: w.weights[39].pop(W.v4_attn_q_norm),
            lambda w: w.weights[39].update(
                {W.v4_attn_q_norm: torch.ones(64, dtype=torch.bfloat16)}
            ),
            lambda w: w.weights[39].update(
                {W.v4_routed_w1_s: w.weights[39][W.v4_routed_w1_s].float()}
            ),
            lambda w: w.weights[39].update(
                {W.v4_compressor_wkv: torch.ones(512, 32, dtype=torch.bfloat16)}
            ),
            lambda w: w.weights[0].update(self.prototypes[0]),
            lambda w: w.global_weights.update(
                {W.embedding: torch.ones(7, 32, dtype=torch.bfloat16)}
            ),
        ):
            weights = self.weights("prefill", 1, 3)
            mutation(weights)
            before = [dict(layer) for layer in weights.weights]
            with patch.object(model, "build_v41_projections") as convert:
                with self.assertRaises(ValueError):
                    self.build(weights=weights)
                convert.assert_not_called()
            for actual, prior in zip(weights.weights, before):
                self.assertEqual(set(actual), set(prior))
                self.assertTrue(all(actual[k] is v for k, v in prior.items()))
        weights = self.weights("prefill", 1, 3)
        with patch.object(
            model, "build_v41_block", side_effect=RuntimeError("provider failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                self.build(weights=weights)
        self.assertIn(W.v4_routed_w1_w, weights.weights[21])

    def test_config_and_topology_mismatches_fail_before_conversion(self):
        for key, value in (
            ("o_lora_rank", 64),
            ("q_lora_rank", 256),
            ("hc_eps", 1e-5),
            ("sliding_window", 64),
        ):
            original = self.config
            self.config = copy.deepcopy(original)
            self.config.text[key] = value
            try:
                with patch.object(model, "build_v41_projections") as convert:
                    with self.assertRaises(ValueError):
                        self.build()
                    convert.assert_not_called()
            finally:
                self.config = original
        for kwargs in (
            dict(world_ranks=(0, 1, 2, 3, 4, 5, 6, 7)),
            dict(ep_rank=4),
            dict(rows_per_rank=33),
            dict(process_group=None),
            dict(generation=""),
            dict(engram=self.engram),
        ):
            with self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_late_unadmitted_or_rebound_ep_fails_before_first_cache_write(self):
        loaded = self.build(pp_rank=0, ep_rank=0)
        stage = loaded.stage
        for op in stage.blocks:
            routed = op.feed_forward.routed
            routed._admitted = True
            routed._identity = routed._runtime_identity()
        batch = fixture.rows()
        initial = loaded.start(torch.tensor([0, 1, -999]), batch.valid_rows)
        plans, _ = fixture.plans_fixture(stage.blocks, batch)
        before = {layer: plan.swa.raw.clone() for layer, plan in plans.items()}
        late = stage.blocks[-1].feed_forward.routed
        for rebound in (False, True):
            late._admitted = rebound
            if rebound:
                late.process_group = object()
            with patch.object(stage.blocks[0], "prepare") as first:
                with self.assertRaisesRegex(RuntimeError, "admission"):
                    stage.run(
                        initial,
                        batch,
                        plans,
                        windows=torch.zeros(3, 4, dtype=torch.int64),
                    )
                first.assert_not_called()
            for layer, plan in plans.items():
                torch.testing.assert_close(plan.swa.raw, before[layer], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

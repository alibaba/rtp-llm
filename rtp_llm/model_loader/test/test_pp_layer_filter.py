"""Stage-local weight ownership through real checkpoint readers and direct modules."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from safetensors.torch import save_file

from rtp_llm.config.layer_partition import local_layer_range
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def topology(counts, stage, ep=1):
    return NS(
        pp_size=len(counts),
        pp_stage_layer_counts=counts,
        pp_rank=stage,
        dp_size=1,
        dp_rank=0,
        tp_size=ep,
        tp_rank=0,
        ep_size=ep,
        ep_rank=0,
        world_size=len(counts) * ep,
        world_rank=stage * ep,
        local_world_size=ep,
    )


def expert_stage(
    counts, stage, moe, *, mode="EPLB", groups=1, replicas=2, ep=1, pc=None
):
    from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer
    from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts

    pc = pc or topology(counts, stage, ep)
    cfg = NS(
        num_layers=sum(counts),
        moe_layer_index=moe,
        activation_type="SiGLU",
        is_mtp=False,
        moe_n_group=groups,
        eplb_config=NS(
            enable_eplb=lambda: mode != "NONE",
            eplb_mode=mode,
            redundant_expert=replicas,
            balance_method="round",
            eplb_force_repack=0,
            eplb_stats_window_size=2,
        ),
    )
    owners = torch.nn.ModuleDict(
        {
            str(i): BaseMoEExperts(
                num_experts=2,
                top_k=1,
                hidden_size=4,
                moe_intermediate_size=4,
                tp_size=ep,
                tp_rank=0,
                ep_size=ep,
                ep_rank=0,
                params_dtype=torch.float32,
                model_config=cfg,
                parallelism_config=pc,
                moe_config=NS(),
                quant_config=None,
                layer_idx=i,
            )
            for i in local_layer_range(cfg, pc)
            if i in moe
        }
    )
    loader = NewModelLoader(
        cfg,
        NewLoaderConfig(device="cpu", tp_size=ep, ep_size=ep, parallelism_config=pc),
    )
    return owners, create_stage_expert_balancer(loader, owners, cfg)


class PPExpertTest(unittest.TestCase):
    def test_stage_balancer_and_redundant_routing(self):
        for counts, moe in (
            ([1, 3, 4], [1, 3, 5, 7]),
            ([8], [1, 3, 5, 7]),
            ([8], []),
            ([2], [0, 1]),
        ):
            for stage in range(len(counts)):
                expected = [
                    i
                    for i in moe
                    if sum(counts[:stage]) <= i < sum(counts[: stage + 1])
                ]
                for mode in ("NONE", "EPLB"):
                    owners, balancer = expert_stage(counts, stage, moe, mode=mode)
                    self.assertEqual(list(owners), list(map(str, expected)))
                    self.assertEqual(balancer is None, not expected or mode == "NONE")
                    if balancer is not None:
                        self.assertEqual(balancer.moe_layer_index, tuple(expected))
                    for owner in owners.values():
                        torch.testing.assert_close(
                            owner._eplb_logic_expert_cnt,
                            torch.tensor([2, 2], dtype=torch.int32),
                        )
                        torch.testing.assert_close(
                            owner._eplb_log2phy,
                            torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32),
                        )

    def test_missing_routed_owners_cannot_publish_partial_eplb(self):
        from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer

        owners, controller = expert_stage([2, 2], 0, [0, 1, 3])
        loader = controller._loader
        cfg = controller._model_config
        with self.assertRaisesRegex(ValueError, r"missing=\[0, 1\]"):
            create_stage_expert_balancer(loader, torch.nn.Module(), cfg)
        del owners["1"]
        with self.assertRaisesRegex(ValueError, r"missing=\[1\]"):
            create_stage_expert_balancer(loader, owners, cfg)
        # A genuinely dense first stage has no controller, as in PPExecutor.
        cfg.moe_layer_index = [3]
        self.assertIsNone(create_stage_expert_balancer(loader, torch.nn.Module(), cfg))
        with self.assertRaisesRegex(ValueError, r"unexpected=\[0\]"):
            create_stage_expert_balancer(loader, owners, cfg)
        cfg.moe_layer_index = [cfg.num_layers]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            create_stage_expert_balancer(loader, torch.nn.Module(), cfg)
        cfg.moe_layer_index = []
        cfg.moe_style = 1
        with self.assertRaisesRegex(ValueError, "no declared MoE layers"):
            create_stage_expert_balancer(loader, torch.nn.Module(), cfg)

    def test_all_selection_policies_stay_on_stage(self):
        from rtp_llm.models_py.eplb_controller import _SelectLayerMethod

        _, balancer = expert_stage([4, 4], 1, [1, 3, 5, 7])
        loads = torch.tensor([[0], [1000], [0], [900], [0], [10], [0], [20]])
        self.assertEqual(
            [balancer._select_layer(loads) for _ in range(4)], [5, 7, 5, 7]
        )
        for method in (
            _SelectLayerMethod.RANDOM,
            _SelectLayerMethod.MIX,
            _SelectLayerMethod.MOST_UNBALANCED_LAYER,
        ):
            balancer.select_layer_method = method
            for step in range(8):
                balancer._update_count = step
                picked = balancer._select_layer(loads)
                self.assertIn(picked, (5, 7))
                if method == _SelectLayerMethod.MOST_UNBALANCED_LAYER:
                    self.assertEqual(picked, 7)

    def test_ep_node_count_and_mode_scoped_hierarchy(self):
        for ep, nodes in ((1, 1), (2, 1), (2, 2)):
            pc = topology([2], 0, ep)
            pc.local_world_size = ep // nodes
            owners, ctl = expert_stage([2], 0, [1], ep=ep, pc=pc)
            self.assertEqual((ctl.num_gpu, ctl.num_nodes), (ep, nodes))
            expected = (
                (0, 1, 0, 1)
                if ep == 1
                else ((0, 1, 1, 0) if nodes == 1 else (0, 0, 1, 1))
            )
            self.assertEqual(tuple(ctl.phy2log[1]), expected)
            self.assertEqual(
                owners["1"]._expert_placement.physical_to_logical, expected
            )
        pc = topology([2], 0)
        pc.world_size = 2
        for groups in (0, 8):
            _, ctl = expert_stage(
                [2], 0, [1], mode="STATS", groups=groups, replicas=0, pc=pc
            )
            self.assertIsNotNone(ctl)
        for mode in ("EPLB", "ALL"):
            with self.assertRaisesRegex(ValueError, "ep_size=1.*num_nodes=2"):
                expert_stage([2], 0, [1], mode=mode, groups=8, replicas=0, pc=pc)


class PPCheckpointTest(unittest.TestCase):
    def test_stacked_and_dense_scopes_match_both_readers(self):
        for moe in ((0, 1), (1,)):
            with tempfile.TemporaryDirectory() as d:
                f = str(Path(d) / "model.safetensors")
                ckpt = {}
                for i in range(2):
                    key = f"model.layers.{i}."
                    if i in moe:
                        ckpt[key + "experts.down_proj"] = torch.tensor(
                            [[10 * i], [10 * i + 1]]
                        )
                    else:
                        ckpt[key + "dense"] = torch.tensor([10 * i])
                save_file(ckpt, f)
                for scope in (range(1), range(1, 2), range(2)):
                    for rank in (0, 1):
                        expert = _ExpertRangeFilter(2, 2, rank)

                        def keep(name):
                            return int(name.split(".")[2]) in scope

                        expected = {
                            (
                                f"model.layers.{i}.experts.{rank}.down_proj.weight"
                                if i in moe
                                else f"model.layers.{i}.dense"
                            ): torch.tensor([10 * i + (rank if i in moe else 0)])
                            for i in scope
                        }
                        for rows in (
                            get_all_weights(
                                [f], name_filter=keep, safetensor_slice_expander=expert
                            ),
                            iter_fst_range_tensors(
                                [f], name_filter=keep, slice_expander=expert
                            ),
                        ):
                            values = dict(rows)
                            self.assertEqual(set(values), set(expected))
                            for key, value in values.items():
                                torch.testing.assert_close(value, expected[key])

    def test_ft_layers_and_globals_first_middle_last_full(self):
        first = {"embedding", "position_encoding.weight"}
        last = {"lm_head", "lm_head_b", "final_layernorm.gamma", "final_layernorm.beta"}
        common = {"__custom__.classifier.weight", "some_other_global"}
        with tempfile.TemporaryDirectory() as d:
            for tp in (0, 1):
                records = [
                    (i, "weight", torch.tensor([i + 100 * tp])) for i in range(12)
                ]
                records.extend(
                    (None, key, torch.tensor([99 + tp]))
                    for key in first | last | common
                )
                write_ft_shards(records, d, tp_rank=tp)
            generate_ft_index(d, {"TP_SIZE": 2, "EP_SIZE": 1})
            for scope in (range(2), range(2, 12), range(2, 6), range(12)):
                globals_ = (
                    common
                    | (first if 0 in scope else set())
                    | (last if 11 in scope else set())
                )
                expected = {f"layers.{i}.weight": i for i in scope}
                expected.update({"global." + name: 99 for name in globals_})
                for method in ("scratch", "fastsafetensors"):
                    ckpt = FTCheckpoint.open(d, CheckpointTopology(tp_size=2))
                    ckpt.read_backend = method
                    ckpt.select_stage(scope, 12)
                    self.assertEqual(set(ckpt.entries), set(expected))
                    self.assertEqual(
                        {name: ckpt.read(name).item() for name in ckpt.entries},
                        expected,
                    )


def qwen_config(n):
    return NS(
        model_type="qwen_2",
        num_layers=n,
        is_mtp=False,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=NS(head_num=2, kv_head_num=1, size_per_head=2),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=False,
        capture_aux_hidden_layer_ids=[],
        hidden_state_capture_layer_ids=[],
        logit_scale=2.0,
    )


def qwen_tensors(n):
    tensors = {
        "model.embed_tokens.weight": torch.arange(32).reshape(8, 4).float(),
        "model.norm.weight": torch.ones(4),
        "lm_head.weight": torch.full((8, 4), 3.0),
        "classifier.weight": torch.arange(8).reshape(2, 4).float(),
    }
    for i in range(n):
        for name, shape in (
            ("input_layernorm.weight", (4,)),
            ("post_attention_layernorm.weight", (4,)),
            ("self_attn.q_proj.weight", (4, 4)),
            ("self_attn.k_proj.weight", (2, 4)),
            ("self_attn.v_proj.weight", (2, 4)),
            ("self_attn.q_proj.bias", (4,)),
            ("self_attn.k_proj.bias", (2,)),
            ("self_attn.v_proj.bias", (2,)),
            ("self_attn.o_proj.weight", (4, 4)),
            ("mlp.gate_proj.weight", (4, 4)),
            ("mlp.up_proj.weight", (4, 4)),
            ("mlp.down_proj.weight", (4, 4)),
        ):
            tensors[f"model.layers.{i}.{name}"] = torch.full(shape, float(i + 1))
    return tensors


def qwen_loader(cfg, pc, method, d, aliases=None):
    return NewModelLoader(
        cfg,
        NewLoaderConfig(
            device="cpu",
            load_method=method,
            force_cpu_load_weights=True,
            compute_dtype=torch.float32,
            parallelism_config=pc,
            custom_weight_mappings=(
                ("__custom__.classifier.weight", "classifier.weight"),
            ),
        ),
        model_path=d,
        parameter_aliases=aliases,
    )


class PPQwenStageTest(unittest.TestCase):
    def test_explicit_layer_reduction_loads_full_qwen_checkpoint(self):
        from rtp_llm.config.model_config import apply_layer_num_override

        cfg = qwen_config(3)
        cfg.hybrid_attention_config = NS(enable_hybrid_attention=False)
        cfg.moe_layer_index = []
        apply_layer_num_override(cfg, 2)
        apply_layer_num_override(cfg, 1)
        checkpoint = qwen_tensors(3)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    save_file(checkpoint, path)
                    loader = qwen_loader(cfg, topology([1], 0), method, directory)
                    model = loader.load()
                    self.assertEqual(loader.loaded_method.value, method)
                    self.assertEqual(tuple(model.layer_ids()), (0,))
                    torch.testing.assert_close(
                        model.layers[0].input_layernorm.weight,
                        checkpoint["model.layers.0.input_layernorm.weight"],
                    )
                    torch.testing.assert_close(
                        model.lm_head.weight,
                        checkpoint["lm_head.weight"] * cfg.logit_scale,
                    )
                    with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
                        qwen_loader(
                            qwen_config(1), topology([1], 0), method, directory
                        ).load()
                    for key in (
                        "model.layers.3.input_layernorm.weight",
                        "model.layers.0.unknown.weight",
                    ):
                        with self.subTest(key=key):
                            save_file({**checkpoint, key: torch.ones(4)}, path)
                            with self.assertRaisesRegex(
                                RuntimeError, "could not dispatch"
                            ):
                                qwen_loader(
                                    cfg, topology([1], 0), method, directory
                                ).load()

    def test_real_scratch_and_fst_stage_owners_alias_and_head_postprocess(self):
        from rtp_llm.models_py.borrowed_parameters import (
            BorrowedParameter,
            bind_borrowed_parameters,
            exclude_borrowed_checkpoint_weights,
        )

        class BorrowingDraft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(8, 4)

            def prepare_borrowed_parameters(self, aliases):
                if set(aliases) != {"embed_tokens.weight"}:
                    raise ValueError("unexpected loan")

        for counts in ([2, 3, 2], [2]):
            cfg = qwen_config(sum(counts))
            with tempfile.TemporaryDirectory() as d:
                checkpoint = qwen_tensors(cfg.num_layers)
                tied = len(counts) == 1
                cfg.tie_word_embeddings = tied
                if tied:
                    del checkpoint["lm_head.weight"]
                save_file(checkpoint, str(Path(d) / "model.safetensors"))
                for method in ("scratch", "fastsafetensors"):
                    for stage in range(len(counts)):
                        pc = topology(counts, stage)
                        loader = qwen_loader(cfg, pc, method, d)
                        model = loader.load()
                        self.assertEqual(loader.loaded_method.value, method)
                        ids = local_layer_range(cfg, pc)
                        self.assertEqual(tuple(model.layer_ids()), tuple(ids))
                        for i in ids:
                            torch.testing.assert_close(
                                model.get_parameter(
                                    f"layers.{i}.input_layernorm.weight"
                                ),
                                checkpoint[f"model.layers.{i}.input_layernorm.weight"],
                            )
                        self.assertEqual(model.embed_tokens is not None, stage == 0)
                        self.assertEqual(
                            model.lm_head is not None, stage == len(counts) - 1
                        )
                        view = model.runtime_weight_view()
                        torch.testing.assert_close(
                            view["__custom__.classifier.weight"],
                            checkpoint["classifier.weight"],
                        )
                        if model.lm_head is not None:
                            source = (
                                "model.embed_tokens.weight"
                                if tied
                                else "lm_head.weight"
                            )
                            torch.testing.assert_close(
                                model.lm_head.weight,
                                checkpoint[source] * cfg.logit_scale,
                            )
                        if not tied and stage == 0:
                            alias = BorrowedParameter(model, "embed_tokens.weight")
                            draft = BorrowingDraft()
                            bind_borrowed_parameters(
                                draft, {"embed_tokens.weight": alias}, "cpu"
                            )
                            self.assertIs(
                                draft.embed_tokens.weight, model.embed_tokens.weight
                            )
                            keep = exclude_borrowed_checkpoint_weights(draft, None)
                            self.assertFalse(keep("embed_tokens.weight"))
                            self.assertTrue(keep("lm_head.weight"))

    def test_non_language_model_has_no_language_projection(self):
        from rtp_llm.ops import TaskType

        cfg = qwen_config(2)
        cfg.task_type = TaskType.DENSE_EMBEDDING
        with tempfile.TemporaryDirectory() as d:
            checkpoint = qwen_tensors(2)
            del checkpoint["lm_head.weight"]
            save_file(checkpoint, str(Path(d) / "model.safetensors"))
            model = qwen_loader(cfg, topology([2], 0), "scratch", d).load()
            self.assertIsNone(model.lm_head)
            self.assertIsNotNone(model.norm)
            torch.testing.assert_close(
                model.runtime_weight_view()["__custom__.classifier.weight"],
                checkpoint["classifier.weight"],
            )

    def test_lora_global_layer_ids_and_scaling(self):
        from rtp_llm.models_py.lora_checkpoint import model_lora_checkpoint_updates

        cfg = qwen_config(7)
        with tempfile.TemporaryDirectory() as d:
            loader = qwen_loader(cfg, topology([2, 3, 2], 1), "scratch", d)
            model = loader._create_model()
            adapter = {}
            for i in range(7):
                stem = f"base_model.model.model.layers.{i}.self_attn.o_proj"
                adapter[stem + ".lora_A.weight"] = torch.full((1, 4), float(i))
                adapter[stem + ".lora_B.weight"] = torch.ones(4, 1)
            (Path(d) / "adapter_config.json").write_text(
                json.dumps({"r": 1, "lora_alpha": 2})
            )
            save_file(adapter, str(Path(d) / "adapter_model.safetensors"))
            updates = {
                u.target: u for u in model_lora_checkpoint_updates(model, loader, d)
            }
            self.assertEqual(
                set(updates), {f"layers.{i}.self_attn.o_proj.weight" for i in (2, 3, 4)}
            )
            for i in (2, 3, 4):
                u = updates[f"layers.{i}.self_attn.o_proj.weight"]
                torch.testing.assert_close(u.a, torch.full((1, 4), float(i)))
                torch.testing.assert_close(u.b * u.scale, torch.full((4, 1), 2.0))


if __name__ == "__main__":
    unittest.main()

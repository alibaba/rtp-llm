import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.ft_checkpoint import (
    CheckpointTopology,
    FTCheckpoint,
    RuntimeTensorBinding,
    ft_layer_bindings,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


@register_model("_rank_local_ft_probe")
class _Stage(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.config = config
        self.stage = load.parallelism_config.pp_rank
        self.expert = torch.nn.Parameter(torch.empty((3, 2)))
        self.boundary = torch.nn.Parameter(torch.empty(2))
        self.placement = ExpertPlacement(4, 2, load.ep_rank, (3, 0, 1, 2, 3, 0))

    def layer_ids(self):
        return range(self.stage, self.stage + 1)

    def load_ft_checkpoint(self, checkpoint):
        layer_id = self.stage
        expert = f"layers.{layer_id}.w13"
        glob = "global.embedding" if self.stage == 0 else "global.lm_head"
        checkpoint.load_parameters(
            self,
            (
                RuntimeTensorBinding(
                    (expert,),
                    "expert",
                    lambda vs: vs[0],
                    lambda: (
                        checkpoint.read_expert_rows(expert, layer_id, self.placement),
                    ),
                ),
                RuntimeTensorBinding(
                    (glob,),
                    "boundary",
                    lambda vs: vs[0],
                    lambda: (
                        torch.cat(
                            [
                                checkpoint.read_rank_slice(glob, peer, (slice(None),))
                                for peer in range(2)
                            ]
                        ),
                    ),
                ),
            ),
        )


def _gloo_load(rank, folder, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        pc = SimpleNamespace(pp_size=2, pp_rank=1, dp_size=1, dp_rank=0)
        config = SimpleNamespace(
            model_type="_rank_local_ft_probe", num_layers=2, expert_num=4
        )
        loader = NewModelLoader(
            config,
            NewLoaderConfig(
                tp_size=2,
                ep_size=2,
                ep_rank=rank,
                tp_rank=rank,
                device="cpu",
                load_method="fastsafetensors",
                parallelism_config=pc,
            ),
            model_path=folder,
        )
        model = loader.load()
        if loader._selected_load_method.value != "fastsafetensors":
            raise AssertionError("FST route was not honored")
        physical = [torch.empty_like(model.expert) for _ in range(2)]
        dist.all_gather(physical, model.expert.detach().contiguous())
        reference = torch.arange(100, 108, dtype=torch.float32).reshape(4, 2)
        torch.testing.assert_close(
            torch.cat(physical), reference[[3, 0, 1, 2, 3, 0]], atol=0, rtol=0
        )
    finally:
        dist.destroy_process_group()


@register_model("_ft_eplb_test_owner")
class _ExpertModel(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts

        self.experts = BaseMoEExperts(
            num_experts=4, top_k=1, hidden_size=16, moe_intermediate_size=32,
            tp_size=load.tp_size, tp_rank=load.tp_rank, ep_size=load.ep_size,
            ep_rank=load.ep_rank, params_dtype=torch.float32, model_config=config,
            parallelism_config=load.parallelism_config, moe_config=None,
            quant_config=None, layer_idx=0, prefix="experts",
        )

    def load_ft_checkpoint(self, cp):
        from rtp_llm.models_py.quant_methods.channel_tensor_moe_ft import nonblock_moe_ft_bindings

        # Simulate a draft: runtime layer zero belongs to archive layer seven.
        bindings = nonblock_moe_ft_bindings(self.experts, "layers.7.", "experts", cp, 7)
        cp.load_parameters(self, bindings)


class RankLocalFTTest(unittest.TestCase):
    def _eplb_loader(self, root, method, *, rank=0, size=1):
        pc = SimpleNamespace(dp_size=1, dp_rank=0, pp_size=1, pp_rank=0)
        cfg = SimpleNamespace(
            model_type="_ft_eplb_test_owner", num_layers=8, expert_num=4,
            quant_config=None, activation_type="SiGLU",
            eplb_config=SimpleNamespace(enable_eplb=True),
        )
        return NewModelLoader(
            cfg, NewLoaderConfig(
                load_method=method, device="cpu", tp_size=size, ep_size=size,
                tp_rank=rank, ep_rank=rank, parallelism_config=pc,
            ), model_path=root,
        )

    def test_ft_eplb_reload_reads_peer_rows_with_selected_backend(self):
        self._check_ft_eplb_reload("EPLB")

    def test_ft_stats_startup_can_promote_to_eplb_and_install_peer_rows(self):
        self._check_ft_eplb_reload("STATS")

    def test_ft_without_reload_layout_preserves_stats_only_serving(self):
        from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer

        original = _ExpertModel.load_ft_checkpoint

        def import_without_reload_layout(model, checkpoint):
            original(model, checkpoint)
            del model.experts._ft_expert_binding_layout

        with tempfile.TemporaryDirectory() as root, patch.object(
            _ExpertModel, "load_ft_checkpoint", import_without_reload_layout
        ):
            records = [
                (7, "partial_moe_weights.intermediate_weight.kernel", torch.zeros(4, 64, 16)),
                (7, "partial_moe_weights.intermediate_weight2.kernel", torch.zeros(4, 16, 32)),
            ]
            write_ft_shards(records, root)
            generate_ft_index(root, {"TP_SIZE": 1, "DP_SIZE": 1, "EP_SIZE": 1})
            loader = self._eplb_loader(root, "scratch")
            cfg = loader.model_config
            cfg.is_mtp = False
            cfg.moe_layer_index = [0]
            cfg.eplb_config = SimpleNamespace(enable_eplb=lambda: True, eplb_mode="STATS")
            model = loader.load()
            self.assertTrue(create_stage_expert_balancer(loader, model, cfg).eplb_stats_only)
            self.assertIsNone(loader._eplb_ft_checkpoint)
            with self.assertRaisesRegex(ValueError, "reload layout"):
                loader.load_moe_layer_weights(
                    model, layer_id=0, physical_to_logical=[3, 1, 0, 2], ep_rank=0, ep_size=1
                )
            cfg.eplb_config.eplb_mode = "EPLB"
            with self.assertRaisesRegex(ValueError, "lacks a reload layout"):
                loader.load()

    def _check_ft_eplb_reload(self, startup_mode):
        from contextlib import nullcontext
        from rtp_llm.models_py import fst_range_reader
        from rtp_llm.utils.model_weight import W

        w13 = torch.stack([torch.full((64, 16), float(i + 1)) for i in range(4)])
        w2 = torch.stack([torch.full((16, 32), float(i + 11)) for i in range(4)])
        with tempfile.TemporaryDirectory() as root:
            for rank in range(2):
                records = [
                    (7, "partial_moe_weights." + name + ".kernel", value[rank*2:rank*2+2])
                    for name, value in (("intermediate_weight", w13), ("intermediate_weight2", w2))
                ]
                write_ft_shards(records, root, tp_rank=rank, ep_rank=rank)
            generate_ft_index(root, {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 2})
            idx = Path(root, "model.safetensors.index.json")
            metadata = json.loads(idx.read_text())
            metadata["ft_export_backend"] = "cuda"
            idx.write_text(json.dumps(metadata))
            for method in ("scratch", "fastsafetensors"):
                for rank in range(2):
                    with self.subTest(method=method, rank=rank):
                        loader = self._eplb_loader(root, method, rank=rank, size=2)
                        loader.model_config.eplb_config.eplb_mode = startup_mode
                        model = loader.load()
                        # CPU startup is checkpoint staging: explicitly prepare the
                        # live tensor view without constructing a GPU MoE executor.
                        model.experts.prepare_eplb_reload_weights()
                        loader.model_config.eplb_config.eplb_mode = "EPLB"
                        rows = [3, 1, 0, 2]
                        selected = rows[rank*2:rank*2+2]
                        ctx = patch.object(
                            fst_range_reader, "iter_fst_range_tensors",
                            wraps=fst_range_reader.iter_fst_range_tensors,
                        ) if method == "fastsafetensors" else nullcontext()
                        with ctx as reader:
                            staged = loader.load_moe_layer_weights(
                                model, layer_id=0, physical_to_logical=rows,
                                ep_rank=rank, ep_size=2,
                            )
                        if reader is not None:
                            self.assertGreater(reader.call_count, 0)
                        for key, expected in ((W.moe_w1, w13), (W.moe_w2, w2)):
                            torch.testing.assert_close(staged[key], expected[selected])
                        torch.testing.assert_close(model.experts.w2, w2[rank*2:rank*2+2])
                        placement = ExpertPlacement(4, 2, rank, tuple(rows))
                        counts, inverse = model.experts._routing_tensors(placement)
                        model.experts.apply_eplb_plan(
                            torch.tensor(rows, dtype=torch.int32), counts, inverse, staged
                        )
                        torch.testing.assert_close(model.experts.w13, w13[selected])
                        torch.testing.assert_close(model.experts.w2, w2[selected])
                        self.assertEqual(model.experts._expert_placement, placement)

    def test_pytorch_ep1_eplb_reload_stacked_and_per_expert(self):
        from rtp_llm.utils.model_weight import W

        for stacked in (False, True):
            with self.subTest(stacked=stacked), tempfile.TemporaryDirectory() as root:
                w13 = torch.stack([torch.full((64, 16), float(i + 1)) for i in range(4)])
                w2 = torch.stack([torch.full((16, 32), float(i + 11)) for i in range(4)])
                if stacked:
                    state = {"experts.gate_up_proj.weight": w13, "experts.down_proj.weight": w2}
                else:
                    state = {}
                    for i in range(4):
                        for name, value in (("gate_proj", w13[i, :32]), ("up_proj", w13[i, 32:]), ("down_proj", w2[i])):
                            state[f"experts.{i}.{name}.weight"] = value
                torch.save({"state_dict": state}, Path(root, "pytorch_model.bin"))
                loader = self._eplb_loader(root, "scratch")
                model = loader.load()
                rows = [3, 0, 1, 2]
                staged = loader.load_moe_layer_weights(
                    model, layer_id=0, physical_to_logical=rows, ep_rank=0, ep_size=1
                )
                for key, expected in ((W.moe_w1, w13), (W.moe_w2, w2)):
                    torch.testing.assert_close(staged[key], expected[rows])
                torch.testing.assert_close(model.experts.w2, w2)
                with self.assertRaisesRegex(RuntimeError, "fastsafetensors"):
                    self._eplb_loader(root, "fastsafetensors").load()

    def test_replicated_ft_classifier_preserves_serialized_role_and_dtype(self):
        from rtp_llm.models_py.downstream_heads import ReplicatedHandlerHeads
        from rtp_llm.models_py.weight_mapper import WeightsMapper

        role = "__custom__.score"
        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with tempfile.TemporaryDirectory() as root:
            write_ft_shards([(None, role, value)], root)
            generate_ft_index(root, {"TP_SIZE": 1, "DP_SIZE": 1, "EP_SIZE": 1})
            for backend in ("scratch", "fastsafetensors"):
                ckpt = FTCheckpoint.open(root, CheckpointTopology())
                ckpt.read_backend = backend
                model = RtpModule()
                model.handler_heads = ReplicatedHandlerHeads(
                    ((role, "classifier.weight", torch.float32),),
                    WeightsMapper(), 3, torch.bfloat16,
                )
                ckpt.load_parameters(model, tuple(model.handler_heads.ft_bindings(ckpt)))
                actual = model.handler_heads.exports()[role]
                torch.testing.assert_close(actual, value, rtol=0, atol=0)

    def test_pp_keys_are_actual_module_paths_not_stage_local_offsets(self):
        leaf = torch.nn.Linear(2, 2)
        model = SimpleNamespace(
            layer_ids=lambda: range(2, 3), layers=torch.nn.ModuleDict({"2": leaf})
        )
        self.assertEqual(ft_layer_bindings(model), ((2, "2", leaf),))
        model.layers = torch.nn.ModuleList([leaf])
        self.assertEqual(ft_layer_bindings(model), ((2, "0", leaf),))
        model.layers = torch.nn.ModuleDict({"0": leaf})
        with self.assertRaisesRegex(ValueError, "global ids"):
            ft_layer_bindings(model)

    def export(self, path):
        for tp, ep in ((0, 0), (1, 1)):
            layers = [
                (
                    i,
                    "w13",
                    torch.arange(4, dtype=torch.float32).reshape(2, 2)
                    + 100 * i
                    + 4 * ep,
                )
                for i in range(2)
            ]
            write_ft_shards(
                layers
                + [
                    (None, "embedding", torch.tensor([float(tp)])),
                    (None, "lm_head", torch.tensor([float(10 + tp)])),
                ],
                path,
                tp_rank=tp,
                ep_rank=ep,
            )
        generate_ft_index(str(path), {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": "0"})

    def test_two_gloo_ranks_reload_physical_slots_from_real_ft_shards(self):
        with tempfile.TemporaryDirectory() as d:
            self.export(d)
            torch.multiprocessing.spawn(
                _gloo_load,
                args=(d, (Path(d) / "rendezvous").as_uri()),
                nprocs=2,
                join=True,
            )

    def test_full_export_proves_global_layers_despite_raw_pp_env(self):
        with tempfile.TemporaryDirectory() as d:
            self.export(d)
            index = Path(d) / "model.safetensors.index.json"
            data = json.loads(index.read_text())
            data["__env__params__"]["PP_SIZE"] = "2"
            index.write_text(json.dumps(data))
            topology = CheckpointTopology(tp_size=2, ep_size=2)
            checkpoint = FTCheckpoint.open(d, topology, num_layers=2)
            checkpoint.select_stage(range(1, 2), 2)
            self.assertEqual(
                set(checkpoint.entries), {"layers.1.w13", "global.lm_head"}
            )
            del data["weight_map"]["rank_01_00_01.layers.1.w13"]
            index.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "full-model layer coverage"):
                FTCheckpoint.open(d, topology, num_layers=2)

    def test_full_export_loads_each_stage_and_moved_or_redundant_expert(self):
        with tempfile.TemporaryDirectory() as d:
            self.export(d)
            for method in ("scratch", "fastsafetensors"):
                for pp in (0, 1):
                    for rank in (0, 1):
                        with self.subTest(method=method, pp=pp, rank=rank):
                            pc = SimpleNamespace(
                                pp_size=2, pp_rank=pp, dp_size=1, dp_rank=0
                            )
                            config = SimpleNamespace(
                                model_type="_rank_local_ft_probe",
                                num_layers=2,
                                expert_num=4,
                            )
                            loader = NewModelLoader(
                                config,
                                NewLoaderConfig(
                                    tp_size=2,
                                    ep_size=2,
                                    ep_rank=rank,
                                    tp_rank=rank,
                                    device="cpu",
                                    load_method=method,
                                    parallelism_config=pc,
                                ),
                                model_path=d,
                            )
                            model = loader.load()
                            self.assertEqual(loader._selected_load_method.value, method)
                            self.assertEqual(
                                model.boundary.tolist(), [10.0 * pp, 10.0 * pp + 1]
                            )
                            all_rows = torch.arange(8).reshape(4, 2) + pp * 100
                            slots = [3, 0, 1] if rank == 0 else [2, 3, 0]
                            self.assertEqual(
                                model.expert.tolist(), all_rows[slots].tolist()
                            )

    def test_explicit_source_placement_and_negative_cases(self):
        with tempfile.TemporaryDirectory() as d:
            self.export(d)
            index = Path(d) / "model.safetensors.index.json"
            meta = json.loads(index.read_text())
            manifests = {}
            for tp in (0, 1):
                rank_key = "rank_%02d_00_%02d" % (tp, tp)
                manifests[rank_key] = dict(
                    binding_schema_version=1,
                    tp_size=2,
                    tp_rank=tp,
                    dp_size=1,
                    dp_rank=0,
                    ep_size=2,
                    ep_rank=tp,
                    modules={
                        "w13": dict(
                            layer_id=0,
                            logical_expert_ids=([3, 0] if tp == 0 else [1, 2]),
                            parameter_bindings={"weight": rank_key + ".layers.0.w13"},
                        )
                    },
                )
            meta["module_export_metadata"] = manifests
            index.write_text(json.dumps(meta))
            top = CheckpointTopology(tp_size=2, ep_size=2)
            place = ExpertPlacement(4, 2, 0, (0, 1, 2, 3))
            ckpt = FTCheckpoint.open(d, top)
            self.assertEqual(
                ckpt.read_expert_rows("layers.0.w13", 0, place).tolist(),
                [[2, 3], [4, 5]],
            )
            with self.assertRaisesRegex(ValueError, "no tensors for rank"):
                FTCheckpoint.open(
                    d, CheckpointTopology(tp_size=2, ep_size=2, ep_rank=1)
                )
            manifests["rank_01_00_01"]["modules"]["w13"]["logical_expert_ids"] = [1, 0]
            index.write_text(json.dumps(meta))
            ckpt = FTCheckpoint.open(d, top)
            with self.assertRaisesRegex(ValueError, "omits logical experts"):
                ckpt.read_expert_rows("layers.0.w13", 0, place)
            ckpt.select_stage(range(1, 2), 2)
            with self.assertRaisesRegex(ValueError, "not owned"):
                ckpt.read_expert_rows("layers.0.w13", 0, place)

    def test_tp_peer_stays_in_dp_replica_and_never_crosses_expert_tp(self):
        with tempfile.TemporaryDirectory() as d:
            for dp in range(2):
                for tp in range(2):
                    ep = 2 * dp + tp
                    write_ft_shards(
                        [
                            (None, "lm_head", torch.tensor([ep])),
                            (0, "w13", torch.tensor([[ep]])),
                        ],
                        d,
                        tp_rank=tp,
                        dp_rank=dp,
                        ep_rank=ep,
                    )
            generate_ft_index(d, {"TP_SIZE": 2, "DP_SIZE": 2, "EP_SIZE": 0})
            top = CheckpointTopology(
                tp_size=2, dp_size=2, ep_size=4, dp_rank=1, ep_rank=2
            )
            ckpt = FTCheckpoint.open(d, top)
            self.assertEqual(
                ckpt.read_rank_slice("global.lm_head", 1, (slice(None),)).item(), 3
            )
            moved = ExpertPlacement(4, 4, 2, (2, 1, 0, 3))
            self.assertEqual(ckpt.read_expert_rows("layers.0.w13", 0, moved).item(), 0)
            with self.assertRaisesRegex(ValueError, "coordinate mismatch"):
                ckpt.read_expert_rows(
                    "layers.0.w13", 0, ExpertPlacement(4, 4, 1, (2, 1, 0, 3))
                )
            unknown_source = ExpertPlacement(2, 4, 2, (0, 1, 1, 0))
            with self.assertRaisesRegex(ValueError, "Ambiguous legacy"):
                ckpt.read_expert_rows("layers.0.w13", 0, unknown_source)


if __name__ == "__main__":
    unittest.main()

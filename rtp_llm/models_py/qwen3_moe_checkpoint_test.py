"""Real CPU checkpoint -> production Qwen3-MoE parameter tree and service publication."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_moe.parameters import (
    install_qwen3_moe_parameters,
)
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.models_py.weight_mapper import WeightsMapper


def weights():
    # Real per-expert HF names and deliberately different values for every plane.
    shapes = {
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (8, 4),
    }
    layer = "model.layers.0."
    for norm in ("input_layernorm", "post_attention_layernorm"):
        shapes[layer + norm + ".weight"] = (4,)
    for leaf, rows in (("q", 4), ("k", 2), ("v", 2), ("o", 4)):
        shapes[layer + "self_attn." + leaf + "_proj.weight"] = (rows, 4)
    for leaf in ("q", "k"):
        shapes[layer + "self_attn." + leaf + "_norm.weight"] = (2,)
    shapes[layer + "mlp.gate.weight"] = (4, 4)
    for i in range(4):
        for leaf in ("gate", "up", "down"):
            shapes[f"{layer}mlp.experts.{i}.{leaf}_proj.weight"] = (4, 4)
    return {
        key: torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        + i
        for i, (key, shape) in enumerate(shapes.items())
    }


@register_model("_qwen3_moe_cpu_parameters")
class _CpuStage(RtpModule):
    """The native scheduler is not needed to exercise checkpoint parameter owners."""

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, model, loader):
        super().__init__()
        cfg = dict(
            model_config=model,
            num_layers=1,
            vocab_size=8,
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            rms_norm_eps=1e-6,
            num_experts=4,
            top_k=2,
            moe_intermediate_size=model.moe_inter_size,
            moe_config=None,
            params_dtype=torch.float32,
            lm_head_params_dtype=torch.float32,
            quant_config=loader.quant_config,
            parallelism_config=loader.parallelism_config,
        )
        for name in ("attn_tp", "ffn_tp", "ep", "lm_head_tp"):
            for field in ("size", "rank"):
                key = f"{name}_{field}"
                cfg[key] = getattr(loader, key)
        install_qwen3_moe_parameters(self, cfg, layer_ids=range(1), partitioned=False)

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def runtime_weight_view(self):
        return {"embedding": self.embed_tokens.weight, "lm_head": self.lm_head.weight}


def configure(model):
    cfg = dict(
        hidden_size=4,
        expert_num=4,
        moe_k=2,
        moe_inter_size=4,
        moe_layer_index=[0],
        n_shared_experts=0,
        qk_norm=True,
        scoring_func=0,
        routed_scaling_factor=1.0,
        has_moe_norm=True,
        moe_n_group=1,
        moe_topk_group=1,
        activation_type="SiGLU",
        tie_word_embeddings=False,
        normalize_lm_head_weight=False,
        logit_scale=1.0,
    )
    for key, value in cfg.items():
        setattr(model.model_config, key, value)


def write_stacked_checkpoint(path):
    tensors = {key: val for key, val in weights().items() if ".experts." not in key}
    prefix = "model.layers.0.mlp.experts."
    tensors[prefix + "gate_up_proj"] = torch.stack(
        [
            torch.cat([torch.full((6, 4), float(base + source)) for base in (10, 20)])
            for source in range(4)
        ]
    )
    tensors[prefix + "down_proj"] = torch.stack(
        [torch.full((4, 6), float(30 + source)) for source in range(4)]
    )
    save_file(tensors, str(path / "model.safetensors"))
    (path / "placement.json").write_text(json.dumps([[2, 2, 1, 3, 0, 0]]))


# This uses the serving publication path and production Qwen3 parameter owners.
# CPU staging has no fused executor; verify local slot contents and the routing
# tables consumed by forward, with real Gloo agreement between the two ranks.
@torch.inference_mode()
def _serving_eplb_rank(rank, directory, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = service(directory, "fastsafetensors", "_qwen3_moe_cpu_parameters")
        configure(model)
        config, pc = model.model_config, model.parallelism_config
        config.is_mtp = False
        config.moe_n_group = 0  # Ungrouped routing is one rebalance group.
        config.moe_inter_size = 6
        config.phy2log_path = str(Path(directory) / "placement.json")
        initial = [2, 2, 1, 3, 0, 0]
        for key, value in dict(
            enable_eplb=lambda: True,
            eplb_mode="ALL",
            redundant_expert=2,
            balance_method="round",
            eplb_force_repack=1,
            eplb_stats_window_size=2,
        ).items():
            setattr(config.eplb_config, key, value)
        pc.world_size = pc.local_world_size = pc.tp_size = pc.ep_size = 2
        pc.world_rank = pc.local_rank = pc.tp_rank = pc.ep_rank = rank
        pc.get_attn_tp_size = pc.get_ffn_tp_size = lambda: 2
        pc.get_attn_tp_rank = pc.get_ffn_tp_rank = lambda: rank
        model.force_cpu_load_weights = True

        with mock.patch(
            "rtp_llm.models_py.weight_mapper.get_all_weights",
            side_effect=AssertionError("explicit EPLB FST reached scratch"),
        ):
            load_owned_model(model)
            assert (
                model.model_weights_loader.loaded_method
                is NewLoaderLoadMethod.FASTSAFETENSORS
            )
            controller = model.py_eplb
            experts = model.py_model.layers[0].mlp.experts
            assert controller.supports_python_weight_install()
            assert experts.is_cpu_checkpoint_staging()
            experts.prepare_eplb_reload_weights()
            logical = torch.tensor([[0, 1], [2, 3], [0, 0]])
            ids_ptr = experts._eplb_log2phy.data_ptr()
            down_ptr = experts.w2.data_ptr()

            def check(placement):
                assert (
                    experts.w2.data_ptr() == down_ptr
                    and experts._eplb_log2phy.data_ptr() == ids_ptr
                )
                assert list(experts._expert_placement.physical_to_logical) == placement
                for slot, source in enumerate(placement[rank * 3 : rank * 3 + 3]):
                    torch.testing.assert_close(
                        experts.w2[slot], torch.full((4, 6), float(30 + source))
                    )
                    torch.testing.assert_close(
                        experts.w13[slot, :6], torch.full((6, 4), float(20 + source))
                    )
                    torch.testing.assert_close(
                        experts.w13[slot, 6:], torch.full((6, 4), float(10 + source))
                    )
                physical = experts._map_logical_to_physical(logical)
                local = physical - rank * 3
                output = torch.where(
                    (local >= 0) & (local < 3), experts.w2[local.clamp(0, 2), 0, 0], 0
                )
                dist.all_reduce(output)
                torch.testing.assert_close(output, (logical + 30).float())

            check(initial)
            counts = torch.tensor([[30, 2, 1, 1]], dtype=torch.int64)
            loads = torch.tensor([[4, 30]], dtype=torch.int64)
            plan = [
                controller.create_balance_plan(counts, loads) if rank == 0 else None
            ]
            dist.broadcast_object_list(plan, src=0)
            layer, replicas, inverse, phy = plan[0]
            desired = phy.tolist()
            assert desired != initial
            for commit in (False, True):
                controller.load_moe_weight(layer, rank, 2, phy)
                check(initial)  # Staging must not publish a partial generation.
                # Simulate one peer voting against an already installed generation.
                controller.install_balance_plan(0, phy, replicas, inverse)
                vote = torch.tensor([int(commit or rank == 0)])
                dist.all_reduce(vote, op=dist.ReduceOp.MIN)
                if vote.item():
                    controller.commit_balance_plan(0)
                else:
                    controller.rollback_balance_plan(0)
                check(desired if commit else initial)
                assert controller.phy2log[0] == (desired if commit else initial)
    finally:
        dist.destroy_process_group()


class Qwen3MoeCpuTest(unittest.TestCase):
    def test_stacked_fst_serving_eplb_ep2_commit_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_stacked_checkpoint(path)
            mp.spawn(_serving_eplb_rank, args=(directory, str(path / "rdv")), nprocs=2)

    def test_single_ep_redundant_stacked_source_populates_every_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_stacked_checkpoint(path)
            for method in ("scratch", "fastsafetensors"):
                model = service(directory, method, "_qwen3_moe_cpu_parameters")
                configure(model)
                model.parallelism_config.local_world_size = 1
                model.model_config.moe_inter_size = 6
                model.model_config.phy2log_path = str(path / "placement.json")
                model.model_config.eplb_config.redundant_expert = 2
                load_owned_model(model)
                self.assertIsNone(model.py_eplb)
                experts = model.py_model.layers[0].mlp.experts
                for slot, source in enumerate([2, 2, 1, 3, 0, 0]):
                    torch.testing.assert_close(
                        experts.w2[slot], torch.full((4, 6), 30.0 + source)
                    )
                    torch.testing.assert_close(
                        experts.w13[slot, :6], torch.full((6, 4), 20.0 + source)
                    )
                    torch.testing.assert_close(
                        experts.w13[slot, 6:], torch.full((6, 4), 10.0 + source)
                    )

    def test_service_publication_scratch_and_fst(self):
        for method in ("scratch", "fastsafetensors"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = service(directory, method, "_qwen3_moe_cpu_parameters")
                model.parallelism_config.local_world_size = 1
                configure(model)
                data = weights()
                save_file(data, str(Path(directory) / "model.safetensors"))
                load_owned_model(model)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                owner = model.py_model
                self.assertIs(
                    model.weight.get_global_weight("lm_head"), owner.lm_head.weight
                )
                self.assertTrue(
                    torch.equal(owner.lm_head.weight, data["lm_head.weight"])
                )
                dec = owner.layers[0]
                self.assertIsNone(dec.self_attn.qkv_proj.bias)
                self.assertTrue(
                    torch.equal(
                        dec.self_attn.qkv_proj.weight,
                        torch.cat(
                            [
                                data[f"model.layers.0.self_attn.{p}_proj.weight"]
                                for p in "qkv"
                            ]
                        ),
                    )
                )
                self.assertIsNone(dec.mlp.shared_experts)
                self.assertTrue(
                    torch.equal(
                        dec.self_attn.q_norm.weight,
                        data["model.layers.0.self_attn.q_norm.weight"],
                    )
                )
                # The original CUDA F16 gate consumes the transpose of a
                # materialized [hidden, experts] matrix. Equal logical values
                # with contiguous [experts, hidden] storage select a different
                # GEMM operand layout, potentially changing a greedy route.
                gate_ref = data["model.layers.0.mlp.gate.weight"].T.contiguous().T
                self.assertEqual(dec.mlp.gate.weight.stride(), gate_ref.stride())
                torch.testing.assert_close(dec.mlp.gate.weight, gate_ref, rtol=0, atol=0)
                tokens = torch.randn(3, 4, generator=torch.Generator().manual_seed(13))
                torch.testing.assert_close(
                    dec.mlp.gate(tokens),
                    torch.nn.functional.linear(tokens, gate_ref),
                    rtol=0,
                    atol=0,
                )
                for i in range(4):
                    prefix = f"model.layers.0.mlp.experts.{i}."
                    self.assertTrue(
                        torch.equal(
                            dec.mlp.experts.w13[i],
                            torch.cat(
                                [
                                    data[prefix + leaf + "_proj.weight"]
                                    for leaf in ("up", "gate")
                                ]
                            ),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            dec.mlp.experts.w2[i], data[prefix + "down_proj.weight"]
                        )
                    )
                logits = torch.tensor([[1.0, 3.0, 2.0, -1.0]])
                self.assertTrue(dec.mlp.experts.is_cpu_checkpoint_staging())
                routes, ids = dec.mlp.route(logits, ids_dtype=torch.int32)
                torch.testing.assert_close(
                    routes, logits.gather(1, ids.long()).softmax(-1)
                )


if __name__ == "__main__":
    unittest.main()

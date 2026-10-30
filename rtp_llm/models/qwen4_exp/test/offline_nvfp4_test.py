import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import QWEN4_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.offline_nvfp4 import (
    Qwen4Nvfp4MoeWeight,
    Qwen4Nvfp4WeightInfo,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, stack_, stack_moe_w1


class Source(TensorSource):
    def __init__(self, values):
        self.values = values
        self.requests = []

    def load_tensor(self, name, data_type=torch.bfloat16):
        self.requests.append(name)
        return [self.values[name].to(data_type)]

    def has_tensor(self, name):
        return name in self.values

    def get_database(self):
        return None


def quant_config():
    return ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=True, model_recipe=QWEN4_NVFP4_RECIPE
    )


def descriptor(name, experts=4):
    projections = ("up", "gate") if name == W.moe_w1 else ("down",)
    source = MoeAtomicWeight(
        name,
        [
            CkptWeightInfo(
                "model.language_model.layers.{i}.mlp.experts.{expert_id}."
                + projection
                + "_proj.weight"
            )
            for projection in projections
        ],
        process_fun=stack_moe_w1 if name == W.moe_w1 else stack_,
        config=MoeConfig(expert_num=experts),
    )
    return Qwen4Nvfp4MoeWeight(source, quant_config())


def load_config(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
    return SimpleNamespace(
        compute_dtype=torch.bfloat16,
        moe_pure_tp_preshard=False,
        moe_pure_tp_mode=True,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ffn_tp_size=tp_size,
        ffn_tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=1,
        dp_rank=0,
        hidden_size=96,
        head_num=6,
        head_num_kv=2,
        size_per_head=16,
        bit=4,
        get_selected_experts=lambda layer, count: list(
            range(count // ep_size * ep_rank, count // ep_size * (ep_rank + 1))
        ),
    )


def checkpoint(experts=4, scalar_shapes=((), (1,), (1, 1))):
    values = {}
    # Non-square packed matrices and distinct row/column/projection/expert marks
    # make an accidental transpose, nibble change or up/gate swap observable.
    for expert in range(experts):
        for projection_id, projection in enumerate(("up", "gate", "down")):
            rows, columns = (96, 64) if projection == "down" else (64, 96)
            prefix = (
                f"model.language_model.layers.0.mlp.experts.{expert}."
                f"{projection}_proj."
            )
            row = torch.arange(rows).unsqueeze(1)
            column = torch.arange(columns // 2).unsqueeze(0)
            values[prefix + "weight"] = (
                (row * 17 + column * 3 + expert * 29 + projection_id * 67)
                .remainder(256)
                .to(torch.uint8)
            )
            values[prefix + "weight_scale"] = (
                (row + torch.arange(columns // 16) + expert + projection_id) % 5 + 1
            ).to(torch.float8_e4m3fn)
            shape = scalar_shapes[(expert + projection_id) % len(scalar_shapes)]
            values[prefix + "weight_scale_2"] = torch.tensor(
                (expert + 1) / 16 if projection != "down" else (expert + 1) / 32
            ).reshape(shape)
            values[prefix + "input_scale"] = torch.tensor(
                0.25 if projection != "down" else 0.125
            ).reshape(shape)
    return values


class OfflineNvfp4Test(unittest.TestCase):
    def test_packed_bytes_scales_and_scalar_shapes_keep_checkpoint_semantics(self):
        values = checkpoint()
        for name in (W.moe_w1, W.moe_w2):
            with self.subTest(weight=name):
                weight = descriptor(name)
                source = Source(values)
                result = weight._load_raw_tensor(source, 0, "cpu", load_config())
                self.assertEqual(result[weight.kernel.name].dtype, torch.uint8)
                self.assertEqual(result[weight.scale.name].dtype, torch.float8_e4m3fn)
                self.assertEqual(result[weight.scale_2.name].shape, (4,))
                self.assertEqual(result[weight.input_scale.name].shape, (4,))
                projections = ("up", "gate") if name == W.moe_w1 else ("down",)
                for expert in range(4):
                    prefixes = [
                        f"model.language_model.layers.0.mlp.experts.{expert}.{p}_proj."
                        for p in projections
                    ]
                    for key, suffix in (
                        (weight.kernel.name, "weight"),
                        (weight.scale.name, "weight_scale"),
                    ):
                        expected = torch.cat([values[p + suffix] for p in prefixes])
                        self.assertTrue(
                            torch.equal(
                                result[key][expert].view(torch.uint8),
                                expected.view(torch.uint8),
                            )
                        )
                    for key, suffix in (
                        (weight.scale_2.name, "weight_scale_2"),
                        (weight.input_scale.name, "input_scale"),
                    ):
                        self.assertEqual(
                            result[key][expert], values[prefixes[0] + suffix].item()
                        )
                self.assertEqual(
                    set(source.requests), weight.get_tensor_names(0, load_config())
                )

    def test_ep_selection_and_tp2_reassemble_identical_packed_bytes(self):
        values = checkpoint()
        for name in (W.moe_w1, W.moe_w2):
            weight = descriptor(name)
            full = weight._load_raw_tensor(Source(values), 0, "cpu", load_config())
            ranks = []
            for ep_rank in range(2):
                source = Source(values)
                cfg = load_config(ep_size=2, ep_rank=ep_rank)
                local = weight._load_raw_tensor(source, 0, "cpu", cfg)
                self.assertTrue(
                    all(
                        f".experts.{expert}." not in key
                        for key in source.requests
                        for expert in range(4)
                        if expert // 2 != ep_rank
                    )
                )
                ranks.append(local)
            for key, value in full.items():
                self.assertTrue(
                    torch.equal(
                        torch.cat([rank[key] for rank in ranks]).view(torch.uint8),
                        value.view(torch.uint8),
                    )
                )

            shards = [
                weight._split(full, load_config(tp_size=2, tp_rank=rank))
                for rank in range(2)
            ]
            for key in (weight.kernel.name, weight.scale.name):
                if name == W.moe_w1:
                    # Every TP rank owns half of each branch, not one branch.
                    reconstructed = torch.cat(
                        [
                            torch.cat(
                                [part[key].chunk(2, dim=1)[branch] for part in shards],
                                dim=1,
                            )
                            for branch in range(2)
                        ],
                        dim=1,
                    )
                else:
                    reconstructed = torch.cat([part[key] for part in shards], dim=-1)
                self.assertTrue(
                    torch.equal(
                        reconstructed.view(torch.uint8), full[key].view(torch.uint8)
                    )
                )
            for part in shards:
                for key in (weight.scale_2.name, weight.input_scale.name):
                    self.assertTrue(torch.equal(part[key], full[key]))

    def test_missing_checkpoint_companions_are_not_synthesized(self):
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            with self.subTest(suffix=suffix):
                values = checkpoint()
                del values[
                    "model.language_model.layers.0.mlp.experts.2.down_proj." + suffix
                ]
                with self.assertRaises(KeyError):
                    descriptor(W.moe_w2)._load_raw_tensor(
                        Source(values), 0, "cpu", load_config()
                    )

    def test_invalid_scalar_scales_are_rejected(self):
        for suffix in ("weight_scale_2", "input_scale"):
            for bad in (float("nan"), float("inf"), -0.25, 0.0):
                with self.subTest(suffix=suffix, value=bad):
                    values = checkpoint()
                    values[
                        "model.language_model.layers.0.mlp.experts.1.down_proj."
                        + suffix
                    ] = torch.tensor(bad)
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        descriptor(W.moe_w2)._load_raw_tensor(
                            Source(values), 0, "cpu", load_config()
                        )
            for shape in ((0,), (2,)):
                values = checkpoint()
                values[
                    "model.language_model.layers.0.mlp.experts.1.down_proj." + suffix
                ] = torch.ones(shape)
                with self.assertRaisesRegex(ValueError, "one scalar per expert"):
                    descriptor(W.moe_w2)._load_raw_tensor(
                        Source(values), 0, "cpu", load_config()
                    )

    def test_unequal_up_gate_and_nonuniform_activation_scales_are_rejected(self):
        for suffix in ("weight_scale_2", "input_scale"):
            values = checkpoint()
            values[
                "model.language_model.layers.0.mlp.experts.1.gate_proj." + suffix
            ] *= 2
            with self.assertRaisesRegex(ValueError, "equal up/gate scales"):
                descriptor(W.moe_w1)._load_raw_tensor(
                    Source(values), 0, "cpu", load_config()
                )
        for name, projections in ((W.moe_w1, ("up", "gate")), (W.moe_w2, ("down",))):
            values = checkpoint()
            for projection in projections:
                values[
                    f"model.language_model.layers.0.mlp.experts.1.{projection}_proj.input_scale"
                ] *= 2
            with self.assertRaisesRegex(ValueError, "uniform activation scales"):
                descriptor(name)._load_raw_tensor(
                    Source(values), 0, "cpu", load_config()
                )

    def test_rejects_noncontiguous_or_indivisible_experts_before_io(self):
        for cfg in (load_config(ep_size=3), load_config(ep_size=2)):
            if cfg.ep_size == 2:
                cfg.get_selected_experts = lambda layer, count: [1, 3]
            source = Source({})
            with self.assertRaises(ValueError):
                descriptor(W.moe_w1)._load_raw_tensor(source, 0, "cpu", cfg)
            self.assertFalse(source.requests)

    def test_bf16_nonexpert_descriptors_and_values_remain_unchanged(self):
        cfg = MoeConfig(expert_num=4)
        routed = MoeWeight(
            [
                descriptor(W.moe_w1).kernel,
                descriptor(W.moe_w2).kernel,
                MoeAtomicWeight(W.moe_gate, [CkptWeightInfo("router")], config=cfg),
            ],
            config=cfg,
        )
        dense = AtomicWeight(W.qwen4_hc_attn_mix_down, [CkptWeightInfo("hc")])
        embedding = AtomicWeight(W.embedding, [CkptWeightInfo("embedding")])
        converted = Qwen4Nvfp4WeightInfo(
            [embedding], [[routed, dense]]
        ).to_quant_weight_info(quant_config())
        self.assertIs(converted.weights[0], embedding)
        self.assertIs(converted.layer_weights[0][1], dense)
        children = converted.layer_weights[0][0].sub_weights
        self.assertIs(children[W.moe_gate], routed.sub_weights[W.moe_gate])
        for name in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(children[name], Qwen4Nvfp4MoeWeight)
        original = torch.tensor([[0.125, -0.75, 1.03125]], dtype=torch.bfloat16)
        for weight, key in ((dense, "hc"), (embedding, "embedding")):
            loaded = weight._load_raw_tensor(
                Source({key: original}), 0, "cpu", load_config()
            )
            self.assertEqual(loaded[weight.name].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(loaded[weight.name], original))


if __name__ == "__main__":
    unittest.main()

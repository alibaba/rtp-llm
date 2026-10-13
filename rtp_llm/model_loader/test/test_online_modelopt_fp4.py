import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineModelOptFp4MoeWeight,
    _cast_to_e2m1,
    quantize_expert_to_nvfp4,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    stack_,
    stack_moe_w1,
    transpose_stack_moe_w1,
)


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


class OnlineNvfp4Test(unittest.TestCase):
    def test_fp8_source_applies_block_scales_before_nvfp4_quantization(self):
        original = MoeAtomicWeight(
            W.moe_w2,
            [CkptWeightInfo("down.{expert_id}.weight")],
            process_fun=stack_,
            config=MoeConfig(expert_num=1),
        )
        quant = OnlineModelOptFp4MoeWeight(
            original,
            ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                source_fp8_block_size=128,
            ),
        )
        weight = torch.full((128, 256), 2.0).to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.25, 0.5]])
        source = Source({"down.0.weight": weight, "down.0.weight_scale_inv": scale})
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            moe_pure_tp_preshard=False,
            get_selected_experts=lambda layer, count: [0],
        )
        names = quant.get_tensor_names(0, config)
        self.assertEqual(names, set(source.values))
        result = quant._load_raw_tensor(source, 0, "cpu", config)
        decoded = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(
            128, 1
        )
        packed, blocks, alpha = quantize_expert_to_nvfp4(decoded.bfloat16())
        self.assertTrue(torch.equal(result[quant.kernel.name][0], packed))
        torch.testing.assert_close(result[quant.scale.name][0].float(), blocks.float())
        torch.testing.assert_close(result[quant.scale_2.name][0], alpha)

    def test_e2m1_ties_and_sign(self):
        x = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
        q = _cast_to_e2m1(x)
        self.assertTrue(
            torch.equal(q, torch.tensor([0, 0, 2, 2, 4, 4, 6, 6, 7], dtype=torch.uint8))
        )
        self.assertTrue(torch.equal(_cast_to_e2m1(-x[1:]), q[1:] + 8))

    def test_zero_weight_is_finite(self):
        p, s, a = quantize_expert_to_nvfp4(torch.zeros(7, 32, dtype=torch.bfloat16))
        self.assertEqual(p.shape, (7, 16))
        self.assertFalse(p.any())
        self.assertTrue(torch.isfinite(s.float()).all() and torch.isfinite(a))

    def test_loader_reuses_private_raw_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            original = MoeAtomicWeight(
                W.moe_w2,
                [CkptWeightInfo("down.{expert_id}")],
                process_fun=stack_,
                config=MoeConfig(expert_num=2),
            )
            quant = OnlineModelOptFp4MoeWeight(
                original,
                ModelOptFp4Config(
                    bits=4,
                    group_size=16,
                    is_quanted=False,
                    activation_amax=32.0,
                    online_cache_dir=directory,
                ),
            )
            config = SimpleNamespace(
                compute_dtype=torch.bfloat16,
                moe_pure_tp_preshard=False,
                get_selected_experts=lambda layer, count: [0, 1],
            )
            source = Source(
                {
                    f"down.{i}": torch.full((4, 32), i + 0.5, dtype=torch.bfloat16)
                    for i in range(2)
                }
            )
            expected = quant._load_raw_tensor(source, 0, "cpu", config)
            with mock.patch(
                "rtp_llm.model_loader.nvfp4_cache.quantize_expert_to_nvfp4",
                side_effect=AssertionError("must reuse cached tensors"),
            ):
                got = quant._load_raw_tensor(source, 0, "cpu", config)
            for name in got:
                self.assertTrue(torch.equal(got[name].float(), expected[name].float()))

    def test_cutlass_rejects_noncontiguous_experts_before_io(self):
        source = Source({})
        original = MoeAtomicWeight(
            W.moe_w2,
            [CkptWeightInfo("down.{expert_id}")],
            process_fun=stack_,
            config=MoeConfig(expert_num=4),
        )
        quant = OnlineModelOptFp4MoeWeight(
            original,
            ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                activation_amax=32.0,
                require_contiguous_experts=True,
            ),
        )
        config = SimpleNamespace(
            ep_size=2, ep_rank=0, get_selected_experts=lambda layer, count: [1, 3]
        )
        with self.assertRaisesRegex(ValueError, "contiguous logical expert ownership"):
            quant._load_raw_tensor(source, 0, "cpu", config)
        self.assertFalse(source.requests)

    def test_stacked_source_is_read_once_and_experts_keep_their_ids(self):
        weight = torch.randn(4, 8, 32, dtype=torch.bfloat16)
        source = Source({"stacked": weight})
        original = MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo("stacked")],
            process_fun=transpose_stack_moe_w1,
            config=MoeConfig(expert_num=4),
            stacked_ckpt_keys=True,
        )
        quant = OnlineModelOptFp4MoeWeight(
            original,
            ModelOptFp4Config(
                bits=4, group_size=16, is_quanted=False, activation_amax=1.0
            ),
        )
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            moe_pure_tp_preshard=False,
            get_selected_experts=lambda layer, count: [1, 3],
        )
        got = quant._load_raw_tensor(source, 0, "cpu", config)
        self.assertEqual(source.requests, ["stacked"])
        for i, e in enumerate([1, 3]):
            ref = quantize_expert_to_nvfp4(
                torch.cat([weight[e, 4:], weight[e, :4]], dim=0)
            )
            self.assertTrue(torch.equal(got[quant.kernel.name][i], ref[0]))
            self.assertTrue(
                torch.equal(got[quant.scale.name][i].float(), ref[1].float())
            )

    def test_streaming_selects_local_experts_and_preserves_gate_up(self):
        for name, keys, fun in (
            (W.moe_w1, ["gate", "up"], stack_moe_w1),
            (W.moe_w2, ["down"], stack_),
        ):
            with self.subTest(weight=name):
                values = {
                    f"{k}.{e}": torch.arange(128, dtype=torch.float32).reshape(4, 32)
                    / 100
                    + (e + 1)
                    + (7 if k == "gate" else 0)
                    for e in range(4)
                    for k in keys
                }
                source = Source(values)
                original = MoeAtomicWeight(
                    name,
                    [CkptWeightInfo(k + ".{expert_id}") for k in keys],
                    process_fun=fun,
                    config=MoeConfig(expert_num=4),
                )
                quant = OnlineModelOptFp4MoeWeight(
                    original,
                    ModelOptFp4Config(
                        bits=4, group_size=16, is_quanted=False, activation_amax=1.0
                    ),
                )
                config = SimpleNamespace(
                    compute_dtype=torch.bfloat16,
                    moe_pure_tp_preshard=False,
                    get_selected_experts=lambda layer, count: [1, 3],
                )
                result = quant._load_raw_tensor(source, 0, "cpu", config)
                self.assertEqual(
                    source.requests, [f"{k}.{e}" for e in [1, 3] for k in keys]
                )
                for i, e in enumerate([1, 3]):
                    w = fun([values[f"{k}.{e}"].bfloat16() for k in keys])[0]
                    p, s, a = quantize_expert_to_nvfp4(w)
                    self.assertTrue(torch.equal(result[quant.kernel.name][i], p))
                    self.assertTrue(
                        torch.equal(result[quant.scale.name][i].float(), s.float())
                    )
                    self.assertEqual(result[quant.scale_2.name][i], a)
                    self.assertAlmostEqual(
                        result[quant.input_scale.name][i].item(), 1 / 2688.0, places=9
                    )


if __name__ == "__main__":
    unittest.main()

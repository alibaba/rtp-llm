"""MLA absorb weights must retain FP32 block scales until dequantization."""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.attn_weight import MlaConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    W8A8Fp8PerBlockMlaAttnAtomicWeight,
    dequant_weight_split_k,
    dequant_weight_split_v,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class _Source(TensorSource):
    def __init__(self, values):
        self.values = values
        self.calls = []

    def load_tensor(self, name, data_type=torch.float16):
        self.calls.append((name, data_type))
        return [self.values[name].to(data_type)]


class MlaFp8ScalePrecisionTest(unittest.TestCase):
    def setUp(self):
        self.key = "model.layers.0.self_attn.kv_b_proj.weight"
        self.scale_key = self.key + "_scale_inv"
        self.raw = (
            torch.linspace(-3, 3, 256 * 128).reshape(256, 128).to(torch.float8_e4m3fn)
        )
        self.scales = torch.tensor([[0.013579246], [0.00493717]], dtype=torch.float32)
        self.config = MlaConfig(
            head_num=1,
            nope_head_dim=128,
            rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=128,
            use_mla=True,
        )
        self.load_config = SimpleNamespace(compute_dtype=torch.bfloat16)

    def weight(self, name):
        process = dequant_weight_split_k if name == W.mla_kc else dequant_weight_split_v
        return W8A8Fp8PerBlockMlaAttnAtomicWeight(
            name,
            [
                CkptWeightInfo(self.key, identity),
                CkptWeightInfo(self.scale_key, identity),
            ],
            functools.partial(
                process,
                block_size=128,
                head_num=1,
                nope_head_dim=128,
                v_head_dim=128,
                lora_rank=128,
            ),
            config=self.config,
        )

    def expected(self, name, scales):
        expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        matrix = (self.raw.float() * expanded.float()).bfloat16()
        return (
            matrix[:128].unsqueeze(0)
            if name == W.mla_kc
            else matrix[128:].T.unsqueeze(0)
        )

    def test_ppu_k_and_v_preserve_scale_until_final_bf16_round(self):
        for name in (W.mla_kc, W.mla_vc):
            with self.subTest(name=name):
                source = _Source({self.key: self.raw, self.scale_key: self.scales})
                with patch(
                    "rtp_llm.model_loader.per_block_fp8_quant_weight.is_ppu",
                    return_value=True,
                ):
                    actual = self.weight(name)._load_raw_tensor(
                        source, 0, "cpu", self.load_config
                    )[name]
                expected = self.expected(name, self.scales)
                rounded_scale = self.expected(name, self.scales.bfloat16())
                self.assertFalse(torch.equal(expected, rounded_scale))
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(
                    source.calls,
                    [(self.key, torch.bfloat16), (self.scale_key, torch.float32)],
                )

    def test_other_backends_keep_existing_load_contract(self):
        source = _Source({self.key: self.raw, self.scale_key: self.scales})
        with patch(
            "rtp_llm.model_loader.per_block_fp8_quant_weight.is_ppu", return_value=False
        ):
            actual = self.weight(W.mla_kc)._load_raw_tensor(
                source, 0, "cpu", self.load_config
            )[W.mla_kc]
        self.assertTrue(
            torch.equal(actual, self.expected(W.mla_kc, self.scales.bfloat16()))
        )
        self.assertEqual(source.calls[-1], (self.scale_key, torch.bfloat16))

    def test_other_mla_weights_keep_existing_load_contract(self):
        weight = W8A8Fp8PerBlockMlaAttnAtomicWeight(
            W.mla_q_b_s,
            [CkptWeightInfo(self.scale_key, identity)],
            identity,
            config=self.config,
        )
        source = _Source({self.scale_key: self.scales})
        with patch(
            "rtp_llm.model_loader.per_block_fp8_quant_weight.is_ppu", return_value=True
        ):
            actual = weight._load_raw_tensor(source, 0, "cpu", self.load_config)[
                W.mla_q_b_s
            ]
        self.assertTrue(torch.equal(actual, self.scales.bfloat16()))
        self.assertEqual(source.calls, [(self.scale_key, torch.bfloat16)])


if __name__ == "__main__":
    unittest.main()

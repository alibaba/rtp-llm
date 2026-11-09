import unittest

import torch

from rtp_llm.config.quant_config import CompressedTensorsNvfp4Config, ModelOptFp4Config
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.compressed_tensors_nvfp4_weight import (
    CompressedTensorsNvfp4Weight,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class CompressedTensorsNvfp4WeightTest(unittest.TestCase):
    def setUp(self):
        self.quant = CompressedTensorsNvfp4Config(ignore_patterns=["lm_head"])

    def qkv(self):
        return AttnAtomicWeight(
            W.attn_qkv_w,
            [
                CkptWeightInfo("model.layers.{i}.self_attn." + p + ".weight")
                for p in ("q_proj", "k_proj", "v_proj")
            ],
            config=AttnConfig(
                hidden_size=128, head_num=4, head_num_kv=2, size_per_head=32
            ),
        )

    def test_registry_and_qkv_names(self):
        src = self.qkv()
        self.assertFalse(PerGroupFp4Weight.support(self.quant, src))
        wrapper = WeightModule.create(src, self.quant)
        self.assertIsInstance(wrapper, CompressedTensorsNvfp4Weight)
        for atomic, suffix in (
            (wrapper.kernel, ".weight_packed"),
            (wrapper.scale, ".weight_scale"),
            (wrapper.scale_2, ".weight_global_scale"),
            (wrapper.input_scale, ".input_global_scale"),
        ):
            self.assertEqual(len(atomic.weights), 3)
            self.assertTrue(all(w.name.endswith(suffix) for w in atomic.weights))
        self.assertTrue(all(w.name.endswith(".weight") for w in src.weights))

    def test_qkv_bytes_scales_and_globals(self):
        wrapper = CompressedTensorsNvfp4Weight(self.qkv(), self.quant)
        blocks = [
            torch.arange(n * 64, dtype=torch.int32).to(torch.uint8).reshape(n, 64)
            for n in (128, 64, 64)
        ]
        actual = wrapper.kernel.process_fun(blocks)
        self.assertTrue(torch.equal(actual, torch.cat(blocks)))
        scales = [torch.ones(n, 8).to(torch.float8_e4m3fn) for n in (128, 64, 64)]
        actual_scales = wrapper.scale.process_fun(scales)
        self.assertEqual(actual_scales.shape, (256, 8))
        self.assertTrue(
            torch.equal(
                actual_scales.view(torch.uint8),
                torch.cat([s.view(torch.uint8) for s in scales]),
            )
        )
        torch.testing.assert_close(
            wrapper.scale_2.process_fun([torch.tensor([2.0])] * 3), torch.tensor([0.5])
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            wrapper.input_scale.process_fun([torch.tensor([2.0]), torch.tensor([3.0])])

    def test_partial_fused_exclusion_fails(self):
        quant = CompressedTensorsNvfp4Config(ignore_patterns=[r"re:q_proj$"])
        with self.assertRaisesRegex(ValueError, "partially exclude"):
            CompressedTensorsNvfp4Weight.support(quant, self.qkv())

    def test_down_padding_uses_original_channels(self):
        src = FfnAtomicWeight(
            W.ffn_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.down_proj.weight")],
            config=FfnConfig(align_size=128),
        )
        wrapper = CompressedTensorsNvfp4Weight(src, self.quant)
        packed = torch.ones(128, 80, dtype=torch.uint8)  # K=160 -> K=256.
        scales = torch.ones(128, 10).to(torch.float8_e4m3fn)
        self.assertEqual(wrapper.kernel.process_fun([packed]).shape, (128, 128))
        self.assertEqual(wrapper.scale.process_fun([scales]).shape, (128, 16))

    def test_qkv_tp_preserves_matching_weight_and_scale_rows(self):
        wrapper = CompressedTensorsNvfp4Weight(self.qkv(), self.quant)
        rows = torch.arange(256).to(torch.uint8)
        packed = rows[:, None].expand(256, 64).contiguous()
        scales = rows[:, None].expand(256, 8).contiguous().view(torch.float8_e4m3fn)
        for rank in (0, 1):
            kwargs = dict(
                head_num=4, head_num_kv=2, size_per_head=32, tp=2, tp_rank=rank
            )
            wp = wrapper.kernel._get_split_func()(packed, **kwargs)
            sp = wrapper.scale._get_split_func()(scales, **kwargs)
            self.assertEqual(wp.shape, (128, 64))
            self.assertTrue(torch.equal(wp[:, 0], sp.view(torch.uint8)[:, 0]))
            expected = torch.cat(
                (
                    rows[rank * 64 : (rank + 1) * 64],
                    rows[128 + rank * 32 : 128 + (rank + 1) * 32],
                    rows[192 + rank * 32 : 192 + (rank + 1) * 32],
                )
            )
            self.assertTrue(torch.equal(wp[:, 0], expected))

    def test_gate_up_fusion_preserves_bytes(self):
        src = FfnAtomicWeight(
            W.ffn_w13,
            [
                CkptWeightInfo("model.layers.{i}.mlp." + name + ".weight")
                for name in ("gate_proj", "up_proj")
            ],
            config=FfnConfig(align_size=128),
        )
        wrapper = CompressedTensorsNvfp4Weight(src, self.quant)
        gate = torch.full((128, 64), 0x38, dtype=torch.uint8)
        up = torch.full((128, 64), 0xB8, dtype=torch.uint8)
        self.assertTrue(
            torch.equal(wrapper.kernel.process_fun([gate, up]), torch.cat((gate, up)))
        )
        self.assertTrue(
            all(w.name.endswith(".weight_packed") for w in wrapper.kernel.weights)
        )

    def test_modelopt_still_uses_original_wrapper(self):
        quant = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
        self.assertIs(type(WeightModule.create(self.qkv(), quant)), PerGroupFp4Weight)


if __name__ == "__main__":
    unittest.main()

import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import CompressedTensorsNvfp4Config, ModelOptFp4Config
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.compressed_tensors_nvfp4_weight import (
    CompressedTensorsNvfp4Weight,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class RecordingSource(TensorSource):
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def load_tensor(self, name, data_type):
        self.reads.append(name)
        return [self.tensors[name].to(data_type)]


class RecordingDevice:
    """Records layout-hook inputs; does not emulate native layout conversion."""

    def __init__(self):
        self.conversions = []

    def maybe_rewrite_weight_by_key(self, name, tensor, **kwargs):
        return tensor

    def convert_fp4_gemm_weight_params(self, packed, scales):
        self.conversions.append((packed.clone(), scales.clone()))
        return packed, scales

    def maybe_prepare_static_weights_for_fp4_moe(
        self, kernel_name, scale_name, packed, scales
    ):
        return packed, scales


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
        wrapper = WeightModule.create(self.qkv(), self.quant)
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
            wrapper.scale_2.process_fun([torch.tensor([2.0])] * 3), torch.tensor(0.5)
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            wrapper.input_scale.process_fun([torch.tensor([2.0]), torch.tensor([3.0])])

    def test_layer_exclusion_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ignore"):
            CompressedTensorsNvfp4Config(ignore_patterns=[r"re:q_proj$"])

    def test_down_padding_uses_original_channels(self):
        src = FfnAtomicWeight(
            W.ffn_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.down_proj.weight")],
            config=FfnConfig(align_size=128),
        )
        wrapper = WeightModule.create(src, self.quant)
        packed = torch.ones(128, 80, dtype=torch.uint8)  # K=160 -> K=256.
        scales = torch.ones(128, 10).to(torch.float8_e4m3fn)
        self.assertEqual(wrapper.kernel.process_fun([packed]).shape, (128, 128))
        self.assertEqual(wrapper.scale.process_fun([scales]).shape, (128, 16))

    def test_qkv_tp_preserves_matching_weight_and_scale_rows(self):
        wrapper = WeightModule.create(self.qkv(), self.quant)
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
        wrapper = WeightModule.create(src, self.quant)
        gate = torch.full((128, 64), 0x38, dtype=torch.uint8)
        up = torch.full((128, 64), 0xB8, dtype=torch.uint8)
        self.assertTrue(
            torch.equal(wrapper.kernel.process_fun([gate, up]), torch.cat((gate, up)))
        )
        self.assertTrue(
            all(w.name.endswith(".weight_packed") for w in wrapper.kernel.weights)
        )

    def real_tile(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "config/test/fixtures/nvfp4/q_proj_tile.json"
        )
        tile = json.loads(path.read_text())
        packed = torch.tensor(
            list(base64.b64decode(tile["weight_packed"])), dtype=torch.uint8
        ).reshape(128, 64)
        scales = (
            torch.tensor(
                list(base64.b64decode(tile["weight_scale"])), dtype=torch.uint8
            )
            .reshape(128, 8)
            .view(torch.float8_e4m3fn)
        )
        return tile, packed, scales

    def load_config(self, rank=0, tp=1):
        return SimpleNamespace(
            compute_dtype=torch.bfloat16,
            merge_lora=False,
            tp_size=tp,
            tp_rank=rank,
            ffn_tp_size=tp,
            ffn_tp_rank=rank,
            dp_size=1,
            dp_rank=0,
            ep_size=1,
            ep_rank=0,
            lm_head_tp_size=tp,
            lm_head_tp_rank=rank,
            hidden_size=128,
            head_num=4,
            head_num_kv=2,
            size_per_head=32,
            moe_pure_tp_mode=False,
            bit=4,
            use_swizzleA=False,
            exported_device=RecordingDevice(),
        )

    def real_qkv_source(self):
        # Structural QKV fixture: q uses the real tile; k/v reuse its first
        # 64 rows. These are not actual k_proj/v_proj checkpoint weights.
        tile, packed, scales = self.real_tile()
        tensors = {}
        for projection, rows in (("q_proj", 128), ("k_proj", 64), ("v_proj", 64)):
            prefix = "model.layers.0.self_attn." + projection
            tensors[prefix + ".weight_packed"] = packed[:rows].clone()
            tensors[prefix + ".weight_scale"] = scales[:rows].clone()
            tensors[prefix + ".weight_global_scale"] = torch.tensor(
                [tile["weight_global_scale"]]
            )
            tensors[prefix + ".input_global_scale"] = torch.tensor(
                [tile["input_global_scale"]]
            )
        return RecordingSource(tensors)

    def test_load_calls_real_read_fuse_split_and_layout_pipeline(self):
        wrapper = WeightModule.create(self.qkv(), self.quant)
        source = self.real_qkv_source()
        config = self.load_config()
        result = wrapper.load(source, 0, "cpu", config)
        tile, packed, scales = self.real_tile()
        self.assertEqual(result[wrapper.scale_2.name].shape, torch.Size([]))
        self.assertEqual(result[wrapper.input_scale.name].shape, torch.Size([]))
        self.assertEqual(len(source.reads), 12)
        self.assertEqual(set(source.reads), set(source.tensors))
        self.assertEqual(len(config.exported_device.conversions), 1)
        expected_p = torch.cat((packed, packed[:64], packed[:64]))
        expected_s = torch.cat(
            (
                scales.view(torch.uint8),
                scales[:64].view(torch.uint8),
                scales[:64].view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(result[wrapper.kernel.name], expected_p))
        self.assertTrue(
            torch.equal(result[wrapper.scale.name].view(torch.uint8), expected_s)
        )
        self.assertAlmostEqual(
            result[wrapper.scale_2.name].item(),
            1 / tile["weight_global_scale"],
            places=10,
        )
        self.assertAlmostEqual(
            result[wrapper.input_scale.name].item(),
            1 / tile["input_global_scale"],
            places=10,
        )

    def test_load_tp2_keeps_scale_rows_and_globals_aligned(self):
        wrapper = WeightModule.create(self.qkv(), self.quant)
        _, packed, scales = self.real_tile()
        for rank in (0, 1):
            config = self.load_config(rank=rank, tp=2)
            result = wrapper.load(self.real_qkv_source(), 0, "cpu", config)
            indices = torch.cat(
                (
                    torch.arange(rank * 64, (rank + 1) * 64),
                    torch.arange(rank * 32, (rank + 1) * 32),
                    torch.arange(rank * 32, (rank + 1) * 32),
                )
            )
            self.assertTrue(torch.equal(result[wrapper.kernel.name], packed[indices]))
            self.assertTrue(
                torch.equal(
                    result[wrapper.scale.name].view(torch.uint8),
                    scales.view(torch.uint8)[indices],
                )
            )
            self.assertEqual(len(config.exported_device.conversions), 1)
            self.assertEqual(config.exported_device.conversions[0][0].shape, (128, 64))

    def test_load_gate_up_fusion_and_row_padding(self):
        tile, packed, scales = self.real_tile()
        src = FfnAtomicWeight(
            W.ffn_w13,
            [
                CkptWeightInfo("model.layers.{i}.mlp." + name + ".weight")
                for name in ("gate_proj", "up_proj")
            ],
            config=FfnConfig(align_size=256),
        )
        tensors = {}
        # Gate/up payloads intentionally reuse the real q tile for assembly tests.
        for name in ("gate_proj", "up_proj"):
            prefix = "model.layers.0.mlp." + name
            tensors.update(
                {
                    prefix + ".weight_packed": packed,
                    prefix + ".weight_scale": scales,
                    prefix
                    + ".weight_global_scale": torch.tensor(
                        [tile["weight_global_scale"]]
                    ),
                    prefix
                    + ".input_global_scale": torch.tensor([tile["input_global_scale"]]),
                }
            )
        config = self.load_config()
        wrapper = WeightModule.create(src, self.quant)
        result = wrapper.load(RecordingSource(tensors), 0, "cpu", config)
        actual = result[wrapper.kernel.name]
        self.assertEqual(actual.shape, (512, 64))
        for start in (0, 256):
            self.assertTrue(torch.equal(actual[start : start + 128], packed))
            self.assertEqual(
                torch.count_nonzero(actual[start + 128 : start + 256]).item(), 0
            )
        scale_bytes = result[wrapper.scale.name].view(torch.uint8)
        self.assertEqual(scale_bytes.shape, (512, 8))
        self.assertTrue(torch.equal(scale_bytes[:128], scales.view(torch.uint8)))
        self.assertEqual(torch.count_nonzero(scale_bytes[128:256]).item(), 0)
        self.assertEqual(len(config.exported_device.conversions), 1)

    def test_load_down_projection_padding_and_tp2(self):
        tile, packed, scales = self.real_tile()
        prefix = "model.layers.0.mlp.down_proj"
        source = RecordingSource(
            {
                prefix + ".weight_packed": packed,
                prefix + ".weight_scale": scales,
                prefix
                + ".weight_global_scale": torch.tensor([tile["weight_global_scale"]]),
                prefix
                + ".input_global_scale": torch.tensor([tile["input_global_scale"]]),
            }
        )
        src = FfnAtomicWeight(
            W.ffn_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.down_proj.weight")],
            config=FfnConfig(align_size=256),
        )
        wrapper = WeightModule.create(src, self.quant)
        for rank in (0, 1):
            config = self.load_config(rank=rank, tp=2)
            result = wrapper.load(source, 0, "cpu", config)
            self.assertEqual(result[wrapper.kernel.name].shape, (128, 64))
            self.assertEqual(result[wrapper.scale.name].shape, (128, 8))
            if rank == 0:
                self.assertTrue(torch.equal(result[wrapper.kernel.name], packed))
                self.assertTrue(
                    torch.equal(
                        result[wrapper.scale.name].view(torch.uint8),
                        scales.view(torch.uint8),
                    )
                )
            else:
                self.assertEqual(
                    torch.count_nonzero(result[wrapper.kernel.name]).item(), 0
                )
                self.assertEqual(
                    torch.count_nonzero(
                        result[wrapper.scale.name].view(torch.uint8)
                    ).item(),
                    0,
                )

    def test_load_rejects_bad_down_scale_shape_before_padding_hides_it(self):
        tile, packed, scales = self.real_tile()
        prefix = "model.layers.0.mlp.down_proj"
        source = RecordingSource(
            {
                prefix + ".weight_packed": packed,
                prefix + ".weight_scale": scales[:, :4],
                prefix
                + ".weight_global_scale": torch.tensor([tile["weight_global_scale"]]),
                prefix
                + ".input_global_scale": torch.tensor([tile["input_global_scale"]]),
            }
        )
        src = FfnAtomicWeight(
            W.ffn_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.down_proj.weight")],
            config=FfnConfig(align_size=256),
        )
        config = self.load_config()
        wrapper = WeightModule.create(src, self.quant)
        with self.assertRaisesRegex(ValueError, "local scales"):
            wrapper.load(source, 0, "cpu", config)
        self.assertEqual(config.exported_device.conversions, [])

    def test_load_rejects_missing_key_before_layout(self):
        source = self.real_qkv_source()
        del source.tensors["model.layers.0.self_attn.k_proj.weight_packed"]
        config = self.load_config()
        wrapper = WeightModule.create(self.qkv(), self.quant)
        with self.assertRaises(KeyError):
            wrapper.load(source, 0, "cpu", config)
        self.assertEqual(config.exported_device.conversions, [])

    def test_load_rejects_unequal_globals_before_layout(self):
        source = self.real_qkv_source()
        source.tensors["model.layers.0.self_attn.k_proj.weight_global_scale"] *= 2
        config = self.load_config()
        wrapper = WeightModule.create(self.qkv(), self.quant)
        with self.assertRaisesRegex(ValueError, "identical") as context:
            wrapper.load(source, 0, "cpu", config)
        self.assertIn("layer 0", str(context.exception))
        self.assertIn("model.layers.0.self_attn.q_proj", str(context.exception))
        self.assertIn("21120", str(context.exception))
        self.assertEqual(config.exported_device.conversions, [])

    def test_load_rejects_invalid_group_shape_before_layout(self):
        source = self.real_qkv_source()
        for name in source.tensors:
            if name.endswith(".weight_scale"):
                source.tensors[name] = source.tensors[name][:, :4]
        config = self.load_config()
        wrapper = WeightModule.create(self.qkv(), self.quant)
        with self.assertRaisesRegex(ValueError, "local scales"):
            wrapper.load(source, 0, "cpu", config)
        self.assertEqual(config.exported_device.conversions, [])

    def test_modelopt_still_uses_original_wrapper(self):
        quant = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
        self.assertIs(type(WeightModule.create(self.qkv(), quant)), PerGroupFp4Weight)


if __name__ == "__main__":
    unittest.main()

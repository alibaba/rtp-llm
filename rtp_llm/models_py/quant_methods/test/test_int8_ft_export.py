"""Real CPU FT export/readback, not an INT8 serving implementation."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.config.quant_config import Int8PerTensorQuantConfig
from rtp_llm.device.device_impl import CpuImpl, CudaImpl, RocmImpl
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_tensor_int8_quant_weight import PerTensorInt8QuantWeight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models_py.checkpoint_reader_test_support import has_native_fst
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.quant_methods.int8 import (
    Int8ExportState,
    PrequantizedInt8ExportModule,
    WeightOnlyInt8ExportModule,
    quantize_weight_only_int8,
)
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.ops import TaskType
from rtp_llm.tools.convert.ft_export_weights import FTExportWeights
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, transpose


class Int8FTExportTest(unittest.TestCase):
    def test_groupwise_offline_dense_and_expert_zero_format(self):
        # Run the real platform weight packers on CPU, not a native GPU kernel.
        for impl in (CudaImpl, RocmImpl):
            dev = impl.__new__(impl)
            dev.py_env_configs = SimpleNamespace(runtime_config=SimpleNamespace(specify_gpu_arch="90"))
            for bits in (4, 8):
                pack = 32 // bits
                weight = torch.zeros(64 // pack, 64, dtype=torch.int32)
                zero = torch.full((4, 64 // pack), -1, dtype=torch.int32)
                scale = torch.ones(4, 64, dtype=torch.float16)
                for v2 in (False, True):
                    args = ("cpu", True, False, bits)
                    _, bias, _ = dev.preprocess_groupwise_weight_params(
                        weight, zero, scale, *args, gptq_v2=v2
                    )
                    expected = torch.full_like(scale, -(1 << (bits - 1)) + int(v2))
                    torch.testing.assert_close(bias, expected, atol=0, rtol=0)
                    stacked = [torch.stack((x.T, x.T)) for x in (weight, zero, scale)]
                    _, expert_bias, _ = dev.preprocess_moe_groupwise_weight_params(
                        *stacked, *args, gptq_v2=v2
                    )
                    torch.testing.assert_close(
                        expert_bias, torch.stack((expected, expected)), atol=0, rtol=0
                    )

    def export(self, source, destination, tp=1):
        # Use production tensor descriptors and LoadConfig. This fixture
        # exercises FTExportWeights._records/write_ft_shards, not server init.
        attention = AttnConfig(
            hidden_size=8, head_num=2, head_num_kv=2, size_per_head=4
        )
        output = PerTensorInt8QuantWeight(
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight")],
                transpose,
                config=attention,
            ),
            Int8PerTensorQuantConfig(),
            name=W.attn_o_w,
        )
        norm = PerTensorInt8QuantWeight(
            AtomicWeight(W.pre_decoder_ln_gamma, [CkptWeightInfo("model.norm.weight")]),
            Int8PerTensorQuantConfig(),
            name=W.pre_decoder_ln_gamma,
        )
        config = LoadConfig(
            database=CkptDatabase(source),
            num_layers=1,
            layer_range=range(1),
            hidden_size=8,
            head_num=2,
            head_num_kv=2,
            size_per_head=4,
            moe_pure_tp_mode=False,
            align_size=8,
            moe_align_size=8,
            moe_layer_index=[],
            moe_n_group=1,
            expert_num=0,
            enable_eplb=False,
            phy_exp_num=0,
            tp_size=tp,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            lm_head_tp_size=tp,
            lm_head_tp_rank=0,
            ffn_tp_size=tp,
            ffn_tp_rank=0,
            num_nodes=1,
            compute_dtype=torch.float32,
            # Real stateless identity method from CpuImpl/DeviceBase. Avoid
            # its unrelated server-CLI parser; no numerical methods are replaced.
            exported_device=CpuImpl.__new__(CpuImpl),
            force_cpu_load_weights=True,
        )
        exporter = FTExportWeights.__new__(FTExportWeights)
        exporter._info = ModelWeightInfo([norm], [[output]])
        exporter._custom = ()
        exporter._head_provenance = {}
        exporter._task_type = TaskType.LANGUAGE_MODEL
        exporter._attention_only = False
        exporter._config = config
        exporter.dump_weight_as_ft_style("cpu", destination)

    @staticmethod
    def checkpoint():
        return {
            "model.layers.0.self_attn.o_proj.qweight": torch.arange(
                64, dtype=torch.int8
            ).reshape(8, 8)
            - 32,
            "model.layers.0.self_attn.o_proj.scales": torch.arange(1, 9).float() / 16,
            "model.layers.0.self_attn.o_proj_input_scale": torch.tensor([0.25]),
            "model.norm.weight": torch.arange(8).float() + 1,
            "model.encoder.layer.0.attention.self.qkv_input_scale": torch.tensor([0.5]),
        }

    def test_per_tensor_real_ft_export_preserves_activation_scales_and_norm(self):
        with tempfile.TemporaryDirectory() as root:
            source, destination = Path(root) / "source", Path(root) / "ft"
            source.mkdir()
            values = self.checkpoint()
            save_file(values, source / "model.safetensors")
            self.export(str(source), str(destination))
            files = sorted(destination.glob("*.safetensors"))
            actual = {
                key: value for path in files for key, value in load_file(path).items()
            }
            layer = "rank_00_00_00.layers.0."
            glob = "rank_00_00_00.global."
            expected = {
                layer
                + W.attn_o_w: values["model.layers.0.self_attn.o_proj.qweight"]
                .T.contiguous()
                .reshape(8, -1),
                layer
                + W.attn_o_s: values["model.layers.0.self_attn.o_proj.scales"].reshape(
                    8, 1
                ),
                layer + W.attention_output_static_quant: torch.tensor([4.0]),
                layer
                + W.attention_output_static_quant_reciprocal: torch.tensor([0.25]),
                glob
                + W.pre_decoder_ln_gamma: values["model.norm.weight"].reshape(8, 1),
                glob + W.pre_decoder_ln_static_quant: torch.tensor([2.0]),
                glob + W.pre_decoder_ln_static_quant_reciprocal: torch.tensor([0.5]),
            }
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            if not has_native_fst():
                return
            fst = {
                name: unwrap_checkpoint_tensor(value)[0]
                for name, value in iter_fst_range_tensors([str(p) for p in files])
            }
            for key, value in expected.items():
                torch.testing.assert_close(fst[key], value, rtol=0, atol=0)

    def test_per_tensor_tp2_replicates_calibration_steps(self):
        with tempfile.TemporaryDirectory() as root:
            save_file(self.checkpoint(), os.path.join(root, "model.safetensors"))
            dest = Path(root) / "ft"
            self.export(root, str(dest), tp=2)
            fields = load_file(next(dest.glob("*.safetensors")))
            key = "rank_00_00_00.layers.0."
            torch.testing.assert_close(
                fields[key + W.attention_output_static_quant], torch.tensor([4.0])
            )
            torch.testing.assert_close(
                fields[key + W.attention_output_static_quant_reciprocal],
                torch.tensor([0.25]),
            )

    def test_weight_only_numerics_dense_and_experts_preserve_max128(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            matrix = torch.tensor([[128.0, -128.0, 0.0], [1.5, -2.5, 0.0]], dtype=dtype)
            for weight in (matrix, matrix * 2):
                quantized, scale = quantize_weight_only_int8(weight)
                reference_scale = weight.abs().max(0).values.clamp(min=1e-8) / 128
                reference = (
                    (weight / reference_scale).round().clamp(-128, 127).to(torch.int8)
                )
                torch.testing.assert_close(quantized, reference, rtol=0, atol=0)
                torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)

    def test_module_metadata_roundtrip_and_negative_contracts(self):
        weight = torch.arange(24, dtype=torch.int8).reshape(3, 8)
        module = PrequantizedInt8ExportModule(weight, torch.ones(8), per_channel=False)
        converted, _ = module.export()
        self.assertEqual(converted.data_ptr(), weight.data_ptr())
        restored = Int8ExportState.from_export(
            module.export_metadata(), module.export_tensors()
        )
        torch.testing.assert_close(
            restored.weight, weight.reshape(8, 3), rtol=0, atol=0
        )
        with self.assertRaisesRegex(RuntimeError, "offline-only"):
            restored(torch.ones(1, 8))
        with self.assertRaisesRegex(RuntimeError, "only once"):
            module.export()
        with self.assertRaises(ValueError):
            PrequantizedInt8ExportModule(weight.float(), per_channel=False)
        with self.assertRaises(ValueError):
            PrequantizedInt8ExportModule(
                weight, torch.ones(8, dtype=torch.int8), per_channel=False
            )

    def test_weight_only_dense_and_expert_bytes_match_main_device_layout(self):
        for platform, impl in (("Cuda", CudaImpl), ("ROCm", RocmImpl)):
            device = impl.__new__(impl)
            for stacked in (False, True):
                weight = (torch.arange(64 * 32).reshape(64, 32) % 127).float() - 63
                if stacked:
                    weight = torch.stack((weight.T, (weight / 2).T))
                module = WeightOnlyInt8ExportModule(weight, stacked=stacked)
                actual, scale = module.export(platform=platform, arch=90)
                matrices = [x.T.contiguous() for x in weight] if stacked else [weight]
                expected, expected_scales = [], []
                for matrix in matrices:
                    scale_ref = matrix.abs().max(0).values.clamp(min=1e-8) / 128
                    q_ref = (matrix / scale_ref).round().clamp(-128, 127).to(torch.int8)
                    expected.append(
                        device.preprocess_weights_for_mixed_gemm(
                            q_ref, torch.int8, "90"
                        )
                    )
                    expected_scales.append(scale_ref)
                reference = (
                    torch.stack(expected)
                    if stacked
                    else expected[0].reshape(weight.shape)
                )
                reference_scale = (
                    torch.stack(expected_scales) if stacked else expected_scales[0]
                )
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as root:
                    path = os.path.join(root, "int8.safetensors")
                    save_file({"weight": actual, "scale": scale}, path)
                    tensors = load_file(path)
                    restored = Int8ExportState.from_export(
                        module.export_metadata(), tensors
                    )
                    torch.testing.assert_close(
                        restored.weight, reference, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()

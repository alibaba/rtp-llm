import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import RowParallelLinear
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.groupwise_export import cuda_groupwise_layout
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _Model(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.proj = RowParallelLinear(
            input_size=config.width,
            output_size=config.rows,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            prefix="proj",
            bias=True,
            quant_config=load.quant_config,
            params_dtype=load.compute_dtype,
        )


def load_fixture(
    device, tp=1, rank=0, method="scratch", dtype=torch.float16,
    rows=192, width=256, startup=False,
):
    gen = torch.Generator().manual_seed(91299458)
    weight = torch.randn((rows, width), generator=gen).to(dtype)
    # Different rank-local max values distinguish shard-before-quantization.
    weight[:, width // 2:] *= 4
    bias = torch.randn(rows, generator=gen).to(dtype)
    with tempfile.TemporaryDirectory() as directory:
        save_file(
            {"proj.weight": weight, "proj.bias": bias},
            os.path.join(directory, "model.safetensors"),
        )
        adapter = None
        if startup:
            path = os.path.join(directory, "lora")
            os.mkdir(path)
            with open(os.path.join(path, "adapter_config.json"), "w") as f:
                json.dump({"r": 2, "lora_alpha": 3}, f)
            a = torch.randn(2, width, generator=gen)
            b = torch.randn(rows, 2, generator=gen)
            save_file(
                {"base_model.model.proj.lora_A.weight": a,
                 "base_model.model.proj.lora_B.weight": b},
                os.path.join(path, "adapter_model.safetensors"),
            )
            adapter = ("one", path)
            weight = weight + ((b @ a) * 1.5).to(dtype)
        with patch.dict(MODEL_REGISTRY, {"w8a16_fixture": _Model}):
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="w8a16_fixture", rows=rows, width=width),
                NewLoaderConfig(
                    device=device,
                    compute_dtype=dtype,
                    tp_size=tp,
                    tp_rank=rank,
                    quant_config=QuantizationConfig("INT8"),
                    load_method=method,
                    force_cpu_load_weights=(method == "fastsafetensors"),
                ),
                model_path=directory,
            )
            layer = loader.load(startup_lora=adapter).proj
            assert loader.loaded_method.value == method
    local = weight.chunk(tp, dim=1)[rank].T
    scale = local.abs().amax(0).clamp(min=1e-8) / 128
    q = (local / scale).round().clamp(-128, 127).to(torch.int8)
    return layer, q, scale, bias


class WeightOnlyInt8CheckpointTest(unittest.TestCase):
    @torch.inference_mode()
    def test_startup_lora_overflow_leaves_unpacked_weight_untouched(self):
        from rtp_llm.models_py.runtime_weight_update import LoRAUpdate

        layer = RowParallelLinear(
            input_size=64, output_size=64, prefix="proj",
            params_dtype=torch.float16, quant_config=QuantizationConfig("INT8"),
        )
        base = torch.full((64, 64), 40000.0, dtype=torch.float16)
        layer.load_weights({"weight": base.clone()})
        original = layer.weight
        update = LoRAUpdate(
            "proj.weight", torch.full((1, 64), 200.0), torch.full((64, 1), 200.0)
        )
        layer.quant_method.stage_startup_lora(layer, update)
        with self.assertRaisesRegex(ValueError, "Non-finite merged LoRA weight.*proj.weight"):
            layer.process_weights_after_loading()
        self.assertIs(layer.weight, original)
        self.assertTrue(torch.equal(layer.weight, base))
        self.assertIsNone(layer.weight_scale)
        self.assertIs(layer._startup_lora_update, update)

    @torch.inference_mode()
    def test_update_requantizes_complete_shard_without_replacing_storage(self):
        from rtp_llm.models_py.runtime_weight_update import RuntimeWeightTransaction

        layer, _, _, _ = load_fixture("cpu", tp=2, rank=1, rows=160, width=192)
        ptrs = [p.data_ptr() for p in (layer.weight, layer.weight_scale, layer.bias)]
        new = torch.arange(160 * 192).reshape(160, 192).remainder(93).half() - 32
        txn = RuntimeWeightTransaction(layer.prepare_weight_updates({"weight": new}))
        txn.install()
        txn.commit()
        local = new[:, 96:].T
        scale = local.abs().amax(0) / 128
        q = (local / scale).round().clamp(-128, 127).to(torch.int8)
        torch.testing.assert_close(
            layer.weight, cuda_groupwise_layout(torch.nn.functional.pad(q, (0, 32, 0, 32)), 8, 80)
        )
        torch.testing.assert_close(layer.weight_scale[:160], scale)
        updates = layer.prepare_weight_updates({"bias": torch.arange(160).half()})
        self.assertEqual([u.name for u in updates], ["bias"])
        txn = RuntimeWeightTransaction(updates)
        txn.install()
        txn.commit()
        torch.testing.assert_close(layer.bias, torch.arange(160).half())
        self.assertEqual(ptrs, [p.data_ptr() for p in (layer.weight, layer.weight_scale, layer.bias)])

    def test_local_rank_before_quantization(self):
        self._check_local_rank_before_quantization("scratch")

    @unittest.skipIf(os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0", "platform lock has no native FST reader")
    def test_local_rank_before_quantization_native_fst(self):
        self._check_local_rank_before_quantization("fastsafetensors")

    def _check_local_rank_before_quantization(self, method):
        for rank in (0, 1):
            with self.subTest(method=method, rank=rank):
                layer, q, scale, bias = load_fixture("cpu", 2, rank, method=method)
                expected = cuda_groupwise_layout(q, 8, 80)
                torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    layer.weight_scale, scale, atol=0, rtol=0
                )
                torch.testing.assert_close(layer.bias, bias, atol=0, rtol=0)
                storage = layer.weight.data_ptr()
                layer.process_weights_after_loading()
                self.assertEqual(layer.weight.data_ptr(), storage)

    def test_peft_merges_before_quantization_of_unaligned_local_shards(self):
        self._check_peft_merges_before_quantization_of_unaligned_local_shards("scratch")

    @unittest.skipIf(os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0", "platform lock has no native FST reader")
    def test_peft_merges_before_quantization_of_unaligned_local_shards_native_fst(self):
        self._check_peft_merges_before_quantization_of_unaligned_local_shards("fastsafetensors")

    def _check_peft_merges_before_quantization_of_unaligned_local_shards(self, method):
        for rank in (0, 1):
            with self.subTest(method=method, rank=rank):
                layer, q, scale, _ = load_fixture(
                    "cpu", 2, rank, method, dtype=torch.bfloat16,
                    rows=160, width=192, startup=True,
                )
                # Logical rank-local K=96/N=160; the GEMM uses K=128/N=192.
                padded = torch.nn.functional.pad(q, (0, 32, 0, 32))
                expected = cuda_groupwise_layout(padded, 8, 80)
                torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    layer.weight_scale, torch.nn.functional.pad(scale, (0, 32), value=1),
                    atol=0, rtol=0,
                )
                self.assertFalse(hasattr(layer, "_startup_lora_update"))
                with self.assertRaisesRegex(NotImplementedError, "Quantized LoRA"):
                    layer.prepare_lora_factors("weight", None, None)

    def test_reject_prequantized_weights(self):
        args = dict(quant_config=QuantizationConfig("INT8"), prefix="proj")
        layer = RowParallelLinear(input_size=128, output_size=192, **args)
        with self.assertRaisesRegex(ValueError, "floating checkpoint"):
            layer.load_weights({"weight": torch.zeros(192, 128, dtype=torch.int8)})


if __name__ == "__main__":
    unittest.main()

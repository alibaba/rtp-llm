"""Native UE8M0 checkpoint loading without pretending CPU executes DeepGEMM."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Linear
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor
from rtp_llm.models_py.weight_mapper import get_all_weights


class Dsv4FP8CheckpointTest(unittest.TestCase):
    def test_loaded_owner_selects_shared_backend_but_retains_regular_dispatch(self):
        layer = Dsv4FP8Linear(512, 256)
        layer.load_weights(self._weights())
        backend = SimpleNamespace(
            quantize_input=mock.Mock(), forward_quantized=mock.Mock()
        )
        dispatcher = mock.Mock(spec=["prequantized_input_backend"])
        dispatcher.prequantized_input_backend = backend
        attention = ModuleType("rtp_llm.models_py.modules.dsv4.fp8.attention")
        attention._v4_fp8_linear = mock.Mock(return_value=dispatcher)
        with mock.patch.dict(
            "sys.modules", {attention.__name__: attention}
        ), mock.patch.object(layer, "validate_runtime_device"):
            dispatcher.prequantized_input_backend = object()
            with self.assertRaisesRegex(
                RuntimeError, "cannot consume shared input scales"
            ):
                layer.process_weights_after_loading()
            self.assertIsNone(layer._executor)
            dispatcher.prequantized_input_backend = backend
            layer.process_weights_after_loading()

        x = torch.zeros(2, 512, dtype=torch.bfloat16)
        pair = (
            x.to(torch.float8_e4m3fn),
            torch.full((2, 1), 0x797B7D7F, dtype=torch.int32),
        )
        out = torch.empty(2, 256, dtype=torch.bfloat16)
        backend.quantize_input.return_value = pair
        self.assertIs(layer.quantize_input(x), pair)
        layer.forward_quantized(*pair, out=out)
        backend.quantize_input.assert_called_once_with(x)
        backend.forward_quantized.assert_called_once_with(*pair, out=out)
        self.assertFalse(layer.supports_fused_prefill_norm_quant)
        dispatcher.assert_not_called()
        layer(x)
        dispatcher.assert_called_once_with(x)
        # Device migration invalidates both execution paths.
        layer.to("cpu")
        self.assertFalse(layer.supports_fused_prefill_norm_quant)
        with self.assertRaisesRegex(RuntimeError, "must be prepared"):
            layer.forward_quantized(*pair, out=out)
        # A factory implementation which directly exposes the packed-input
        # API retains its original fusion eligibility.
        attention._v4_fp8_linear.return_value = backend
        with mock.patch.dict(
            "sys.modules", {attention.__name__: attention}
        ), mock.patch.object(layer, "validate_runtime_device"):
            layer.process_weights_after_loading()
        self.assertTrue(layer.supports_fused_prefill_norm_quant)

    def test_preflight_requires_main_gemm_and_native_scale_packer(self):
        # Isolate symbol resolution, not a GPU execution test. main has no
        # recovery-branch is_deep_gemm_runtime_available() helper.
        module = ModuleType("deep_gemm")
        utils = ModuleType("deep_gemm.utils")
        layout = ModuleType("deep_gemm.utils.layout")
        modules = {m.__name__: m for m in (module, utils, layout)}
        module.fp8_gemm_nt = lambda: None
        layout.get_mn_major_tma_aligned_packed_ue8m0_tensor = lambda: None
        layer = Dsv4FP8Linear(512, 256)
        with mock.patch.dict("sys.modules", modules), mock.patch.object(
            torch.version, "hip", None
        ), mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            layer.validate_runtime_device("cuda:0")
            with mock.patch.object(module, "fp8_gemm_nt", None), self.assertRaisesRegex(
                RuntimeError, "not callable"
            ):
                layer.validate_runtime_device("cuda:0")
            del layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                layer.validate_runtime_device("cuda:0")

    @unittest.skipUnless(
        os.getenv("DSV4_FLASH_0731_CHECKPOINT"), "release asset opt-in"
    )
    def test_release_attention_wkv_bytes(self):
        # Read only one layer's ~2MB MQA projection from the exact smoke release.
        model = Path(os.environ["DSV4_FLASH_0731_CHECKPOINT"])
        index = json.loads((model / "model.safetensors.index.json").read_bytes())
        prefix = "layers.0.attn.wkv."
        names = [prefix + "weight", prefix + "scale"]
        paths = sorted({str(model / index["weight_map"][name]) for name in names})
        layer = Dsv4FP8Linear(4096, 512)
        layer.load_weights(
            (name[len(prefix) :], value)
            for name, value in get_all_weights(
                paths,
                name_filter=names.__contains__,
                read_spec_resolver=lambda n, s: layer.checkpoint_read_spec(
                    n[len(prefix) :], s
                ).for_checkpoint_name(n),
            )
        )
        layer.validate_weights_loaded()
        for name, dtype in zip(names, ("F8_E4M3", "F8_E8M0")):
            # Compare the owned storage against original file payload bytes,
            # independent of the safetensors reader and its dtype decoder.
            with (model / index["weight_map"][name]).open("rb") as stream:
                (length,) = struct.unpack("<Q", stream.read(8))
                entry = json.loads(stream.read(length))[name]
                self.assertEqual(entry["dtype"], dtype)
                lo, hi = entry["data_offsets"]
                stream.seek(8 + length + lo)
                self.assertEqual(
                    getattr(layer, name[len(prefix) :])
                    .view(torch.uint8)
                    .numpy()
                    .tobytes(),
                    stream.read(hi - lo),
                )

    def _weights(self):
        weight = ((torch.arange(256 * 512).reshape(256, 512) % 16) - 8).to(
            torch.float8_e4m3fn
        )
        scale = torch.tensor(
            [[126, 127, 128, 129], [130, 131, 132, 133]], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        return {"weight": weight, "scale": scale}

    def test_real_pytorch_checkpoint_reads_local_weight_and_native_scale_blocks(self):
        weights = self._weights()
        with tempfile.TemporaryDirectory() as folder:
            # The pinned ROCm environment's safetensors writer predates
            # float8_e8m0fnu. A real PyTorch checkpoint preserves native bytes
            # without relabelling scales as uint8 or ordinary float32.
            filename = os.path.join(folder, "model.pt")
            torch.save(weights, filename)
            for partition in ("replicated", "column", "row"):
                for rank in range(2):
                    with self.subTest(partition=partition, rank=rank):
                        layer = Dsv4FP8Linear(
                            512, 256, tp_size=2, tp_rank=rank, partition=partition
                        )
                        layer.load_weights(
                            get_all_weights(
                                [filename],
                                read_spec_resolver=layer.checkpoint_read_spec,
                            )
                        )
                        NewModelLoader._validate_loaded_weights(layer)
                        for name, expected in weights.items():
                            if partition != "replicated":
                                expected = expected.chunk(
                                    2, dim=0 if partition == "column" else 1
                                )[rank]
                            actual = getattr(layer, name)
                            self.assertEqual(actual.dtype, weights[name].dtype)
                            torch.testing.assert_close(
                                actual.view(torch.uint8),
                                expected.contiguous().view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )

    def test_full_tensor_dispatch_matches_presliced_reads(self):
        weights = self._weights()
        layer = Dsv4FP8Linear(512, 256, tp_size=2, tp_rank=1, partition="row")
        layer.load_weights(weights)
        torch.testing.assert_close(
            layer.scale.view(torch.uint8),
            weights["scale"][:, 2:].contiguous().view(torch.uint8),
        )
        layer.validate_weights_loaded()

    def test_float_scales_cannot_silently_replace_ue8m0_bytes(self):
        layer = Dsv4FP8Linear(512, 256)
        with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
            layer.load_weights({"scale": torch.ones(2, 4)})

    def test_wrong_rank_readspec_is_rejected(self):
        rank0 = Dsv4FP8Linear(512, 256, tp_size=2, tp_rank=0, partition="row")
        rank1 = Dsv4FP8Linear(512, 256, tp_size=2, tp_rank=1, partition="row")
        value = rank0.checkpoint_read_spec("scale", (2, 4)).materialize(
            self._weights()["scale"]
        )
        self.assertIsInstance(value, LoadedCheckpointTensor)
        with self.assertRaisesRegex(ValueError, "different shard"):
            rank1.load_weights({"scale": value})

    def test_missing_scale_duplicate_and_unknown_parameters_fail(self):
        layer = Dsv4FP8Linear(512, 256)
        weight = self._weights()["weight"]
        layer.load_weights({"weight": weight})
        with self.assertRaisesRegex(RuntimeError, "scale"):
            layer.validate_weights_loaded()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights({"weight": weight})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            layer.load_weights({"weight_scale_inv": torch.ones(2, 4)})

    def test_tp_scale_blocks_must_not_be_fractional(self):
        with self.assertRaisesRegex(ValueError, "128-element"):
            Dsv4FP8Linear(512, 256, tp_size=4, partition="column")
        with self.assertRaisesRegex(ValueError, "align to 128"):
            Dsv4FP8Linear(513, 256)

    def test_cpu_backend_rejected_and_unprepared_forward_fails(self):
        layer = Dsv4FP8Linear(512, 256)
        layer.load_weights(self._weights())
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            layer.validate_runtime_device(torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            layer.process_weights_after_loading()
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            layer(torch.zeros(1, 512, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()

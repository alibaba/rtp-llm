"""Load completeness and input-layout regression tests for newloader."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

try:
    import librtp_compute_ops  # noqa: F401
    import librtp_compute_ops.rtp_llm_ops  # noqa: F401
except ImportError:
    librtp_stub = types.ModuleType("librtp_compute_ops")
    rtp_ops_stub = types.ModuleType("librtp_compute_ops.rtp_llm_ops")
    compute_ops_stub = types.ModuleType("rtp_llm.ops.compute_ops")

    class _MissingComputeOp:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "librtp_compute_ops is unavailable in this CPU test environment"
            )

    class _RtpOpsStub:
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _MissingComputeOp

    def _missing_compute_op(*args, **kwargs):
        raise RuntimeError("librtp_compute_ops is unavailable in this CPU test environment")

    def _compute_ops_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "get_device_id":
            return lambda: 0
        if name == "rtp_llm_ops":
            return _RtpOpsStub()
        return _MissingComputeOp

    librtp_stub.get_device_id = lambda: 0
    librtp_stub.preprocess_gemm_weight_by_key = _missing_compute_op
    librtp_stub.preprocess_weight_scale = _missing_compute_op
    librtp_stub.rtp_llm_ops = rtp_ops_stub
    def _rtp_ops_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _MissingComputeOp

    librtp_stub.__file__ = "<librtp_compute_ops test stub>"
    rtp_ops_stub.__file__ = "<librtp_compute_ops.rtp_llm_ops test stub>"
    compute_ops_stub.__file__ = "<rtp_llm.ops.compute_ops test stub>"
    rtp_ops_stub.__getattr__ = _rtp_ops_getattr
    compute_ops_stub.__getattr__ = _compute_ops_getattr
    compute_ops_stub.get_device_id = lambda: 0
    compute_ops_stub.preprocess_gemm_weight_by_key = _missing_compute_op
    compute_ops_stub.preprocess_weight_scale = _missing_compute_op
    compute_ops_stub.rtp_llm_ops = _RtpOpsStub()
    sys.modules.setdefault("librtp_compute_ops", librtp_stub)
    sys.modules.setdefault("librtp_compute_ops.rtp_llm_ops", rtp_ops_stub)
    sys.modules.setdefault("rtp_llm.ops.compute_ops", compute_ops_stub)

from rtp_llm.models_py.layers.attention import MMEncoderAttention
from rtp_llm.models_py.layers.embedding import HiddenParallelEmbedding, ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import LayerNorm, RMSNorm
from rtp_llm.models_py.layers.conv import Conv3dLayer
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.model_loader import (
    LoadConfig,
    LoadMethod,
    NewModelLoader,
    _create_ep_filter,
    _discover_ckpt_files,
    _get_all_weights,
    _is_quantized_load,
    _normalize_fastsafetensors_stacked_moe_weights,
    _tp_slice,
)
from rtp_llm.models_py.new_models.bert import BertForEmbedding, RobertaForEmbedding
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import Fp8LinearMethod, _runtime_fp8_dtype
from rtp_llm.models_py.quant_methods.awq_triton import awq_dequantize_triton

BaseMoEExperts = None


def _ensure_moe_test_deps():
    global BaseMoEExperts
    if BaseMoEExperts is not None:
        return
    try:
        from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts as _BaseMoEExperts
        import rtp_llm.models_py.quant_methods.fp8_moe  # noqa: F401
        import rtp_llm.models_py.quant_methods.w4a8_moe  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"MoE test dependencies unavailable: {exc}") from exc
    BaseMoEExperts = _BaseMoEExperts


def _qc(quant_type: str) -> QuantizationConfig:
    return QuantizationConfig(quant_type=quant_type)


class TestNewModelRegistry(unittest.TestCase):
    def test_existing_embedding_models_resolve_via_newloader_registry(self):
        for model_type, expected_name in (
            ("bert", "BertForEmbedding"),
            ("roberta", "RobertaForEmbedding"),
        ):
            config = types.SimpleNamespace(model_type=model_type, num_layers=1)
            loader = NewModelLoader(
                config,
                LoadConfig(compute_dtype=torch.float32, device="cpu"),
                model_path="/tmp",
                device="cpu",
            )
            model = loader._create_model()
            self.assertEqual(type(model).__name__, expected_name)


class TestQuantMethodValidation(unittest.TestCase):
    def test_awq_dequant_rejects_empty_scale_before_group_size_division(self):
        with self.assertRaisesRegex(ValueError, "at least one group"):
            awq_dequantize_triton(
                torch.empty(4, 1, dtype=torch.int32),
                torch.empty(0, 8, dtype=torch.float16),
                torch.empty(0, 1, dtype=torch.int32),
            )

    def test_fp8_online_post_load_rejects_non_2d_weight(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_qc("fp8_online"),
            prefix="fp8_online",
            params_dtype=torch.float32,
        )
        layer.weight = torch.nn.Parameter(torch.zeros(2, 2, 2), requires_grad=False)
        layer._new_loader_force_cpu_load_weights = True
        with self.assertRaisesRegex(ValueError, "expected 2D weight"):
            layer.quant_method.process_weights_after_loading(layer)


class TestWeightMapperStreaming(unittest.TestCase):
    def test_safetensors_loader_reads_one_tensor_at_a_time(self):
        class FakeSafeOpen:
            def __enter__(self):
                self.read_names = []
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def keys(self):
                return ["a.weight", "b.weight"]

            def get_tensor(self, name):
                self.read_names.append(name)
                return torch.full((1,), len(self.read_names), dtype=torch.float32)

        fake = FakeSafeOpen()

        def fake_safe_open(path, framework, device):
            self.assertEqual(path, "model.safetensors")
            self.assertEqual(framework, "pt")
            self.assertEqual(device, "cpu")
            return fake

        with mock.patch("safetensors.safe_open", side_effect=fake_safe_open):
            items = list(
                weight_mapper.get_all_weights(["model.safetensors"], device="cpu")
            )

        self.assertEqual([name for name, _ in items], ["a.weight", "b.weight"])
        self.assertEqual(fake.read_names, ["a.weight", "b.weight"])



    def test_pytorch_loader_unwraps_state_dict_and_model_containers(self):
        tensors = {
            "linear.weight": torch.ones(2, 2),
            "linear.bias": torch.zeros(2),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for container_key in ("state_dict", "model"):
                path = os.path.join(tmpdir, f"wrapped_{container_key}.pt")
                torch.save({container_key: dict(tensors), "epoch": 1}, path)
                loaded = dict(weight_mapper.get_all_weights([path], device="cpu"))
                self.assertEqual(set(loaded), set(tensors))
                self.assertTrue(torch.equal(loaded["linear.weight"], tensors["linear.weight"]))

    def test_scratch_loader_unwraps_wrapped_pytorch_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wrapped.bin")
            torch.save({"state_dict": {"x.weight": torch.ones(1)}}, path)
            loaded = list(_get_all_weights([path], device="cpu"))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0], "x.weight")
        self.assertTrue(torch.equal(loaded[0][1], torch.ones(1)))

    def test_lora_tp_slice_rejects_non_divisible_dim(self):
        with self.assertRaisesRegex(ValueError, "LoRA TP slice.*divisible"):
            _tp_slice(torch.zeros(5, 4), tp_size=2, tp_rank=0, rule=(0, "split"))

    def test_scratch_ep_filter_expands_stacked_moe_to_local_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stacked.pt")
            gate_up = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
            down = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
            gate_scale = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
            torch.save(
                {
                    "model.layers.0.mlp.experts.gate_up_proj": gate_up,
                    "model.layers.0.mlp.experts.down_proj": down,
                    "model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv": gate_scale,
                    "model.layers.0.input_layernorm.weight": torch.ones(2),
                },
                path,
            )
            ep_filter = _create_ep_filter(ep_size=2, ep_rank=1, num_experts=4)
            loaded = list(_get_all_weights([path], device="cpu", name_filter=ep_filter))

        self.assertEqual(
            [name for name, _ in loaded],
            [
                "model.layers.0.mlp.experts.2.gate_up_proj.weight",
                "model.layers.0.mlp.experts.3.gate_up_proj.weight",
                "model.layers.0.mlp.experts.2.down_proj.weight",
                "model.layers.0.mlp.experts.3.down_proj.weight",
                "model.layers.0.mlp.experts.2.gate_up_proj.weight_scale_inv",
                "model.layers.0.mlp.experts.3.gate_up_proj.weight_scale_inv",
                "model.layers.0.input_layernorm.weight",
            ],
        )
        torch.testing.assert_close(loaded[0][1], gate_up[2])
        torch.testing.assert_close(loaded[1][1], gate_up[3])
        torch.testing.assert_close(loaded[2][1], down[2])
        torch.testing.assert_close(loaded[3][1], down[3])
        torch.testing.assert_close(loaded[4][1], gate_scale[2])
        torch.testing.assert_close(loaded[5][1], gate_scale[3])



    def test_fastsafetensors_path_normalizes_stacked_moe_names_before_load(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = []

            def to_empty(self, device):
                return self

            def load_weights(self, weights_iter):
                self.loaded = list(weights_iter)

        model = CaptureModel()
        config = types.SimpleNamespace(model_type="dummy", num_layers=1)
        loader = NewModelLoader(
            config,
            LoadConfig(compute_dtype=torch.float32, device="cpu", load_method="fastsafetensors"),
            model_path="/tmp/nonexistent",
            device="cpu",
        )
        gate_up = torch.zeros(4, 16)
        down = torch.zeros(8, 4)
        with mock.patch.object(loader, "_create_model", return_value=model), \
             mock.patch.object(loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]), \
             mock.patch.object(loader, "_run_post_load_hooks", return_value=None), \
             mock.patch.object(loader, "_log_peak_gpu_memory", return_value=None), \
             mock.patch(
                 "rtp_llm.models_py.model_loader._get_fastsafetensors_weights",
                 return_value=iter(
                     [
                         ("model.layers.0.mlp.experts.0.gate_up_proj", gate_up),
                         ("model.layers.0.mlp.experts.0.down_proj", down),
                     ]
                 ),
             ):
            loaded_model = loader._load_via_fastsafetensors()
        self.assertIs(loaded_model, model)
        self.assertEqual(
            [name for name, _ in model.loaded],
            [
                "model.layers.0.mlp.experts.0.gate_up_proj.weight",
                "model.layers.0.mlp.experts.0.down_proj.weight",
            ],
        )

    def test_fastsafetensors_path_ep_filters_stacked_moe_before_load(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = []

            def to_empty(self, device):
                return self

            def load_weights(self, weights_iter):
                self.loaded = list(weights_iter)

        model = CaptureModel()
        config = types.SimpleNamespace(model_type="dummy", num_layers=1, expert_num=4)
        loader = NewModelLoader(
            config,
            LoadConfig(
                compute_dtype=torch.float32,
                device="cpu",
                load_method="fastsafetensors",
                ep_size=2,
                ep_rank=1,
            ),
            model_path="/tmp/nonexistent",
            device="cpu",
        )
        gate_up = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        down = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        gate_scale = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
        with mock.patch.object(loader, "_create_model", return_value=model), \
             mock.patch.object(loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]), \
             mock.patch.object(loader, "_run_post_load_hooks", return_value=None), \
             mock.patch.object(loader, "_log_peak_gpu_memory", return_value=None), \
             mock.patch(
                 "rtp_llm.models_py.model_loader._get_fastsafetensors_weights",
                 return_value=iter(
                     [
                         ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                         ("model.layers.0.mlp.experts.down_proj", down),
                         ("model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv", gate_scale),
                     ]
                 ),
             ):
            loaded_model = loader._load_via_fastsafetensors()
        self.assertIs(loaded_model, model)
        self.assertEqual(
            [name for name, _ in model.loaded],
            [
                "model.layers.0.mlp.experts.2.gate_up_proj.weight",
                "model.layers.0.mlp.experts.3.gate_up_proj.weight",
                "model.layers.0.mlp.experts.2.down_proj.weight",
                "model.layers.0.mlp.experts.3.down_proj.weight",
                "model.layers.0.mlp.experts.2.gate_up_proj.weight_scale_inv",
                "model.layers.0.mlp.experts.3.gate_up_proj.weight_scale_inv",
            ],
        )
        torch.testing.assert_close(model.loaded[0][1], gate_up[2])
        torch.testing.assert_close(model.loaded[1][1], gate_up[3])
        torch.testing.assert_close(model.loaded[2][1], down[2])
        torch.testing.assert_close(model.loaded[3][1], down[3])
        torch.testing.assert_close(model.loaded[4][1], gate_scale[2])
        torch.testing.assert_close(model.loaded[5][1], gate_scale[3])

    def test_load_config_device_cpu_is_used_by_scratch_loader(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_device = None

            def load_weights(self, weights_iter):
                list(weights_iter)

            def to(self, device):
                self.to_device = device
                return self

        model = CaptureModel()
        loader = NewModelLoader(
            types.SimpleNamespace(model_type="dummy", num_layers=1),
            LoadConfig(compute_dtype=torch.float32, device="cpu"),
            model_path="/tmp/nonexistent",
        )
        with mock.patch.object(
            loader, "_create_model", return_value=model
        ), mock.patch.object(
            loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]
        ), mock.patch(
            "rtp_llm.models_py.model_loader._get_all_weights", return_value=iter([])
        ), mock.patch.object(
            loader, "_run_post_load_hooks", return_value=None
        ), mock.patch.object(
            loader, "_log_peak_gpu_memory", return_value=None
        ):
            self.assertIs(loader._load_via_scratch(), model)
        self.assertEqual(loader.device, "cpu")
        self.assertEqual(model.to_device, "cpu")

    def test_force_cpu_scratch_runs_post_load_before_device_move(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.events = []
                self.child = torch.nn.Module()

                def hook():
                    self.events.append(
                        (
                            "hook",
                            getattr(
                                self.child,
                                "_new_loader_force_cpu_load_weights",
                                None,
                            ),
                        )
                    )

                self.child.process_weights_after_loading = hook

            def load_weights(self, weights_iter):
                list(weights_iter)

            def to(self, device):
                self.events.append(("to", device))
                return self

        model = CaptureModel()
        loader = NewModelLoader(
            types.SimpleNamespace(model_type="dummy", num_layers=1),
            LoadConfig(
                compute_dtype=torch.float32,
                device="cuda",
                force_cpu_load_weights=True,
            ),
            model_path="/tmp/nonexistent",
        )
        with mock.patch.object(
            loader, "_create_model", return_value=model
        ), mock.patch.object(
            loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]
        ), mock.patch(
            "rtp_llm.models_py.model_loader._get_all_weights", return_value=iter([])
        ), mock.patch.object(
            loader, "_log_peak_gpu_memory", return_value=None
        ), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertIs(loader._load_via_scratch(), model)
        self.assertEqual(model.events, [("hook", True), ("to", "cuda")])

    def test_auto_fastsafetensors_fallback_releases_before_scratch(self):
        loader = NewModelLoader(
            types.SimpleNamespace(model_type="dummy", num_layers=1),
            LoadConfig(compute_dtype=torch.float32, device="cuda"),
            model_path="/tmp/nonexistent",
            device="cuda",
        )
        scratch_model = torch.nn.Module()
        with mock.patch.object(
            loader, "_resolve_load_method", return_value=(LoadMethod.FASTSAFETENSORS, False)
        ), mock.patch.object(
            loader, "_load_via_fastsafetensors", side_effect=RuntimeError("fast failed")
        ), mock.patch(
            "gc.collect"
        ) as collect, mock.patch(
            "torch.cuda.is_available", return_value=True
        ), mock.patch(
            "torch.cuda.empty_cache"
        ) as empty_cache, mock.patch.object(
            loader, "_load_via_scratch", return_value=scratch_model
        ) as scratch:
            self.assertIs(loader.load(), scratch_model)
        collect.assert_called_once()
        empty_cache.assert_called_once()
        scratch.assert_called_once()

    def test_load_config_device_cpu_is_used_by_fastsafetensors_loader(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_empty_device = None

            def to_empty(self, device):
                self.to_empty_device = device
                return self

            def load_weights(self, weights_iter):
                self.loaded = list(weights_iter)

        model = CaptureModel()
        loader = NewModelLoader(
            types.SimpleNamespace(model_type="dummy", num_layers=1),
            LoadConfig(
                compute_dtype=torch.float32,
                device="cpu",
                load_method=LoadMethod.FASTSAFETENSORS,
            ),
            model_path="/tmp/nonexistent",
        )
        with mock.patch.object(
            loader, "_create_model", return_value=model
        ), mock.patch.object(
            loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]
        ), mock.patch.object(
            loader, "_run_post_load_hooks", return_value=None
        ), mock.patch.object(
            loader, "_log_peak_gpu_memory", return_value=None
        ), mock.patch(
            "rtp_llm.models_py.model_loader._get_fastsafetensors_weights",
            return_value=iter([]),
        ):
            self.assertIs(loader._load_via_fastsafetensors(), model)
        self.assertEqual(loader.device, "cpu")
        self.assertEqual(model.to_empty_device, "cpu")

    def test_fastsafetensors_cuda_target_uses_current_device_not_global_rank(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = []
                self.to_empty_device = None

            def to_empty(self, device):
                self.to_empty_device = device
                return self

            def load_weights(self, weights_iter):
                self.loaded = list(weights_iter)

        def fake_get_fastsafetensors_weights(ckpt_files, device):
            self.assertEqual(device, "cuda:0")
            return iter([("dummy.weight", torch.ones(1))])

        model = CaptureModel()
        config = types.SimpleNamespace(model_type="dummy", num_layers=1)
        loader = NewModelLoader(
            config,
            LoadConfig(
                compute_dtype=torch.float32,
                device="cuda",
                load_method="fastsafetensors",
            ),
            model_path="/tmp/nonexistent",
            device="cuda",
        )
        with mock.patch.object(loader, "_create_model", return_value=model), \
             mock.patch.object(loader, "_discover_ckpt_files_cached", return_value=["dummy.safetensors"]), \
             mock.patch.object(loader, "_run_post_load_hooks", return_value=None), \
             mock.patch.object(loader, "_log_peak_gpu_memory", return_value=None), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.distributed.is_available", return_value=True), \
             mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.get_rank", return_value=17), \
             mock.patch(
                 "rtp_llm.models_py.model_loader._get_fastsafetensors_weights",
                 side_effect=fake_get_fastsafetensors_weights,
             ):
            loaded_model = loader._load_via_fastsafetensors()

        self.assertIs(loaded_model, model)
        self.assertEqual(model.to_empty_device, "cuda:0")
        self.assertEqual([name for name, _ in model.loaded], ["dummy.weight"])


class TestCheckpointDiscovery(unittest.TestCase):
    def test_optimizer_only_bin_does_not_hide_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = os.path.join(tmpdir, "optimizer.bin")
            model_pt = os.path.join(tmpdir, "model.pt")
            open(optimizer, "wb").close()
            open(model_pt, "wb").close()
            self.assertEqual(weight_mapper.discover_ckpt_files(tmpdir), [model_pt])
            self.assertEqual(_discover_ckpt_files(tmpdir), [model_pt])


class TestFastSafetensorsFallbackSemantics(unittest.TestCase):
    def _new_loader(self, load_method=LoadMethod.AUTO):
        return NewModelLoader(
            types.SimpleNamespace(model_type="dummy", num_layers=1),
            LoadConfig(
                compute_dtype=torch.float32,
                device="cuda",
                load_method=load_method,
            ),
            model_path="/tmp/nonexistent",
            device="cuda",
        )

    def test_is_quanted_method_false_uses_full_rank_share_for_memory_gate(self):
        class NotQuantizedConfig:
            def is_quanted(self):
                return False

        loader = self._new_loader(LoadMethod.FASTSAFETENSORS)
        loader.load_config.quant_source_config = NotQuantizedConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.safetensors")
            with open(ckpt, "wb") as f:
                f.truncate(1000)
            loader.model_path = tmpdir
            with mock.patch.dict(
                sys.modules, {"fastsafetensors": types.ModuleType("fastsafetensors")}
            ), mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.mem_get_info", return_value=(3600, 10000)
            ):
                ok, reason = loader._fastsafetensors_eligible()
        self.assertFalse(ok)
        self.assertEqual(reason, "insufficient GPU free memory")

    def test_quantized_load_treats_empty_defaults_as_unquantized(self):
        load_config = LoadConfig(compute_dtype=torch.float32, device="cpu")
        for value in (None, "", "none", "None", " null "):
            with self.subTest(value=value):
                self.assertFalse(
                    _is_quantized_load(
                        types.SimpleNamespace(quantization=value), load_config
                    )
                )

    def test_quantized_load_detects_real_quantization_configs(self):
        class FakeQuantConfig:
            def __init__(self, quanted):
                self.quanted = quanted

            def is_quanted(self):
                return self.quanted

        self.assertTrue(
            _is_quantized_load(
                types.SimpleNamespace(quantization="fp8"),
                LoadConfig(compute_dtype=torch.float32, device="cpu"),
            )
        )
        self.assertTrue(
            _is_quantized_load(
                types.SimpleNamespace(quantization=""),
                LoadConfig(
                    compute_dtype=torch.float32,
                    device="cpu",
                    quant_source_config=FakeQuantConfig(True),
                ),
            )
        )
        self.assertTrue(
            _is_quantized_load(
                types.SimpleNamespace(
                    quantization="", quant_config=FakeQuantConfig(True)
                ),
                LoadConfig(compute_dtype=torch.float32, device="cpu"),
            )
        )
        self.assertFalse(
            _is_quantized_load(
                types.SimpleNamespace(
                    quantization="", quant_config=FakeQuantConfig(False)
                ),
                LoadConfig(compute_dtype=torch.float32, device="cpu"),
            )
        )

    def test_explicit_config_fastsafetensors_failure_does_not_fallback(self):
        loader = self._new_loader(LoadMethod.FASTSAFETENSORS)
        err = RuntimeError("fast path failed")
        with mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(True, "")
        ), mock.patch.object(
            loader, "_load_via_fastsafetensors", side_effect=err
        ), mock.patch.object(
            loader, "_load_via_scratch"
        ) as scratch:
            with self.assertRaisesRegex(RuntimeError, "fast path failed"):
                loader.load()
        scratch.assert_not_called()

    def test_env_fastsafetensors_failure_does_not_fallback(self):
        loader = self._new_loader(LoadMethod.AUTO)
        err = RuntimeError("env fast path failed")
        with mock.patch.dict(
            os.environ, {"LOAD_METHOD": LoadMethod.FASTSAFETENSORS}
        ), mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(True, "")
        ), mock.patch.object(
            loader, "_load_via_fastsafetensors", side_effect=err
        ), mock.patch.object(
            loader, "_load_via_scratch"
        ) as scratch:
            with self.assertRaisesRegex(RuntimeError, "env fast path failed"):
                loader.load()
        scratch.assert_not_called()

    def test_auto_fastsafetensors_failure_can_fallback(self):
        loader = self._new_loader(LoadMethod.AUTO)
        scratch_model = torch.nn.Module()
        with mock.patch.object(
            loader,
            "_resolve_load_method",
            return_value=(LoadMethod.FASTSAFETENSORS, False),
        ), mock.patch.object(
            loader,
            "_load_via_fastsafetensors",
            side_effect=RuntimeError("auto fast path failed"),
        ), mock.patch.object(
            loader, "_load_via_scratch", return_value=scratch_model
        ) as scratch:
            self.assertIs(loader.load(), scratch_model)
        scratch.assert_called_once()

    def test_auto_selects_fastsafetensors_when_eligible_without_mocking_resolve(self):
        loader = self._new_loader(LoadMethod.AUTO)
        fast_model = torch.nn.Module()
        with mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(True, "ok")
        ) as eligible, mock.patch.object(
            loader, "_load_via_fastsafetensors", return_value=fast_model
        ) as fast, mock.patch.object(
            loader, "_load_via_scratch"
        ) as scratch:
            self.assertIs(loader.load(), fast_model)
        eligible.assert_called_once()
        fast.assert_called_once()
        scratch.assert_not_called()

    def test_auto_fastsafetensors_failure_falls_back_without_mocking_resolve(self):
        loader = self._new_loader(LoadMethod.AUTO)
        scratch_model = torch.nn.Module()
        with mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(True, "ok")
        ), mock.patch.object(
            loader, "_load_via_fastsafetensors", side_effect=RuntimeError("auto failed")
        ), mock.patch.object(
            loader, "_load_via_scratch", return_value=scratch_model
        ) as scratch:
            self.assertIs(loader.load(), scratch_model)
        scratch.assert_called_once()

    def test_auto_uses_scratch_when_fastsafetensors_is_not_eligible(self):
        loader = self._new_loader(LoadMethod.AUTO)
        scratch_model = torch.nn.Module()
        with mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(False, "not eligible")
        ) as eligible, mock.patch.object(
            loader, "_load_via_fastsafetensors"
        ) as fast, mock.patch.object(
            loader, "_load_via_scratch", return_value=scratch_model
        ) as scratch:
            self.assertIs(loader.load(), scratch_model)
        eligible.assert_called_once()
        fast.assert_not_called()
        scratch.assert_called_once()

    def test_force_cpu_auto_uses_scratch_without_fastsafetensors(self):
        loader = self._new_loader(LoadMethod.AUTO)
        loader.load_config.force_cpu_load_weights = True
        scratch_model = torch.nn.Module()
        with mock.patch.object(
            loader, "_load_via_fastsafetensors"
        ) as fast, mock.patch.object(
            loader, "_load_via_scratch", return_value=scratch_model
        ) as scratch:
            self.assertIs(loader.load(), scratch_model)
        fast.assert_not_called()
        scratch.assert_called_once()

    def test_force_cpu_explicit_fastsafetensors_fails_fast(self):
        loader = self._new_loader(LoadMethod.FASTSAFETENSORS)
        loader.load_config.force_cpu_load_weights = True
        with self.assertRaisesRegex(RuntimeError, "force_cpu_load_weights"):
            loader._resolve_load_method()

    def test_invalid_explicit_load_method_fails_fast(self):
        loader = self._new_loader("typo")
        with self.assertRaisesRegex(ValueError, "Unsupported load_method"):
            loader._resolve_load_method()

    def test_invalid_load_method_env_fails_fast(self):
        loader = self._new_loader(LoadMethod.AUTO)
        with mock.patch.dict(os.environ, {"LOAD_METHOD": "typo"}):
            with self.assertRaisesRegex(ValueError, "Unsupported LOAD_METHOD"):
                loader._resolve_load_method()

    def test_auto_load_method_env_keeps_auto_selection(self):
        loader = self._new_loader(LoadMethod.AUTO)
        with mock.patch.dict(os.environ, {"LOAD_METHOD": LoadMethod.AUTO}), mock.patch.object(
            loader, "_fastsafetensors_eligible", return_value=(True, "ok")
        ):
            self.assertEqual(
                loader._resolve_load_method(), (LoadMethod.FASTSAFETENSORS, False)
            )

    def test_fastsafetensors_memory_gate_uses_target_device(self):
        loader = self._new_loader(LoadMethod.FASTSAFETENSORS)
        loader.device = "cuda:1"
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.safetensors")
            with open(ckpt, "wb") as f:
                f.truncate(1000)
            loader.model_path = tmpdir
            with mock.patch.dict(
                sys.modules, {"fastsafetensors": types.ModuleType("fastsafetensors")}
            ), mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.mem_get_info", return_value=(10000, 20000)
            ) as mem_info:
                ok, reason = loader._fastsafetensors_eligible()
        self.assertTrue(ok, reason)
        mem_info.assert_called_once_with("cuda:1")

    def test_explicit_fastsafetensors_memory_recheck_uses_target_device(self):
        loader = self._new_loader(LoadMethod.FASTSAFETENSORS)
        loader.device = "cuda:1"
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.safetensors")
            with open(ckpt, "wb") as f:
                f.truncate(1000)
            loader.model_path = tmpdir
            with mock.patch.object(
                loader,
                "_fastsafetensors_eligible",
                return_value=(False, "insufficient GPU free memory"),
            ), mock.patch(
                "torch.cuda.mem_get_info", return_value=(2000, 20000)
            ) as mem_info:
                method, explicit = loader._resolve_load_method()
        self.assertEqual((method, explicit), (LoadMethod.FASTSAFETENSORS, True))
        mem_info.assert_called_once_with("cuda:1")


class TestRtpModuleDispatchIntegrity(unittest.TestCase):
    class Container(RtpModule):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(2, 2, bias=False)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    def test_nested_missing_leaf_fails(self):
        module = self.Container()
        with self.assertRaisesRegex(RuntimeError, "child.missing"):
            module.load_weights({"child.missing": torch.zeros(2, 2)})

    def test_module_list_bad_index_fails(self):
        module = self.Container()
        with self.assertRaisesRegex(RuntimeError, "layers.3.weight"):
            module.load_weights({"layers.3.weight": torch.zeros(2, 2)})


class TestLinearLoadCompleteness(unittest.TestCase):
    def test_column_missing_weight_fails_before_post_load(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_qc("none"),
            prefix="linear",
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "missing required checkpoint tensors"):
            layer.process_weights_after_loading()

    def test_fp8_missing_scale_fails(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_qc("fp8"),
            prefix="fp8_linear",
            params_dtype=torch.float32,
        )
        layer.load_weights({"fp8_linear.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype())})
        with self.assertRaisesRegex(RuntimeError, "weight_scale"):
            layer.process_weights_after_loading()

    def test_merged_missing_shard_fails(self):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_size=8,
            quant_config=_qc("none"),
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        layer.load_weights({"gate_up_proj.gate_proj.weight": torch.zeros(4, 4)})
        with self.assertRaisesRegex(RuntimeError, "up_proj|1"):
            layer.process_weights_after_loading()

    def test_merged_shard_rejects_non_divisible_tp_checkpoint_dim(self):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_size=12,
            tp_size=3,
            tp_rank=0,
            quant_config=_qc("none"),
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            layer.load_weights({"gate_up_proj.gate_proj.weight": torch.zeros(5, 4)})

    def test_qkv_missing_v_shard_fails(self):
        layer = QKVParallelLinear(
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            quant_config=_qc("none"),
            prefix="qkv_proj",
            params_dtype=torch.float32,
        )
        layer.load_weights(
            {
                "qkv_proj.q_proj.weight": torch.zeros(4, 4),
                "qkv_proj.k_proj.weight": torch.zeros(2, 4),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "v"):
            layer.process_weights_after_loading()


    def test_qkv_fp8_block_scale_rejects_non_aligned_tp_rows(self):
        qc = _qc("fp8_block")
        qc.weight_block_size = [128, 128]
        layer = QKVParallelLinear(
            hidden_size=384,
            num_heads=6,
            num_kv_heads=6,
            head_dim=64,
            tp_size=2,
            tp_rank=0,
            quant_config=qc,
            prefix="qkv_proj",
            params_dtype=torch.float32,
        )
        scale = torch.ones(3, 3, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "FP8 block QKV scale"):
            layer.load_weights({"qkv_proj.q_proj.weight_scale_inv": scale})

    def test_merged_fp8_block_scale_rejects_non_aligned_gate_up_boundary(self):
        qc = _qc("fp8_block")
        qc.weight_block_size = [128, 128]
        layer = MergedColumnParallelLinear(
            input_size=256,
            output_size=384,
            quant_config=qc,
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        scale = torch.ones(2, 2, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "FP8 block merged scale"):
            layer.load_weights({"gate_up_proj.gate_proj.weight_scale_inv": scale})


    def test_row_parallel_adds_bias_after_all_reduce(self):
        layer = RowParallelLinear(
            input_size=4,
            output_size=3,
            tp_size=2,
            tp_rank=0,
            quant_config=_qc("none"),
            prefix="row",
            bias=True,
            params_dtype=torch.float32,
        )
        layer.weight.data.fill_(1.0)
        layer.bias.data.fill_(3.0)
        x = torch.ones(1, 2, dtype=torch.float32)
        with mock.patch(
            "rtp_llm.models_py.layers.linear.all_reduce",
            side_effect=lambda tensor, group: tensor * 2,
        ):
            out = layer(x)
        torch.testing.assert_close(out, torch.full((1, 3), 7.0), rtol=0, atol=0)

    def test_qkv_rejects_non_divisible_kv_heads(self):
        with self.assertRaisesRegex(ValueError, "num_kv_heads"):
            QKVParallelLinear(
                hidden_size=8,
                num_heads=8,
                num_kv_heads=6,
                head_dim=1,
                tp_size=4,
                tp_rank=0,
                quant_config=_qc("none"),
                prefix="qkv_proj",
                params_dtype=torch.float32,
            )

    def test_column_rejects_non_divisible_output_tp(self):
        with self.assertRaisesRegex(ValueError, "output_size.*divisible"):
            ColumnParallelLinear(
                input_size=4,
                output_size=10,
                tp_size=3,
                quant_config=_qc("none"),
                prefix="linear",
                params_dtype=torch.float32,
            )

    def test_row_rejects_non_divisible_input_tp(self):
        with self.assertRaisesRegex(ValueError, "input_size.*divisible"):
            RowParallelLinear(
                input_size=10,
                output_size=4,
                tp_size=3,
                quant_config=_qc("none"),
                prefix="linear",
                params_dtype=torch.float32,
            )

    def test_linear_rejects_invalid_tp_rank(self):
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            ColumnParallelLinear(
                input_size=4,
                output_size=4,
                tp_size=2,
                tp_rank=2,
                quant_config=_qc("none"),
                prefix="linear",
                params_dtype=torch.float32,
            )

    def test_embedding_rejects_invalid_tp_rank(self):
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            VocabParallelEmbedding(
                vocab_size=4,
                embedding_dim=4,
                tp_size=2,
                tp_rank=2,
                params_dtype=torch.float32,
            )

    def test_merged_rejects_non_divisible_shards_at_construction(self):
        with self.assertRaisesRegex(ValueError, "num_shards"):
            MergedColumnParallelLinear(
                input_size=4,
                output_size=10,
                tp_size=1,
                tp_rank=0,
                quant_config=_qc("none"),
                prefix="gate_up_proj",
                shard_names=["gate_proj", "up_proj", "extra_proj"],
                params_dtype=torch.float32,
            )

    def test_split_weight_rejects_non_divisible_checkpoint_dim(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=9,
            tp_size=3,
            quant_config=_qc("none"),
            prefix="linear",
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "tensor dim 0.*divisible"):
            layer._split_weight(torch.zeros(10, 4), dim=0)

    def test_lora_tp_slice_rejects_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            _tp_slice(torch.zeros(4, 4), tp_size=2, tp_rank=2, rule=(0, "split"))


class TestMoELoadCompleteness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_moe_test_deps()

    def test_fastsafetensors_stacked_moe_names_are_normalized_to_expert_weights(self):
        gate_up = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
        down = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        items = list(
            _normalize_fastsafetensors_stacked_moe_weights(
                iter(
                    [
                        ("model.layers.0.mlp.experts.0.gate_up_proj", gate_up),
                        ("model.layers.0.mlp.experts.0.down_proj", down),
                    ]
                )
            )
        )
        self.assertEqual(
            [name for name, _ in items],
            [
                "model.layers.0.mlp.experts.0.gate_up_proj.weight",
                "model.layers.0.mlp.experts.0.down_proj.weight",
            ],
        )
        self.assertIs(items[0][1], gate_up)
        self.assertIs(items[1][1], down)

    def test_ep_filter_rejects_non_divisible_experts(self):
        with self.assertRaisesRegex(ValueError, "num_experts.*divisible"):
            _create_ep_filter(ep_size=2, ep_rank=0, num_experts=3)

    def test_ep_filter_rejects_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            _create_ep_filter(ep_size=2, ep_rank=2, num_experts=4)

    def test_moe_rejects_non_divisible_ep_experts(self):
        with self.assertRaisesRegex(ValueError, "num_experts.*divisible"):
            BaseMoEExperts(
                num_experts=3,
                hidden_size=4,
                moe_intermediate_size=8,
                tp_size=1,
                tp_rank=0,
                ep_size=2,
                ep_rank=0,
                params_dtype=torch.float32,
                model_config=types.SimpleNamespace(
                    data_type="fp32", quant_config=None, exported_device=None
                ),
                parallelism_config=types.SimpleNamespace(dp_size=1),
                moe_config=types.SimpleNamespace(),
                quant_config=_qc("none"),
                layer_idx=0,
            )

    def test_moe_rejects_invalid_ep_rank(self):
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            BaseMoEExperts(
                num_experts=4,
                hidden_size=4,
                moe_intermediate_size=8,
                tp_size=1,
                tp_rank=0,
                ep_size=2,
                ep_rank=2,
                params_dtype=torch.float32,
                model_config=types.SimpleNamespace(
                    data_type="fp32", quant_config=None, exported_device=None
                ),
                parallelism_config=types.SimpleNamespace(dp_size=1),
                moe_config=types.SimpleNamespace(),
                quant_config=_qc("none"),
                layer_idx=0,
            )

    def test_moe_fp8_block_scale_rejects_non_aligned_tp_intermediate(self):
        qc = _qc("fp8_block")
        qc.weight_block_size = [128, 128]
        with self.assertRaisesRegex(ValueError, "MoE FP8 block scale TP shard"):
            BaseMoEExperts(
                num_experts=1,
                hidden_size=256,
                moe_intermediate_size=384,
                tp_size=2,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.float32,
                model_config=types.SimpleNamespace(
                    data_type="fp32", quant_config=None, exported_device=None
                ),
                parallelism_config=types.SimpleNamespace(dp_size=1),
                moe_config=types.SimpleNamespace(),
                quant_config=qc,
                layer_idx=0,
            )

    def _make_unquant_moe(self, hidden_size=8, moe_intermediate_size=2):
        return BaseMoEExperts(
            num_experts=1,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("none"),
            layer_idx=0,
        )

    def test_moe_gate_up_weight_uses_config_for_h_2m_layout(self):
        layer = self._make_unquant_moe(hidden_size=8, moe_intermediate_size=2)
        gate_up = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        layer.load_weights({"0.gate_up_proj.weight": gate_up})
        torch.testing.assert_close(layer.w13[0, :2], gate_up[:, 2:].t())
        torch.testing.assert_close(layer.w13[0, 2:], gate_up[:, :2].t())

    def test_moe_gate_up_weight_uses_config_for_2m_h_layout(self):
        layer = self._make_unquant_moe(hidden_size=8, moe_intermediate_size=2)
        gate_up = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        layer.load_weights({"0.gate_up_proj.weight": gate_up})
        torch.testing.assert_close(layer.w13[0, :2], gate_up[2:])
        torch.testing.assert_close(layer.w13[0, 2:], gate_up[:2])

    def test_stacked_moe_gate_up_scale_scratch_name_splits_gate_up(self):
        layer = BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("fp8_per_channel"),
            layer_idx=0,
        )
        scale = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        layer.load_weights({"gate_up_proj.weight_scale": scale})
        torch.testing.assert_close(layer._gate_ch_scales[0], scale[0, :4])
        torch.testing.assert_close(layer._up_ch_scales[0], scale[0, 4:])

    def test_stacked_moe_gate_up_scale_fastsafetensors_name_splits_gate_up(self):
        layer = BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("fp8_per_channel"),
            layer_idx=0,
        )
        scale = torch.arange(8, dtype=torch.float32)
        layer.load_weights({"0.gate_up_proj.weight_scale": scale})
        torch.testing.assert_close(layer._gate_ch_scales[0], scale[:4])
        torch.testing.assert_close(layer._up_ch_scales[0], scale[4:])

    def test_moe_accepts_normalized_fastsafetensors_stacked_projection_names(self):
        layer = BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=8,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("none"),
            layer_idx=0,
        )
        gate_up = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
        down = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        for full_name, tensor in _normalize_fastsafetensors_stacked_moe_weights(
            iter(
                [
                    ("model.layers.0.mlp.experts.0.gate_up_proj", gate_up),
                    ("model.layers.0.mlp.experts.0.down_proj", down),
                ]
            )
        ):
            expert_name = full_name.split("experts.", 1)[1]
            layer.load_weights({expert_name: tensor})
        with mock.patch.object(
            BaseMoEExperts, "_maybe_build_fused_moe", return_value=None
        ):
            layer.process_weights_after_loading()
        self.assertEqual(
            layer._loaded_keys, {(0, "gate_proj"), (0, "up_proj"), (0, "down_proj")}
        )

    def test_w4a8_online_rejects_odd_hidden_or_intermediate_size(self):
        with self.assertRaisesRegex(ValueError, "expects even H/M"):
            BaseMoEExperts(
                num_experts=2,
                hidden_size=5,
                moe_intermediate_size=8,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.float32,
                model_config=types.SimpleNamespace(
                    data_type="fp32", quant_config=None, exported_device=None
                ),
                parallelism_config=types.SimpleNamespace(dp_size=1),
                moe_config=types.SimpleNamespace(),
                quant_config=QuantizationConfig(
                    quant_type="W4A8_INT4_PER_CHANNEL",
                    source_config=types.SimpleNamespace(group_size=lambda: 1),
                ),
                layer_idx=0,
            )

    def test_w4a8_online_rejects_non_divisible_group_size(self):
        with self.assertRaisesRegex(ValueError, "group_size=4.*divide"):
            BaseMoEExperts(
                num_experts=2,
                hidden_size=8,
                moe_intermediate_size=10,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.float32,
                model_config=types.SimpleNamespace(
                    data_type="fp32", quant_config=None, exported_device=None
                ),
                parallelism_config=types.SimpleNamespace(dp_size=1),
                moe_config=types.SimpleNamespace(),
                quant_config=QuantizationConfig(
                    quant_type="W4A8_INT4_PER_CHANNEL",
                    source_config=types.SimpleNamespace(group_size=lambda: 4),
                ),
                layer_idx=0,
            )

    def _make_moe(self):
        return BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("fp8"),
            layer_idx=0,
        )

    def test_missing_quant_scale_fails(self):
        layer = self._make_moe()
        layer.load_weights(
            {
                "0.gate_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
                "0.up_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
                "0.down_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "auxiliary tensors"):
            layer.process_weights_after_loading()


class TestAttentionConstructionIntegrity(unittest.TestCase):
    def test_mm_encoder_attention_rejects_hidden_not_divisible_by_heads(self):
        with self.assertRaisesRegex(ValueError, "hidden_size.*divisible"):
            MMEncoderAttention(
                hidden_size=10,
                num_heads=3,
                tp_size=1,
                params_dtype=torch.float32,
            )

    def test_mm_encoder_attention_rejects_heads_not_divisible_by_tp(self):
        with self.assertRaisesRegex(ValueError, "num_heads.*divisible"):
            MMEncoderAttention(
                hidden_size=12,
                num_heads=3,
                tp_size=2,
                params_dtype=torch.float32,
            )

    def test_mm_encoder_attention_rejects_invalid_tp_rank(self):
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            MMEncoderAttention(
                hidden_size=12,
                num_heads=3,
                tp_size=1,
                tp_rank=1,
                params_dtype=torch.float32,
            )

    def test_mm_encoder_attention_qkv_block_scale_splits_qkv_aware(self):
        qc = _qc("fp8_block")
        qc.weight_block_size = [4, 4]
        layer = MMEncoderAttention(
            hidden_size=8,
            num_heads=4,
            tp_size=2,
            tp_rank=1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=qc,
        )
        weight = torch.zeros(24, 8, dtype=torch.float8_e4m3fn)
        scale = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        layer.load_weights({"qkv.weight": weight, "qkv.weight_scale_inv": scale})
        expected = torch.stack([scale[1], scale[3], scale[5]], dim=0)
        torch.testing.assert_close(layer.qkv.weight_scale_inv.detach(), expected)

    def test_mm_encoder_attention_tp2_matches_full_attention(self):
        torch.manual_seed(123)
        hidden = 8
        heads = 4
        x = torch.randn(2, 3, hidden, dtype=torch.float32)
        qkv_w = torch.randn(3 * hidden, hidden, dtype=torch.float32) * 0.1
        proj_w = torch.randn(hidden, hidden, dtype=torch.float32) * 0.1

        full = MMEncoderAttention(
            hidden_size=hidden,
            num_heads=heads,
            tp_size=1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=_qc("none"),
        )
        full.load_weights({"qkv.weight": qkv_w, "proj.weight": proj_w})

        parts = []
        for rank in range(2):
            part = MMEncoderAttention(
                hidden_size=hidden,
                num_heads=heads,
                tp_size=2,
                tp_rank=rank,
                bias=False,
                params_dtype=torch.float32,
                quant_config=_qc("none"),
            )
            part.proj.reduce_output = False
            part.load_weights({"qkv.weight": qkv_w, "proj.weight": proj_w})
            parts.append(part(x))

        torch.testing.assert_close(
            parts[0] + parts[1], full(x), rtol=1e-5, atol=1e-5
        )


class TestOtherLayerLoadCompleteness(unittest.TestCase):
    def test_embedding_missing_weight_fails(self):
        layer = VocabParallelEmbedding(8, 4, params_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "weight"):
            layer.process_weights_after_loading()

    def test_embedding_tp_dims_fail_fast_when_non_divisible(self):
        with self.assertRaisesRegex(ValueError, "vocab_size.*divisible"):
            VocabParallelEmbedding(10, 4, tp_size=3, params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "embedding_dim.*divisible"):
            HiddenParallelEmbedding(8, 10, tp_size=3, params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "vocab_size.*divisible"):
            ParallelLMHead(10, 4, tp_size=3, params_dtype=torch.float32)

    def test_norm_missing_weight_fails(self):
        layer = RMSNorm(4, params_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "weight"):
            layer.process_weights_after_loading()

    def test_layernorm_missing_bias_fails(self):
        layer = LayerNorm(4, params_dtype=torch.float32)
        layer.load_weights({"weight": torch.ones(4)})
        with self.assertRaisesRegex(RuntimeError, "bias"):
            layer.process_weights_after_loading()

    def test_conv_missing_bias_fails(self):
        layer = Conv3dLayer(1, 1, (1, 1, 1), params_dtype=torch.float32)
        layer.load_weights({"weight": torch.ones_like(layer.conv.weight)})
        with self.assertRaisesRegex(RuntimeError, "bias"):
            layer.process_weights_after_loading()



class TestBertNewLoaderCompatibility(unittest.TestCase):
    def _bert_config(self):
        return types.SimpleNamespace(
            model_type="bert",
            num_layers=1,
            hidden_size=4,
            inter_size=8,
            type_vocab_size=2,
            max_generate_batch_size=1,
            quant_config=None,
        )

    def _gamma_beta_weights(self, prefix="bert"):
        h = 4
        inter = 8
        return {
            f"{prefix}.embeddings.word_embeddings.weight": torch.ones(8, h),
            f"{prefix}.embeddings.position_embeddings.weight": torch.ones(16, h),
            f"{prefix}.embeddings.token_type_embeddings.weight": torch.ones(2, h),
            f"{prefix}.embeddings.LayerNorm.gamma": torch.full((h,), 2.0),
            f"{prefix}.embeddings.LayerNorm.beta": torch.full((h,), 3.0),
            f"{prefix}.encoder.layer.0.attention.self.query.weight": torch.ones(h, h),
            f"{prefix}.encoder.layer.0.attention.self.query.bias": torch.zeros(h),
            f"{prefix}.encoder.layer.0.attention.self.key.weight": torch.ones(h, h),
            f"{prefix}.encoder.layer.0.attention.self.key.bias": torch.zeros(h),
            f"{prefix}.encoder.layer.0.attention.self.value.weight": torch.ones(h, h),
            f"{prefix}.encoder.layer.0.attention.self.value.bias": torch.zeros(h),
            f"{prefix}.encoder.layer.0.attention.output.dense.weight": torch.ones(h, h),
            f"{prefix}.encoder.layer.0.attention.output.dense.bias": torch.zeros(h),
            f"{prefix}.encoder.layer.0.attention.output.LayerNorm.gamma": torch.full((h,), 4.0),
            f"{prefix}.encoder.layer.0.attention.output.LayerNorm.beta": torch.full((h,), 5.0),
            f"{prefix}.encoder.layer.0.intermediate.dense.weight": torch.ones(inter, h),
            f"{prefix}.encoder.layer.0.intermediate.dense.bias": torch.zeros(inter),
            f"{prefix}.encoder.layer.0.output.dense.weight": torch.ones(h, inter),
            f"{prefix}.encoder.layer.0.output.dense.bias": torch.zeros(h),
            f"{prefix}.encoder.layer.0.output.LayerNorm.gamma": torch.full((h,), 6.0),
            f"{prefix}.encoder.layer.0.output.LayerNorm.beta": torch.full((h,), 7.0),
        }

    def test_bert_accepts_layernorm_gamma_beta_checkpoint_names(self):
        model = BertForEmbedding(self._bert_config(), LoadConfig(compute_dtype=torch.float32, device="cpu"))
        with mock.patch.object(BertForEmbedding, "_build_inner_model", return_value=None):
            model.load_weights(self._gamma_beta_weights("bert"))
        self.assertTrue(torch.equal(model.embeddings.layernorm_weight, torch.full((4,), 2.0)))
        self.assertTrue(torch.equal(model.layers[0].attention_output_layernorm_weight, torch.full((4,), 4.0)))
        self.assertTrue(torch.equal(model.layers[0].output_layernorm_bias, torch.full((4,), 7.0)))

    def test_roberta_accepts_layernorm_gamma_beta_checkpoint_names(self):
        model = RobertaForEmbedding(self._bert_config(), LoadConfig(compute_dtype=torch.float32, device="cpu"))
        with mock.patch.object(RobertaForEmbedding, "_build_inner_model", return_value=None):
            model.load_weights(self._gamma_beta_weights("roberta"))
        self.assertTrue(torch.equal(model.embeddings.layernorm_bias, torch.full((4,), 3.0)))


class TestAwqRegistrationSafety(unittest.TestCase):
    def test_awq_reference_path_is_not_registered_for_hot_forward(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=8,
            quant_config=_qc("none"),
            prefix="linear",
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported linear quant_type 'awq'"):
            _qc("awq").get_quant_method(layer, "linear")

class TestFp8ForwardInputLayout(unittest.TestCase):
    def test_fp8_per_tensor_accepts_non_contiguous_input(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=3,
            quant_config=_qc("fp8"),
            prefix="fp8_linear",
            params_dtype=torch.float32,
        )
        layer.load_weights(
            {
                "fp8_linear.weight": torch.zeros(3, 4, dtype=_runtime_fp8_dtype()),
                "fp8_linear.weight_scale": torch.tensor([1.0], dtype=torch.float32),
            }
        )
        layer.process_weights_after_loading()
        x = torch.randn(2, 4, 3).transpose(1, 2)
        self.assertFalse(x.is_contiguous())

        def fake_quant(inp):
            self.assertEqual(inp.shape, (6, 4))
            self.assertTrue(inp.is_contiguous())
            return inp.to(_runtime_fp8_dtype()), torch.ones(1, dtype=torch.float32)

        def fake_scaled_mm(a, b, **kwargs):
            return torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)

        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._resolve_per_tensor_quant",
            return_value=fake_quant,
        ), mock.patch.object(torch, "_scaled_mm", side_effect=fake_scaled_mm):
            out = layer(x)
        self.assertEqual(out.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()

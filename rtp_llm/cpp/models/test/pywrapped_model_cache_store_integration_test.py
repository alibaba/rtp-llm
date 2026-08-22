import importlib
import inspect
import os
import re
import sys
import types
import unittest
from typing import Any, ClassVar, cast

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    clear_comm_ops,
    register_comm_ops,
    run_scenario,
)

_KMONITOR_GAUGE = 0
_KMONITOR_QPS = 3
_KMONITOR_STATUS = 5
_CAPTURE_METRIC_PREFIX = "rtp_llm_hidden_state_capture_"
_CAPTURE_QPS_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}batch_qps",
    f"{_CAPTURE_METRIC_PREFIX}publish_success_qps",
    f"{_CAPTURE_METRIC_PREFIX}failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}initialization_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}layout_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}prepare_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}quantize_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}store_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}shutdown_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}hard_contract_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}request_error_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}operational_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}duplicate_request_id_qps",
    f"{_CAPTURE_METRIC_PREFIX}fail_open_disable_qps",
    f"{_CAPTURE_METRIC_PREFIX}disabled_skip_qps",
    f"{_CAPTURE_METRIC_PREFIX}broken_rejection_qps",
    f"{_CAPTURE_METRIC_PREFIX}bf16_publish_qps",
    f"{_CAPTURE_METRIC_PREFIX}fp8_publish_qps",
}
_CAPTURE_STATUS_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}enabled",
    f"{_CAPTURE_METRIC_PREFIX}broken",
    f"{_CAPTURE_METRIC_PREFIX}fail_open_enabled",
}
_CAPTURE_PUBLISH_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}publish_latency_us",
    f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us",
    f"{_CAPTURE_METRIC_PREFIX}publish_request_count",
    f"{_CAPTURE_METRIC_PREFIX}publish_token_count",
    f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes",
}
_MOONCAKE_STORE_KEY_NAMESPACE_ENV = "MOONCAKE_STORE_KEY_NAMESPACE"
_MOONCAKE_RTP_LAYOUT_ENV = {
    "MOONCAKE_HIDDEN_DIM": "hidden_dim",
    "MOONCAKE_MAX_SEQ_LEN": "max_seq_len",
    "MOONCAKE_GET_BATCH_SIZE": "get_batch_size",
    "MOONCAKE_GPU_BUFFER_SIZE": "gpu_buffer_size",
    "MOONCAKE_NUM_AUX_LAYERS": "num_aux_layers",
}


def _snapshot_mooncake_layout(config) -> dict[str, Any]:
    return {
        attr_name: getattr(config, attr_name)
        for attr_name in _MOONCAKE_RTP_LAYOUT_ENV.values()
    }


def _restore_mooncake_layout_env(previous_env: dict[str, str | None]) -> None:
    for env_name, value in previous_env.items():
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = value


class _FakeMooncakeConfig:
    hidden_dim = 1
    max_seq_len = 8192
    get_batch_size = 1
    gpu_buffer_size = None
    num_aux_layers = 2
    store_key_namespace = "TestActor"
    make_store_key_calls: ClassVar[list[int]] = []

    @classmethod
    def from_env(cls):
        config = cls()
        namespace = os.environ.get(_MOONCAKE_STORE_KEY_NAMESPACE_ENV)
        if not namespace or re.fullmatch(r"[A-Za-z0-9]+", namespace) is None:
            raise ValueError(
                "Mooncake store_key_namespace must be non-empty and contain only "
                "ASCII letters and digits"
            )
        config.store_key_namespace = namespace
        for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items():
            value = os.environ.get(env_name)
            if value is not None:
                setattr(config, attr_name, int(value))
        return config

    def make_store_key(self, request_id: int) -> str:
        type(self).make_store_key_calls.append(request_id)
        return f"rtp_{self.store_key_namespace}_{request_id}"


class _FakeEagleMooncakeStore:
    instances: ClassVar[list["_FakeEagleMooncakeStore"]] = []
    events = None
    fail_put = False
    fail_put_count = 0
    fail_flush = False
    fail_warmup = False
    async_error_after_accept_count = 0

    def __init__(self, config) -> None:
        self.config = config
        self.config_at_init = _snapshot_mooncake_layout(config)
        self.config_at_setup: dict[str, Any] = {}
        self.puts = []
        self.put_batch_calls = []
        self.put_attempts = 0
        self.keys: set[str] = set()
        self.pending_async_errors: list[BaseException] = []
        self.setup_device = None
        self.warmed_up = False
        self.flushed = False
        self.closed = False
        type(self).instances.append(self)

    @classmethod
    def _record(cls, event: str) -> None:
        if cls.events is not None:
            cls.events.append(event)

    def setup(self, device=None) -> None:
        type(self)._record("store.setup")
        self.config_at_setup = _snapshot_mooncake_layout(self.config)
        self.setup_device = device

    def warmup_rdma(self) -> None:
        type(self)._record("store.warmup")
        self.warmed_up = True
        if type(self).fail_warmup:
            raise RuntimeError("injected Mooncake warmup failure")

    def take_async_errors(self) -> list[BaseException]:
        type(self)._record("store.take_async_errors")
        errors = self.pending_async_errors
        self.pending_async_errors = []
        return errors

    def _record_put(
        self,
        key,
        hidden_states,
        input_ids,
        last_hidden_states,
        hidden_states_scale=None,
    ) -> dict[str, Any]:
        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
        assert input_ids.is_contiguous(), "input_ids must be contiguous"
        assert input_ids.dtype == torch.int64, "input_ids must be int64"
        assert (
            input_ids.device == last_hidden_states.device
        ), "input_ids and last_hidden_states must be on the same device"
        assert (
            last_hidden_states.is_contiguous()
        ), "last_hidden_states must be contiguous"
        assert (
            last_hidden_states.dtype == torch.bfloat16
        ), "last_hidden_states must be bfloat16 at the store boundary"
        if hidden_states_scale is not None:
            assert (
                hidden_states_scale.is_contiguous()
            ), "hidden_states_scale must be contiguous"
            assert (
                hidden_states.dtype == torch.float8_e4m3fn
            ), "FP8 hidden_states must be float8_e4m3fn at the store boundary"
        else:
            assert (
                hidden_states.dtype == torch.bfloat16
            ), "BF16 hidden_states must be bfloat16 at the store boundary"

        dequantized_hidden_states = None
        if hidden_states_scale is not None:
            num_layers = hidden_states_scale.shape[-1]
            assert hidden_states.shape[-1] % num_layers == 0
            layer_width = hidden_states.shape[-1] // num_layers
            dequantized_hidden_states = (
                (
                    hidden_states.detach()
                    .to(torch.float32)
                    .reshape(*hidden_states.shape[:-1], num_layers, layer_width)
                    * hidden_states_scale.detach().to(torch.float32).unsqueeze(-1)
                )
                .reshape(hidden_states.shape)
                .cpu()
            )

        self.puts.append(
            {
                "key": key,
                "hidden_states": hidden_states.detach().cpu(),
                "dequantized_hidden_states": dequantized_hidden_states,
                "input_ids": input_ids.detach().cpu(),
                "last_hidden_states": last_hidden_states.detach().cpu(),
                "hidden_states_scale": (
                    hidden_states_scale.detach().cpu()
                    if hidden_states_scale is not None
                    else None
                ),
                "source_hidden_dtype": hidden_states.dtype,
                "source_last_hidden_dtype": last_hidden_states.dtype,
            }
        )
        self.keys.add(key)
        return {
            "shapes": {
                "hidden_states": tuple(hidden_states.shape),
                "input_ids": tuple(input_ids.shape),
                "last_hidden_states": tuple(last_hidden_states.shape),
            },
            "dtypes": {
                "hidden_states": hidden_states.dtype,
                "input_ids": input_ids.dtype,
                "last_hidden_states": last_hidden_states.dtype,
            },
        }

    def put_batch(
        self,
        batch_id,
        request_ids,
        hidden_states,
        input_ids,
        last_hidden_states=None,
        target=None,
        hidden_states_scale=None,
    ) -> list[dict[str, Any]]:
        del target
        request_ids = list(request_ids)
        hidden_states = list(hidden_states)
        input_ids = list(input_ids)
        last_hidden_states = (
            list(last_hidden_states)
            if last_hidden_states is not None
            else [None] * len(request_ids)
        )
        hidden_states_scale = (
            list(hidden_states_scale)
            if hidden_states_scale is not None
            else [None] * len(request_ids)
        )
        values = (hidden_states, input_ids, last_hidden_states, hidden_states_scale)
        if not request_ids or any(len(items) != len(request_ids) for items in values):
            raise ValueError("invalid fake Mooncake put_batch lengths")

        type(self)._record(f"store.put_batch:{batch_id}")
        self.put_attempts += 1
        self.put_batch_calls.append(
            {
                "batch_id": batch_id,
                "request_ids": request_ids,
                "lengths": [tensor.shape[0] for tensor in hidden_states],
            }
        )
        duplicate_key = next((key for key in request_ids if key in self.keys), None)
        if duplicate_key is not None:
            raise FileExistsError(f"Mooncake key already exists: {duplicate_key}")
        if type(self).fail_put_count > 0:
            type(self).fail_put_count -= 1
            raise RuntimeError("injected Mooncake put_batch admission failure")
        if type(self).fail_put:
            raise RuntimeError("injected Mooncake put_batch admission failure")

        metadata = [
            self._record_put(key, aux, ids, last, scale)
            for key, aux, ids, last, scale in zip(
                request_ids,
                hidden_states,
                input_ids,
                last_hidden_states,
                hidden_states_scale,
                strict=True,
            )
        ]
        if type(self).async_error_after_accept_count > 0:
            type(self).async_error_after_accept_count -= 1
            self.pending_async_errors.append(
                RuntimeError(
                    "injected historical Mooncake async failure "
                    f"(batch_id={batch_id!r}, request_ids={request_ids!r}, "
                    "stage=data, attempts=1)"
                )
            )
        return metadata

    def flush(self) -> None:
        type(self)._record("store.flush")
        self.flushed = True
        if type(self).fail_flush:
            raise RuntimeError("injected Mooncake flush failure")

    def close(self) -> None:
        type(self)._record("store.close")
        self.closed = True


def _fake_quantize_aux_hidden_states(hidden_states, num_layers):
    layer_width = hidden_states.shape[-1] // num_layers
    scales = torch.ones(
        (*hidden_states.shape[:-1], num_layers),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    quantized = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    for layer_idx in range(num_layers):
        start = layer_idx * layer_width
        quantized[..., start : start + layer_width].copy_(
            hidden_states[..., start : start + layer_width]
        )
    return quantized, scales


_FAKE_TORCHSPEC_MODULE_NAMES = (
    "torchspec",
    "torchspec.transfer",
    "torchspec.transfer.mooncake",
    "torchspec.utils",
    "torchspec.utils.fp8",
)


def _install_fake_torchspec_modules() -> dict[str, types.ModuleType]:
    previous_modules = {
        name: sys.modules[name]
        for name in _FAKE_TORCHSPEC_MODULE_NAMES
        if name in sys.modules
    }
    torchspec = types.ModuleType("torchspec")
    torchspec.__path__ = []
    transfer = types.ModuleType("torchspec.transfer")
    transfer.__path__ = []
    mooncake = types.ModuleType("torchspec.transfer.mooncake")
    mooncake_attrs = cast(Any, mooncake)
    mooncake_attrs.MooncakeConfig = _FakeMooncakeConfig
    mooncake_attrs.EagleMooncakeStore = _FakeEagleMooncakeStore
    utils = types.ModuleType("torchspec.utils")
    utils.__path__ = []
    fp8 = types.ModuleType("torchspec.utils.fp8")
    cast(Any, fp8).quantize_aux_hidden_states = _fake_quantize_aux_hidden_states
    sys.modules["torchspec"] = torchspec
    sys.modules["torchspec.transfer"] = transfer
    sys.modules["torchspec.transfer.mooncake"] = mooncake
    sys.modules["torchspec.utils"] = utils
    sys.modules["torchspec.utils.fp8"] = fp8
    return previous_modules


def _restore_torchspec_modules(
    previous_modules: dict[str, types.ModuleType],
) -> None:
    for name in _FAKE_TORCHSPEC_MODULE_NAMES:
        if name in previous_modules:
            sys.modules[name] = previous_modules[name]
        else:
            sys.modules.pop(name, None)


def _register_no_collectives(allreduce=None) -> None:
    def unexpected_collective(*args):
        raise AssertionError("hidden-state capture issued an unexpected collective")

    register_comm_ops(
        unexpected_collective,
        unexpected_collective if allreduce is None else allreduce,
        unexpected_collective,
    )


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    supports_hidden_state_capture = True

    def __init__(self, hidden_size: int = 1) -> None:
        self.hidden_size = hidden_size
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []
        self.capture_flags: list[bool] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        self.capture_flags.append(inputs.capture_hidden_states)
        attention_inputs = inputs.attention_inputs
        first_inputs = (
            next(iter(attention_inputs.values()))
            if isinstance(attention_inputs, dict)
            else attention_inputs
        )
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )

        hidden_states = torch.zeros(
            (inputs.input_ids.numel(), self.hidden_size),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        if inputs.capture_hidden_states:
            hidden_states = torch.cat(
                (hidden_states + 10, hidden_states + 20, hidden_states + 30),
                dim=-1,
            )
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class PositionAwareCaptureModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            positions = inputs.input_ids.to(torch.float16).reshape(-1, 1)
            return PyModelOutputs(
                torch.cat((positions + 10, positions + 20, positions + 30), dim=-1)
            )
        return output


class FinalNormParityModel(CacheStoreForwardModel):
    _FINAL_NORM_GAMMA = (0.5, 1.5)
    _FINAL_NORM_EPS = 1e-5

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        super()._forward_one(inputs)
        token_values = inputs.input_ids.to(torch.float16).reshape(-1, 1)
        hidden_states = torch.cat((token_values, token_values + 1), dim=-1)
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        final_hidden_states = (
            hidden_states * torch.rsqrt(variance + self._FINAL_NORM_EPS)
        ) * hidden_states.new_tensor(self._FINAL_NORM_GAMMA)
        final_hidden_states = final_hidden_states.to(hidden_states.dtype)
        if inputs.capture_hidden_states:
            return PyModelOutputs(
                torch.cat(
                    (
                        hidden_states + 10,
                        hidden_states + 20,
                        final_hidden_states,
                    ),
                    dim=-1,
                )
            )
        return PyModelOutputs(final_hidden_states)


class MalformedCaptureLayoutModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            return PyModelOutputs(output.hidden_states[:, :2])
        return output


class MalformedCaptureRankModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            return PyModelOutputs(output.hidden_states.flatten())
        return output


class MalformedCaptureRowCountModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            return PyModelOutputs(output.hidden_states[:-1])
        return output


class MalformedCaptureDtypeModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            return PyModelOutputs(output.hidden_states.float())
        return output


class MalformedCaptureDeviceModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            return PyModelOutputs(output.hidden_states.cpu())
        return output


class UnsupportedCaptureModel:
    supports_hidden_state_capture = False

    def initialize(self, resources) -> bool:
        return True

    def forward(self, inputs, fmha_impl=None) -> PyModelOutputs:
        raise AssertionError("unsupported capture model must fail during construction")


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        previous_torchspec_modules = _install_fake_torchspec_modules()
        self.addCleanup(_restore_torchspec_modules, previous_torchspec_modules)
        _FakeEagleMooncakeStore.instances.clear()
        _FakeEagleMooncakeStore.events = None
        _FakeMooncakeConfig.store_key_namespace = "TestActor"
        _FakeMooncakeConfig.make_store_key_calls.clear()
        _FakeMooncakeConfig.hidden_dim = 1
        _FakeMooncakeConfig.max_seq_len = 8192
        _FakeMooncakeConfig.get_batch_size = 1
        _FakeMooncakeConfig.gpu_buffer_size = None
        _FakeMooncakeConfig.num_aux_layers = 2
        mooncake_env_names = (
            *_MOONCAKE_RTP_LAYOUT_ENV,
            _MOONCAKE_STORE_KEY_NAMESPACE_ENV,
        )
        previous_layout_env = {
            env_name: os.environ.get(env_name) for env_name in mooncake_env_names
        }
        for env_name in mooncake_env_names:
            os.environ.pop(env_name, None)
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "TestActor"
        self.addCleanup(_restore_mooncake_layout_env, previous_layout_env)
        _FakeEagleMooncakeStore.fail_put = False
        _FakeEagleMooncakeStore.fail_put_count = 0
        _FakeEagleMooncakeStore.fail_flush = False
        _FakeEagleMooncakeStore.fail_warmup = False
        _FakeEagleMooncakeStore.async_error_after_accept_count = 0

    def assertCaptureQps(
        self,
        metrics: dict[str, float],
        *,
        batch: int,
        publish_success: int,
        failure: int,
        initialization_failure: int = 0,
        layout_failure: int = 0,
        prepare_failure: int = 0,
        quantize_failure: int = 0,
        store_failure: int = 0,
        shutdown_failure: int = 0,
        hard_contract_failure: int = 0,
        request_error_failure: int = 0,
        operational_failure: int = 0,
        duplicate_request_id: int = 0,
        fail_open_disable: int = 0,
        disabled_skip: int = 0,
        broken_rejection: int = 0,
        bf16_publish: int = 0,
        fp8_publish: int = 0,
    ) -> None:
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_QPS_METRICS},
            {
                f"{_CAPTURE_METRIC_PREFIX}batch_qps": batch,
                f"{_CAPTURE_METRIC_PREFIX}publish_success_qps": publish_success,
                f"{_CAPTURE_METRIC_PREFIX}failure_qps": failure,
                f"{_CAPTURE_METRIC_PREFIX}initialization_failure_qps": initialization_failure,
                f"{_CAPTURE_METRIC_PREFIX}layout_failure_qps": layout_failure,
                f"{_CAPTURE_METRIC_PREFIX}prepare_failure_qps": prepare_failure,
                f"{_CAPTURE_METRIC_PREFIX}quantize_failure_qps": quantize_failure,
                f"{_CAPTURE_METRIC_PREFIX}store_failure_qps": store_failure,
                f"{_CAPTURE_METRIC_PREFIX}shutdown_failure_qps": shutdown_failure,
                f"{_CAPTURE_METRIC_PREFIX}hard_contract_failure_qps": hard_contract_failure,
                f"{_CAPTURE_METRIC_PREFIX}request_error_failure_qps": request_error_failure,
                f"{_CAPTURE_METRIC_PREFIX}operational_failure_qps": operational_failure,
                f"{_CAPTURE_METRIC_PREFIX}duplicate_request_id_qps": duplicate_request_id,
                f"{_CAPTURE_METRIC_PREFIX}fail_open_disable_qps": fail_open_disable,
                f"{_CAPTURE_METRIC_PREFIX}disabled_skip_qps": disabled_skip,
                f"{_CAPTURE_METRIC_PREFIX}broken_rejection_qps": broken_rejection,
                f"{_CAPTURE_METRIC_PREFIX}bf16_publish_qps": bf16_publish,
                f"{_CAPTURE_METRIC_PREFIX}fp8_publish_qps": fp8_publish,
            },
        )

    def assertCaptureStatus(
        self,
        metrics: dict[str, float],
        *,
        enabled: int,
        broken: int,
        fail_open: int,
    ) -> None:
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_STATUS_METRICS},
            {
                f"{_CAPTURE_METRIC_PREFIX}enabled": enabled,
                f"{_CAPTURE_METRIC_PREFIX}broken": broken,
                f"{_CAPTURE_METRIC_PREFIX}fail_open_enabled": fail_open,
            },
        )

    def assertCaptureMetricTypes(self, result: dict) -> None:
        metrics = result["capture_metrics"]
        metric_types = result["capture_metric_types"]
        self.assertEqual(set(metric_types), set(metrics))
        for name in metrics:
            if name in _CAPTURE_QPS_METRICS:
                self.assertEqual(metric_types[name], _KMONITOR_QPS)
            elif name in _CAPTURE_STATUS_METRICS:
                self.assertEqual(metric_types[name], _KMONITOR_STATUS)
            else:
                self.assertEqual(metric_types[name], _KMONITOR_GAUGE)

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertFalse(result["logits_defined"])
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertFalse(result["logits_defined"])
        record = _record_for_request(result, 301)
        self.assertEqual(len(record["blocks"]), 3)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                token_key
                for token_key in (3102, 3104, 3106)
                if any(
                    f"_token_id_str_{token_key}_" in block["key"]
                    for block in record["blocks"]
                )
            ),
            [3102, 3104, 3106],
        )

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))

    def test_prefill_only_pd_does_not_write_decode_handoff(self) -> None:
        result = run_scenario(CacheStoreForwardModel(), "prefill_only_pd")

        self.assertFalse(result["logits_defined"])
        self.assertEqual(result["records"], [])

    def test_skip_lm_head_without_explicit_capture_does_not_publish(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_explicit_off")

        self.assertEqual(model.capture_flags, [False])
        self.assertFalse(result["logits_defined"])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        self.assertCaptureQps(
            result["capture_metrics"], batch=0, publish_success=0, failure=0
        )

    def test_explicit_capture_with_lm_head_publishes_and_returns_logits(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_with_lm_head")

        self.assertEqual(model.capture_flags, [True])
        self.assertTrue(result["logits_defined"])
        self.assertEqual(len(_FakeEagleMooncakeStore.instances[0].puts), 2)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )

    def test_capture_bf16_store_contract_slices_each_request_and_flushes_on_shutdown(
        self,
    ) -> None:
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")

        self.assertEqual(result["hidden_width"], 1)
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureMetricTypes(result)
        self.assertEqual(
            set(metrics),
            _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_PUBLISH_METRICS,
        )
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=0)
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_latency_us"], 0
        )
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us"], 0
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_request_count"], 2)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_token_count"], 5)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes"], 70)
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes"], 40
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes"], 20
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes"], 10
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes"], 0)
        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        expected_default_layout = {
            "hidden_dim": 1,
            "max_seq_len": 8192,
            "get_batch_size": 1,
            "gpu_buffer_size": None,
            "num_aux_layers": 2,
        }
        self.assertEqual(store.config_at_init, expected_default_layout)
        self.assertEqual(store.config_at_setup, expected_default_layout)
        self.assertTrue(store.warmed_up)
        self.assertTrue(store.flushed)
        self.assertTrue(store.closed)
        self.assertEqual(len(store.put_batch_calls), 1)
        self.assertEqual(
            store.put_batch_calls[0]["request_ids"],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(_FakeMooncakeConfig.make_store_key_calls, [0, 501, 502])
        self.assertEqual(store.put_batch_calls[0]["lengths"], [2, 3])
        self.assertIn("requests-501-502", store.put_batch_calls[0]["batch_id"])
        self.assertEqual(
            [tuple(put["hidden_states"].shape) for put in store.puts],
            [(2, 2), (3, 2)],
        )
        self.assertEqual(
            [tuple(put["last_hidden_states"].shape) for put in store.puts],
            [(2, 1), (3, 1)],
        )
        self.assertEqual(store.puts[0]["input_ids"].tolist(), [1, 2])
        self.assertEqual(store.puts[1]["input_ids"].tolist(), [3, 4, 5])
        self.assertTrue(
            all(put["input_ids"].dtype == torch.int64 for put in store.puts)
        )
        self.assertTrue(
            all(put["hidden_states"].dtype == torch.bfloat16 for put in store.puts)
        )
        self.assertTrue(
            all(put["source_hidden_dtype"] == torch.bfloat16 for put in store.puts)
        )
        self.assertTrue(
            all(put["source_last_hidden_dtype"] == torch.bfloat16 for put in store.puts)
        )
        self.assertEqual(result["forward_hidden_states"][0].dtype, torch.float16)
        self.assertTrue(all(put["hidden_states_scale"] is None for put in store.puts))
        expected_auxiliary = [
            [[10.0, 20.0], [10.0, 20.0]],
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
        ]
        expected_last = [
            [[30.0], [30.0]],
            [[30.0], [30.0], [30.0]],
        ]
        self.assertEqual(
            [put["hidden_states"].tolist() for put in store.puts],
            expected_auxiliary,
        )
        self.assertEqual(
            [put["last_hidden_states"].tolist() for put in store.puts],
            expected_last,
        )

    def test_capture_publisher_consumes_non_4096_hidden_env(self) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2048"

        result = run_scenario(
            CacheStoreForwardModel(hidden_size=2048), "capture_non_4096_hidden"
        )

        self.assertEqual(result["hidden_width"], 2048)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.config_at_init["hidden_dim"], 2048)
        self.assertEqual(store.config_at_setup["hidden_dim"], 2048)
        self.assertEqual(
            [tuple(put["hidden_states"].shape) for put in store.puts],
            [(2, 4096), (3, 4096)],
        )
        self.assertEqual(
            [tuple(put["last_hidden_states"].shape) for put in store.puts],
            [(2, 2048), (3, 2048)],
        )

    def test_capture_publisher_consumes_gpu_direct_layout_env_before_store_setup(
        self,
    ) -> None:
        exported_layout = {
            "MOONCAKE_HIDDEN_DIM": "1",
            "MOONCAKE_MAX_SEQ_LEN": "16384",
            "MOONCAKE_GET_BATCH_SIZE": "8",
            "MOONCAKE_GPU_BUFFER_SIZE": "268435456",
            "MOONCAKE_NUM_AUX_LAYERS": "2",
        }
        os.environ.update(exported_layout)

        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        expected_layout = {
            attr_name: int(exported_layout[env_name])
            for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items()
        }
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.config_at_init, expected_layout)
        self.assertEqual(store.config_at_setup, expected_layout)

    def test_capture_missing_namespace_fails_before_store_construction(self) -> None:
        os.environ.pop(_MOONCAKE_STORE_KEY_NAMESPACE_ENV)

        with self.assertRaisesRegex(RuntimeError, "store_key_namespace"):
            run_scenario(CacheStoreForwardModel(), "capture_bf16")

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_namespace_matches_rtp_engine_key_contract(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorA1"

        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_ActorA1_501", "rtp_ActorA1_502"],
        )
        self.assertEqual(
            store.put_batch_calls[0]["request_ids"],
            ["rtp_ActorA1_501", "rtp_ActorA1_502"],
        )
        self.assertEqual(
            store.put_batch_calls[0]["batch_id"],
            "rtp-forward-ActorA1-1-requests-501-502",
        )

    def test_capture_same_request_id_differs_across_namespaces(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorA1"
        run_scenario(CacheStoreForwardModel(), "capture_bf16")
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorB2"
        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        first_key = _FakeEagleMooncakeStore.instances[0].puts[0]["key"]
        second_key = _FakeEagleMooncakeStore.instances[1].puts[0]["key"]
        self.assertEqual(first_key, "rtp_ActorA1_501")
        self.assertEqual(second_key, "rtp_ActorB2_501")
        self.assertNotEqual(first_key, second_key)

    def test_capture_invalid_namespace_fails_before_store_construction(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "invalid_namespace"

        with self.assertRaisesRegex(RuntimeError, "store_key_namespace"):
            run_scenario(CacheStoreForwardModel(), "capture_bf16")

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_cp_validates_before_existing_restore_and_store(self) -> None:
        events = []
        _FakeEagleMooncakeStore.events = events

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            events.append(f"allreduce:{int(tensor.item())}")
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)

        result = run_scenario(PositionAwareCaptureModel(), "capture_cp")

        self.assertEqual(result["cp_handle_outputs_calls"], 1)
        self.assertEqual(
            events,
            [
                "store.setup",
                "store.warmup",
                "allreduce:1",
                "store.take_async_errors",
                "store.put_batch:rtp-forward-TestActor-1-requests-601-601",
                "store.flush",
                "store.take_async_errors",
                "store.close",
            ],
        )

    def test_shutdown_failure_is_counted_and_close_still_runs(self) -> None:
        events = []
        _FakeEagleMooncakeStore.events = events
        _FakeEagleMooncakeStore.fail_flush = True

        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertTrue(store.flushed)
        self.assertTrue(store.closed)
        self.assertEqual(
            events,
            [
                "store.setup",
                "store.warmup",
                "store.take_async_errors",
                "store.put_batch:rtp-forward-TestActor-1-requests-501-502",
                "store.flush",
                "store.take_async_errors",
                "store.close",
            ],
        )
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=1,
            shutdown_failure=1,
            operational_failure=1,
            bf16_publish=1,
        )

    def test_capture_fp8_passes_per_token_per_layer_scale(self) -> None:
        result = run_scenario(CacheStoreForwardModel(), "capture_fp8")

        metrics = result["capture_metrics"]
        self.assertCaptureMetricTypes(result)
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=1,
            failure=0,
            fp8_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=0)
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}quantize_latency_us"], 0
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes"], 100)
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes"], 40
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes"], 10
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes"], 10
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes"], 40)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertTrue(
            all(put["source_hidden_dtype"] == torch.float8_e4m3fn for put in store.puts)
        )
        self.assertEqual(
            [tuple(put["hidden_states_scale"].shape) for put in store.puts],
            [(2, 2), (3, 2)],
        )
        self.assertTrue(
            all(put["hidden_states_scale"].dtype == torch.float32 for put in store.puts)
        )
        expected_auxiliary = [
            [[10.0, 20.0], [10.0, 20.0]],
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
        ]
        for put, expected in zip(store.puts, expected_auxiliary):
            torch.testing.assert_close(
                put["dequantized_hidden_states"],
                torch.tensor(expected, dtype=torch.float32),
                rtol=0.05,
                atol=0.5,
            )
        self.assertEqual(
            [put["last_hidden_states"].tolist() for put in store.puts],
            [
                [[30.0], [30.0]],
                [[30.0], [30.0], [30.0]],
            ],
        )

    def test_capture_micro_batch_merges_packed_width_before_publish(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(result["hidden_width"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertEqual(
            [
                put["input_ids"].tolist()
                for put in _FakeEagleMooncakeStore.instances[0].puts
            ],
            [[1, 2], [3, 4, 5]],
        )

    def test_capture_disabled_micro_batch_ignores_canonical_fake_lane(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_disabled_micro_batch_fake_lane")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.capture_flags, [True, True])
        self.assertEqual(model.seen_input_lengths, [[2], [2]])
        self.assertEqual(result["hidden_width"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        puts = _FakeEagleMooncakeStore.instances[0].puts
        self.assertEqual(len(puts), 1)
        self.assertEqual(puts[0]["input_ids"].tolist(), [1, 2])

    def test_disabled_micro_batch_zero_tokens_does_not_narrow_fake_lane(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "disabled_micro_batch_zero_tokens")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.capture_flags, [False, False])
        self.assertEqual(model.seen_input_lengths, [[0], [0]])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        self.assertEqual(tuple(result["forward_hidden_states"][0].shape), (0, 1))

    def test_micro_batch_final_hidden_numeric_parity_capture_on_off(self) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        capture_off_model = FinalNormParityModel(hidden_size=2)
        capture_on_model = FinalNormParityModel(hidden_size=2)

        capture_off_result = run_scenario(
            capture_off_model, "micro_batch_final_norm_parity_capture_off"
        )
        capture_on_result = run_scenario(
            capture_on_model, "micro_batch_final_norm_parity_capture_on"
        )

        self.assertTrue(capture_off_model.capture_flags)
        self.assertTrue(capture_on_model.capture_flags)
        self.assertTrue(all(flag is False for flag in capture_off_model.capture_flags))
        self.assertTrue(all(flag is True for flag in capture_on_model.capture_flags))
        capture_off_hidden = capture_off_result["forward_hidden_states"][0]
        capture_on_hidden = capture_on_result["forward_hidden_states"][0]
        self.assertEqual(tuple(capture_off_hidden.shape), (5, 2))
        self.assertEqual(tuple(capture_on_hidden.shape), (5, 2))
        torch.testing.assert_close(
            capture_on_hidden,
            capture_off_hidden,
            rtol=0,
            atol=0,
        )

    def test_capture_micro_batch_layout_failure_is_deferred_without_collective(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        with self.assertRaisesRegex(
            Exception,
            "capture micro-batch output 0 width 2 must match expected packed width 3",
        ):
            run_scenario(MalformedCaptureLayoutModel(), "capture_micro_batch_tp")

        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_context_parallel_and_micro_batch_is_explicitly_unsupported(
        self,
    ) -> None:
        model = CacheStoreForwardModel()
        with self.assertRaisesRegex(
            Exception,
            "hidden-state capture does not support context parallel combined with layer micro-batching",
        ):
            run_scenario(model, "capture_cp_micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 0)
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_context_parallel_rejects_invalid_num_valid_tokens(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        with self.assertRaisesRegex(
            Exception,
            "context parallel restored captured hidden-state row count 6 must match expected row count 5",
        ):
            run_scenario(
                CacheStoreForwardModel(), "capture_cp_invalid_num_valid_tokens"
            )

        self.assertEqual(layout_statuses, [1])

    def test_capture_context_parallel_rejects_local_layout_before_restore(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureLayoutModel(),
            "capture_cp_malformed_layout",
            ignore_deferred_errors=True,
        )

        self.assertEqual(layout_statuses, [0])
        self.assertEqual(result["cp_handle_outputs_calls"], 0)
        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn(
            "captured hidden-state width 2 must match expected packed width 3",
            result["deferred_capture_errors"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_context_parallel_remote_invalid_rank_skips_restore(self) -> None:
        layout_reduce_calls = []

        def allreduce(tensor, op, mode, dest):
            layout_reduce_calls.append((int(tensor.item()), op, mode, dest))
            return torch.zeros_like(tensor)

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_cp",
            ignore_deferred_errors=True,
        )

        self.assertEqual(layout_reduce_calls, [(1, 3, 0, None)])
        self.assertEqual(result["cp_handle_outputs_calls"], 0)
        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn(
            "layout contract violation on another TP rank",
            result["deferred_capture_errors"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_context_parallel_rechecks_width_after_restore(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_cp_malformed_restored_width",
            ignore_deferred_errors=True,
        )

        self.assertEqual(layout_statuses, [1])
        self.assertEqual(result["cp_handle_outputs_calls"], 1)
        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn(
            "context parallel restored captured hidden-state width 2 must match expected packed width 3",
            result["deferred_capture_errors"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_packed_layout_reports_rows_dtype_and_device_errors(self) -> None:
        cases = [
            (
                MalformedCaptureRowCountModel(),
                "captured hidden-state row count 4 must match expected row count 5",
            ),
            (
                MalformedCaptureDtypeModel(),
                "captured hidden-state dtype must match the model dtype",
            ),
            (
                MalformedCaptureDeviceModel(),
                "captured hidden-state must be on the model input device",
            ),
        ]
        for model, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(Exception, expected_error):
                    run_scenario(model, "capture_bf16")
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_capture_malformed_rank_uses_fallback_before_any_narrow(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureRankModel(),
            "capture_bf16_tp",
            ignore_deferred_errors=True,
        )

        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn("must be a rank-2 tensor", result["deferred_capture_errors"][0])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_context_parallel_uses_original_ids_and_lengths(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        model = PositionAwareCaptureModel()
        result = run_scenario(model, "capture_cp")

        self.assertEqual(layout_statuses, [1])
        self.assertEqual(result["cp_handle_outputs_calls"], 1)
        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertEqual(result["hidden_width"], 1)
        puts = _FakeEagleMooncakeStore.instances[0].puts
        self.assertEqual(len(puts), 1)
        self.assertEqual(puts[0]["key"], "rtp_TestActor_601")
        self.assertEqual(puts[0]["input_ids"].tolist(), [1, 2, 3, 4, 5, 6])
        torch.testing.assert_close(
            puts[0]["hidden_states"],
            torch.tensor(
                [[11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [16, 26]],
                dtype=torch.bfloat16,
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            puts[0]["last_hidden_states"],
            torch.tensor([[31], [32], [33], [34], [35], [36]], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["forward_hidden_states"][0],
            torch.tensor([[31], [32], [33], [34], [35], [36]], dtype=torch.float16),
            rtol=0,
            atol=0,
        )

    def test_capture_rejects_negative_request_id_before_put(self) -> None:
        with self.assertRaisesRegex(Exception, "request id must be non-negative"):
            run_scenario(CacheStoreForwardModel(), "capture_negative_request_id")

        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_duplicate_request_id_only_rejects_each_current_batch(
        self,
    ) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_duplicate_request_id",
            3,
            True,
            hidden_state_capture_fail_open=True,
        )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(store.put_attempts, 0)
        self.assertEqual(store.puts, [])
        self.assertEqual(result["capture_failure_count"], 3)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(len(result["deferred_capture_errors"]), 3)
        self.assertTrue(
            all(
                "duplicate capture request id 501" in error
                for error in result["deferred_capture_errors"]
            )
        )
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=3,
            prepare_failure=3,
            request_error_failure=3,
            duplicate_request_id=3,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=1)

    def test_capture_cpu_prepare_failure_only_rejects_each_current_batch(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_prepare_failure_tp", 3, True)

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(result["capture_failure_count"], 3)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(len(result["deferred_capture_errors"]), 3)
        self.assertTrue(
            all(
                "capture request metadata" in error
                for error in result["deferred_capture_errors"]
            )
        )
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=3,
            prepare_failure=3,
            operational_failure=3,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=0)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_non_owner_does_not_materialize_request_metadata(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_prepare_failure_non_owner")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(result["deferred_capture_errors"], [])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_historical_async_error_is_not_attributed_to_current_batch(
        self,
    ) -> None:
        _FakeEagleMooncakeStore.async_error_after_accept_count = 1
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_async_history",
            2,
            hidden_state_capture_fail_open=True,
        )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True])
        self.assertEqual(store.put_attempts, 2)
        self.assertEqual(len(store.put_batch_calls), 2)
        self.assertEqual(
            [call["request_ids"] for call in store.put_batch_calls],
            [
                ["rtp_TestActor_501", "rtp_TestActor_502"],
                ["rtp_TestActor_1501", "rtp_TestActor_1502"],
            ],
        )
        self.assertEqual(store.pending_async_errors, [])
        self.assertEqual(result["deferred_capture_errors"], [])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=2,
            publish_success=2,
            failure=1,
            store_failure=1,
            operational_failure=1,
            bf16_publish=2,
        )
        self.assertCaptureStatus(
            result["capture_metrics"], enabled=1, broken=0, fail_open=1
        )

    def test_capture_fp8_rejects_malformed_quantizer_result_even_when_fail_open(
        self,
    ) -> None:
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        fp8_module.quantize_aux_hidden_states = lambda hidden_states, num_layers: (
            hidden_states,
        )
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )

        with self.assertRaisesRegex(Exception, "must return tensor and scale"):
            run_scenario(
                CacheStoreForwardModel(),
                "capture_fp8",
                hidden_state_capture_fail_open=True,
            )

        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_quantizer_execution_failure_recovers_on_next_batch(self) -> None:
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        call_count = 0

        def fail_once(hidden_states, num_layers):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("injected TorchSpec quantizer execution failure")
            return original_quantizer(hidden_states, num_layers)

        fp8_module.quantize_aux_hidden_states = fail_once
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )

        result = run_scenario(CacheStoreForwardModel(), "capture_fp8_tp", 2, True)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.put_attempts, 1)
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn(
            "injected TorchSpec quantizer execution failure",
            result["deferred_capture_errors"][0],
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=2,
            publish_success=1,
            failure=1,
            quantize_failure=1,
            operational_failure=1,
            fp8_publish=1,
        )
        self.assertCaptureStatus(
            result["capture_metrics"], enabled=1, broken=0, fail_open=0
        )

    def test_capture_non_owner_does_not_duplicate_success_metrics(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_non_owner")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS)
        self.assertCaptureQps(metrics, batch=0, publish_success=0, failure=0)
        self.assertCaptureStatus(metrics, enabled=0, broken=0, fail_open=0)

    def test_capture_fp8_hard_failure_disables_owner_store_before_next_forward(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        fp8_module.quantize_aux_hidden_states = lambda hidden_states, num_layers: (
            hidden_states,
        )
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_fp8_tp", 3, True)

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(len(result["deferred_capture_errors"]), 3)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 2)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=1,
            quantize_failure=1,
            hard_contract_failure=1,
            broken_rejection=2,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=1, fail_open=0)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}quantize_latency_us", metrics)
        self.assertTrue(
            all(
                "must return tensor and scale" in error
                for error in result["deferred_capture_errors"]
            )
        )

    def test_capture_non_owner_keeps_packed_shape_without_owner_status_sync(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_non_owner", 3, True)

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(result["deferred_capture_errors"], [])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS)
        self.assertCaptureQps(metrics, batch=0, publish_success=0, failure=0)
        self.assertCaptureStatus(metrics, enabled=0, broken=0, fail_open=0)

    def test_capture_rejects_duplicate_layer_ids_at_startup(self) -> None:
        with self.assertRaisesRegex(
            Exception, "layer ids must be unique; duplicate id 0"
        ):
            run_scenario(CacheStoreForwardModel(), "capture_duplicate_layer_id")

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_invalid_dtype_at_startup(self) -> None:
        with self.assertRaisesRegex(
            Exception, "unsupported hidden-state capture dtype: 99"
        ):
            run_scenario(CacheStoreForwardModel(), "capture_invalid_dtype")

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_unsupported_python_model_at_startup(self) -> None:
        with self.assertRaisesRegex(Exception, "does not support hidden-state capture"):
            run_scenario(UnsupportedCaptureModel(), "capture_bf16")

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_rejects_layer_count_mismatch_even_when_fail_open(
        self,
    ) -> None:
        os.environ["MOONCAKE_NUM_AUX_LAYERS"] = "3"

        with self.assertRaisesRegex(
            Exception, "does not match RTP capture layer count"
        ):
            run_scenario(
                CacheStoreForwardModel(),
                "capture_bf16",
                hidden_state_capture_fail_open=True,
            )

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_rejects_hidden_dim_mismatch_even_when_fail_open(
        self,
    ) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"

        with self.assertRaisesRegex(Exception, "does not match RTP hidden size"):
            run_scenario(
                CacheStoreForwardModel(),
                "capture_bf16",
                hidden_state_capture_fail_open=True,
            )

        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_requires_batch_and_async_error_contract_methods(
        self,
    ) -> None:
        for method_name in ("put_batch", "take_async_errors"):
            with self.subTest(method_name=method_name):
                original = getattr(_FakeEagleMooncakeStore, method_name)
                setattr(_FakeEagleMooncakeStore, method_name, None)
                try:
                    with self.assertRaisesRegex(
                        Exception,
                        rf"EagleMooncakeStore\.{method_name} must be callable",
                    ):
                        run_scenario(
                            CacheStoreForwardModel(),
                            "capture_bf16",
                            hidden_state_capture_fail_open=True,
                        )
                finally:
                    setattr(_FakeEagleMooncakeStore, method_name, original)

    def test_capture_publisher_init_failure_is_fail_closed_by_default(self) -> None:
        _FakeEagleMooncakeStore.fail_warmup = True

        with self.assertRaisesRegex(Exception, "injected Mooncake warmup failure"):
            run_scenario(CacheStoreForwardModel(), "capture_bf16")

    def test_capture_publisher_init_failure_fail_open_disables_store_only(self) -> None:
        _FakeEagleMooncakeStore.fail_warmup = True
        model = CacheStoreForwardModel()

        result = run_scenario(
            model, "capture_bf16", hidden_state_capture_fail_open=True
        )

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS)
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=0,
            failure=1,
            initialization_failure=1,
            operational_failure=1,
            fail_open_disable=1,
            disabled_skip=1,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0, fail_open=1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.put_attempts, 0)
        self.assertTrue(store.closed)
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.full_like(hidden_states, 30))

    def test_capture_is_disabled_on_ffn_service_model(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_ffn_service")

        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_non_owner_uses_static_capture_shape_without_status_sync(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_non_owner")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.full_like(hidden_states, 30))

    def test_capture_runtime_fail_open_returns_last_hidden_and_disables_store(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        _FakeEagleMooncakeStore.fail_put = True
        model = CacheStoreForwardModel()
        result = run_scenario(
            model,
            "capture_bf16_tp",
            3,
            hidden_state_capture_fail_open=True,
        )

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].put_attempts, 1)
        for hidden_states in result["forward_hidden_states"]:
            torch.testing.assert_close(
                hidden_states, torch.full_like(hidden_states, 30)
            )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=1,
            store_failure=1,
            operational_failure=1,
            fail_open_disable=1,
            disabled_skip=2,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0, fail_open=1)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us", metrics)

    def test_capture_duplicate_store_key_only_rejects_each_current_batch(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_bf16_tp",
            3,
            True,
            hidden_state_capture_fail_open=True,
        )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(store.put_attempts, 3)
        self.assertEqual(result["capture_failure_count"], 2)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(len(result["deferred_capture_errors"]), 2)
        self.assertTrue(
            all(
                "Mooncake key already exists: rtp_TestActor_501" in error
                for error in result["deferred_capture_errors"]
            )
        )
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=1,
            failure=2,
            store_failure=2,
            request_error_failure=2,
            duplicate_request_id=2,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=1)

    def test_capture_non_owner_never_receives_owner_runtime_state(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_non_owner", 3)

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        for hidden_states in result["forward_hidden_states"]:
            torch.testing.assert_close(
                hidden_states, torch.full_like(hidden_states, 30)
            )
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)

    def test_capture_rejects_packed_layout_even_when_fail_open(self) -> None:
        model = MalformedCaptureLayoutModel()
        result = run_scenario(
            model,
            "capture_bf16",
            3,
            True,
            hidden_state_capture_fail_open=True,
        )

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(len(result["deferred_capture_errors"]), 3)
        self.assertTrue(
            all(
                "captured hidden-state width 2 must match expected packed width 3"
                in error
                for error in result["deferred_capture_errors"]
            )
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 2)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=1,
            layout_failure=1,
            hard_contract_failure=1,
            broken_rejection=2,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=1, fail_open=1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_fail_closed_synchronous_admission_error_fails_current_batch(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        _FakeEagleMooncakeStore.fail_put_count = 1
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_bf16_tp", 2, True)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True])
        self.assertEqual(store.put_attempts, 2)
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(len(result["deferred_capture_errors"]), 1)
        self.assertIn(
            "injected Mooncake put_batch admission failure",
            result["deferred_capture_errors"][0],
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=2,
            publish_success=1,
            failure=1,
            store_failure=1,
            operational_failure=1,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0, fail_open=0)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us", metrics)


class TorchSpecContractGuardTest(unittest.TestCase):
    def test_fake_torchspec_is_not_installed_during_module_import(self) -> None:
        mooncake = sys.modules.get("torchspec.transfer.mooncake")
        if mooncake is not None:
            self.assertIsNot(
                getattr(mooncake, "EagleMooncakeStore", None),
                _FakeEagleMooncakeStore,
            )
            self.assertIsNot(
                getattr(mooncake, "MooncakeConfig", None),
                _FakeMooncakeConfig,
            )

        fp8 = sys.modules.get("torchspec.utils.fp8")
        if fp8 is not None:
            self.assertIsNot(
                getattr(fp8, "quantize_aux_hidden_states", None),
                _fake_quantize_aux_hidden_states,
            )

    def assertSignatureBinds(self, callable_object, *args, **kwargs) -> None:
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError) as error:
            self.fail(f"TorchSpec callable has no inspectable signature: {error}")
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            self.fail(f"TorchSpec callable does not satisfy capture contract: {error}")

    def test_real_torchspec_contract_when_available(self) -> None:
        try:
            importlib.import_module("torchspec")
        except ModuleNotFoundError as error:
            if error.name == "torchspec":
                self.skipTest("real TorchSpec is not installed")
            self.skipTest(f"real TorchSpec dependency is unavailable: {error.name}")
        except ImportError as error:
            self.skipTest(
                f"real TorchSpec cannot be loaded in this environment: {error}"
            )

        try:
            mooncake = importlib.import_module("torchspec.transfer.mooncake")
            fp8 = importlib.import_module("torchspec.utils.fp8")
        except ModuleNotFoundError as error:
            if error.name is not None and error.name.startswith("torchspec"):
                self.fail(f"real TorchSpec is missing capture API module: {error.name}")
            self.skipTest(f"real TorchSpec dependency is unavailable: {error.name}")
        except ImportError as error:
            self.skipTest(f"real TorchSpec capture API cannot be loaded: {error}")

        try:
            mooncake_config = cast(Any, getattr(mooncake, "MooncakeConfig", None))
            mooncake_store = cast(Any, getattr(mooncake, "EagleMooncakeStore", None))
            quantize_aux_hidden_states = cast(
                Any, getattr(fp8, "quantize_aux_hidden_states", None)
            )
        except (ImportError, OSError) as error:
            self.skipTest(f"real TorchSpec capture API is unavailable: {error}")
        self.assertIsNotNone(mooncake_config)
        self.assertIsNotNone(mooncake_store)
        self.assertTrue(callable(quantize_aux_hidden_states))

        self.assertSignatureBinds(mooncake_config.from_env)
        layout_env = {
            "MOONCAKE_HIDDEN_DIM": "2048",
            "MOONCAKE_MAX_SEQ_LEN": "16384",
            "MOONCAKE_GET_BATCH_SIZE": "8",
            "MOONCAKE_GPU_BUFFER_SIZE": "268435456",
            "MOONCAKE_NUM_AUX_LAYERS": "2",
            _MOONCAKE_STORE_KEY_NAMESPACE_ENV: "ContractActor1",
        }
        previous_layout_env = {
            env_name: os.environ.get(env_name) for env_name in layout_env
        }
        try:
            os.environ.update(layout_env)
            restored_config = mooncake_config.from_env()
        finally:
            _restore_mooncake_layout_env(previous_layout_env)
        for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items():
            self.assertEqual(
                getattr(restored_config, attr_name), int(layout_env[env_name])
            )
        self.assertEqual(restored_config.store_key_namespace, "ContractActor1")
        self.assertSignatureBinds(restored_config.make_store_key, 501)
        self.assertEqual(
            restored_config.make_store_key(501),
            "rtp_ContractActor1_501",
        )

        self.assertSignatureBinds(mooncake_store, object())
        self.assertSignatureBinds(mooncake_store.setup, object(), torch.device("cuda"))
        for method_name in ("warmup_rdma", "take_async_errors", "flush", "close"):
            method = getattr(mooncake_store, method_name, None)
            self.assertTrue(callable(method), method_name)
            self.assertSignatureBinds(method, object())
        self.assertSignatureBinds(
            mooncake_store.put_batch,
            object(),
            batch_id="contract-batch",
            request_ids=["contract-key"],
            hidden_states=[object()],
            input_ids=[object()],
            last_hidden_states=[object()],
            hidden_states_scale=None,
        )
        self.assertSignatureBinds(quantize_aux_hidden_states, object(), 2)


if __name__ == "__main__":
    unittest.main()

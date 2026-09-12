import importlib
import os
import sys
import types
import unittest
from typing import Optional

import torch

from rtp_llm.config.cuda_graph import GenerationPrefillCudaGraphUnsupportedBackend


def _install_single_rank_rocm_graph_capture_stub() -> None:
    """Keep this focused integration test independent of the production ops SO.

    The C++ test extension registers the same pybind types as the production
    extension, so importing ``rocm_rccl`` here would register those types a
    second time. Single-rank capture has no RCCL work; these are the exact
    no-op hooks used by the device shim for that case.
    """

    models_py = types.ModuleType("rtp_llm.models_py")
    models_py.__path__ = []
    distributed = types.ModuleType("rtp_llm.models_py.distributed")
    distributed.__path__ = []
    rocm_rccl = types.ModuleType("rtp_llm.models_py.distributed.rocm_rccl")
    rocm_rccl.set_graph_capture_nccl_comm = lambda *_args: None
    rocm_rccl.enter_graph_capture_mode = lambda *_args: None
    rocm_rccl.exit_graph_capture_mode = lambda: None
    rocm_rccl.finish_hipgraph_capture_session = lambda: None

    models_py.distributed = distributed
    distributed.rocm_rccl = rocm_rccl
    sys.modules["rtp_llm.models_py"] = models_py
    sys.modules["rtp_llm.models_py.distributed"] = distributed
    sys.modules["rtp_llm.models_py.distributed.rocm_rccl"] = rocm_rccl


if os.environ.get("TEST_USING_DEVICE") == "ROCM":
    _install_single_rank_rocm_graph_capture_stub()

_extension = importlib.import_module(
    "rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test"
)
PyModelInputs = _extension.PyModelInputs
PyModelOutputs = _extension.PyModelOutputs
run_generation_prefill_capture_scenario = (
    _extension.run_generation_prefill_capture_scenario
)
run_dirty_generation_prefill_capture_scenario = (
    _extension.run_dirty_generation_prefill_capture_scenario
)
run_scenario = _extension.run_scenario


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
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
            (inputs.input_ids.numel(), 1),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class DirtyGenerationPrefillCaptureModel:
    """Fail on the graph-capture body after all eager generation-prefill warmups."""

    def __init__(self) -> None:
        self.prefill_forward_calls = 0
        self._is_generation_prefill = False

    def initialize(self, _resources) -> bool:
        return True

    def prepare_fmha_impl(
        self,
        _inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode=None,
    ):
        self._is_generation_prefill = (
            cuda_graph_selection_mode == "generation_prefill_graph"
        )
        return None

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
            # The datatype probe and two eager warmups precede capture.
            if self.prefill_forward_calls == 4:
                raise RuntimeError("injected PyWrappedModel capture-body failure")
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class SuccessfulGenerationPrefillCaptureModel(DirtyGenerationPrefillCaptureModel):
    """Capture-safe model used to verify block-0 capture needs no allocation."""

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class LateInitializeFailureGenerationPrefillCaptureModel(
    SuccessfulGenerationPrefillCaptureModel
):
    """Fail the post-capture initialization check after both graphs exist."""

    def __init__(self) -> None:
        super().__init__()
        self.initialize_calls = 0

    def initialize(self, _resources) -> bool:
        self.initialize_calls += 1
        # PyWrappedModel initializes once before graph setup and once before
        # capture. The second result is checked only after decode and prefill
        # and generation-prefill runners have been constructed, exercising
        # constructor unwind.
        return self.initialize_calls == 1


class CleanFallbackGenerationPrefillCaptureModel(DirtyGenerationPrefillCaptureModel):
    """Fail before capture begins to verify clean fail-fast teardown."""

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
            if self.prefill_forward_calls == 1:
                raise RuntimeError("injected clean generation prefill graph fallback")
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class UnsupportedBackendGenerationPrefillCaptureModel(
    DirtyGenerationPrefillCaptureModel
):
    """Reject generation-prefill graph selection before capture begins."""

    def prepare_fmha_impl(
        self,
        _inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode=None,
    ):
        if cuda_graph_selection_mode == "generation_prefill_graph":
            raise GenerationPrefillCudaGraphUnsupportedBackend(
                "injected unsupported backend"
            )
        return None


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
    def test_successful_generation_prefill_capture_does_not_reserve_request_blocks(
        self,
    ) -> None:
        model = SuccessfulGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(model)

        self.assertFalse(result["saw_capture_error"])
        self.assertEqual(result["capture_error_message"], "")
        self.assertTrue(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_clean_generation_prefill_capture_failure_fails_init_without_cache_allocation(
        self,
    ) -> None:
        model = CleanFallbackGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "injected clean generation prefill graph fallback"
        )

        self.assertTrue(result["saw_capture_error"])
        self.assertIn(
            "injected clean generation prefill graph fallback",
            result["capture_error_message"],
        )
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 1)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_unsupported_generation_prefill_backend_fails_init_without_cache_allocation(
        self,
    ) -> None:
        model = UnsupportedBackendGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "injected unsupported backend"
        )

        self.assertTrue(result["saw_capture_error"])
        self.assertIn("injected unsupported backend", result["capture_error_message"])
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 0)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_late_constructor_failure_destroys_clean_graphs_without_cache_allocation(
        self,
    ) -> None:
        model = LateInitializeFailureGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "Python model initialization failed"
        )

        self.assertEqual(model.initialize_calls, 2)
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertTrue(result["saw_capture_error"])
        self.assertIn(
            "Python model initialization failed", result["capture_error_message"]
        )
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_dirty_generation_prefill_capture_does_not_retain_cache_manager(
        self,
    ) -> None:
        # Dirty capture poisons process-local graph/allocator state by design;
        # the dedicated Bazel target runs this method alone.
        if os.environ.get("RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST") != "1":
            self.skipTest(
                "requires an isolated process for a dirty device graph capture"
            )

        model = DirtyGenerationPrefillCaptureModel()
        result = run_dirty_generation_prefill_capture_scenario(model)

        self.assertTrue(result["saw_dirty_capture_error"])
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertEqual(result["available_after"], result["available_before"])
        self.assertFalse(result["manager_retained"])

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
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

    @unittest.skipIf(
        os.environ.get("TEST_USING_DEVICE") == "ROCM",
        "context parallel is not supported on ROCm",
    )
    def test_context_parallel_publishes_original_lengths_to_every_tag(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)
        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 3)
        self.assertEqual(len(linear_blocks), 6)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144, 168, 192],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})
        for token_key in range(3101, 3107):
            self.assertTrue(
                any(f"_token_id_str_{token_key}_" in key for key in linear_blocks)
            )
        for token_key in (3102, 3104, 3106):
            self.assertTrue(
                any(f"_token_id_str_{token_key}_" in key for key in full_blocks)
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


if __name__ == "__main__":
    unittest.main()

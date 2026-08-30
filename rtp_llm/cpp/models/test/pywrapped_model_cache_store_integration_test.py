import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    run_invalid_boundary_diagnostics,
    run_scenario,
)


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

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
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


def _offsets_by_tag(result: dict) -> dict:
    blocks = _blocks_by_key(result)
    offsets = {}
    for tag in ("full", "linear"):
        offsets[tag] = sorted(
            block["address"] - result["base_addresses"][tag]
            for key, block in blocks.items()
            if ("_tag_" + tag) in key
        )
    return offsets


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def _assert_boundary_failure_has_no_publication(self, scenario: str) -> None:
        model = CacheStoreForwardModel()
        result = run_invalid_boundary_diagnostics(model, scenario)
        self.assertTrue(result["message"])
        self.assertEqual(result["held_delta"], 0)
        self.assertEqual(result["device_copy_delta"], 0)
        self.assertEqual(result["store_records"], 0)
        self.assertEqual(model.forward_calls, 0)

    def test_tagged_boundary_rejects_missing_tags_before_python_forward(self) -> None:
        model = CacheStoreForwardModel()
        with self.assertRaisesRegex(RuntimeError, "cache tags"):
            run_scenario(model, "missing_boundary_tags")
        self.assertEqual(model.forward_calls, 0)

    def test_multi_tag_binding_ignores_cache_group_declaration_order(self) -> None:
        # The block-table group dimension is ordered by sorted tags, so
        # declaring the same two groups in the other order must publish exactly
        # the same per-tag addresses.
        unsorted_result = run_scenario(CacheStoreForwardModel(), "multi_tag")
        sorted_result = run_scenario(
            CacheStoreForwardModel(), "multi_tag_sorted_declaration"
        )

        self.assertEqual(
            _offsets_by_tag(unsorted_result), _offsets_by_tag(sorted_result)
        )
        self.assertEqual(
            _offsets_by_tag(sorted_result),
            {"full": [16, 32], "linear": [72, 96, 120, 144]},
        )

    def test_multi_tag_boundary_reorders_tags_and_all_parallel_rows_together(
        self,
    ) -> None:
        canonical = run_scenario(CacheStoreForwardModel(), "multi_tag")
        reordered = run_scenario(CacheStoreForwardModel(), "reordered_boundary_tags")
        self.assertEqual(_offsets_by_tag(reordered), _offsets_by_tag(canonical))

    def test_tp_non_root_reconstructs_single_tag_after_tensor_sync(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "tp_non_root_single_tag")
        self.assertEqual(model.forward_calls, 1)
        record = _record_for_request(result, 351)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32],
        )

    def test_tp_non_root_reconstructs_reordered_multi_group_parallel_rows(self) -> None:
        canonical = run_scenario(CacheStoreForwardModel(), "multi_tag")
        reconstructed = run_scenario(CacheStoreForwardModel(), "tp_non_root_multi_tag")
        self.assertEqual(_offsets_by_tag(reconstructed), _offsets_by_tag(canonical))

    def test_tagged_boundary_rejects_all_invalid_parallel_payloads_before_publication(
        self,
    ) -> None:
        for scenario in (
            "duplicate_boundary_tags",
            "unknown_boundary_tag",
            "unequal_group_types",
            "late_group_type_mismatch",
            "tp_non_root_group_type_mismatch",
        ):
            with self.subTest(scenario=scenario):
                self._assert_boundary_failure_has_no_publication(scenario)

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

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
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


if __name__ == "__main__":
    unittest.main()

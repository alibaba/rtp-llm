import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    run_scenario,
)


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []
        self.seen_kernel_tables: dict[str, list[list[int]]] = {}
        self.seen_cache_tables: list[dict[str, dict[str, list[list[int]]]]] = []
        self.seen_attention_tables: list[dict[str, list[list[int]]]] = []
        self.seen_direct_inputs: list[bool] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        self.seen_direct_inputs.append(not isinstance(attention_inputs, dict))
        if isinstance(attention_inputs, dict):
            self.seen_attention_tables.append(
                {
                    tag: item.kv_cache_kernel_block_id_device.cpu().tolist()
                    for tag, item in attention_inputs.items()
                }
            )
        first_inputs = (
            next(iter(attention_inputs.values()))
            if isinstance(attention_inputs, dict)
            else attention_inputs
        )
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        cache_tables = {}
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            self.seen_kernel_tables[layer_cache.tag] = (
                tag_inputs.kv_cache_kernel_block_id.tolist()
            )
            cache_tables[layer_cache.tag] = {
                "physical": tag_inputs.kv_cache_block_id.tolist(),
                "kernel": tag_inputs.kv_cache_kernel_block_id.tolist(),
            }
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )
        self.seen_cache_tables.append(cache_tables)

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


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_mtp_placeholders_preserve_model_tag_set_with_extra_payload_rows(self):
        for scenario in ("mtp_placeholders", "mtp_placeholders_extra_payload"):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                result = run_scenario(model, scenario)
                self.assertEqual(model.seen_direct_inputs, [False])
                self.assertEqual(
                    model.seen_attention_tables,
                    [{"unused": [[3, 4]], "draft": [[1, 2]], "other": [[5, 6]]}],
                )
                self.assertEqual(set(model.seen_cache_tables[0]), {"draft"})
                self.assertEqual(set(result["base_addresses"]), {"draft"})
                record = _record_for_request(result, 401)
                self.assertEqual(
                    sorted(
                        block["address"] - result["base_addresses"]["draft"]
                        for block in record["blocks"]
                    ),
                    [32, 64],
                )

    def test_missing_mtp_placeholder_is_rejected_before_forward(self):
        model = CacheStoreForwardModel()
        with self.assertRaisesRegex(
            RuntimeError, "missing model cache group tag=unused"
        ):
            run_scenario(model, "mtp_missing_placeholder")
        self.assertEqual(model.forward_calls, 0)

    def test_single_model_group_selects_direct_input_from_shared_payload(self):
        model = CacheStoreForwardModel()
        result = run_scenario(model, "single_model_extra_payload")
        self.assertEqual(model.seen_direct_inputs, [True])
        self.assertEqual(
            model.seen_cache_tables,
            [{"draft": {"physical": [[1, 2]], "kernel": [[1, 2]]}}],
        )
        record = _record_for_request(result, 401)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["draft"]
                for block in record["blocks"]
            ),
            [32, 64],
        )

    def test_cacheless_multigroup_warmup_exposes_single_input(self):
        class WarmupModel(CacheStoreForwardModel):
            def _forward_one(self, inputs):
                assert self.kv_cache is None
                attention_inputs = inputs.attention_inputs
                assert not isinstance(attention_inputs, dict)
                assert attention_inputs.is_prefill
                assert not attention_inputs.is_target_verify
                assert attention_inputs.kv_cache_block_id is None
                assert attention_inputs.kv_cache_block_id_device is None
                assert attention_inputs.kv_cache_kernel_block_id is None
                assert attention_inputs.kv_cache_kernel_block_id_device is None
                assert attention_inputs.cache_store_inputs is None
                return PyModelOutputs(
                    torch.zeros(
                        (inputs.input_ids.numel(), 1),
                        dtype=torch.float16,
                        device=inputs.input_ids.device,
                    )
                )

        model = WarmupModel()
        result = run_scenario(model, "cacheless_warmup")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(result["records"], [])

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(model.seen_kernel_tables["full"], [[1, 2, -1, -1]])
        self.assertEqual(model.seen_kernel_tables["linear"], [[3, 4, 5, 6]])
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

    def test_invalid_payload_identity_is_rejected_before_forward(self) -> None:
        for scenario in (
            "duplicate_tags",
            "empty_tag",
            "missing_tags",
            "unknown_tag",
            "physical_group_mismatch",
            "type_group_mismatch",
            "multi_group_2d",
            "single_group_2d_unknown_tag",
        ):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                with self.assertRaises(RuntimeError):
                    run_scenario(model, scenario)
                self.assertEqual(model.forward_calls, 0)

    def test_single_group_2d_preserves_direct_cache_store_path(self) -> None:
        for scenario in ("single_group_2d", "single_group_2d_no_tags"):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                result = run_scenario(model, scenario)
                record = _record_for_request(result, 401)
                base = result["base_addresses"]["draft"]
                self.assertEqual(
                    sorted(block["address"] - base for block in record["blocks"]),
                    [32, 64],
                )
                self.assertEqual(model.seen_kernel_tables["draft"], [[1, 2]])

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

    def test_micro_batch_preserves_reordered_group_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch_multi_tag")
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(model.seen_cache_tables), 2)
        expected_tables = [
            {
                "full": [[1, -1, -1, -1], [2, 3, -1, -1]],
                "linear": [[2, 3, -1, -1], [4, 5, 6, 7]],
            },
            {"full": [[4, -1, -1, -1]], "linear": [[1, 2, -1, -1]]},
        ]
        for actual, expected in zip(model.seen_cache_tables, expected_tables):
            self.assertEqual(set(actual), set(expected))
            for tag, table in expected.items():
                self.assertEqual(actual[tag]["physical"], table)
                self.assertEqual(actual[tag]["kernel"], table)

        blocks = _blocks_by_key(result)
        for tag, offsets in (
            ("full", [16, 32, 48, 64]),
            ("linear", [24, 48, 48, 72, 96, 120, 144, 168]),
        ):
            self.assertEqual(
                sorted(
                    block["address"] - result["base_addresses"][tag]
                    for key, block in blocks.items()
                    if f"_tag_{tag}" in key
                ),
                offsets,
            )

    def test_micro_batch_split_preserves_cache_storage_contract(self) -> None:
        # Synthetic core contract: group names are identities, both policies are FULL.
        # This does not assert support for a production Full+Linear model.
        multi_group_tables = [
            [[1, -1, -1, -1], [2, 3, -1, -1], [4, -1, -1, -1]],
            [[2, 3, -1, -1], [4, 5, 6, 7], [1, 2, -1, -1]],
        ]
        single_group_tables = [[[1, -1], [2, 3], [4, -1]]]
        for scenario, on_cuda, two_dimensional, tags, tables in (
            (
                "micro_batch_split_pinned",
                False,
                False,
                ["full", "linear"],
                multi_group_tables,
            ),
            (
                "micro_batch_split_cuda",
                True,
                False,
                ["full", "linear"],
                multi_group_tables,
            ),
            (
                "micro_batch_split_single_group",
                False,
                False,
                ["default"],
                single_group_tables,
            ),
            (
                "micro_batch_split_2d",
                False,
                True,
                ["default"],
                single_group_tables,
            ),
        ):
            with self.subTest(scenario=scenario):
                result = run_scenario(CacheStoreForwardModel(), scenario)
                self.assertEqual(result["source_tags"], tags)
                self.assertEqual(len(result["batches"]), 2)
                physical = torch.tensor(tables, dtype=torch.int32)
                kernel = torch.where(physical >= 0, physical + 100, physical)
                if two_dimensional:
                    physical = physical.squeeze(0)
                    kernel = kernel.squeeze(0)
                batch_axis = 0 if two_dimensional else 1
                for field, expected in (("physical", physical), ("kernel", kernel)):
                    source = result[f"source_{field}"]
                    self.assertEqual(source.dtype, torch.int32)
                    self.assertEqual(source.is_cuda, on_cuda)
                    self.assertEqual(source.is_pinned(), not on_cuda)
                    torch.testing.assert_close(source.cpu(), expected)
                    for batch, start, count in zip(result["batches"], (0, 2), (2, 1)):
                        self.assertEqual(batch["tags"], tags)
                        tensor = batch[field]
                        self.assertEqual(tensor.dtype, torch.int32)
                        self.assertEqual(tensor.device, source.device)
                        self.assertEqual(tensor.is_pinned(), not on_cuda)
                        self.assertTrue(tensor.is_contiguous())
                        torch.testing.assert_close(
                            tensor.cpu(), expected.narrow(batch_axis, start, count)
                        )
                self.assertEqual(
                    [batch["input_lengths"].tolist() for batch in result["batches"]],
                    [[2, 4], [2]],
                )

    def test_fake_micro_batch_has_no_cache_identity(self) -> None:
        result = run_scenario(CacheStoreForwardModel(), "fake_micro_batch")
        self.assertEqual(result["real_tags"], ["full", "linear"])
        self.assertEqual(result["fake_tags"], [])
        self.assertFalse(result["fake_physical_defined"])
        self.assertFalse(result["fake_kernel_defined"])

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

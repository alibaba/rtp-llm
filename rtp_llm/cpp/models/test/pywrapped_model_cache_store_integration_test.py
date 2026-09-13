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


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_device_metadata_preserves_long_prefix_counts_without_scalar_sync(self) -> None:
        for batch_size in (1, 3, 8, 16):
            with self.subTest(batch_size=batch_size):
                query = torch.full((batch_size,), 4, dtype=torch.int32, device="cuda")
                prefix = torch.tensor(
                    [524284, 999996, 1048563] * ((batch_size + 2) // 3),
                    dtype=torch.int32,
                    device="cuda",
                )[:batch_size]
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    attention = run_scenario(
                        CacheStoreForwardModel(),
                        "metadata",
                        {"input_lengths": query, "prefix_lengths": prefix, "total_tokens": 4 * batch_size},
                    )["attention"]
                builds = [event for event in profile.events() if event.name == "py_model.buildPyAttentionInputs"]
                self.assertEqual(len(builds), 1)
                descendants = list(builds[0].cpu_children)
                names = []
                while descendants:
                    event = descendants.pop()
                    names.append(event.name)
                    descendants.extend(event.cpu_children)
                self.assertNotIn("aten::_local_scalar_dense", names)
                expected_q = torch.cat([torch.zeros(1, dtype=torch.int32), query.cpu().cumsum(0).int()])
                expected_kv = torch.cat([torch.zeros(1, dtype=torch.int32), (query + prefix).cpu().cumsum(0).int()])
                torch.testing.assert_close(attention.cu_seqlens_device.cpu(), expected_q)
                torch.testing.assert_close(attention.cu_kv_seqlens_device.cpu(), expected_kv)
                self.assertEqual(attention.context_total_kv_length, int(expected_kv[-1]))

    def test_metadata_handles_mixed_length_tensor_residency(self) -> None:
        for input_device, prefix_device in (("cuda", "cpu"), ("cpu", "cuda"), ("cpu", "cpu")):
            with self.subTest(input_device=input_device, prefix_device=prefix_device):
                query = torch.tensor([2, 4, 3], dtype=torch.int32, device=input_device)
                prefix = torch.tensor([512, 1048560, 999990], dtype=torch.int32, device=prefix_device)
                if input_device == "cpu":
                    query = query.pin_memory()
                if prefix_device == "cpu":
                    prefix = prefix.pin_memory()
                attention = run_scenario(
                    CacheStoreForwardModel(),
                    "metadata",
                    {"input_lengths": query, "prefix_lengths": prefix, "total_tokens": 9},
                )["attention"]
                torch.testing.assert_close(attention.cu_seqlens_device.cpu(), torch.tensor([0, 2, 6, 9], dtype=torch.int32))
                expected = torch.tensor([0, 514, 1049078, 2049071], dtype=torch.int32)
                torch.testing.assert_close(attention.cu_kv_seqlens_device.cpu(), expected)
                self.assertEqual(attention.context_total_kv_length, int(expected[-1]))

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
        self.assertEqual(
            {block["length"] for block in linear_blocks.values()}, {24}
        )
        for token_key in range(3101, 3107):
            self.assertTrue(
                any(
                    f"_token_id_str_{token_key}_" in key
                    for key in linear_blocks
                )
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

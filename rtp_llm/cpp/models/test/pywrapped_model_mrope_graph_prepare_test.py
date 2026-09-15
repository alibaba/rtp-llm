import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    run_mrope_graph_preparation,
)


class MetadataGraphModel:
    """Expose real graph-held metadata in its output, without attention math."""

    def __init__(self) -> None:
        self.forward_calls = 0

    def initialize(self, resources) -> bool:
        # Exercise the actual MHA view construction, so a correctly named
        # opaque/byte-only cache cannot silently stand in for this fixture.
        assert list(resources.kv_cache.group_tags) == ["full"]
        cache = resources.kv_cache.get_layer_cache(0, "full")
        assert cache.tag == "full"
        assert cache.seq_size_per_block == 64
        assert cache.kv_cache_base.dtype == torch.float32
        assert cache.kv_cache_base.is_cuda
        assert tuple(cache.kv_cache_base.shape) == (8, 2, 1, 64, 4)
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        # Capture owns max-batch position capacity, which may exceed this graph's
        # token capacity (five rows for the four accepted tokens in this test).
        rows = inputs.input_ids.numel()
        positions = inputs.combo_position_ids.reshape(-1, 3)[:rows].float()
        # One cache group uses PyAttentionInputs directly; only multi-group
        # models expose a tag mapping (in both warmup and captured replay).
        attention = inputs.attention_inputs
        assert not isinstance(attention, dict)
        lengths = attention.input_lengths_device
        block = attention.kv_cache_kernel_block_id_device[0, 0]
        # Order-sensitive: [1,3] and [3,1] have the same total token count.
        signature = (lengths[0] * 16 + lengths[1] + block * 256).float()
        hidden = torch.cat((positions, signature.expand(rows, 1)), dim=1)
        return PyModelOutputs(hidden)


class PyWrappedModelMropeGraphPrepareTest(unittest.TestCase):
    def test_compacted_positions_reach_prepared_graph_and_replay(self) -> None:
        results = run_mrope_graph_preparation(MetadataGraphModel())
        # Independent token-major [T,H,W] oracle; request1 starts at original
        # row5, never at the row1 that a stale dense first-N copy would use.
        expected_positions = [
            [[0, 70, 140], [300, 370, 440], [301, 371, 441], [302, 372, 442]],
            [
                [1000, 1070, 1140],
                [1001, 1071, 1141],
                [1002, 1072, 1142],
                [1300, 1370, 1440],
            ],
        ]
        expected_lengths = [[1, 3], [3, 1]]
        self.assertEqual(len(results), 2)
        for round_index, result in enumerate(results):
            with self.subTest(round=round_index):
                self.assertTrue(result["prepared_before_forward"])
                self.assertEqual(result["attention_group_count"], 1)
                self.assertTrue(result["legacy_attention_inputs"])
                self.assertEqual(result["graph_key"], 5)
                self.assertEqual(result["token_count"], 4)
                self.assertEqual(result["capture_token_capacity"], 5)
                self.assertEqual(result["capture_position_capacity"], 30)
                self.assertEqual(
                    result["positions_before_forward"].tolist(),
                    expected_positions[round_index],
                )
                self.assertEqual(
                    result["lengths_before_forward"].tolist(),
                    expected_lengths[round_index],
                )
                self.assertEqual(result["prefixes_before_forward"].tolist(), [63, 127])
                block_id = round_index + 3
                self.assertEqual(result["block_before_forward"], block_id)
                first, second = expected_lengths[round_index]
                signature = first * 16 + second + block_id * 256
                expected = torch.tensor(
                    [row + [signature] for row in expected_positions[round_index]],
                    dtype=torch.float32,
                )
                self.assertEqual(tuple(result["output"].shape), (4, 4))
                torch.testing.assert_close(result["output"], expected, rtol=0, atol=0)
                # A normal Python fallback could produce the right values while
                # masking the missing early graph preparation being regressed.
                self.assertEqual(result["python_forward_delta"], 0)
                self.assertFalse(result["prepared_after_forward"])


if __name__ == "__main__":
    unittest.main()

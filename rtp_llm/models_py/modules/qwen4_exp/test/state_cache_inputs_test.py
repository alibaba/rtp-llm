import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.state_cache_inputs import (
    prepare_state_cache_inputs,
    state_page_table,
)
from rtp_llm.models_py.utils.cache_payload_view import same_tensor_metadata


class _CacheGeometry:
    def __init__(self, factors):
        self.factors = factors

    def get_seq_size_per_block(self, tag):
        return 128 * self.factors[tag]

    def get_kernel_seq_size_per_block(self, tag):
        return 128


def _expand(table, factor):
    offsets = torch.arange(factor, dtype=torch.int32, device=table.device)
    expanded = table[..., None] * factor + offsets
    return torch.where(table[..., None] > 0, expanded, 0).flatten(1).contiguous()


class StateCacheInputsTest(unittest.TestCase):
    def test_equal_metadata_is_read_once_and_inplace_changes_are_rechecked(self):
        left = torch.tensor([1, 2], dtype=torch.int32)
        right = left.clone()
        cache = {}
        with patch("torch.equal", wraps=torch.equal) as equal:
            self.assertTrue(same_tensor_metadata(left, right, cache=cache))
            self.assertTrue(same_tensor_metadata(left, right, cache=cache))
            equal.assert_called_once()
            right[0] = 3
            self.assertFalse(same_tensor_metadata(left, right, cache=cache))
            self.assertEqual(equal.call_count, 2)

    def test_inference_metadata_is_rechecked_with_each_forward_cache(self):
        with torch.inference_mode():
            left = torch.tensor([1, 2], dtype=torch.int32)
            right = left.clone()
            self.assertTrue(same_tensor_metadata(left, right, cache={}))
            right[0] = 3
            self.assertFalse(same_tensor_metadata(left, right, cache={}))

    def test_tag_local_geometry_and_metadata(self):
        factors = {PLE_STATE_TAG: 1, PLE_NGRAM_CTX_TAG: 2, INDEXER_STATE_TAG: 4}
        groups = {}
        expected = {}
        lengths = torch.tensor([1, 1], dtype=torch.int32)
        for index, (tag, factor) in enumerate(factors.items()):
            table = torch.tensor(
                [[5 + index, 7 + index], [0, 32768]], dtype=torch.int32
            )
            expected[tag] = table
            groups[tag] = SimpleNamespace(
                is_cuda_graph=True,
                input_lengths_device=lengths,
                kv_cache_kernel_block_id_device=_expand(table, factor),
                kv_cache_block_id_device=torch.full_like(table, 999),
                kv_cache_block_id=torch.full_like(table, 998),
            )
        attention = SimpleNamespace(is_cuda_graph=True)
        groups["full"] = attention
        views = prepare_state_cache_inputs(groups, _CacheGeometry(factors))
        self.assertIs(views["full"], attention)
        for tag, factor in factors.items():
            with self.subTest(tag=tag):
                torch.testing.assert_close(
                    views[tag].kv_cache_block_id_device, expected[tag]
                )
                self.assertTrue(views[tag].kv_cache_block_id_device.is_contiguous())
                self.assertIsNone(views[tag].kv_cache_block_id)
                self.assertIs(views[tag].input_lengths_device, lengths)
                self.assertEqual(int(groups[tag].kv_cache_block_id_device[0, 0]), 999)
                if factor == 1:
                    self.assertIs(
                        views[tag].kv_cache_block_id_device,
                        groups[tag].kv_cache_kernel_block_id_device,
                    )

    def test_eager_inputs_retain_native_identity_and_host_mirrors(self):
        groups = {
            PLE_STATE_TAG: SimpleNamespace(is_cuda_graph=False),
            INDEXER_STATE_TAG: SimpleNamespace(is_cuda_graph=False),
        }
        self.assertIs(prepare_state_cache_inputs(groups, object()), groups)
        self.assertIs(prepare_state_cache_inputs(groups, None), groups)

    def test_common_padding_width_and_strided_sources(self):
        for factor in (2, 4, 8):
            table = torch.tensor([[0, 5, 7], [32768, 9, 0]], dtype=torch.int32)
            kernel = torch.nn.functional.pad(_expand(table, factor), (0, factor - 1))
            storage = torch.zeros((2, kernel.shape[1] * 2), dtype=torch.int32)
            storage[:, ::2] = kernel
            result = state_page_table(storage[:, ::2], factor)
            torch.testing.assert_close(result, table)
            self.assertTrue(result.is_contiguous())

    def test_invalid_geometry_and_table_rejected(self):
        for table, factor in (
            (torch.ones(2, 3), 1),
            (torch.ones(2, dtype=torch.int32), 1),
            (torch.empty(2, 0, dtype=torch.int32), 1),
            (torch.ones(2, 3, dtype=torch.int32), 0),
            (torch.ones(2, 1, dtype=torch.int32), 2),
        ):
            with self.subTest(shape=table.shape, factor=factor):
                with self.assertRaises(RuntimeError):
                    state_page_table(table, factor)
        cache = _CacheGeometry({PLE_STATE_TAG: 1})
        cache.get_kernel_seq_size_per_block = lambda tag: 96
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            prepare_state_cache_inputs(
                {PLE_STATE_TAG: SimpleNamespace(is_cuda_graph=True)}, cache
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph requires a GPU")
    def test_graph_views_refresh_after_mapping_changes_and_batch_shrink(self):
        for factor in (1, 2, 4, 8):
            with self.subTest(factor=factor):
                source = SimpleNamespace(
                    is_cuda_graph=True,
                    kv_cache_kernel_block_id_device=torch.zeros(
                        (3, 3 * factor), device="cuda", dtype=torch.int32
                    ),
                )
                groups = {PLE_STATE_TAG: source}
                cache = _CacheGeometry({PLE_STATE_TAG: factor})
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    prepare_state_cache_inputs(groups, cache)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    retained = prepare_state_cache_inputs(groups, cache)[PLE_STATE_TAG]
                    output = retained.kv_cache_block_id_device.clone()
                # A transaction may outlive the local mapping used during capture.
                del groups
                gc.collect()
                for batch, width, value in ((3, 3, 5), (1, 1, 7), (2, 2, 11)):
                    physical = torch.zeros((3, 3), device="cuda", dtype=torch.int32)
                    physical[:batch, :width].fill_(value)
                    source.kv_cache_kernel_block_id_device.copy_(
                        _expand(physical, factor)
                    )
                    graph.replay()
                    torch.testing.assert_close(output, physical)
                    torch.testing.assert_close(
                        retained.kv_cache_block_id_device, physical
                    )


if __name__ == "__main__":
    unittest.main()

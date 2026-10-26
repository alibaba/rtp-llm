"""Explicit SM103 coverage for MiniMax M3.1 NVFP4 kernels."""

import unittest

import torch

from rtp_llm.models.test import minimax_m31_kv_cache_test as kv

_GPU_KV_TESTS = (
    "test_target_verify_uses_grouped_index_score_only_on_supported_shape",
    "test_grouped_index_score_matches_row_scores_after_graph_replay",
    "test_cp_prefill_tiled_index_score_matches_row_scores",
    "test_cp_prefix_cuda_page_copy_matches_tag_local_table",
    "test_cp_native_prefill_matches_page_reader",
    "test_fp32_prefill_router_preserves_logits_and_topk",
    "test_sparse_topk_preserves_ties_and_invalid_pages",
)


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("MiniMax M3.1 GPU tests require an SM103 worker")
    suite = unittest.TestSuite()
    for name in _GPU_KV_TESTS:
        suite.addTest(kv.MiniMaxM31KVCacheTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()

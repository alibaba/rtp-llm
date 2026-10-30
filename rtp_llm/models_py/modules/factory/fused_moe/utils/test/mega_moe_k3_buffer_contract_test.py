"""Check K3 versus generic buffer protocols without CUDA or RTP bindings."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_buffer_module():
    path = Path(__file__).resolve().parents[1] / "mega_moe" / "buffer.py"
    spec = importlib.util.spec_from_file_location("mega_moe_buffer_contract", path)
    module = importlib.util.module_from_spec(spec)
    # These tests exercise allocation dispatch and size metadata. Tensor
    # operations are deliberately outside this CPU-only contract boundary.
    with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
        spec.loader.exec_module(module)
    return module


class _Group:
    def size(self):
        return 8


class _Backend:
    def __init__(self):
        self.calls = []
        self.size_calls = []
        self._C = types.SimpleNamespace(
            get_token_alignment_for_mega_moe_k3=lambda: 16,
            get_symm_buffer_size_for_mega_moe_k3=self._k3_size,
            get_symm_buffer_size_for_mega_moe=self._generic_size,
        )

    def _k3_size(self, ranks, experts, capacity, topk, hidden, inter, mma, activation):
        self.size_calls.append(("k3", capacity, mma, activation))
        return (capacity * hidden, None)

    def _generic_size(
        self, ranks, experts, capacity, topk, hidden, inter, fp8, activation
    ):
        self.size_calls.append(("generic", capacity, fp8, activation))
        return (capacity * hidden, None)

    # Strict signatures catch accidentally passing the generic bool dispatch
    # argument to the separate K3 API, or vice versa.
    def get_symm_buffer_for_mega_moe_k3(
        self,
        *,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        mma_type,
        activation,
    ):
        self.calls.append(("k3", num_max_tokens_per_rank, mma_type, activation))
        return object()

    def get_symm_buffer_for_mega_moe(
        self,
        *,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
    ):
        self.calls.append(
            ("generic", num_max_tokens_per_rank, use_fp8_dispatch, activation)
        )
        return object()


class K3BufferContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_buffer_module()
        self.backend = _Backend()
        self.group = _Group()
        self.shape = dict(
            group=self.group,
            num_experts=384,
            num_max_tokens_per_rank=17,
            num_topk=8,
            hidden=3584,
            intermediate_hidden=3072,
            use_fp8_dispatch=True,
            backend=self.backend,
        )

    def test_k3_uses_its_factory_and_aligned_size_query(self):
        first = self.module._get_or_create_mega_buf(**self.shape, activation="situ")
        reused = self.module._get_or_create_mega_buf(**self.shape, activation="situ")
        self.assertIs(first, reused)
        self.assertEqual(self.backend.calls, [("k3", 17, "fp8xfp4", "situ")])
        self.assertEqual(self.backend.size_calls, [("k3", 32, "fp8xfp4", "situ")])

    def test_k3_cannot_reuse_a_legacy_generic_buffer(self):
        legacy_key = (
            id(self.group),
            384,
            17,
            8,
            3584,
            3072,
            True,
            "situ",
            self.backend,
        )
        legacy_buffer = object()
        self.module._MEGA_BUF_CACHE[legacy_key] = legacy_buffer
        actual = self.module._get_or_create_mega_buf(**self.shape, activation="situ")
        self.assertIsNot(actual, legacy_buffer)
        self.assertEqual(self.backend.calls[0][0], "k3")

    def test_generic_api_and_cache_remain_independent(self):
        generic = self.module._get_or_create_mega_buf(**self.shape, activation="swiglu")
        k3 = self.module._get_or_create_mega_buf(**self.shape, activation="situ")
        self.assertIsNot(generic, k3)
        self.assertEqual(
            self.backend.calls,
            [
                ("generic", 17, True, "swiglu"),
                ("k3", 17, "fp8xfp4", "situ"),
            ],
        )
        self.assertEqual(self.backend.size_calls[0], ("generic", 17, True, "swiglu"))

    def test_missing_k3_api_does_not_fall_back_to_generic_layout(self):
        backend = types.SimpleNamespace(
            _C=self.backend._C,
            get_symm_buffer_for_mega_moe=self.backend.get_symm_buffer_for_mega_moe,
        )
        shape = dict(self.shape, backend=None)
        with patch.dict(sys.modules, {"deep_gemm": backend}):
            with self.assertRaisesRegex(
                AttributeError, "get_symm_buffer_for_mega_moe_k3"
            ):
                self.module._get_or_create_mega_buf(**shape, activation="situ")
        self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main()

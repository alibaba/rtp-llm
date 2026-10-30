"""CPU contract checks for legacy FP8 metadata and SM100 dispatch."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_contract(backend):
    # Load the production dispatch without importing CUDA/Triton model modules.
    source = Path(__file__).resolve().parents[1] / "_indexer_score.py"
    tree = ast.parse(source.read_text())
    names = {
        "get_fp8_paged_mqa_logits_metadata",
        "_sm100_needs_triton_fallback",
        "deepgemm_supports_block_kv",
        "supports_paged_mqa_logits",
        "supports_mqa_logits",
        "fp8_paged_indexer_score",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            or isinstance(node, ast.FunctionDef)
            and node.name in names
            or isinstance(node, ast.Try)
        )
    ]
    namespace = {
        "torch": SimpleNamespace(
            float8_e4m3fn="fp8",
            float32="fp32",
            uint8="uint8",
            int32="int32",
            cuda=SimpleNamespace(get_device_capability=lambda device: (10, 0)),
        ),
        "INDEXER_HEAD_DIM": 128,
        "INDEXER_ENTRY_BYTES": 132,
        "_get_num_sms": lambda device: 148,
    }
    with patch.dict(sys.modules, {"deep_gemm": backend}):
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
            namespace,
        )
    return SimpleNamespace(**namespace)


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "cuda"

    def dim(self):
        return len(self.shape)

    def contiguous(self):
        return self

    def view(self, *shape):
        return _Tensor(shape, self.dtype)


class LegacyMqaMetadataTest(unittest.TestCase):
    @staticmethod
    def _backend(version, *, legacy=False):
        backend = SimpleNamespace(
            __version__=version,
            fp8_mqa_logits=Mock(),
            fp8_paged_mqa_logits=Mock(return_value="scores"),
            get_paged_mqa_logits_metadata=Mock(return_value="modern-schedule"),
        )
        if legacy:
            backend.get_legacy_paged_mqa_logits_metadata = Mock(
                return_value="legacy-schedule"
            )
        return backend

    def test_new_wheel_uses_legacy_schedule_in_production_score(self):
        backend = self._backend("2.8.2+a5165ad.cu132", legacy=True)
        contract = _load_contract(backend)
        self.assertTrue(contract.supports_mqa_logits("cuda"))
        self.assertTrue(contract.supports_paged_mqa_logits("cuda", 64))
        lengths = _Tensor((2, 1), "int32")
        result = contract.fp8_paged_indexer_score(
            _Tensor((2, 1, 32, 128), "fp8"),
            _Tensor((2, 32), "fp32"),
            _Tensor((256, 132), "uint8"),
            _Tensor((2, 2), "int32"),
            lengths,
            64,
            128,
        )
        self.assertEqual(result, "scores")
        backend.get_legacy_paged_mqa_logits_metadata.assert_called_once_with(
            lengths, 64, 148
        )
        backend.get_paged_mqa_logits_metadata.assert_not_called()
        self.assertEqual(
            backend.fp8_paged_mqa_logits.call_args.args[5], "legacy-schedule"
        )

    def test_old_wheel_keeps_shared_schedule(self):
        backend = self._backend("2.8.0+122e18b.cu132")
        contract = _load_contract(backend)
        self.assertTrue(contract.supports_mqa_logits("cuda"))
        self.assertTrue(contract.supports_paged_mqa_logits("cuda", 64))
        self.assertEqual(
            contract.get_fp8_paged_mqa_logits_metadata(
                backend, "lengths", 64, 148, indices="request-ids"
            ),
            "modern-schedule",
        )
        backend.get_paged_mqa_logits_metadata.assert_called_once_with(
            "lengths", 64, 148, indices="request-ids"
        )

    def test_mx_only_wheel_retains_sm100_fallback(self):
        contract = _load_contract(self._backend("2.8.1"))
        self.assertFalse(contract.supports_mqa_logits("cuda"))
        self.assertFalse(contract.supports_paged_mqa_logits("cuda", 64))

    def test_noncallable_legacy_attribute_does_not_enable_native_path(self):
        backend = self._backend("2.8.2")
        backend.get_legacy_paged_mqa_logits_metadata = None
        contract = _load_contract(backend)
        self.assertFalse(contract.supports_mqa_logits("cuda"))
        self.assertEqual(
            contract.get_fp8_paged_mqa_logits_metadata(backend, "lengths", 64, 148),
            "modern-schedule",
        )

    def test_native_page_sizes_match_architecture_contract(self):
        contract = _load_contract(self._backend("2.8.2", legacy=True))
        for major, expected in (
            (8, set()),
            (9, {64}),
            (10, {32, 64, 128}),
            (12, set()),
        ):
            contract.torch.cuda.get_device_capability = lambda device: (major, 0)
            self.assertEqual(contract.supports_mqa_logits("cuda"), major in (9, 10))
            self.assertEqual(
                contract.supports_paged_mqa_logits("cuda"), major in (9, 10)
            )
            for page in (16, 32, 64, 128, 256):
                with self.subTest(major=major, page=page):
                    self.assertEqual(
                        contract.deepgemm_supports_block_kv("cuda", page),
                        page in expected,
                    )

    def test_vendor_wheel_preserves_existing_page_gates(self):
        contract = _load_contract(self._backend("1.0.0+ppu2.2.0"))
        contract.torch.cuda.get_device_capability = lambda device: (8, 9)
        self.assertTrue(contract.supports_mqa_logits("cuda"))
        self.assertTrue(contract.supports_paged_mqa_logits("cuda"))
        self.assertTrue(contract.deepgemm_supports_block_kv("cuda", 128))
        self.assertFalse(contract.deepgemm_supports_block_kv("cuda", 64))


if __name__ == "__main__":
    unittest.main()

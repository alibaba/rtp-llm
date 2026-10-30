"""The real FP8 MLA adapter must not require wheel metadata for logging."""

import builtins
import unittest
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.kimi_k3 import flashinfer_prefill


class FlashinferSourceInstallIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires the CUDA13 runtime")
        flashinfer_prefill._load_backend.cache_clear()

    def tearDown(self):
        flashinfer_prefill._load_backend.cache_clear()

    def test_importable_backend_without_distribution_metadata_runs_prefill(self):
        # Load the actual pinned backend. Only its optional logging lookup is
        # fault-injected; there is no fake module or replacement attention kernel.
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        with mock.patch.object(
            flashinfer_prefill,
            "version",
            side_effect=PackageNotFoundError("flashinfer-python"),
        ) as lookup:
            # K3 TP8 has 96 / 8 = 12 expanded heads at the FP8 Prefill boundary.
            adapter = flashinfer_prefill.FlashinferMlaPrefill(12, 192**-0.5)
        lookup.assert_called_once_with("flashinfer-python")
        self.assertIs(adapter._run, trtllm_ragged_attention_deepseek)
        self._assert_one_key_prefill(adapter)

    def test_installed_backend_runs_prefill(self):
        adapter = flashinfer_prefill.FlashinferMlaPrefill(12, 192**-0.5)
        self._assert_one_key_prefill(adapter)

    def test_rtp_fia_and_prefill_do_not_import_removed_backend(self):
        original_import = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "tokenspeed_mla" or name.startswith("tokenspeed_mla."):
                raise AssertionError("Removed MLA package must not be imported")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=importing):
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_fia2a_backend import (
                PageRRFIA2AMLABackend,
            )

            self.assertEqual(
                PageRRFIA2AMLABackend.__module__,
                "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_fia2a_backend",
            )
            adapter = flashinfer_prefill.FlashinferMlaPrefill(12, 192**-0.5)
            self._assert_one_key_prefill(adapter)

    def _assert_one_key_prefill(self, adapter):
        indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        params = SimpleNamespace(
            q_tokens=1,
            kv_tokens=1,
            qo_indptr_d=indptr,
            kv_lens_d=torch.ones(1, dtype=torch.int32, device="cuda"),
            prefill_ragged_kv_len_indptr_d=indptr,
            q_lens_host=[1],
            max_q=1,
            max_k=1,
        )
        q = torch.zeros((1, 12, 192), dtype=torch.float8_e4m3fn, device="cuda")
        k = torch.zeros_like(q)
        v = torch.full((1, 12, 128), 2, dtype=torch.float8_e4m3fn, device="cuda")
        actual = adapter.run(q, k, v, params)
        self.assertEqual(actual.shape, (1, 12, 128))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, torch.full_like(actual, 2), rtol=0, atol=0)

    def test_backend_import_error_still_fails_before_version_lookup(self):
        original_import = builtins.__import__
        failure = ImportError("backend dependency is unavailable")

        def importing(name, *args, **kwargs):
            if name == "flashinfer.prefill":
                raise failure
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=importing):
            with mock.patch.object(flashinfer_prefill, "version") as lookup:
                with self.assertRaisesRegex(
                    ImportError, "pinned FlashInfer ragged TRTLLM API"
                ) as error:
                    flashinfer_prefill.FlashinferMlaPrefill(12, 192**-0.5)
        self.assertIs(error.exception.__cause__, failure)
        lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()

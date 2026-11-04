import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.prefill_preflight import (
    main_geometry_preflight,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_validation


class PrefillPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

    def fixture(self, *, padded=False):
        table = torch.tensor([[1, 2, 3, 0]], device="cuda", dtype=torch.int32)
        cache = torch.zeros(5, 2, 1, 4, 16, device="cuda", dtype=torch.bfloat16)
        bases = torch.tensor([4], dtype=torch.int32)
        lengths = torch.tensor([4], dtype=torch.int32)
        impl = SimpleNamespace(
            attn_inputs=SimpleNamespace(
                is_s_padded=padded, kv_cache_kernel_block_id_device=table
            ),
            attn_configs=SimpleNamespace(kernel_tokens_per_block=4),
            head_dim=16,
            kv_head_num=1,
        )
        return impl, cache, bases, lengths

    def validate(self, entry):
        impl, cache, bases, lengths = entry
        return fmha_validation._validate_main_paged_geometry(
            impl,
            cache,
            impl.attn_inputs.kv_cache_kernel_block_id_device,
            bases,
            lengths,
            cache.device,
        )

    def test_receipts_are_forward_scoped_and_metadata_mutation_is_rechecked(self):
        entries = [self.fixture(), self.fixture()]
        with main_geometry_preflight(entries):
            with patch.object(
                fmha_validation,
                "_try_validate_main_paged_geometry",
                side_effect=AssertionError("redundant scan"),
            ):
                for entry in entries:
                    self.assertEqual(self.validate(entry), 4)
            entries[0][0].attn_inputs.kv_cache_kernel_block_id_device[0, 1] = 0
            with self.assertRaisesRegex(RuntimeError, "unallocated"):
                self.validate(entries[0])
        self.assertTrue(
            all(entry[0]._model_prefill_geometry is None for entry in entries)
        )
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            with main_geometry_preflight(entries):
                self.fail("Invalid live metadata entered the model body")

    def test_bad_unused_page_is_rejected_before_any_receipt_is_published(self):
        entries = [self.fixture(), self.fixture()]
        entries[1][0].attn_inputs.kv_cache_kernel_block_id_device[0, -1] = 5
        with self.assertRaisesRegex(RuntimeError, "out-of-range physical block"):
            with main_geometry_preflight(entries):
                self.fail("Invalid page entered the model body")
        for impl, *_ in entries:
            self.assertIsNone(getattr(impl, "_model_prefill_geometry", None))

    def test_receipt_cleanup_after_model_failure_and_new_host_lengths(self):
        entry = self.fixture()
        with self.assertRaisesRegex(ValueError, "model failure"):
            with main_geometry_preflight([entry]):
                raise ValueError("model failure")
        self.assertIsNone(entry[0]._model_prefill_geometry)
        entry[2].fill_(16)
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            with main_geometry_preflight([entry]):
                self.fail("New host lengths must be validated")

    def test_empty_padded_rows_keep_the_existing_geometry_semantics(self):
        entry = self.fixture(padded=True)
        entry[0].attn_inputs.kv_cache_kernel_block_id_device.zero_()
        entry[2].fill_(128)
        with main_geometry_preflight([entry]):
            self.assertEqual(self.validate(entry), 4)

    def test_native_inference_tensors_are_validated_fresh_for_each_forward(self):
        with torch.inference_mode():
            entry = self.fixture()
            with main_geometry_preflight([entry]):
                with patch.object(
                    fmha_validation,
                    "_try_validate_main_paged_geometry",
                    side_effect=AssertionError("inference receipt was bypassed"),
                ):
                    self.assertEqual(self.validate(entry), 4)
            self.assertIsNone(entry[0]._model_prefill_geometry)
            entry[0].attn_inputs.kv_cache_kernel_block_id_device[0, 1] = 0
            with self.assertRaisesRegex(RuntimeError, "unallocated"):
                with main_geometry_preflight([entry]):
                    self.fail("Stale inference metadata survived a forward")


if __name__ == "__main__":
    unittest.main()

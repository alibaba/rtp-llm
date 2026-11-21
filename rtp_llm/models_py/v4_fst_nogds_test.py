import unittest
from unittest.mock import Mock

import torch

from rtp_llm.models_py.v4_fst_nogds import _CheckedReader
from rtp_llm.models_py.v41_fst_file_pool import v41_file_staging_loader


class CompletionTest(unittest.TestCase):
    def run_case(self, result):
        raw = Mock()
        raw.submit_read.return_value = 4
        raw.wait_read.return_value = result
        transfer = Mock(return_value=107)
        gbuf = Mock()
        gbuf.get_base_address.return_value = 100
        reader = _CheckedReader(raw, "device", transfer)
        token = reader.submit_read(3, gbuf, 4096, 20, 7)
        return reader, token, transfer, gbuf

    def test_success_is_never_recopied(self):
        reader, token, transfer, _ = self.run_case(107)
        self.assertEqual(reader.wait_read(token), 107)
        transfer.assert_not_called()

    def test_null_repairs_exact_extent(self):
        reader, token, transfer, gbuf = self.run_case(0)
        self.assertEqual(reader.wait_read(token), 107)
        transfer.assert_called_once_with(3, gbuf, 4096, 20, 7, "device")

    def test_unexpected_result_or_failed_retry_cannot_publish(self):
        reader, token, transfer, _ = self.run_case(9)
        with self.assertRaises(OSError):
            reader.wait_read(token)
        transfer.assert_not_called()
        for err in (0, OSError("short read")):
            reader, token, transfer, _ = self.run_case(0)
            if isinstance(err, Exception):
                transfer.side_effect = err
            else:
                transfer.return_value = err
            with self.assertRaises(OSError):
                reader.wait_read(token)

    def test_scope(self):
        class Raw:
            def __init__(self, **opts):
                self.loader = Mock()
                self.loader.copier_constructor = self.make = Mock()

        for model, dev, vendor, nogds, shm, protect in (
            ("deepseek_v4", "cuda", "Cuda", True, False, True),
            ("deepseek_v4_dspark", "cuda", "Cuda", True, False, True),
            ("qwen3", "cuda", "Cuda", True, False, False),
            ("deepseek_v4", "cpu", "Cpu", True, False, False),
            ("deepseek_v4", "cuda", "ROCm", True, False, False),
            ("deepseek_v4", "cuda", "Cuda", False, False, False),
            ("deepseek_v4", "cuda", "Cuda", True, True, False),
        ):
            cls = v41_file_staging_loader(
                Raw, model, torch.device(dev), cuda_backend=vendor, use_shm=shm
            )
            feeder = cls(nogds=nogds, use_shm=shm)
            self.assertEqual(
                feeder.loader.copier_constructor is not feeder.make, protect
            )


if __name__ == "__main__":
    unittest.main()

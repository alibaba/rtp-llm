import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.v4_fst_nogds import _CheckedReader
from rtp_llm.models_py.v41_fst_file_pool import cuda_file_loader


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
            ("qwen3", "cuda", "Cuda", True, False, True),
            ("deepseek_v4_mtp", "cuda", "Cuda", True, False, True),
            ("deepseek_v41_dspark", "cuda", "Cuda", True, False, True),
            ("deepseek_v4", "cpu", "Cpu", True, False, False),
            ("deepseek_v4", "cuda", "ROCm", True, False, False),
            ("deepseek_v4", "cuda", "Cuda", False, False, False),
            ("deepseek_v4", "cuda", "Cuda", True, True, False),
        ):
            cls = cuda_file_loader(
                Raw,
                model,
                torch.device(dev),
                cuda_backend=vendor,
                use_shm=shm,
                nogds=nogds,
            )
            feeder = cls(nogds=nogds, use_shm=shm)
            self.assertEqual(
                feeder.loader.copier_constructor is not feeder.make, protect
            )

    def test_staged_v41_checks_every_producer_including_failed_extent(self):
        class Raw:
            def __init__(self, queue_size=0, **opts):
                self.q = queue_size
                self.loaders = [Mock(), Mock()]

            def _load_single_batch(self):
                pass

            def _consumer_worker(self):
                yield from ()

        with (
            patch.object(torch.cuda, "MemPool", create=True),
            patch.object(torch.cuda, "use_mem_pool", create=True),
            patch.object(torch.cuda, "get_allocator_backend", return_value="native", create=True),
        ):
            cls = cuda_file_loader(
                Raw,
                "deepseek_v41",
                torch.device("cuda"),
                cuda_backend="Cuda",
                use_shm=False,
                nogds=True,
            )
            feeder = cls(nogds=True, use_shm=False)
        self.assertEqual(feeder.q, -1)
        for producer in feeder.loaders:
            with patch(
                "rtp_llm.models_py.v4_fst_nogds._transfer", return_value=107
            ) as retry:
                copier = producer.copier_constructor(None, "device", Mock())
                reader = copier.reader
                reader.reader.submit_read.return_value = 4
                reader.reader.wait_read.return_value = 0
                buf = Mock()
                buf.get_base_address.return_value = 100
                token = reader.submit_read(3, buf, 4096, 20, 7)
                self.assertEqual(reader.wait_read(token), 107)
                retry.assert_called_once_with(3, buf, 4096, 20, 7, "device")


if __name__ == "__main__":
    unittest.main()

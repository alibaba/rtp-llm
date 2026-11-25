import gc
import multiprocessing
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

import rtp_llm.ops  # Loads the native config module.
from rtp_llm.multimodal import preprocess_ipc
from rtp_llm.multimodal.mm_process_engine import _worker_initializer


def _make_views():
    values = torch.arange(18, dtype=torch.int32).reshape(3, 6)
    return values, values[1:, ::2], torch.empty(0, dtype=torch.bfloat16)


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and torch.multiprocessing.get_sharing_strategy() == "file_descriptor",
    "requires PyTorch file descriptor sharing on Linux",
)
class PreprocessIpcTest(unittest.TestCase):
    def test_spawn_worker_preserves_tensor_views_after_worker_exit(self):
        from libth_transformer_config import map_preprocess_storage

        with patch(
            "libth_transformer_config.map_preprocess_storage",
            wraps=map_preprocess_storage,
        ) as mapper:
            with multiprocessing.get_context("spawn").Pool(
                1, initializer=_worker_initializer, initargs=(None, {}, _make_views)
            ) as pool:
                values, view, empty = pool.apply_async(_make_views).get(40)

        mapper.assert_called_once()
        self.assertTrue(values.is_shared())
        self.assertEqual(
            values.untyped_storage().data_ptr(), view.untyped_storage().data_ptr()
        )
        self.assertEqual(view.dtype, torch.int32)
        self.assertEqual(view.stride(), (6, 2))
        self.assertEqual(view.storage_offset(), 6)
        self.assertEqual(view.tolist(), [[6, 8, 10], [12, 14, 16]])
        values[1, 2] = 99
        self.assertEqual(view[0, 1].item(), 99)
        self.assertEqual(empty.dtype, torch.bfloat16)
        self.assertEqual(empty.numel(), 0)

    def test_rebuild_owns_cached_mapping_and_closes_each_transferred_fd(self):
        with tempfile.TemporaryFile() as backing:
            backing.write(b"test")
            backing.flush()
            transferred = [os.dup(backing.fileno()), os.dup(backing.fileno())]
            first, second = [
                preprocess_ipc._rebuild_storage_fd(
                    torch.UntypedStorage, Mock(detach=Mock(return_value=fd)), 4
                )
                for fd in transferred
            ]
            owned_fd = first._get_shared_fd()
            self.assertFalse(os.get_inheritable(owned_fd))
            self.assertEqual(os.fstat(owned_fd), os.fstat(backing.fileno()))
            self.assertEqual(first.data_ptr(), second.data_ptr())
            self.assertEqual(bytes(first), b"test")
            for fd in transferred:
                with self.assertRaises(OSError):
                    os.fstat(fd)

            del first, second
            gc.collect()
            with self.assertRaises(OSError):
                os.fstat(owned_fd)

    def test_oversized_storage_closes_fd_without_extending_backing_file(self):
        with tempfile.TemporaryFile() as backing:
            backing.write(b"test")
            backing.flush()
            transferred = os.dup(backing.fileno())
            with self.assertRaisesRegex(RuntimeError, "exceeds shared file size"):
                preprocess_ipc._rebuild_storage_fd(
                    torch.UntypedStorage,
                    Mock(detach=Mock(return_value=transferred)),
                    5,
                )
            with self.assertRaises(OSError):
                os.fstat(transferred)
            self.assertEqual(os.fstat(backing.fileno()).st_size, 4)

    def test_missing_native_mapper_uses_pytorch_reconstruction(self):
        with tempfile.TemporaryFile() as backing:
            backing.write(b"test")
            backing.flush()
            transferred = os.dup(backing.fileno())
            with patch.dict(sys.modules, {"libth_transformer_config": None}):
                storage = preprocess_ipc._rebuild_storage_fd(
                    torch.UntypedStorage,
                    Mock(detach=Mock(return_value=transferred)),
                    4,
                )
            self.assertEqual(bytes(storage), b"test")
            with self.assertRaises(OSError):
                os.fstat(transferred)


if __name__ == "__main__":
    unittest.main()

import multiprocessing
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.preprocess_ipc import (
    _rebuild_storage_fd,
    _reduce_storage,
    register_preprocess_storage_reducer,
)


def _make_views():
    data = torch.arange(24).reshape(4, 6)
    return data, data[1:, ::2], torch.empty(0, dtype=torch.bfloat16)


class PreprocessIpcTest(unittest.TestCase):
    def test_spawn_worker_preserves_shared_views_after_exit(self):
        import rtp_llm.ops  # Load the native config module as a ViT parent does.

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1, initializer=register_preprocess_storage_reducer) as pool:
            data, view, empty = pool.apply_async(_make_views).get(timeout=30)
        self.assertTrue(data.is_shared())
        self.assertEqual(view.stride(), (6, 2))
        self.assertEqual(view.storage_offset(), 6)
        torch.testing.assert_close(view, _make_views()[1])
        data[1, 2] = -1
        self.assertEqual(view[0, 1].item(), -1)
        self.assertEqual((empty.numel(), empty.dtype), (0, torch.bfloat16))

    def test_fd_reducer_uses_native_map_once_and_releases_received_fd(self):
        import rtp_llm.ops
        from libth_transformer_config import map_preprocess_storage

        with tempfile.TemporaryFile() as backing:
            backing.write(b"data")
            backing.flush()
            with patch(
                "libth_transformer_config.map_preprocess_storage",
                wraps=map_preprocess_storage,
            ) as map_fd:
                storages = []
                for _ in range(2):
                    fd = os.dup(backing.fileno())
                    handle = Mock()
                    handle.detach.return_value = fd
                    storages.append(
                        _rebuild_storage_fd(torch.UntypedStorage, handle, 4)
                    )
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                map_fd.assert_called_once()
                self.assertEqual(bytes(storages[0]), b"data")
                self.assertEqual(storages[0].data_ptr(), storages[1].data_ptr())
                rebuild, args = _reduce_storage(storages[0])
                self.assertIs(rebuild, _rebuild_storage_fd)
                self.assertEqual(rebuild(*args).data_ptr(), storages[0].data_ptr())


if __name__ == "__main__":
    unittest.main()

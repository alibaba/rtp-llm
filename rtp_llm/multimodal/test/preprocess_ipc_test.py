import multiprocessing
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.preprocess_ipc import (
    SharedPreprocessSources,
    _SourceSlice,
    _rebuild_storage_fd,
    _reduce_storage,
    pack_preprocess_tensors,
    read_preprocess_sources,
    register_preprocess_storage_reducer,
)


def _make_views():
    data = torch.arange(24).reshape(4, 6)
    return data, data[1:, ::2], torch.empty(0, dtype=torch.bfloat16)


def _make_packed():
    return pack_preprocess_tensors(
        (
            torch.arange(17, dtype=torch.uint8).reshape(1, 17),
            torch.arange(9, dtype=torch.int32).reshape(3, 3),
            torch.ones(5, dtype=torch.bfloat16),
        )
    )


_source_view = None


def _init_sources(buffer):
    global _source_view
    _source_view = memoryview(buffer).cast("B")
    register_preprocess_storage_reducer()


def _read_sources(sources):
    from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import (
        prepare_resize_tables,
    )

    return read_preprocess_sources(sources, _source_view), prepare_resize_tables(
        (32, 32), (16, 16), patch_size=16
    )


class PreprocessIpcTest(unittest.TestCase):
    def test_shared_source_outlives_slot_reuse_and_worker_exit(self):
        ctx = multiprocessing.get_context("spawn")
        with patch.object(SharedPreprocessSources, "_SLOTS", 1):
            arena = SharedPreprocessSources(ctx)
        source = b"a" * arena._MIN_BYTES
        replacement = b"b" * len(source)
        transported, lease = arena.prepare((source, b"small", None))
        self.assertIsInstance(transported[0], _SourceSlice)
        self.assertEqual(arena.prepare((replacement,)), ((replacement,), None))
        with patch.dict(os.environ, {"MM_VIT_SHARED_PREPROCESS_IPC": "1"}):
            os.environ.pop("MM_VIT_PACKED_PREPROCESS_IPC", None)
            with ctx.Pool(1, initializer=_init_sources, initargs=(arena.buffer,)) as pool:
                actual, tables = pool.apply_async(_read_sources, (transported,)).get(
                    timeout=30
                )
                arena.release(lease)
                next_sources, next_lease = arena.prepare((replacement,))
                self.assertEqual(next_sources, (transported[0],))
                arena.release(lease)
                self.assertEqual(arena.prepare((source,)), ((source,), None))
                arena.release(next_lease)
        expected, expected_tables = _read_sources((source, b"small", None))
        self.assertEqual(actual, expected)
        for actual_axis, expected_axis in (
            (tables.horizontal, expected_tables.horizontal),
            (tables.vertical, expected_tables.vertical),
        ):
            for actual_tensor, expected_tensor in zip(actual_axis, expected_axis):
                self.assertFalse(actual_tensor.is_shared())
                self.assertTrue(torch.equal(actual_tensor, expected_tensor))
        self.assertTrue(torch.equal(tables.normalization, expected_tables.normalization))

    def test_reset_ignores_late_release(self):
        ctx = multiprocessing.get_context("spawn")
        with patch.object(SharedPreprocessSources, "_SLOTS", 1):
            arena = SharedPreprocessSources(ctx)
        sources = (b"a" * arena._MIN_BYTES,)
        _, abandoned = arena.prepare(sources)
        arena.reset()
        _, current = arena.prepare(sources)
        arena.release(abandoned)
        self.assertEqual(arena.prepare(sources), (sources, None))
        arena.release(current)
        self.assertIsNotNone(arena.prepare(sources)[1])
        with self.assertRaises(RuntimeError):
            read_preprocess_sources((_SourceSlice(0, len(arena._view) + 1),), arena._view)

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

    def test_packed_mixed_dtypes_share_storage_after_worker_exit(self):
        import rtp_llm.ops

        if torch.multiprocessing.get_sharing_strategy() != "file_descriptor":
            self.skipTest("packed preprocessing requires file-descriptor sharing")
        ctx = multiprocessing.get_context("spawn")
        with patch.dict(os.environ, {"MM_VIT_PACKED_PREPROCESS_IPC": "1"}):
            with ctx.Pool(1, initializer=register_preprocess_storage_reducer) as pool:
                rgb, coeffs, lut = pool.apply_async(_make_packed).get(timeout=30)
        storages = {x.untyped_storage().data_ptr() for x in (rgb, coeffs, lut)}
        self.assertEqual(len(storages), 1)
        torch.testing.assert_close(rgb, torch.arange(17, dtype=torch.uint8)[None, :])
        torch.testing.assert_close(coeffs, torch.arange(9, dtype=torch.int32).reshape(3, 3))
        torch.testing.assert_close(lut, torch.ones(5, dtype=torch.bfloat16))

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

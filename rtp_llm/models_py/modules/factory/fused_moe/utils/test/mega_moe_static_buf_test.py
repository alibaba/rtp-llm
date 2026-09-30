"""Tests for the production MegaMoE output-buffer cache."""

import unittest
import weakref
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import (
    buffer as mega_buf,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _MEGA_OUTPUT_CACHE,
    _get_or_create_mega_output,
)


class MegaMoEStaticBufferTest(unittest.TestCase):
    def setUp(self):
        _MEGA_OUTPUT_CACHE.clear()

    def tearDown(self):
        _MEGA_OUTPUT_CACHE.clear()

    def test_reuses_production_cache_and_live_slice_storage(self):
        buf = _get_or_create_mega_output(64, 128, torch.bfloat16, torch.device("cpu"))
        reused = _get_or_create_mega_output(
            16, 128, torch.bfloat16, torch.device("cpu")
        )
        live = reused[:16]

        self.assertIs(reused, buf)
        self.assertEqual(live.shape, (16, 128))
        self.assertEqual(live.data_ptr(), buf.data_ptr())
        live.fill_(1)
        self.assertTrue(torch.all(buf[:16] == 1))

    def test_grows_capacity_and_replaces_cached_storage(self):
        initial = _get_or_create_mega_output(8, 64, torch.bfloat16, torch.device("cpu"))
        grown = _get_or_create_mega_output(32, 64, torch.bfloat16, torch.device("cpu"))
        reused = _get_or_create_mega_output(16, 64, torch.bfloat16, torch.device("cpu"))

        self.assertIsNot(grown, initial)
        self.assertEqual(grown.shape, (32, 64))
        self.assertIs(reused, grown)

    def test_cache_key_separates_hidden_size_and_dtype(self):
        base = _get_or_create_mega_output(4, 32, torch.bfloat16, torch.device("cpu"))
        different_hidden = _get_or_create_mega_output(
            4, 64, torch.bfloat16, torch.device("cpu")
        )
        different_dtype = _get_or_create_mega_output(
            4, 32, torch.float32, torch.device("cpu")
        )

        self.assertIsNot(base, different_hidden)
        self.assertIsNot(base, different_dtype)


class MegaBufferSleepTest(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_MEGA_BUF_CACHE", {}),
            ("_MEGA_OUTPUT_CACHE", {}),
            ("_MEGA_STRATEGY_REGISTRY", weakref.WeakSet()),
        ):
            patcher = mock.patch.object(mega_buf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.executor = torch.nn.Module()
        self.executor._mega_y = torch.ones(4, 8)
        self.executor._mega_buf = mock.Mock(buffer=torch.empty(8))
        mega_buf._MEGA_OUTPUT_CACHE["output"] = self.executor._mega_y
        mega_buf._MEGA_BUF_CACHE["symm"] = self.executor._mega_buf
        mega_buf.register_mega_executor(self.executor)

    def test_graph_guard_preserves_all_cached_storages(self):
        output, buffer = self.executor._mega_y, self.executor._mega_buf
        with mock.patch.object(mega_buf, "mega_buffers_graph_baked", return_value=True):
            self.assertEqual(mega_buf.release_mega_symm_buffers(), 0)
        self.assertIs(self.executor._mega_y, output)
        self.assertIs(self.executor._mega_buf, buffer)
        self.assertGreater(mega_buf.mega_output_buffer_gib(), 0)
        buffer.destroy.assert_not_called()

    def test_release_clears_every_owner_and_is_idempotent(self):
        other = torch.nn.Module()
        other._mega_y, other._mega_buf = self.executor._mega_y, self.executor._mega_buf
        mega_buf.register_mega_executor(other)
        buffer = self.executor._mega_buf
        with mock.patch.object(
            mega_buf, "mega_buffers_graph_baked", return_value=False
        ):
            self.assertGreater(mega_buf.release_mega_symm_buffers(), 0)
            self.assertEqual(mega_buf.release_mega_symm_buffers(), 0)
        buffer.destroy.assert_called_once_with()
        for executor in (self.executor, other):
            self.assertIsNone(executor._mega_y)
            self.assertIsNone(executor._mega_buf)
        self.assertEqual(mega_buf._MEGA_BUF_CACHE, {})
        self.assertEqual(mega_buf._MEGA_OUTPUT_CACHE, {})

    def test_failed_destroy_keeps_failed_buffer_and_output(self):
        failed = self.executor._mega_buf
        failed.destroy.side_effect = RuntimeError("destroy failure")
        with mock.patch.object(
            mega_buf, "mega_buffers_graph_baked", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "destroy failure"):
                mega_buf.release_mega_symm_buffers()
        self.assertIs(mega_buf._MEGA_BUF_CACHE["symm"], failed)
        self.assertIs(self.executor._mega_buf, failed)
        self.assertIsNotNone(self.executor._mega_y)

    def test_partial_destroy_only_retires_successful_entries(self):
        good = self.executor._mega_buf
        failed = mock.Mock(buffer=torch.empty(8))
        failed.destroy.side_effect = RuntimeError("second buffer failure")
        mega_buf._MEGA_BUF_CACHE["failed"] = failed
        with mock.patch.object(
            mega_buf, "mega_buffers_graph_baked", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "second buffer failure"):
                mega_buf.release_mega_symm_buffers()
        good.destroy.assert_called_once_with()
        self.assertIsNone(self.executor._mega_buf)
        self.assertEqual(list(mega_buf._MEGA_BUF_CACHE), ["failed"])


if __name__ == "__main__":
    unittest.main()

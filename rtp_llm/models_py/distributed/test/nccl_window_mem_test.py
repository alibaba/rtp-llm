import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.distributed import nccl_window_mem


class _FakeShard:
    def __init__(self, *, dtype, numel, elem_size, is_cuda=True, contiguous=True, dim=2):
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._numel = numel
        self._elem_size = elem_size
        self._contiguous = contiguous
        self._dim = dim

    def is_contiguous(self):
        return self._contiguous

    def dim(self):
        return self._dim

    def numel(self):
        return self._numel

    def element_size(self):
        return self._elem_size


class NcclWindowMemTest(unittest.TestCase):
    def test_enable_switch_defaults_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(nccl_window_mem.window_allgather_enabled())
        with mock.patch.dict(
            os.environ, {nccl_window_mem.ENABLE_ENV: "0"}, clear=True
        ):
            self.assertFalse(nccl_window_mem.window_allgather_enabled())

    def test_size_policy(self):
        shard = _FakeShard(
            dtype=torch.bfloat16,
            numel=1024 * 1024,
            elem_size=torch.empty((), dtype=torch.bfloat16).element_size(),
            is_cuda=False,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(nccl_window_mem.should_use_window_allgather(shard, 4))

        shard.is_cuda = True
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(nccl_window_mem.should_use_window_allgather(shard, 4))

        with mock.patch.dict(
            os.environ, {nccl_window_mem.ENABLE_ENV: "0"}, clear=True
        ):
            self.assertFalse(nccl_window_mem.should_use_window_allgather(shard, 4))

        large_fp32 = _FakeShard(
            dtype=torch.float32,
            numel=5 * 1024 * 1024 * 4,
            elem_size=torch.empty((), dtype=torch.float32).element_size(),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                nccl_window_mem.should_use_window_allgather(large_fp32, 4)
            )


if __name__ == "__main__":
    unittest.main()

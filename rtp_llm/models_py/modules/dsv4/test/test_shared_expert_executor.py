import os
import unittest
from contextlib import contextmanager

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    OverlapSharedExpertExecutor,
    SequentialSharedExpertExecutor,
    combine_routed_and_shared,
    get_shared_expert_executor,
)


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class _Shared(nn.Module):
    def forward(self, x):
        return (x.float() * 0.25).to(x.dtype)


class TestSharedExpertExecutor(unittest.TestCase):
    def test_combine_preserves_fp32_accumulate_semantics(self):
        routed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        shared = torch.tensor([[0.5, -0.25], [0.125, -0.5]], dtype=torch.float32)
        got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.float() + shared.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_bf16_add_experimental_switch(self):
        routed = torch.randn(4, 8, dtype=torch.float32)
        shared = torch.randn(4, 8, dtype=torch.float32)
        with _env("DSV4_SHARED_EXPERT_BF16_ADD", "1"):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.to(torch.bfloat16) + shared.to(torch.bfloat16)).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_executor_dispatch(self):
        with _env("DSV4_SHARED_EXPERT_MODE", "sequential"):
            self.assertIsInstance(get_shared_expert_executor(), SequentialSharedExpertExecutor)
        with _env("DSV4_SHARED_EXPERT_MODE", "overlap"):
            self.assertIsInstance(get_shared_expert_executor(), OverlapSharedExpertExecutor)

    def test_sequential_executor(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        executor.start(_Shared(), x)
        got = executor.finish()
        ref = _Shared()(x).float()
        self.assertTrue(torch.equal(got, ref))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_matches_sequential(self):
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        overlap = OverlapSharedExpertExecutor()
        overlap.start(shared, x)
        got = overlap.finish()
        ref = shared(x).float()
        self.assertTrue(torch.equal(got.cpu(), ref.cpu()))


if __name__ == "__main__":
    unittest.main()

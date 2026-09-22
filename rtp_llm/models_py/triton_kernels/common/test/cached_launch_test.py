"""Manual CUDA tests for tensor-free, stream-safe warm launch caching."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.cached_launch import CachedLaunch, knobs


@triton.jit
def _affine(X, Y, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Y + offsets, value * SCALE, offsets < N)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CachedLaunchTest(unittest.TestCase):
    def launcher(self):
        return CachedLaunch(_affine)

    def warmed_launch(self):
        launch = self.launcher()
        x = torch.ones(16, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        launch((1, 1, 1), (x, y), (16, 2, 256))
        return launch, x, y

    def test_warm_launch_uses_fresh_tensors_without_retaining_them(self):
        launch = self.launcher()
        for value in (1, 3, -7):
            x = torch.full((513,), value, device="cuda", dtype=torch.bfloat16)
            y = torch.empty_like(x)
            launch((3, 1, 1), (x, y), (513, 2, 256))
            torch.testing.assert_close(y, torch.full_like(y, 2 * value), rtol=0, atol=0)
        ref = weakref.ref(x)
        del x
        gc.collect()
        self.assertIsNone(ref(), "launcher cache must not retain input tensors")

    def test_alignment_dtype_scalar_and_grid_changes_do_not_reuse_wrong_code(self):
        launch = self.launcher()
        for dtype in (torch.bfloat16, torch.float32):
            for n in (1, 257, 1025):
                for offset in (0, 1, 3):
                    for scale in (2, -3):
                        x = torch.arange(n + offset, device="cuda", dtype=dtype)[
                            offset:
                        ]
                        y = torch.full((n + offset,), -11, device="cuda", dtype=dtype)
                        launch(
                            ((n + 255) // 256, 1, 1), (x, y[offset:]), (n, scale, 256)
                        )
                        torch.testing.assert_close(
                            y[offset:], x * scale, rtol=0, atol=0
                        )
                        self.assertTrue(torch.all(y[:offset] == -11))

    def test_current_stream_and_graph_replay_are_not_cached(self):
        launch = self.launcher()
        for _ in range(2):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                x = torch.ones(513, device="cuda", dtype=torch.bfloat16)
                y = torch.empty_like(x)
                launch((3, 1, 1), (x, y), (513, 2, 256))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch((3, 1, 1), (x, y), (513, 2, 256))
                for value in (3, -4):
                    x.fill_(value)
                    graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(
                        y, torch.full_like(y, value * 2), rtol=0, atol=0
                    )
                graph.reset()
            torch.cuda.current_stream().wait_stream(stream)

    def test_hooks_installed_after_warmup_receive_launch_metadata(self):
        launch, x, y = self.warmed_launch()
        if not launch.enabled:
            self.skipTest("runtime HookChain API requires Triton 3.6")
        seen = []

        def enter(metadata):
            seen.append(("enter", metadata.get()["name"]))

        def leave(metadata):
            seen.append(("leave", metadata.get()["name"]))

        with patch.object(knobs.runtime, "launch_enter_hook", enter), patch.object(
            knobs.runtime, "launch_exit_hook", leave
        ):
            launch((1, 1, 1), (x, y), (16, 2, 256))
        self.assertEqual(seen, [("enter", "_affine"), ("leave", "_affine")])
        torch.testing.assert_close(y, x * 2, rtol=0, atol=0)

    def test_debug_and_pre_run_hooks_keep_the_normal_jit_path(self):
        launch, x, y = self.warmed_launch()
        if not launch.enabled:
            self.skipTest("runtime debug/pre-run hook API requires Triton 3.6")
        seen = []
        hook = lambda *args, **kwargs: seen.append(True)
        with patch.object(_affine, "pre_run_hooks", [hook]):
            launch((1, 1, 1), (x, y), (16, 2, 256))
        self.assertEqual(seen, [True])
        with patch.object(knobs.runtime, "debug", True), patch.object(
            _affine, "run", wraps=_affine.run
        ) as jit:
            launch((1, 1, 1), (x, y), (16, 2, 256))
            self.assertEqual(jit.call_count, 1)
        torch.testing.assert_close(y, x * 2, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

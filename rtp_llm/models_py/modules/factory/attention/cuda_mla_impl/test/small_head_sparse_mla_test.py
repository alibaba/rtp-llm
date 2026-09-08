"""Real H8 sparse attention: independent oracle, ABI holes, tails and graphs."""

import os
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla import flashinfer_bf16_small_head as fi


def reference(q, kv, ids, scale):
    valid = (ids[:, 0] >= 0) & (ids[:, 0] < kv.shape[0])
    selected = kv[ids[:, 0].clamp(0, kv.shape[0] - 1).long(), 0].float()
    logits = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    logits.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(logits, -1).nan_to_num(0)
    return torch.einsum("thk,tkd->thd", probs, selected).to(q.dtype)


def tilelang(q, kv, ids, scale):
    from rtp_llm.models_py.triton_kernels.sparse_mla.sglang_bf16_small_head import (
        tilelang_sparse_fwd,
    )

    width = max(64, (ids.shape[-1] + 63) // 64 * 64)
    padded = torch.full(
        (*ids.shape[:-1], width), -1, dtype=ids.dtype, device=ids.device
    )
    padded[..., : ids.shape[-1]] = ids
    return tilelang_sparse_fwd(q, kv, padded, scale)


class BackendSelectionTest(unittest.TestCase):
    def test_automatic_capability_and_version_guard(self):
        for capability, version, expected in (
            ((10, 0), "0.6.14", True),
            ((10, 3), "0.6.14", True),
            ((10, 3), "0.6.12", False),
            ((9, 0), "0.6.14", False),
            ((8, 9), "0.6.14", False),
        ):
            with self.subTest(capability=capability, version=version):
                with patch.object(torch.version, "hip", None), patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ), patch.object(fi, "version", return_value=version):
                    fi.flashinfer_sparse_supported.cache_clear()
                    self.assertEqual(
                        fi.flashinfer_sparse_supported(torch.device("cuda", 0)),
                        expected,
                    )
        with patch.object(torch.version, "hip", None), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.object(fi, "version", side_effect=PackageNotFoundError):
            fi.flashinfer_sparse_supported.cache_clear()
            self.assertFalse(fi.flashinfer_sparse_supported(torch.device("cuda", 0)))
        fi.flashinfer_sparse_supported.cache_clear()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallHeadSparseMlaTest(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        fi.flashinfer_sparse_supported.cache_clear()
        if not fi.flashinfer_sparse_supported(
            torch.device("cuda", torch.cuda.current_device())
        ):
            self.skipTest(
                "both backends require the supported FlashInfer H8 environment"
            )
        torch.manual_seed(31)

    def test_reference_holes_upper_bounds_and_strided_inputs(self):
        for backend in (tilelang, fi.flashinfer_sparse_fwd):
            for width in (1, 3, 127, 128, 129, 2051, 2176):
                with self.subTest(backend=backend.__name__, width=width):
                    q = torch.randn(9, 16, 512, device="cuda", dtype=torch.bfloat16)[
                        :, ::2
                    ]
                    kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
                    ids = torch.randint(
                        0, 4096, (9, 1, width + 7), device="cuda", dtype=torch.int32
                    )[..., :width]
                    ids[0] = -1
                    ids[1, :, : min(width - 1, 64)] = -1
                    ids[2, :, -1] = 4096
                    ids[3, :, ::3] = -1
                    actual = backend(q, kv, ids, 0.0625)
                    torch.testing.assert_close(
                        actual, reference(q, kv, ids, 0.0625), atol=0.016, rtol=0.015
                    )
                    self.assertEqual(actual.shape, (9, 8, 512))
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(actual[0].count_nonzero().item(), 0)

    def test_constant_kv_detects_padding_in_softmax_denominator(self):
        q = torch.zeros(6, 8, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.full((4096, 1, 512), 2.0, device="cuda", dtype=torch.bfloat16)
        ids = torch.full((6, 1, 2051), -1, device="cuda", dtype=torch.int32)
        for row, count in enumerate((0, 1, 127, 128, 129, 2051)):
            ids[row, 0, -count:] = torch.arange(count, device="cuda") if count else -1
        for backend in (tilelang, fi.flashinfer_sparse_fwd):
            actual = backend(q, kv, ids, 0.0625)
            torch.testing.assert_close(
                actual[0], torch.zeros_like(actual[0]), atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual[1:], torch.full_like(actual[1:], 2), atol=0, rtol=0
            )

    def test_graph_replay_with_changed_queries_and_indices(self):
        for backend in (tilelang, fi.flashinfer_sparse_fwd):
            q = torch.randn(9, 8, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(0, 4096, (9, 1, 2051), device="cuda", dtype=torch.int32)
            backend(q, kv, ids, 0.0625)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = backend(q, kv, ids, 0.0625)
            for step in range(3):
                q.mul_(0.5)
                ids[step] = -1
                graph.replay()
                torch.testing.assert_close(
                    out, reference(q, kv, ids, 0.0625), atol=0.016, rtol=0.015
                )
            self.assertEqual(backend(q[:0], kv, ids[:0], 0.0625).shape, (0, 8, 512))

    def test_flashinfer_concurrent_streams(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        pending = []
        for stream in streams:
            with torch.cuda.stream(stream):
                q = torch.randn(17, 8, 512, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
                ids = torch.randint(
                    0, 4096, (17, 1, 2051), device="cuda", dtype=torch.int32
                )
                pending.append(
                    (fi.flashinfer_sparse_fwd(q, kv, ids, 0.0625), q, kv, ids)
                )
        for stream, (out, q, kv, ids) in zip(streams, pending):
            stream.synchronize()
            torch.testing.assert_close(
                out, reference(q, kv, ids, 0.0625), atol=0.016, rtol=0.015
            )


if __name__ == "__main__":
    unittest.main()

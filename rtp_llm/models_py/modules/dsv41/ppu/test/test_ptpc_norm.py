"""PTPC compatibility wrapper uses the shared standalone norm policy."""

import os
import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv41.ppu import ptpc_norm


class PTPCNormCPU(unittest.TestCase):
    def test_wrapper_preserves_arguments_and_keyword_abi(self):
        gamma = torch.ones(512, dtype=torch.bfloat16)
        base = torch.randn(3, 1024, dtype=torch.bfloat16)
        for value in (base[:, :512], base[:, ::2], base[:, :512].contiguous()):
            for allow_strided in (False, True):
                sentinel = object()
                with patch.object(
                    ptpc_norm, "rms_norm", return_value=sentinel
                ) as native:
                    self.assertIs(
                        ptpc_norm.ptpc_rms_norm(
                            value, gamma, 1e-6, allow_strided=allow_strided
                        ),
                        sentinel,
                    )
                native.assert_called_once_with(value, gamma, 1e-6)

    def test_autograd_is_explicitly_unsupported(self):
        value = torch.randn(3, 512, dtype=torch.bfloat16, requires_grad=True)
        gamma = torch.ones(512, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "inference"):
            ptpc_norm.ptpc_rms_norm(value, gamma, 1e-6)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU component run"
)
class PTPCNormDevice(unittest.TestCase):
    @torch.no_grad()
    def test_dense_uses_actual_flashinfer_including_odd_widths(self):
        from flashinfer.norm import rmsnorm

        for width in (512, 513, 1280, 1281, 5120, 5121):
            value = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            actual = ptpc_norm.ptpc_rms_norm(value, gamma, 1e-6)
            expected = rmsnorm(value, gamma, eps=1e-6)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )

    @torch.no_grad()
    def test_graph_replays_changed_inputs_and_gamma(self):
        from flashinfer.norm import rmsnorm

        for width in (512, 1280, 5120):
            value = torch.randn(4, width, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            ptpc_norm.ptpc_rms_norm(value, gamma, 1e-6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = ptpc_norm.ptpc_rms_norm(value, gamma, 1e-6)
            for magnitude in (0.0, 0.125, 1.0, 16.0):
                value.copy_(torch.randn_like(value) * magnitude)
                gamma.copy_(torch.randn_like(gamma))
                graph.replay()
                expected = rmsnorm(value, gamma, eps=1e-6)
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )


def load_tests(loader, tests, pattern):
    # Keep the existing fused producer's independent arithmetic/byte tests.
    from rtp_llm.models_py.modules.dsv41.ppu.test import test_ptpc_norm_quant

    tests.addTests(loader.loadTestsFromModule(test_ptpc_norm_quant))
    return tests


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.scaled_rope import (
    _qk_offset,
    scaled_rope_qk_inplace,
)
from rtp_llm.models_py.kernels.cuda.test.scaled_rope_reference import reference_rope


@triton.jit
def _qk_offset_probe(
    output,
    token,
    head,
    Q_TOKEN: tl.constexpr,
    K_TOKEN: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
):
    # Exercise the exact address arithmetic used for Q AND K, without needing
    # a multi-gigabyte QKV allocation merely to reach the overflow boundary.
    tl.store(output, _qk_offset(token, head, Q_TOKEN, HEAD_STRIDE))
    tl.store(output + 1, _qk_offset(token, head, K_TOKEN, HEAD_STRIDE))


class ScaledRopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter = os.environ.get("TRITON_INTERPRET") == "1"
        cls.device = "cpu" if cls.interpreter else "cuda"
        if not cls.interpreter and not torch.cuda.is_available():
            raise RuntimeError(
                "scaled RoPE GPU target requires its assigned accelerator"
            )

    def test_qk_offsets_cross_int32_boundary(self):
        output = torch.empty(2, dtype=torch.int64, device=self.device)
        for q_stride, k_stride in ((10240, 6144), (5120, 4096)):
            for stride in (q_stride, k_stride):
                boundary = (2**31 + stride - 1) // stride
                for token in (boundary - 1, boundary, boundary + 1):
                    with self.subTest(
                        token=token, q_stride=q_stride, k_stride=k_stride
                    ):
                        _qk_offset_probe[(1,)](
                            output, token, 3, q_stride, k_stride, 128
                        )
                        self.assertEqual(
                            output.tolist(),
                            [token * q_stride + 3 * 128, token * k_stride + 3 * 128],
                        )
        # Head offsets must also be widened before multiplication, not just
        # the token-stride product; Q/K are allowed to have strided head views.
        _qk_offset_probe[(1,)](output, 1, 32769, 128, 256, 65536)
        self.assertEqual(output.tolist(), [128 + 32769 * 65536, 256 + 32769 * 65536])

    def fixture(self, dtype):
        torch.manual_seed(523)
        qkv = torch.randn(9, 8, 128, device=self.device, dtype=dtype)
        positions = torch.tensor(
            [0, 1, 3, 31, 32, 45, 63, 81, 127], device=self.device, dtype=torch.int32
        )
        batches = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2], device=self.device, dtype=torch.int32
        )
        lengths = torch.tensor([4, 46, 128], device=self.device, dtype=torch.int32)
        return qkv, positions, batches, lengths

    def assert_result(self, actual, original, positions, batches, lengths, **kwargs):
        token_lengths = lengths[batches.long()]
        for start, end in ((0, 4), (4, 6)):
            expected = reference_rope(
                original[:, start:end], positions, token_lengths, **kwargs
            )
            torch.testing.assert_close(
                actual[:, start:end], expected, rtol=4e-3, atol=2e-2
            )
        torch.testing.assert_close(actual[:, 6:], original[:, 6:], rtol=0, atol=0)
        dim = kwargs["rotary_dim"]
        torch.testing.assert_close(
            actual[:, :6, dim:], original[:, :6, dim:], rtol=0, atol=0
        )

    def test_strided_gqa_full_and_partial_rotary_against_reference(self):
        # The three requests straddle the NTK threshold; Llama3 exercises low,
        # high and interpolated frequencies. Q/K are non-contiguous QKV views.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for llama3 in (True, False):
                for dim in (96, 128):
                    with self.subTest(dtype=dtype, llama3=llama3, dim=dim):
                        qkv, positions, batches, lengths = self.fixture(dtype)
                        original = qkv.clone()
                        kwargs = dict(
                            rotary_dim=dim,
                            base=10000.0,
                            scale=8.0,
                            original_max_position=32,
                            llama3=llama3,
                        )
                        scaled_rope_qk_inplace(
                            qkv[:, :4],
                            qkv[:, 4:6],
                            positions,
                            batches,
                            lengths,
                            **kwargs
                        )
                        self.assert_result(
                            qkv, original, positions, batches, lengths, **kwargs
                        )
                        base = dict(kwargs, scale=1.0, llama3=False)
                        unscaled = reference_rope(
                            original[:, :4], positions, lengths[batches.long()], **base
                        )
                        self.assertGreater(
                            (qkv[:, :4].float() - unscaled.float()).abs().max().item(),
                            0.1,
                        )

    def test_style_only_override_matches_unscaled_rope(self):
        self._check_style_only_override_matches_unscaled_rope()

    def test_dynamic_ntk_factor_one_with_original_length_still_extrapolates(self):
        qkv, positions, batches, lengths = self.fixture(torch.float32)
        original = qkv.clone()
        kwargs = dict(
            rotary_dim=128,
            base=10000.0,
            scale=1.0,
            original_max_position=32,
            llama3=False,
        )
        scaled_rope_qk_inplace(
            qkv[:, :4], qkv[:, 4:6], positions, batches, lengths, **kwargs
        )
        self.assert_result(qkv, original, positions, batches, lengths, **kwargs)
        unscaled = reference_rope(
            original[:, :4],
            positions,
            lengths[batches.long()],
            **dict(kwargs, original_max_position=1024)
        )
        self.assertGreater((qkv[:, :4] - unscaled).abs().max().item(), 0.1)

    def _check_style_only_override_matches_unscaled_rope(self):
        for llama3 in (True, False):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(llama3=llama3, dtype=dtype):
                    qkv, positions, batches, lengths = self.fixture(dtype)
                    original = qkv.clone()
                    # Defaults produced by a style-only JSON override. No KV
                    # lengths are needed when the scaling factor is one.
                    scaled_rope_qk_inplace(
                        qkv[:, :4],
                        qkv[:, 4:6],
                        positions,
                        batches,
                        lengths[:0],
                        rotary_dim=128,
                        base=10000.0,
                        scale=1.0,
                        original_max_position=0,
                        llama3=llama3,
                        low_freq_factor=1.0,
                        high_freq_factor=1.0,
                    )
                    self.assert_result(
                        qkv,
                        original,
                        positions,
                        batches,
                        lengths,
                        rotary_dim=128,
                        base=10000.0,
                        scale=1.0,
                        original_max_position=1024,
                        llama3=False,
                    )

    def test_llama3_long_context_positions(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                qkv, positions, batches, lengths = self.fixture(dtype)
                positions.copy_(
                    torch.tensor(
                        [0, 8191, 8192, 16383, 32767, 65535, 65536, 98304, 131071],
                        device=self.device,
                        dtype=torch.int32,
                    )
                )
                original = qkv.clone()
                kwargs = dict(
                    rotary_dim=128,
                    base=500000.0,
                    scale=8.0,
                    original_max_position=8192,
                    llama3=True,
                )
                scaled_rope_qk_inplace(
                    qkv[:, :4], qkv[:, 4:6], positions, batches, lengths, **kwargs
                )
                self.assert_result(qkv, original, positions, batches, lengths, **kwargs)

    def test_graph_replay_reads_updated_positions_and_request_lengths(self):
        if self.interpreter:
            self.skipTest("Triton interpreter verifies math, not CUDA graph execution")
        for llama3 in (True, False):
            qkv, positions, batches, lengths = self.fixture(torch.float16)
            original = qkv.clone()
            kwargs = dict(
                rotary_dim=128,
                base=10000.0,
                scale=8.0,
                original_max_position=32,
                llama3=llama3,
            )

            def run():
                scaled_rope_qk_inplace(
                    qkv[:, :4], qkv[:, 4:6], positions, batches, lengths, **kwargs
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    qkv.copy_(original)
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for offset in (7, 39):
                qkv.copy_(original)
                positions.add_(offset)
                lengths.add_(offset)
                graph.replay()
                self.assert_result(qkv, original, positions, batches, lengths, **kwargs)


if __name__ == "__main__":
    unittest.main()

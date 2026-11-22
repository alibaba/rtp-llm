import os
import unittest
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.scaled_rope import (
    _qk_offset,
    scaled_rope_qk_inplace,
    scaled_rope_qkv,
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

    def test_packed_adapter_prefill_decode_metadata_and_input_preservation(self):
        for llama3 in (False, True):
            config = SimpleNamespace(
                dim=96,
                base=10000,
                scale=8.0,
                max_pos=32,
                factor1=1.0,
                factor2=4.0,
                index_factor=1,
            )
            # Empty middle request exercises right-sided cumulative lookup;
            # unequal totals distinguish real lengths from a batch maximum.
            counts = torch.tensor([3, 0, 2, 4], device=self.device, dtype=torch.int32)
            cu = torch.cat((counts.new_zeros(1), counts.cumsum(0).int()))
            prefixes = counts.new_tensor([0, 9, 31, 64])
            batches = counts.new_tensor([0, 0, 0, 2, 2, 3, 3, 3, 3]).long()
            inferred = (
                torch.arange(9, device=self.device) - cu[batches] + prefixes[batches]
            )
            raw = torch.randn(9, 8, 128, device=self.device)
            saved = raw.clone()
            kwargs = dict(
                rotary_dim=96,
                base=10000,
                scale=8.0,
                original_max_position=32,
                llama3=llama3,
            )
            for explicit in (None, inferred.flip(0).int()):
                with self.subTest(llama3=llama3, explicit=explicit is not None):
                    actual = scaled_rope_qkv(
                        raw.reshape(9, -1),
                        num_heads=4,
                        num_kv_heads=2,
                        head_size=128,
                        rope_config=config,
                        llama3=llama3,
                        positions=explicit,
                        cu_seqlens=cu,
                        prefix_lengths=prefixes,
                    ).reshape_as(raw)
                    self.assert_result(
                        actual,
                        raw,
                        inferred if explicit is None else explicit,
                        batches,
                        counts + prefixes,
                        **kwargs
                    )
                    torch.testing.assert_close(raw, saved, atol=0, rtol=0)
                    self.assertNotEqual(actual.data_ptr(), raw.data_ptr())
            history = counts.new_tensor([0, 1, 7, 31, 32, 63, 64, 127, 5])
            for explicit in (None, history.flip(0)):
                actual = scaled_rope_qkv(
                    raw.reshape(9, -1),
                    num_heads=4,
                    num_kv_heads=2,
                    head_size=128,
                    rope_config=config,
                    llama3=llama3,
                    positions=explicit,
                    sequence_lengths=history,
                ).reshape_as(raw)
                self.assert_result(
                    actual,
                    raw,
                    history if explicit is None else explicit,
                    torch.arange(9, device=self.device),
                    history + 1,
                    **kwargs
                )
                torch.testing.assert_close(raw, saved, atol=0, rtol=0)

    def test_prefill_graph_bucket_can_have_more_rows_than_live_tokens(self):
        raw = torch.randn(9, 8, 128, device=self.device)
        cu = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device=self.device)
        prefixes = cu.new_tensor([0, 65, 70])
        config = SimpleNamespace(dim=96, base=10000, scale=8.0, max_pos=32,
                                 factor1=1.0, factor2=4.0, index_factor=1)
        for llama3 in (False, True):
            def run():
                return scaled_rope_qkv(
                    raw.flatten(1), num_heads=4, num_kv_heads=2, head_size=128,
                    rope_config=config, llama3=llama3, positions=None,
                    cu_seqlens=cu, prefix_lengths=prefixes,
                ).reshape_as(raw)

            cu.copy_(cu.new_tensor([0, 3, 6, 9]))
            run()  # Compile before capture.
            if not self.interpreter:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
            # Two live requests, one empty request, five discarded bucket rows.
            cu.copy_(cu.new_tensor([0, 1, 4, 4]))
            if self.interpreter:
                actual = run()
            else:
                graph.replay()
            self.assert_result(
                actual[:4], raw[:4], cu.new_tensor([0, 65, 66, 67]),
                cu.new_tensor([0, 1, 1, 1]), cu.new_tensor([1, 68, 70]),
                rotary_dim=96, base=10000, scale=8.0, original_max_position=32,
                llama3=llama3,
            )

    def test_packed_adapter_accepts_capacity_positions_and_preserves_stride(self):
        config = SimpleNamespace(
            dim=128,
            base=10000,
            scale=8.0,
            max_pos=32,
            factor1=1.0,
            factor2=4.0,
            index_factor=3,
        )
        raw = torch.randn(2, 8, 128, device=self.device)
        history = torch.tensor([31, 64], device=self.device, dtype=torch.int32)
        # Two live tokens in a four-token, three-axis capacity buffer. Only
        # axis 0 defines this single-axis RoPE; unused storage is poisoned.
        positions = history.new_tensor(
            [0, 999, 999, 77, 999, 999, -777, -777, -777, -777, -777, -777]
        )
        for llama3 in (False, True):
            actual = scaled_rope_qkv(
                raw.reshape(2, -1),
                num_heads=4,
                num_kv_heads=2,
                head_size=128,
                rope_config=config,
                llama3=llama3,
                positions=positions,
                sequence_lengths=history,
            ).reshape_as(raw)
            self.assert_result(
                actual,
                raw,
                history.new_tensor([0, 77]),
                torch.arange(2, device=self.device),
                history + 1,
                rotary_dim=128,
                base=10000,
                scale=8.0,
                original_max_position=32,
                llama3=llama3,
            )

    def test_packed_adapter_rejects_ambiguous_or_short_metadata(self):
        config = SimpleNamespace(
            dim=128,
            base=10000,
            scale=8.0,
            max_pos=32,
            factor1=1.0,
            factor2=4.0,
            index_factor=1,
        )
        raw = torch.zeros(2, 8 * 128, device=self.device)
        lengths = torch.ones(2, device=self.device, dtype=torch.int32)
        cu = lengths.new_tensor([0, 2])
        kwargs = dict(
            num_heads=4,
            num_kv_heads=2,
            head_size=128,
            rope_config=config,
            llama3=False,
            positions=None,
        )
        for metadata in (
            {},
            dict(sequence_lengths=lengths, cu_seqlens=cu),
            dict(sequence_lengths=lengths[:1]),
            dict(cu_seqlens=cu, prefix_lengths=lengths),
        ):
            with self.subTest(metadata=tuple(metadata)), self.assertRaises(ValueError):
                scaled_rope_qkv(raw, **kwargs, **metadata)

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

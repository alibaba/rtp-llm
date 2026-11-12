"""Exact BF16-domain qualification for the two-round native Verify TopK.

Inputs are constructed as BF16 bit patterns expanded into FP32 storage. The
specialized API is never used to round an arbitrary FP32 score. Native tie
order/IDs are unspecified; the original ordered-32 score multiset, unique IDs,
valid domain and unique-score IDs are checked without changing that contract.
"""

import unittest

import torch


def _ordered_integer(bits):
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def _ordered32(values):
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits & 0x80000000 != 0, ~bits & 0xFFFFFFFF, bits | 0x80000000)


def _expand_bf16_bits(bits):
    # Integer construction also retains signaling/signed NaN payloads, which
    # an arithmetic float/BF16 conversion is allowed to quiet or canonicalize.
    return ((bits.to(torch.int64) & 0xFFFF) << 16).to(torch.int32).view(torch.float32)


class VerifyBF16TopKKeyTest(unittest.TestCase):
    def test_all_bf16_keys_are_bijective_and_preserve_original_order(self):
        full = [_ordered_integer(bits << 16) for bits in range(65536)]
        compressed = [key >> 16 for key in full]
        self.assertEqual(len(set(compressed)), 65536)
        self.assertEqual(
            sorted(range(65536), key=full.__getitem__),
            sorted(range(65536), key=compressed.__getitem__),
        )
        self.assertLess(_ordered_integer(0x80000000), _ordered_integer(0x00000000))
        # The qualification is a typed producer precondition, not a generic
        # FP32 approximation: these distinct FP32 values would collide.
        self.assertNotEqual(_ordered_integer(0x3F800000), _ordered_integer(0x3F800008))
        self.assertEqual(_ordered_integer(0x3F800000) >> 16, _ordered_integer(0x3F800008) >> 16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/PPU execution required")
class VerifyBF16TopKDeviceTest(unittest.TestCase):
    topk = 512

    def setUp(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.native = rtp_llm_ops.dsv4_persistent_topk_bf16_scores
        self.generic = rtp_llm_ops.dsv4_persistent_topk
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _buffers(self, rows):
        output_bank = torch.full((rows * self.topk + 2,), -9191, dtype=torch.int32, device=self.device)
        output = output_bank[1:-1].reshape(rows, self.topk)
        workspace_bank = torch.full((1024 * 1024 + 16,), 173, dtype=torch.uint8, device=self.device)
        return output, workspace_bank[:-16], output_bank, workspace_bank

    def _check(self, scores, lengths, output, *, unique_rows=(), phase=""):
        keys = _ordered32(scores)
        for row, length in enumerate(lengths.cpu().flatten().tolist()):
            actual = output[row].long()
            context = f"phase={phase}, row={row}, length={length}, width={scores.shape[1]}"
            if length <= self.topk:
                ordinal = torch.arange(self.topk, device=self.device)
                expected = torch.where(ordinal < length, ordinal, -1)
                self.assertTrue(torch.equal(actual, expected), context)
                continue
            in_domain = (actual >= 0) & (actual < length)
            if not in_domain.all().item():
                self.fail(
                    f"{context}, invalid={actual[~in_domain][:32].tolist()}, "
                    f"unwritten_sentinel={(actual == -98765).sum().item()}, "
                    f"head={actual[:16].tolist()}, tail={actual[-16:].tolist()}"
                )
            self.assertEqual(actual.unique().numel(), self.topk, context)
            selected = keys[row, actual].sort().values
            expected = keys[row, :length].topk(self.topk).values.sort().values
            self.assertTrue(torch.equal(selected, expected), context)
            if row in unique_rows:
                self.assertEqual(keys[row, :length].unique().numel(), length, context)
                expected_ids = keys[row, :length].topk(self.topk).indices.sort().values
                self.assertTrue(torch.equal(actual.sort().values, expected_ids), context)

    def _invoke_and_check(self, scores, lengths, *, generic=True, unique_rows=()):
        rows, width = scores.shape
        output, workspace, output_bank, workspace_bank = self._buffers(rows)
        original_bits, original_lengths = scores.contiguous().view(torch.int32).clone(), lengths.clone()
        self.native(scores, lengths, output, workspace, self.topk, width)
        self._check(scores, lengths, output, unique_rows=unique_rows)
        self.assertTrue(torch.equal(scores.contiguous().view(torch.int32), original_bits))
        self.assertTrue(torch.equal(lengths, original_lengths))
        self.assertEqual(output_bank[0].item(), -9191)
        self.assertEqual(output_bank[-1].item(), -9191)
        self.assertTrue((workspace_bank[-16:] == 173).all().item())
        if generic:
            old = torch.empty_like(output)
            old_workspace = torch.empty_like(workspace)
            self.generic(scores, lengths, old, old_workspace, self.topk, width)
            self._check(scores, lengths, old, unique_rows=unique_rows)

    def test_every_trivial_length_and_supported_row_counts(self):
        rows, width = 32, 512
        scores = torch.full((rows, width), float("inf"), device=self.device)
        for start in range(0, self.topk + 1, rows):
            lengths = torch.arange(start, start + rows, dtype=torch.int32, device=self.device).clamp_max_(width)
            self._invoke_and_check(scores, lengths)
        for count in (1, 3, 8, 31):
            lengths = torch.arange(count, dtype=torch.int32, device=self.device)
            self._invoke_and_check(scores[:count], lengths)

    def test_partial_vectors_medium_boundaries_and_real_widths(self):
        rows = 32
        for width in (513, 514, 515, 8193, 32768, 32769, 65920, 131840):
            with self.subTest(width=width):
                # The reference's legacy medium refinement must not be given
                # an oversized equal-key bucket. This bounded finite pattern
                # admits the old four-round implementation independently.
                column = torch.arange(width, device=self.device)
                values = ((column % 1024) - 512).to(torch.bfloat16).float()
                scores = values[None, :].expand(rows, width).contiguous()
                pattern = (0, 511, 512, 513, 514, 515, 8193, 32768, 32769, width)
                lengths = torch.tensor([min(pattern[i % len(pattern)], width) for i in range(rows)],
                                       dtype=torch.int32, device=self.device)
                self._invoke_and_check(scores, lengths)

    def test_all_bf16_payloads_and_wide_equal_boundaries(self):
        # All 65536 encodings include +/-normal, subnormal, zero, infinity and
        # every signed NaN payload. Full ordered32 keys are the dense oracle;
        # Torch floating TopK has a different NaN/signed-zero policy.
        bits = torch.arange(65536, device=self.device)
        values = _expand_bf16_bits((bits * 40503 + 17) & 0xFFFF)
        scores = values[None, :].expand(32, -1).contiguous()
        lengths = torch.full((32,), 65536, dtype=torch.int32, device=self.device)
        self._invoke_and_check(scores, lengths, unique_rows=(0, 31))
        # More than 4096 equal keys intentionally exercises only the new
        # cooperative path and the dense oracle, never the old medium buffer.
        for width in (8193, 32768):
            for bit_pattern in (0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80):
                with self.subTest(width=width, bits=bit_pattern):
                    scores = _expand_bf16_bits(torch.full((32, width), bit_pattern, device=self.device))
                    lengths.fill_(width)
                    self._invoke_and_check(scores, lengths, generic=False)
        # Put signed zero, subnormal, infinity and NaN distinctions exactly
        # at the Kth boundary instead of merely loading them below the pivot.
        pairs = (
            (0xFF80, 0xFF7F), (0x8080, 0x807F), (0x8001, 0x8000),
            (0x8000, 0x0000), (0x0000, 0x0001), (0x007F, 0x0080),
            (0x7F7F, 0x7F80), (0xFFC1, 0xFF80), (0x7F80, 0x7F81),
            (0x7F81, 0x7FC1), (0xFFFF, 0xFF81),
        )
        boundary = torch.full((32, 515), 0x7FFF, device=self.device)
        for row in range(32):
            a, b = pairs[row % len(pairs)]
            boundary[row, 511:] = torch.tensor((a, b, a, b), device=self.device)
        lengths.fill_(515)
        self._invoke_and_check(_expand_bf16_bits(boundary), lengths, generic=False)

    def test_padded_and_unaligned_rows(self):
        # Row stride and base-pointer alignment jointly determine vector size.
        for padding, offset in ((1, 0), (2, 0), (4, 0), (4, 1), (4, 2)):
            width, rows = 32769, 3
            stride = ((width + 3) // 4) * 4 + padding
            bank = torch.empty(rows * stride + offset, device=self.device)
            scores = bank[offset:].view(rows, stride)[:, :width]
            values = (torch.arange(width, device=self.device) % 127).to(torch.bfloat16).float()
            scores.copy_(values[None, :])
            lengths = torch.full((rows,), width, dtype=torch.int32, device=self.device)
            # The old API did not admit unaligned float4/float2 pointers.
            self._invoke_and_check(scores, lengths, generic=offset == 0)

    def test_graph_replay_changes_scores_lengths_and_workspace(self):
        rows, width = 32, 131840
        column = torch.arange(width, device=self.device)[None, :]
        row = torch.arange(rows, device=self.device)[:, None]
        scores = torch.empty((rows, width), device=self.device)
        lengths = torch.empty(rows, dtype=torch.int32, device=self.device)
        output, workspace, _, _ = self._buffers(rows)
        old, old_workspace, _, _ = self._buffers(rows)
        patterns = (
            [width, 0, width, 511, width, 512, width, 513],
            [width, 8193, width, 32768, width, 32769, width, 515],
            [513, 514, 515, 0, 511, 512, width, width - 3],
        )

        def populate(iteration):
            scores.copy_(((column * (17 + iteration * 2) + row * 3) % 2048 - 1024).to(torch.bfloat16).float())
            lengths.copy_(torch.tensor(patterns[iteration % len(patterns)] * 4,
                                       dtype=torch.int32, device=self.device))
            # Each native call must initialize all state it consumes. Large,
            # trivial and short nontrivial rows rotate histogram reuse.
            workspace.fill_(37 + iteration)
            old_workspace.fill_(83 + iteration)
            output.fill_(-98765)
            old.fill_(-98765)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            populate(0)
            self.native(scores, lengths, output, workspace, self.topk, width)
            self.generic(scores, lengths, old, old_workspace, self.topk, width)
        stream.synchronize()
        self._check(scores, lengths, output, phase="pre-capture specialized eager")
        self._check(scores, lengths, old, phase="pre-capture generic eager")
        graph, reference_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.native(scores, lengths, output, workspace, self.topk, width)
        with torch.cuda.graph(reference_graph, stream=stream):
            self.generic(scores, lengths, old, old_workspace, self.topk, width)
        for iteration in range(6):
            # Use the identical row-varying inputs for eager and both Graphs.
            # Identical rows can hide one persistent row overwriting shared
            # keys while another thread still collects the preceding row.
            with torch.cuda.stream(stream):
                populate(iteration)
                self.native(scores, lengths, output, workspace, self.topk, width)
                self.generic(scores, lengths, old, old_workspace, self.topk, width)
            stream.synchronize()
            self._check(scores, lengths, output, phase=f"iteration={iteration} specialized eager")
            self._check(scores, lengths, old, phase=f"iteration={iteration} generic eager")
            with torch.cuda.stream(stream):
                populate(iteration)
                graph.replay()
                reference_graph.replay()
            stream.synchronize()
            self._check(scores, lengths, output, phase=f"iteration={iteration} specialized Graph")
            self._check(scores, lengths, old, phase=f"iteration={iteration} generic Graph")

    def test_metadata_refusal_keeps_specialization_bounded(self):
        scores = torch.zeros((32, 515), device=self.device)
        lengths = torch.full((32,), 515, dtype=torch.int32, device=self.device)
        output, workspace, _, _ = self._buffers(32)
        cases = (
            (scores.bfloat16(), lengths, output, workspace, 512, 515),
            (scores, lengths.long(), output, workspace, 512, 515),
            (scores, lengths.cpu(), output, workspace, 512, 515),
            (scores, lengths, output.long(), workspace, 512, 515),
            (scores, lengths, output, workspace[:-1], 512, 515),
            (scores, lengths, output, workspace[1:], 512, 515),
            (scores, lengths, output, workspace, 1024, 515),
            (scores, lengths, output, workspace, 512, 0),
            (scores, lengths, output, workspace, 512, 516),
            (scores.t(), lengths, output, workspace, 512, 32),
            (scores, lengths, output.t(), workspace, 512, 515),
            (scores[:0], lengths[:0], output[:0], workspace, 512, 515),
            (torch.zeros((33, 515), device=self.device),
             torch.full((33,), 515, dtype=torch.int32, device=self.device),
             torch.empty((33, 512), dtype=torch.int32, device=self.device), workspace, 512, 515),
        )
        for args in cases:
            with self.assertRaises(RuntimeError):
                self.native(*args)


if __name__ == "__main__":
    unittest.main()

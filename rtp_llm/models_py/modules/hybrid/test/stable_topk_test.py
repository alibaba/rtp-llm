import unittest

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class StableTopkTest(unittest.TestCase):
    def check_case(self, scores, starts, ends, k, replay=False):
        width = scores.shape[1]
        expected = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
        # The public contract orders IEEE float keys (including signed zero and
        # NaN payloads), then breaks equal-score ties by the smaller column.
        for row, (start, end) in enumerate(zip(starts, ends)):
            start = max(0, min(start, width))
            end = max(start, min(end, width))
            bits = scores[row, start:end].cpu().view(torch.int32).tolist()
            keys = [
                (~(b & 0xFFFFFFFF) & 0xFFFFFFFF) if b < 0 else b ^ 0x80000000
                for b in bits
            ]
            selected = sorted(range(len(bits)), key=lambda i: (-keys[i], i))[:k]
            expected[row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        starts_d = torch.tensor(starts, device="cuda", dtype=torch.int32)
        ends_d = torch.tensor(ends, device="cuda", dtype=torch.int32)
        output = torch.empty_like(expected, device="cuda")

        def run():
            rtp_llm_ops.topk_v3_tie_break(scores, starts_d, ends_d, output, k, width)

        run()
        # Output order is unspecified; the chosen set and padding are exact.
        torch.testing.assert_close(output.cpu().sort().values, expected.sort().values)
        if replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(3):
                output.fill_(-99)
                graph.replay()
                torch.testing.assert_close(
                    output.cpu().sort().values, expected.sort().values
                )

    def test_dispatch_boundaries_and_strided_rows(self):
        for k in (512, 1024, 2048):
            for width in (4096, 4097, 8192, 8193, 16384, 16385):
                with self.subTest(k=k, width=width):
                    scores = (
                        torch.arange(width + 7, device="cuda", dtype=torch.float32)
                        .remainder(37)
                        .repeat(4, 1)[:, :width]
                    )
                    self.check_case(scores, [0, 1, 3, 0], [0, 129, width, width], k)

    def test_equal_scores_overflow_and_graph_replay(self):
        for k in (512, 1024, 2048):
            self.check_case(
                torch.ones(2, 32771, device="cuda"),
                [1, 0],
                [32771, 32771],
                k,
                replay=True,
            )

    def test_negative_midpoints_signed_zero_and_nonfinite(self):
        patterns = (
            torch.tensor([-1.00390625, -1.00390637, -1.00390613, -1.0]),
            torch.tensor(
                [-0.0, 0.0, float("inf"), -float("inf"), float("nan"), -float("nan")]
            ),
        )
        for pattern in patterns:
            for k in (512, 1024, 2048):
                with self.subTest(k=k, pattern=pattern):
                    scores = pattern.repeat(4097)[:16385].repeat(2, 1).cuda()
                    self.check_case(scores, [0, 1], [16385, 16384], k)

    def test_signed_nan_bins_preserve_exact_score_order(self):
        # Exercise every dispatcher and both sides of the 2048-candidate limit.
        # A NaN bin above the threshold must not emit negative NaNs either.
        for width in (4096, 8192, 16384, 16385):
            for positive_count, negative_count in (
                (0, 128),
                (128, 512),
                (128, 1920),
                (128, 1921),
            ):
                scores = torch.ones((2, width + 3), device="cuda")[:, :width]
                scores[:, :positive_count] = float("nan")
                scores[:, positive_count : positive_count + negative_count] = -float(
                    "nan"
                )
                for k in (512, 1024, 2048):
                    with self.subTest(
                        width=width,
                        positive=positive_count,
                        negative=negative_count,
                        k=k,
                    ):
                        self.check_case(
                            scores,
                            [0, 1],
                            [width, width],
                            k,
                            replay=width == 16385
                            and negative_count == 512
                            and k == 512,
                        )


if __name__ == "__main__":
    unittest.main()

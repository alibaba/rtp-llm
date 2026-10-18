import os
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_launch_fusion import try_write_topk
from rtp_llm.models_py.modules.base.cuda.indexer_op import _write_prefill_topk


def reference_topk(
    indices: torch.Tensor, offsets: torch.Tensor, canonicalize: bool
) -> torch.Tensor:
    if canonicalize:
        sentinel = torch.iinfo(indices.dtype).max
        sortable = torch.where(
            indices >= 0, indices, torch.full_like(indices, sentinel)
        )
        indices = torch.sort(sortable, dim=-1).values
        indices = torch.where(
            indices == sentinel, torch.full_like(indices, -1), indices
        )
    return torch.where(indices >= 0, indices + offsets.unsqueeze(1), indices)


class IndexerLaunchFusionTest(TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "This target requires an NVIDIA GPU")
        env = patch.dict(os.environ, {"RTP_INDEXER_FUSED_TOPK": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_integer_extremes_promotion_and_strided_destination(self) -> None:
        for dtype in (torch.int32, torch.int64):
            limits = torch.iinfo(dtype)
            x = torch.tensor(
                [[limits.min, -7, -1, 0, 1, limits.max - 1, limits.max]] * 4,
                dtype=dtype,
                device="cuda",
            )
            for off_dtype in (torch.int32, torch.int64):
                off_limits = torch.iinfo(off_dtype)
                offsets = torch.tensor(
                    [off_limits.max, off_limits.min, -7, 11],
                    device="cuda",
                    dtype=off_dtype,
                )
                for canonicalize in (False, True):
                    for out_dtype in (torch.int32, torch.int64):
                        with self.subTest(
                            dtype=dtype,
                            offset=off_dtype,
                            canonical=canonicalize,
                            output=out_dtype,
                        ):
                            backing = torch.full(
                                (6, 11), 123, device="cuda", dtype=out_dtype
                            )
                            output = backing[1:5, 2:9]
                            expected = backing.clone()
                            expected[1:5, 2:9] = reference_topk(
                                x, offsets, canonicalize
                            )
                            source = x.clone()

                            self.assertTrue(
                                try_write_topk(x, offsets, output, canonicalize)
                            )

                            self.assertTrue(torch.equal(backing, expected))
                            self.assertTrue(torch.equal(x, source))

    def test_production_writer_matches_disabled_path(self) -> None:
        for rows, width in ((0, 2048), (3, 17), (1247, 2048)):
            x = torch.randint(
                -7, 100000, (rows, width), device="cuda", dtype=torch.int32
            )
            offsets = torch.arange(rows, device="cuda", dtype=torch.int64) * 100001
            for canonicalize in ("0", "1"):
                with patch.dict(
                    os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": canonicalize}
                ):
                    expected = torch.empty_like(x)
                    with patch.dict(os.environ, {"RTP_INDEXER_FUSED_TOPK": "0"}):
                        _write_prefill_topk(x, offsets, expected)
                    output = torch.empty_like(x)

                    _write_prefill_topk(x, offsets, output)

                    self.assertTrue(torch.equal(output, expected))

    def test_output_alias_and_cpu_use_fallback(self) -> None:
        x = torch.zeros((2, 7), dtype=torch.int32, device="cuda")
        offsets = torch.ones(2, dtype=torch.int32, device="cuda")

        self.assertFalse(try_write_topk(x, offsets, x, True))
        self.assertFalse(
            try_write_topk(x, offsets, offsets[:, None].expand(2, 7), False)
        )
        self.assertFalse(try_write_topk(x.cpu(), offsets.cpu(), x.cpu(), True))

    def test_sort_boundaries_strided_sources_and_invalid_rows(self) -> None:
        for width in (1, 31, 33, 127, 1023, 2047, 2048, 2049):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(width=width, dtype=dtype):
                    x = torch.randint(
                        -3, 100, (19, width * 2), device="cuda", dtype=dtype
                    )[1::2, ::2]
                    x[0].fill_(-7)
                    x[1].fill_(torch.iinfo(dtype).max)
                    x[2].fill_(17)
                    offsets = torch.arange(27, device="cuda", dtype=dtype)[1::3]
                    backing = torch.full(
                        (11, width + 4), 123, device="cuda", dtype=dtype
                    )
                    out = backing[1:-1, 2:-2]
                    expected = backing.clone()
                    expected[1:-1, 2:-2] = reference_topk(x, offsets, True)

                    self.assertTrue(try_write_topk(x, offsets, out, True))

                    self.assertTrue(torch.equal(backing, expected))

    def test_graph_refreshes_topk_offsets(self) -> None:
        x = torch.randint(-7, 1000, (17, 33), device="cuda", dtype=torch.int32)
        offsets = torch.arange(17, device="cuda", dtype=torch.int64)
        output = torch.empty_like(x)
        for _ in range(3):
            try_write_topk(x, offsets, output, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertTrue(try_write_topk(x, offsets, output, True))
        for _ in range(3):
            offsets.add_(100)
            x.copy_(torch.randint(-7, 2000, x.shape, device="cuda", dtype=x.dtype))

            graph.replay()

            self.assertTrue(
                torch.equal(output, reference_topk(x, offsets, True).to(output.dtype))
            )

    def test_profiler_observes_fused_launches(self) -> None:
        x = torch.randint(-7, 100000, (1247, 2048), device="cuda", dtype=torch.int32)
        offsets = torch.arange(1247, device="cuda", dtype=torch.int32)
        out = torch.empty_like(x)
        try_write_topk(x, offsets, out, True)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            try_write_topk(x, offsets, out, True)
            torch.cuda.synchronize()

        kernels = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        print("INDEXER_FUSION_GPU_EVENTS", kernels)
        self.assertEqual(len(kernels), 1)
        self.assertIn("_sort_and_write_topk_kernel", kernels[0])


if __name__ == "__main__":
    main()

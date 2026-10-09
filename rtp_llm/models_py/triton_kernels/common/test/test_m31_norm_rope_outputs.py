"""Small GPU regression for optional M3.1 norm/RoPE contiguous outputs."""

import unittest

import torch
import triton

from rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope import (
    _gemma_norm_rope,
    minimax_m31_gemma_norm_rope_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ContiguousNormRopeTest(unittest.TestCase):
    def test_live_rows_reuse_binary_and_mask_tail(self):
        params = {param.name: param for param in _gemma_norm_rope.params}
        self.assertFalse(params["ROWS"].is_constexpr)
        self.assertTrue(params["ROWS"].do_not_specialize)
        previous = None
        for rows in (129, 131, 143, 129):
            with self.subTest(rows=rows):
                storage = torch.full((rows + 9, 9856), 7, device="cuda", dtype=torch.bfloat16)
                storage[1:rows + 1].fill_(2)
                expected = storage.clone()
                expected[1:rows + 1, :8704] = 1
                expected[1:rows + 1, 9216:] = 1
                weights = tuple(
                    torch.zeros(128, device="cuda", dtype=torch.bfloat16)
                    for _ in range(4)
                )
                positions = torch.zeros(rows + 8, device="cuda", dtype=torch.int32)
                cache = torch.cat(
                    (torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")),
                    dim=1,
                )
                compiled = _gemma_norm_rope[(triton.cdiv(rows, 8), 73)](
                    storage[1:rows + 1, :9216],
                    storage[1:rows + 1, 9216:9728],
                    storage[1:rows + 1, 9728:],
                    *weights,
                    positions,
                    cache,
                    9856, 9856, 9856, 1, 64, 64, 4, 4, 1e-6,
                    rows, 8, num_warps=1, enable_fp_fusion=False,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(storage, expected))
                if previous is not None:
                    self.assertEqual(compiled.hash, previous)
                previous = compiled.hash

    def test_query_fp8_outputs_preserve_bf16_rounding_and_graph(self):
        torch.manual_seed(20261006)
        for rows in (0, 1, 2047, 2048, 2049):
            with self.subTest(rows=rows):
                source = torch.randn(rows, 9856, device="cuda", dtype=torch.bfloat16)
                old, new = source.clone(), source.clone()
                positions = torch.arange(rows, device="cuda", dtype=torch.int64) % 257
                cache = torch.randn(257, 64, device="cuda", dtype=torch.float32)
                weights = tuple(
                    torch.randn(128, device="cuda", dtype=torch.bfloat16)
                    for _ in range(4)
                )
                outputs = tuple(
                    torch.empty(rows, h, 128, device="cuda", dtype=torch.float8_e4m3fn)
                    for h in (64, 4)
                )

                def run(storage, output=None):
                    minimax_m31_gemma_norm_rope_(
                        storage[:, :9216],
                        storage[:, 9216:9728],
                        storage[:, 9728:],
                        weights,
                        positions,
                        cache,
                        num_q_heads=64,
                        num_kv_heads=4,
                        num_index_heads=4,
                        query_fp8_outputs=output,
                    )

                def check():
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(old.view(torch.uint8), new.view(torch.uint8))
                    )
                    for expected, actual in zip(
                        (old[:, :8192], old[:, 9216:9728]), outputs
                    ):
                        expected = expected.reshape_as(actual).to(torch.float8_e4m3fn)
                        self.assertTrue(
                            torch.equal(
                                expected.view(torch.uint8), actual.view(torch.uint8)
                            )
                        )

                run(old)
                run(new, outputs)
                check()
                if not rows:
                    continue
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    new.copy_(source)
                    run(new, outputs)
                source.normal_()
                positions.add_(7).remainder_(257)
                old.copy_(source)
                run(old)
                graph.replay()
                check()
                del graph

    def test_grouped_inplace_threshold_and_tail_rows(self):
        torch.manual_seed(20261006)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for rows in (2047, 2048, 2049, 3377, 32767, 32769):
                with self.subTest(rows=rows):
                    source = torch.randn(
                        rows + 2, 9856, device="cuda", dtype=torch.bfloat16
                    )
                    old, new = source.clone(), source.clone()
                    positions = (
                        torch.arange(rows * 2, device="cuda", dtype=torch.int64) % 256
                    )[::2]
                    cache = torch.randn(256, 64, device="cuda", dtype=torch.float32)
                    weights = tuple(
                        torch.randn(128, device="cuda", dtype=torch.bfloat16)
                        for _ in range(4)
                    )

                    def reference():
                        _gemma_norm_rope[(rows, 73)](
                            old[1:-1, :9216],
                            old[1:-1, 9216:9728],
                            old[1:-1, 9728:],
                            *weights,
                            positions,
                            cache,
                            9856,
                            9856,
                            9856,
                            positions.stride(0),
                            cache.stride(0),
                            64,
                            4,
                            4,
                            1e-6,
                            rows,
                            1,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )

                    def candidate():
                        new.copy_(source)
                        minimax_m31_gemma_norm_rope_(
                            new[1:-1, :9216],
                            new[1:-1, 9216:9728],
                            new[1:-1, 9728:],
                            weights,
                            positions,
                            cache,
                            num_q_heads=64,
                            num_kv_heads=4,
                            num_index_heads=4,
                        )

                    old.copy_(source)
                    reference()
                    candidate()
                    stream.synchronize()
                    self.assertTrue(
                        torch.equal(old.view(torch.uint8), new.view(torch.uint8))
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        candidate()
                    for tiny in (False, True):
                        source.normal_()
                        if tiny:
                            source.mul_(1e-8)
                        positions.add_(7).remainder_(256)
                        old.copy_(source)
                        reference()
                        graph.replay()
                        stream.synchronize()
                        self.assertTrue(
                            torch.equal(old.view(torch.uint8), new.view(torch.uint8))
                        )
                    del graph

    def test_inplace_outputs_and_changed_graph_inputs(self):
        torch.manual_seed(20261004)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for rows in (0, 1, 2, 3, 16, 64, 65, 79, 80, 81, 127, 128, 129, 143, 159, 160):
                with self.subTest(rows=rows):
                    source = torch.randn(
                        rows, 9856, device="cuda", dtype=torch.bfloat16
                    )
                    old, new = source.clone(), source.clone()
                    positions = (
                        torch.arange(rows, device="cuda", dtype=torch.int64) % 256
                    )
                    cache = torch.randn(256, 64, device="cuda", dtype=torch.float32)
                    weights = tuple(
                        torch.randn(128, device="cuda", dtype=torch.bfloat16)
                        for _ in range(4)
                    )
                    outputs = tuple(
                        torch.empty(
                            rows, heads, 128, device="cuda", dtype=torch.bfloat16
                        )
                        for heads in (64, 4, 4, 1)
                    )

                    def run(storage, contiguous=None):
                        minimax_m31_gemma_norm_rope_(
                            storage[:, :9216],
                            storage[:, 9216:9728],
                            storage[:, 9728:],
                            weights,
                            positions,
                            cache,
                            num_q_heads=64,
                            num_kv_heads=4,
                            num_index_heads=4,
                            contiguous_outputs=contiguous,
                        )

                    def check():
                        stream.synchronize()
                        self.assertTrue(torch.equal(old, new))
                        self.assertTrue(
                            torch.equal(new[:, 8704:9216], source[:, 8704:9216])
                        )
                        slices = (
                            old[:, :8192],
                            old[:, 8192:8704],
                            old[:, 9216:9728],
                            old[:, 9728:],
                        )
                        for expected, actual in zip(slices, outputs):
                            self.assertTrue(
                                torch.equal(expected.reshape_as(actual), actual)
                            )

                    run(old)
                    run(new, outputs)
                    check()
                    if not rows:
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        new.copy_(source)
                        run(new, outputs)
                    for pattern in ("zero", "tiny", "random", "padded"):
                        source.normal_()
                        if pattern == "zero":
                            source.zero_()
                        elif pattern == "tiny":
                            source.mul_(1e-8)
                        elif pattern == "padded":
                            source[rows // 2 :].zero_()
                        positions.add_(7).remainder_(256)
                        old.copy_(source)
                        run(old)
                        graph.replay()
                        check()
                    with self.assertRaisesRegex(ValueError, "independent"):
                        run(new, (outputs[0], outputs[1], outputs[1], outputs[3]))
                    with self.assertRaisesRegex(ValueError, "four contiguous"):
                        run(new, outputs[:3])
                    del graph


if __name__ == "__main__":
    unittest.main()

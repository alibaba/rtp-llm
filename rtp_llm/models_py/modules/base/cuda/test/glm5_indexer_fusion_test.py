import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import (
    fused_indexer_cp_assemble as assemble,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
    fused_logits_head_gate,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    fused_prefill_rope_hadamard_qk,
)


class Glm5IndexerFusionTest(unittest.TestCase):
    def test_head_gate_dispatch_and_transposed_weights(self):
        torch.manual_seed(29)
        for rows in (0, 1, 32, 33):
            for transposed in (False, True):
                with self.subTest(rows=rows, transposed=transposed):
                    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                    weight = torch.randn(
                        512, 16, device="cuda", dtype=torch.bfloat16
                    ).t()
                    if not transposed:
                        weight = weight.contiguous()
                    scales = torch.rand(rows, 16, 1, device="cuda")
                    expected = (
                        (x.float() @ weight.float().t()).unsqueeze(-1) * scales * 0.125
                    )
                    actual = fused_logits_head_gate(x, scales, weight, 0.125)
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_prefill_fusion_falls_back_for_unsupported_geometry(self):
        from rtp_llm.models_py.modules.base.cuda.indexer_op import (
            _try_fused_prefill_rope_hadamard_qk,
        )

        for heads, rope, cache_width in ((3, 64, 64), (4, 32, 32), (4, 64, 128)):
            result = _try_fused_prefill_rope_hadamard_qk(
                torch.empty(1, heads, 128),
                torch.empty(1, 128),
                torch.tensor([0]),
                torch.empty(1, cache_width),
                rope,
                False,
            )
            self.assertIsNone(result)

    def test_fp32_loaded_bf16_head_gate_weights(self):
        # GLM5.3 stores these weights in BF16, then loads them as FP32.
        torch.manual_seed(42)
        x = torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16).float()
        scales = torch.rand(128, 32, 1, device="cuda")
        for current in (weight, weight.t().contiguous().t()):
            actual = fused_logits_head_gate(x, scales, current, 0.125)
            expected = (x.float() @ current.t()).unsqueeze(-1) * scales * 0.125
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_scatter_graph_replay_replaces_indices(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.triton_kv_scatter import (
            triton_kv_scatter,
        )

        src = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 2, 4)
        ids = torch.tensor([0, 2, 4], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            triton_kv_scatter(src, ids, 7)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = triton_kv_scatter(src, ids, 7)
        for indices in ([0, 2, 4], [1, 3, 6], [2, 4, 5]):
            ids.copy_(torch.tensor(indices, device="cuda"))
            graph.replay()
            expected = torch.zeros_like(actual)
            expected[ids] = src
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            triton_kv_scatter(src[:0], ids[:0], 7), torch.zeros_like(actual)
        )
        self.assertEqual(triton_kv_scatter(src[:0], ids[:0], 0).shape, (0, 2, 4))

    def test_slice_copy_non_flat_rows(self):
        from rtp_llm.models_py.triton_kernels.common.strided_slice_copy import (
            strided_slice_copy_,
        )

        for strided_src in (False, True):
            for strided_dst in (False, True):
                with self.subTest(src=strided_src, dst=strided_dst):
                    src = torch.randn(3, 4, 64, device="cuda", dtype=torch.bfloat16)
                    dst = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
                    if strided_src:
                        src = src[:, ::2]
                    else:
                        src = src[:, :2].contiguous()
                    if strided_dst:
                        dst = dst[:, ::2]
                    else:
                        dst = dst[:, :2].contiguous()
                    expected = dst.clone()
                    expected[..., 64:] = src
                    strided_slice_copy_(dst, src, 64)
                    torch.testing.assert_close(dst, expected)

    def test_cp_padding_restore_both_collective_layouts(self):
        for packed in (False, True):
            for empty_rank in (False, True):
                with self.subTest(packed=packed, empty_rank=empty_rank):
                    count = 0 if empty_rank else 3
                    quant = (
                        torch.arange(count * 128, device="cuda", dtype=torch.int32)
                        .to(torch.uint8)
                        .reshape(count, 128)
                    )
                    scale = torch.arange(
                        count * 4, device="cuda", dtype=torch.uint8
                    ).reshape(count, 4)
                    inverse = torch.tensor(
                        [-1, -1, -1, -1] if empty_rank else [0, -1, 1, 2], device="cuda"
                    )
                    restore = torch.tensor([7, 0, 5, 2, 6], device="cuda")
                    plan = SimpleNamespace(
                        total_local_T=4,
                        total_actual_local_T=count,
                        restore_indices=restore,
                    )
                    out_q = torch.empty(5, 128, device="cuda", dtype=torch.uint8)
                    out_s = torch.empty(5, 4, device="cuda", dtype=torch.uint8)
                    expected_q = torch.zeros(4, 128, device="cuda", dtype=torch.uint8)
                    expected_s = torch.zeros(4, 4, device="cuda", dtype=torch.uint8)
                    if count:
                        expected_q[[0, 2, 3]] = quant
                        expected_s[[0, 2, 3]] = scale

                    def gather(local, **kwargs):
                        return torch.cat([local, local], dim=0)

                    with patch.object(
                        assemble, "cp_opt_enabled", return_value=packed
                    ), patch.object(
                        assemble, "all_gather", side_effect=gather
                    ) as collective:
                        assemble.fused_copy_and_assemble_indexer_k(
                            plan=plan,
                            actual_k_quant=quant,
                            actual_k_scale=scale,
                            out_k_quant=out_q,
                            out_k_scale=out_s,
                            src_for_padded=inverse,
                        )
                    self.assertEqual(collective.call_count, 1 if packed else 2)
                    torch.testing.assert_close(out_q, expected_q.repeat(2, 1)[restore])
                    torch.testing.assert_close(out_s, expected_s.repeat(2, 1)[restore])

    def test_prefill_rope_hadamard_styles_and_strides(self):
        torch.manual_seed(31)
        hadamard = torch.ones(1, 1, device="cuda")
        while hadamard.size(0) < 128:
            hadamard = torch.cat(
                [
                    torch.cat([hadamard, hadamard], 1),
                    torch.cat([hadamard, -hadamard], 1),
                ],
                0,
            )
        # The Q GEMM consumes a BF16-scaled Hadamard matrix.
        hadamard = (hadamard / 128**0.5).bfloat16()
        for rows in (0, 1, 33):
            for neox in (False, True):
                with self.subTest(rows=rows, neox=neox):
                    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)[
                        :, :2
                    ]
                    k = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)[
                        :, :128
                    ]
                    positions = torch.arange(rows, device="cuda", dtype=torch.int32)
                    angles = torch.randn(max(rows, 1), 32, device="cuda")
                    cache = torch.cat([angles.cos(), angles.sin()], -1)

                    def rope(x):
                        result = x.clone()
                        cos, sin = cache[:rows, :32], cache[:rows, 32:]
                        if x.dim() == 3:
                            cos, sin = cos[:, None], sin[:, None]
                        first, second = (
                            (x[..., :32].float(), x[..., 32:64].float())
                            if neox
                            else (x[..., :64:2].float(), x[..., 1:64:2].float())
                        )
                        a, b = first * cos - second * sin, second * cos + first * sin
                        if neox:
                            result[..., :32], result[..., 32:64] = a, b
                        else:
                            result[..., :64:2], result[..., 1:64:2] = a, b
                        return result

                    q_ref, k_ref = rope(q), rope(k)
                    actual_q, actual_k = fused_prefill_rope_hadamard_qk(
                        q, k, positions, cache, 64, neox
                    )
                    torch.testing.assert_close(q, q_ref)
                    torch.testing.assert_close(actual_q, q_ref @ hadamard)
                    # K uses FP32 butterflies before the final BF16 store.
                    exact_h = hadamard.float().sign() / 128**0.5
                    torch.testing.assert_close(
                        actual_k, (k_ref.float() @ exact_h).bfloat16()
                    )


if __name__ == "__main__":
    unittest.main()

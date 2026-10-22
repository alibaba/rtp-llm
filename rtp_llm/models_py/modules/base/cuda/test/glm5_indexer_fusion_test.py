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
        for rows in (1, 32, 33):
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

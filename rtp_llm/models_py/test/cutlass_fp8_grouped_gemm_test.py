import itertools
import math
import random
from typing import Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import cutlass_moe_mm  # isort:skip


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match the
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output


class Fp8GroupedGemmOpTest(TestCase):
    NUM_EXPERT = [8, 128]
    PER_ACT_TOKEN = [False]
    PER_OUT_CHANNEL = [False]
    USE_BIAS = [False]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")
        self.device = "cuda"

    def _run_fp8_groupedgemm_op_test(
        self,
        num_expert: int,
        per_act_token: bool,
        per_out_ch: bool,
        use_bias: bool,
    ):
        out_dtype = torch.half

        a_tensors = []
        b_tensors = []
        a_scales_tensors = []
        b_scales_tensors = []
        baseline_tensors = []

        expert_offsets = torch.zeros(
            (num_expert + 1), device=self.device, dtype=torch.int32
        )

        problem_sizes = torch.zeros(
            (num_expert, 3), device=self.device, dtype=torch.int32
        )

        if not per_act_token:
            one_scale_a = torch.randn((1, 1), device=self.device, dtype=torch.float32)

        alignment = 16

        n_g = alignment * random.randint(1, 64)
        k_g = alignment * random.randint(1, 64)

        for g in range(num_expert):
            m_g = alignment * random.randint(1, 64)

            expert_offsets[g + 1] = expert_offsets[g] + m_g
            problem_sizes[g][0] = m_g
            problem_sizes[g][1] = n_g
            problem_sizes[g][2] = k_g

            m_a_scales = m_g if per_act_token else 1
            n_b_scales = n_g if per_out_ch else 1

            # Create group-specific A and B (FP8) and output (FP16/FP32)
            a_g = to_fp8(torch.randn((m_g, k_g), device=self.device))
            b_g = to_fp8(torch.randn((n_g, k_g), device=self.device).t())
            a_tensors.append(a_g)
            b_tensors.append(b_g)

            # Set up A/B scales
            scale_b = torch.randn(
                (1, n_b_scales), device=self.device, dtype=torch.float32
            )
            b_scales_tensors.append(scale_b)

            if per_act_token:
                scale_a = torch.randn(
                    (m_a_scales, 1), device=self.device, dtype=torch.float32
                )
                a_scales_tensors.append(scale_a)
            else:
                scale_a = one_scale_a

            # Compute baseline result for this group
            baseline_g = baseline_scaled_mm(a_g, b_g, scale_a, scale_b, out_dtype, None)
            baseline_tensors.append(baseline_g)

        a_tensors_stacked = torch.empty(
            (expert_offsets[num_expert], k_g),
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        b_tensors_stacked = torch.empty(
            (num_expert, n_g, k_g), device=self.device, dtype=torch.float8_e4m3fn
        )

        for g in range(num_expert):
            a_tensors_stacked[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
            b_tensors_stacked[g] = b_tensors[g].t()

        b_tensors_stacked = b_tensors_stacked.transpose(1, 2)

        if per_act_token:
            a_scales_tensors_stacked = torch.empty(
                (expert_offsets[num_expert], 1), device=self.device, dtype=torch.float32
            )
            for g in range(num_expert):
                a_scales_tensors_stacked[expert_offsets[g] : expert_offsets[g + 1]] = (
                    a_scales_tensors[g]
                )
        else:
            a_scales_tensors_stacked = one_scale_a

        b_scales_tensors_stacked = torch.empty(
            (num_expert, n_b_scales), device=self.device, dtype=torch.float32
        )
        for g in range(num_expert):
            b_scales_tensors_stacked[g] = b_scales_tensors[g]

        out_tensors_stacked = torch.zeros(
            (expert_offsets[num_expert], n_g), device=self.device, dtype=out_dtype
        )

        ab_strides = torch.full(
            (num_expert,),
            a_tensors_stacked.stride(0),
            device=self.device,
            dtype=torch.int64,
        )
        c_strides = torch.full(
            (num_expert,),
            out_tensors_stacked.stride(0),
            device=self.device,
            dtype=torch.int64,
        )
        cutlass_moe_mm(
            out_tensors_stacked,
            a_tensors_stacked,
            b_tensors_stacked,
            a_scales_tensors_stacked,
            b_scales_tensors_stacked,
            expert_offsets[:-1],
            problem_sizes,
            ab_strides,
            ab_strides,
            c_strides,
            per_act_token,
            per_out_ch,
        )

        # Validate each group's result against the baseline
        for g in range(num_expert):
            baseline = baseline_tensors[g]
            c = out_tensors_stacked[expert_offsets[g] : expert_offsets[g + 1]]
            torch.testing.assert_close(c, baseline, rtol=1e-2, atol=5e-4)

    def test_fp8_groupedgemm(self):
        for params in itertools.product(
            self.NUM_EXPERT, self.PER_ACT_TOKEN, self.PER_OUT_CHANNEL, self.USE_BIAS
        ):
            with self.subTest(
                num_expert=params[0],
                per_act_token=params[1],
                per_out_ch=params[2],
                use_bias=params[3],
            ):
                self._run_fp8_groupedgemm_op_test(*params)


if __name__ == "__main__":
    main()

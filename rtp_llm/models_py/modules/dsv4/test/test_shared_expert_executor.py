import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.dsv4.moe.expert import Expert
from rtp_llm.models_py.modules.dsv4.moe._shared_expert_triton import (
    quant_bf16_fp8_packed_ue8m0,
)
from rtp_llm.models_py.modules.dsv4.moe._silu_mul_fp8_quant_triton import (
    silu_mul_fp8_quant_packed_from_parts,
)
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    FusedSharedExpertExecutor,
    FusedSharedExpertFastPath,
    OverlapSharedExpertExecutor,
    SequentialSharedExpertExecutor,
    W13SharedExpert,
    combine_routed_and_shared,
    get_shared_expert_executor,
)
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import concat_0


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class _Shared(nn.Module):
    def forward(self, x):
        return (x.float() * 0.25).to(x.dtype)


def _quant_weight(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight_bf16.shape
    if is_deep_gemm_e8m0_used():
        k_packed = (in_features + 511) // 512
        aligned_rows = FusedSharedExpertFastPath._tma_aligned_rows(
            out_features,
            torch.empty((), dtype=torch.int32).element_size(),
        )
        scale_storage = torch.empty(
            (k_packed, aligned_rows),
            dtype=torch.int32,
            device=weight_bf16.device,
        )
        scales = scale_storage.as_strided(
            (out_features, k_packed),
            (1, aligned_rows),
        )
        scales.fill_(0x7F7F7F7F)
        return weight_bf16.to(torch.float8_e4m3fn), scales
    return (
        weight_bf16.t().contiguous().to(torch.float8_e4m3fn),
        torch.ones(
            (in_features + 127) // 128,
            (out_features + 127) // 128,
            dtype=torch.float32,
            device=weight_bf16.device,
        ),
    )


def _make_shared_expert(
    dim: int = 256,
    inter: int = 256,
    swiglu_limit: float = 0.0,
) -> tuple[W13SharedExpert, Expert]:
    torch.manual_seed(123)
    device = torch.device("cuda")
    w1_bf16 = torch.randn((inter, dim), device=device, dtype=torch.bfloat16) * 0.05
    w2_bf16 = torch.randn((dim, inter), device=device, dtype=torch.bfloat16) * 0.05
    w3_bf16 = torch.randn((inter, dim), device=device, dtype=torch.bfloat16) * 0.05
    w1_w, w1_s = _quant_weight(w1_bf16)
    w2_w, w2_s = _quant_weight(w2_bf16)
    w3_w, w3_s = _quant_weight(w3_bf16)
    split_ref = Expert(
        dim,
        inter,
        swiglu_limit=swiglu_limit,
        storage="fp8",
        expert_weights={
            "w1_w": w1_w,
            "w1_s": w1_s,
            "w2_w": w2_w,
            "w2_s": w2_s,
            "w3_w": w3_w,
            "w3_s": w3_s,
        },
    )
    shared_w13 = W13SharedExpert(
        dim,
        inter,
        expert_weights={
            "w13_w": concat_0([w1_w, w3_w]),
            "w13_s": FusedSharedExpertFastPath._merge_weight_scales(w1_s, w3_s),
            "w2_w": w2_w,
            "w2_s": w2_s,
        },
        swiglu_limit=swiglu_limit,
    )
    return shared_w13, split_ref


def _split_reference(
    shared: Expert,
    x: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

    T, D = x.shape
    inter = shared.w1.weight.shape[0]
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_scale_storage = FusedSharedExpertFastPath._scale_storage(
        (D // 128 + 3) // 4,
        max(T, 1),
        x.device,
    )
    x_scale = FusedSharedExpertFastPath._scale_view(x_scale_storage, T)
    gate = torch.empty((T, inter), dtype=torch.bfloat16, device=x.device)
    up = torch.empty((T, inter), dtype=torch.bfloat16, device=x.device)
    hidden_fp8 = torch.empty((T, inter), dtype=torch.float8_e4m3fn, device=x.device)
    hidden_scale_storage = FusedSharedExpertFastPath._scale_storage(
        (inter // 128 + 3) // 4,
        max(T, 1),
        x.device,
    )
    hidden_scale = FusedSharedExpertFastPath._scale_view(hidden_scale_storage, T)
    out = torch.empty((T, D), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    quant_bf16_fp8_packed_ue8m0(x, x_fp8, x_scale, group_size=128, eps=1.0e-4)
    fp8_gemm_nt(
        (x_fp8, x_scale),
        (shared.w1.weight, shared.w1.weight_scales),
        gate,
        disable_ue8m0_cast=False,
    )
    fp8_gemm_nt(
        (x_fp8, x_scale),
        (shared.w3.weight, shared.w3.weight_scales),
        up,
        disable_ue8m0_cast=False,
    )
    silu_mul_fp8_quant_packed_from_parts(
        gate,
        up,
        clamp_limit=swiglu_limit,
        group_size=128,
        output_q=hidden_fp8,
        output_scale=hidden_scale,
    )
    fp8_gemm_nt(
        (hidden_fp8, hidden_scale),
        (shared.w2.weight, shared.w2.weight_scales),
        out,
        disable_ue8m0_cast=False,
    )
    return out


def _fake_fp8_gemm_nt(a, b, output, *args, **kwargs) -> None:
    del args, kwargs
    a_q = a[0]
    b_q = b[0]
    output.copy_((a_q.float() @ b_q.float().t()).to(torch.bfloat16))


class TestSharedExpertExecutor(unittest.TestCase):
    def test_combine_preserves_fp32_accumulate_semantics(self):
        routed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        shared = torch.tensor([[0.5, -0.25], [0.125, -0.5]], dtype=torch.float32)
        with _env("DSV4_MOE_STRICT_FUSED", "0"):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.float() + shared.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_bf16_add_experimental_switch(self):
        routed = torch.randn(4, 8, dtype=torch.float32)
        shared = torch.randn(4, 8, dtype=torch.float32)
        with _env("DSV4_MOE_STRICT_FUSED", "0"), _env(
            "DSV4_SHARED_EXPERT_BF16_ADD", "1"
        ):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.to(torch.bfloat16) + shared.to(torch.bfloat16)).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_strict_rejects_bf16_add_switch(self):
        routed = torch.randn(4, 8, dtype=torch.float32)
        shared = torch.randn(4, 8, dtype=torch.float32)
        with _env("DSV4_SHARED_EXPERT_BF16_ADD", "1"):
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                combine_routed_and_shared(routed, shared, torch.bfloat16)

    def test_strict_rejects_generic_shared_path(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        with self.assertRaisesRegex(RuntimeError, "generic Expert.forward"):
            executor.start(_Shared(), x)

    def test_executor_dispatch(self):
        with _env("DSV4_SHARED_EXPERT_MODE", "sequential"):
            self.assertIsInstance(get_shared_expert_executor(), SequentialSharedExpertExecutor)
        with _env("DSV4_SHARED_EXPERT_MODE", "overlap"):
            self.assertIsInstance(get_shared_expert_executor(), OverlapSharedExpertExecutor)

    def test_sequential_executor(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        with _env("DSV4_MOE_STRICT_FUSED", "0"):
            executor.start(_Shared(), x)
        got = executor.finish()
        ref = _Shared()(x).float()
        self.assertTrue(torch.equal(got, ref))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_matches_sequential(self):
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        overlap = OverlapSharedExpertExecutor()
        with _env("DSV4_MOE_STRICT_FUSED", "0"):
            overlap.start(shared, x)
        got = overlap.finish()
        ref = shared(x).float()
        self.assertTrue(torch.equal(got.cpu(), ref.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_merged_w13_matches_split_reference(self):
        swiglu_limit = 1.5
        shared, split_ref = _make_shared_expert(swiglu_limit=swiglu_limit)
        executor = FusedSharedExpertExecutor(
            max_tokens_per_rank=64,
            dim=256,
            inter_dim=256,
            swiglu_limit=swiglu_limit,
        )
        executor.prepare(shared)
        self.assertTrue(FusedSharedExpertFastPath.has_merged_w13(shared))
        w13_w, w13_s = FusedSharedExpertFastPath._linear_parts(shared.w13)
        self.assertEqual(
            w13_w.shape,
            (512, 256),
        )
        self.assertEqual(w13_s.shape[0], 512)
        if w13_s.dtype == torch.int32:
            self.assertEqual(w13_s.stride(0), 1)
        self.assertFalse(hasattr(shared, "w1"))
        self.assertFalse(hasattr(shared, "w3"))

        for tokens in (0, 1, 33):
            with self.subTest(tokens=tokens):
                torch.manual_seed(1000 + tokens)
                x = torch.randn(
                    (tokens, 256),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                gemm_calls = []

                def fake_with_record(a, b, output, *args, **kwargs):
                    gemm_calls.append((tuple(a[0].shape), tuple(b[0].shape), tuple(output.shape)))
                    _fake_fp8_gemm_nt(a, b, output, *args, **kwargs)

                with mock.patch(
                    "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.fp8_gemm_nt",
                    side_effect=fake_with_record,
                ):
                    got = executor.run(shared, x)
                    merged_calls = list(gemm_calls)
                    ref = _split_reference(split_ref, x, swiglu_limit)
                if tokens == 0:
                    self.assertEqual(tuple(got.shape), (0, 256))
                    self.assertEqual(merged_calls, [])
                    continue
                self.assertEqual(
                    merged_calls,
                    [
                        ((tokens, 256), (512, 256), (tokens, 512)),
                        ((tokens, 256), (256, 256), (tokens, 256)),
                    ],
                )
                self.assertLess(calc_diff(got, ref), 0.0011)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_w13_generic_forward_matches_split_expert(self):
        swiglu_limit = 1.5
        shared, split_ref = _make_shared_expert(swiglu_limit=swiglu_limit)
        x = torch.randn((5, 256), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(hasattr(shared, "w1"))
        self.assertFalse(hasattr(shared, "w3"))

        with mock.patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear.fp8_gemm_nt",
            side_effect=_fake_fp8_gemm_nt,
        ):
            got = shared(x)
            ref = split_ref(x)
        self.assertLess(calc_diff(got, ref), 0.0011)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_strict_fused_requires_prepared_w13(self):
        _, shared = _make_shared_expert()
        executor = FusedSharedExpertExecutor(
            max_tokens_per_rank=8,
            dim=256,
            inter_dim=256,
        )
        with self.assertRaisesRegex(RuntimeError, "loader-prepared w13"):
            executor.prepare(shared)


if __name__ == "__main__":
    unittest.main()

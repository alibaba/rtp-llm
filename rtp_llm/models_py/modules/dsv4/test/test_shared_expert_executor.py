import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.dsv4.moe._shared_expert_triton import (
    quant_bf16_fp8_packed_ue8m0,
)
from rtp_llm.models_py.modules.dsv4.moe._silu_mul_fp8_quant_triton import (
    silu_mul_fp8_quant_packed_from_parts,
)
from rtp_llm.models_py.modules.dsv4.moe.expert import Expert
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    _SHARED_EXPERT_STREAM_CACHE,
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


class _SharedWithCudaWeight(_Shared):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(1, device="cuda")


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
        os.environ.pop("DSV4_SHARED_EXPERT_MODE", None)
        self.assertIsInstance(
            get_shared_expert_executor(), SequentialSharedExpertExecutor
        )
        with _env("DSV4_SHARED_EXPERT_MODE", "sequential"):
            self.assertIsInstance(
                get_shared_expert_executor(), SequentialSharedExpertExecutor
            )
        with _env("DSV4_SHARED_EXPERT_MODE", "overlap"):
            self.assertIsInstance(
                get_shared_expert_executor(), OverlapSharedExpertExecutor
            )
        with _env("DSV4_SHARED_EXPERT_MODE", "auto"):
            self.assertIsInstance(
                get_shared_expert_executor(), OverlapSharedExpertExecutor
            )

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
    def test_overlap_executor_reuses_fixed_device_stream(self):
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        first = OverlapSharedExpertExecutor()
        second = OverlapSharedExpertExecutor()

        with _env("DSV4_MOE_STRICT_FUSED", "0"):
            first.start(shared, x)
            first_stream = first._active_stream
            self.assertIsNotNone(first_stream)
            first.finish()

            second.start(shared, x)
            second_stream = second._active_stream
            self.assertIs(second_stream, first_stream)
            second.finish()

        device_index = x.device.index
        assert device_index is not None
        self.assertIs(_SHARED_EXPERT_STREAM_CACHE[device_index], first_stream)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_prepare_precreates_device_stream(self):
        _SHARED_EXPERT_STREAM_CACHE.clear()
        shared = _SharedWithCudaWeight()
        executor = OverlapSharedExpertExecutor()

        executor.prepare(shared)

        device_index = shared.weight.device.index
        assert device_index is not None
        self.assertIn(device_index, _SHARED_EXPERT_STREAM_CACHE)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_capture_requires_precreated_stream(self):
        _SHARED_EXPERT_STREAM_CACHE.clear()
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        executor = OverlapSharedExpertExecutor()

        with _env("DSV4_MOE_STRICT_FUSED", "0"), mock.patch(
            "torch.cuda.is_current_stream_capturing",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "not created before CUDA graph capture"
            ):
                executor.start(shared, x)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_captures_with_precreated_stream(self):
        _SHARED_EXPERT_STREAM_CACHE.clear()
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        executor = OverlapSharedExpertExecutor()
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)

        with _env("DSV4_MOE_STRICT_FUSED", "0"):
            executor.start(shared, x)
            warmup_stream = executor._active_stream
            self.assertIsNotNone(warmup_stream)
            executor.finish()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                executor.start(shared, x)
                self.assertIs(executor._active_stream, warmup_stream)
                out.copy_(executor.finish())

            x.mul_(2.0)
            graph.replay()
            torch.cuda.synchronize()

        ref = shared(x).float()
        self.assertTrue(torch.equal(out.cpu(), ref.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_threshold_falls_back_to_sequential(self):
        x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
        shared = _Shared().cuda()
        executor = OverlapSharedExpertExecutor()

        with _env("DSV4_MOE_STRICT_FUSED", "0"), _env(
            "DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "1"
        ):
            executor.start(shared, x)
            self.assertIsNone(executor._active_stream)
            got = executor.finish()

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
                    gemm_calls.append(
                        (tuple(a[0].shape), tuple(b[0].shape), tuple(output.shape))
                    )
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


def _make_shared_expert_e8m0(
    dim: int = 256,
    inter: int = 256,
    swiglu_limit: float = 0.0,
) -> W13SharedExpert:
    """Shared expert whose scales use the checkpoint UE8M0 (e8m0) format —
    the format the SM120 fused path repacks from at prepare() time."""
    torch.manual_seed(777)
    device = torch.device("cuda")

    def q(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w8 = w_bf16.to(torch.float8_e4m3fn)
        s = torch.ones(
            w8.shape[0] // 128,
            w8.shape[1] // 128,
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e8m0fnu)
        return w8, s

    w13_w = torch.randn((2 * inter, dim), device=device, dtype=torch.bfloat16) * 0.05
    w2_w = torch.randn((dim, inter), device=device, dtype=torch.bfloat16) * 0.05
    w13_w, w13_s = q(w13_w)
    w2_w, w2_s = q(w2_w)
    return W13SharedExpert(
        dim,
        inter,
        expert_weights={"w13_w": w13_w, "w13_s": w13_s, "w2_w": w2_w, "w2_s": w2_s},
        swiglu_limit=swiglu_limit,
    )


class TestSharedExpertSM120FusedPath(unittest.TestCase):
    """SM120 fused shared expert: capability gating + real-kernel numerics."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from rtp_llm.models_py.utils.arch import is_sm120

        if not is_sm120(torch.device("cuda")):
            self.skipTest("SM120-only capability path")
        enabled = mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.shared_expert._SM120_FUSED_ENABLED",
            True,
        )
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_disabled_preserves_sm120_fallback(self):
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.shared_expert._SM120_FUSED_ENABLED",
            False,
        ):
            shared = _make_shared_expert_e8m0()
            fp = FusedSharedExpertFastPath(dim=256, inter_dim=256)
            fp.prepare(shared)
            x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(getattr(shared.w13, "_dsv4_sm120_packed_scale", None))
            self.assertFalse(fp.can_run(shared, x))

    def test_can_run_requires_packed_scales(self):
        # Without e8m0 originals (the generic fixture) nothing is packed, so
        # the SM120 gate stays closed and the generic fallback is preserved.
        shared, _ = _make_shared_expert()
        fp = FusedSharedExpertFastPath(dim=256, inter_dim=256)
        fp.prepare(shared)
        x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(getattr(shared.w13, "_dsv4_sm120_packed_scale", None))
        self.assertFalse(FusedSharedExpertFastPath.can_run(shared, x))

    def test_prepare_builds_packed_scales_idempotent(self):
        shared = _make_shared_expert_e8m0()
        fp = FusedSharedExpertFastPath(dim=256, inter_dim=256)
        fp.prepare(shared)
        p13 = shared.w13._dsv4_sm120_packed_scale
        p2 = shared.w2._dsv4_sm120_packed_scale
        self.assertIsNotNone(p13)
        self.assertIsNotNone(p2)
        self.assertEqual(p13.dtype, torch.int32)
        # packed [N, K/512] int32 views: w13 N=512,K=256 -> k_packed 1; w2 N=256,K=256 -> 1
        self.assertEqual(p13.shape[0], 512)
        self.assertEqual(p2.shape[0], 256)
        x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
        self.assertTrue(FusedSharedExpertFastPath.can_run(shared, x))
        fp.prepare(shared)  # idempotent: same object, no rebuild
        self.assertIs(shared.w13._dsv4_sm120_packed_scale, p13)

    def test_linear_parts_prefers_packed_scale(self):
        shared = _make_shared_expert_e8m0()
        FusedSharedExpertFastPath(dim=256, inter_dim=256).prepare(shared)
        _, scale = FusedSharedExpertFastPath._linear_parts(shared.w13)
        self.assertIs(scale, shared.w13._dsv4_sm120_packed_scale)

    def test_sm120_fused_real_kernel_numerics(self):
        # Real kernels end-to-end (no mocked GEMM), compared against a
        # same-weights, same-quantizer-contract split reference: the merged
        # w13 is split back into w1/w3 halves, all scales are repacked from
        # the same e8m0 originals, and the reference runs the same real
        # fp8_gemm_nt + silu_mul_fp8_quant_packed kernels. Same-contract
        # comparisons use the established FP8 tolerance 0.0011; declared
        # before seeing results.
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt
        from rtp_llm.models_py.modules.dsv4.utils import _repack_v4_fp8_scale_to_int32

        swiglu_limit = 1.5
        dim = inter = 256
        shared = _make_shared_expert_e8m0(swiglu_limit=swiglu_limit)
        executor = FusedSharedExpertExecutor(
            max_tokens_per_rank=64, dim=dim, inter_dim=inter, swiglu_limit=swiglu_limit
        )
        executor.prepare(shared)
        self.assertTrue(
            FusedSharedExpertFastPath.can_run(
                shared, torch.randn((1, dim), device="cuda", dtype=torch.bfloat16)
            )
        )
        # same-weight split parts from the checkpoint originals (the linear
        # keeps the fp8 weight verbatim; the e8m0 scales are the stashed ones)
        w13_w = shared.w13.weight
        w13_s = shared._dsv4_w13_scale_e8m0
        w2_w = shared.w2.weight
        w2_s = shared._dsv4_w2_scale_e8m0
        w1_w, w3_w = w13_w[:inter], w13_w[inter:]
        w1_s = _repack_v4_fp8_scale_to_int32(w13_s[: inter // 128])
        w3_s = _repack_v4_fp8_scale_to_int32(w13_s[inter // 128 :])
        w2_sp = _repack_v4_fp8_scale_to_int32(w2_s)

        for tokens in (1, 33, 64):
            with self.subTest(tokens=tokens):
                torch.manual_seed(2000 + tokens)
                x = torch.randn((tokens, dim), device="cuda", dtype=torch.bfloat16)
                got = executor.run(shared, x)

                # split reference, same kernels and contract
                x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                xs_st = FusedSharedExpertFastPath._scale_storage(
                    (dim // 128 + 3) // 4, max(tokens, 1), x.device
                )
                xs = FusedSharedExpertFastPath._scale_view(xs_st, tokens)
                quant_bf16_fp8_packed_ue8m0(x, x_fp8, xs, group_size=128, eps=1.0e-4)
                gate = torch.empty((tokens, inter), dtype=torch.bfloat16, device="cuda")
                up = torch.empty((tokens, inter), dtype=torch.bfloat16, device="cuda")
                fp8_gemm_nt((x_fp8, xs), (w1_w, w1_s), gate, disable_ue8m0_cast=False)
                fp8_gemm_nt((x_fp8, xs), (w3_w, w3_s), up, disable_ue8m0_cast=False)
                h_fp8 = torch.empty(
                    (tokens, inter), dtype=torch.float8_e4m3fn, device="cuda"
                )
                hs_st = FusedSharedExpertFastPath._scale_storage(
                    (inter // 128 + 3) // 4, max(tokens, 1), x.device
                )
                hs = FusedSharedExpertFastPath._scale_view(hs_st, tokens)
                silu_mul_fp8_quant_packed_from_parts(
                    gate,
                    up,
                    clamp_limit=swiglu_limit,
                    group_size=128,
                    output_q=h_fp8,
                    output_scale=hs,
                )
                ref = torch.empty((tokens, dim), dtype=torch.bfloat16, device="cuda")
                fp8_gemm_nt((h_fp8, hs), (w2_w, w2_sp), ref, disable_ue8m0_cast=False)

                diff = calc_diff(got, ref)
                print(
                    f"sm120 fused vs same-contract split calc_diff tokens={tokens}: {diff:.6f}"
                )
                self.assertLess(diff, 0.0011)

    def test_sm120_overlap_matches_sequential(self):
        # The overlap executor must produce the same outputs as sequential
        # when the fused path is engaged on SM120 (producer/consumer stream
        # ordering is the only difference).
        swiglu_limit = 1.5
        shared = _make_shared_expert_e8m0(swiglu_limit=swiglu_limit)
        seq = SequentialSharedExpertExecutor(
            FusedSharedExpertExecutor(
                max_tokens_per_rank=64,
                dim=256,
                inter_dim=256,
                swiglu_limit=swiglu_limit,
            )
        )
        ovl = OverlapSharedExpertExecutor(
            FusedSharedExpertExecutor(
                max_tokens_per_rank=64,
                dim=256,
                inter_dim=256,
                swiglu_limit=swiglu_limit,
            )
        )
        seq.prepare(shared)
        ovl.prepare(shared)
        for tokens in (1, 33, 64):
            with self.subTest(tokens=tokens):
                torch.manual_seed(3000 + tokens)
                x = torch.randn((tokens, 256), device="cuda", dtype=torch.bfloat16)
                seq.start(shared, x)
                seq_out = seq.finish()
                ovl.start(shared, x)
                ovl_out = ovl.finish()
                self.assertTrue(
                    torch.equal(seq_out, ovl_out),
                    f"overlap != sequential at tokens={tokens}",
                )

    def test_sm120_workspace_grow_shrink_reuse(self):
        # Grow-only workspace: outputs must stay correct when the token count
        # grows past the initial capacity and shrinks again (buffer reuse).
        swiglu_limit = 1.5
        shared = _make_shared_expert_e8m0(swiglu_limit=swiglu_limit)
        executor = FusedSharedExpertExecutor(
            max_tokens_per_rank=8, dim=256, inter_dim=256, swiglu_limit=swiglu_limit
        )
        executor.prepare(shared)
        outs = {}
        for tokens in (4, 64, 3, 48, 1):  # grow, grow, shrink, grow, shrink
            with self.subTest(tokens=tokens):
                torch.manual_seed(4000 + tokens)
                x = torch.randn((tokens, 256), device="cuda", dtype=torch.bfloat16)
                got = executor.run(shared, x)
                self.assertEqual(tuple(got.shape), (tokens, 256))
                self.assertTrue(torch.isfinite(got.float()).all())
                outs[tokens] = got
        # deterministic: re-running a size reproduces the earlier output
        torch.manual_seed(4003)
        x = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.equal(executor.run(shared, x), outs[3]))


if __name__ == "__main__":
    unittest.main()

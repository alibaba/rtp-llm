import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import MoeWorkspace
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import Expert
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    FusedSharedExpertExecutor,
    FusedSharedExpertFastPath,
    SequentialSharedExpertExecutor,
    W13SharedExpert,
    _run_shared_expert,
    combine_routed_and_shared,
)
from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
    quant_bf16_fp8_packed_ue8m0,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed_from_parts,
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
    w13_w, w13_s = _quant_weight(concat_0([w1_bf16, w3_bf16]))
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
            "w13_w": w13_w,
            "w13_s": w13_s,
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
    @staticmethod
    def _tma_scale(rows: int, k: int, device: str | torch.device = "cpu") -> torch.Tensor:
        packed_k = (k // 128 + 3) // 4
        aligned = ((rows + 3) // 4) * 4
        storage = torch.empty((packed_k, aligned), dtype=torch.int32, device=device)
        return storage.as_strided((rows, packed_k), (1, aligned)).fill_(0x7F7F7F7F)

    @classmethod
    def _prepare_only_shared(
        cls, inter: int = 256, dim: int = 256, device: str | torch.device = "cpu"
    ) -> nn.Module:
        shared = nn.Module()
        shared.w13 = mock.Mock(
            weight=torch.empty((2 * inter, dim), dtype=torch.float8_e4m3fn, device=device),
            weight_scales=cls._tma_scale(2 * inter, dim, device=device),
        )
        shared.w2 = mock.Mock(
            weight=torch.empty((dim, inter), dtype=torch.float8_e4m3fn, device=device),
            weight_scales=cls._tma_scale(dim, inter, device=device),
        )
        return shared

    def test_fast_path_prepare_infers_inter_dim_and_syncs_dim(self):
        fast_path = FusedSharedExpertFastPath(dim=128, inter_dim=None)

        fast_path.prepare(self._prepare_only_shared(inter=256, dim=256))

        self.assertEqual(fast_path.inter_dim, 256)
        self.assertEqual(fast_path.dim, 256)

    def test_fast_path_prepare_uses_weight_inter_dim(self):
        fast_path = FusedSharedExpertFastPath(dim=256, inter_dim=128)

        fast_path.prepare(self._prepare_only_shared(inter=256))

        self.assertEqual(fast_path.inter_dim, 256)

    def test_fast_path_prepare_rejects_malformed_w2_or_strides(self):
        fast_path = FusedSharedExpertFastPath(dim=256, inter_dim=256)
        bad_w2_shape = self._prepare_only_shared()
        bad_w2_shape.w2.weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(RuntimeError, "w2 weight shape mismatch"):
            fast_path.prepare(bad_w2_shape)

        bad_w2_dtype = self._prepare_only_shared()
        bad_w2_dtype.w2.weight = torch.empty((256, 256), dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "float8_e4m3fn"):
            fast_path.prepare(bad_w2_dtype)

        bad_stride = self._prepare_only_shared(dim=640, inter=256)
        bad_stride.w13.weight_scales = torch.empty((512, 2), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "TMA-aligned"):
            fast_path.prepare(bad_stride)

        bad_k = self._prepare_only_shared(dim=256, inter=192)
        with self.assertRaisesRegex(RuntimeError, "divisible by 128"):
            fast_path.prepare(bad_k)

    def test_soft_prepare_allows_non_ue8m0_unless_strict(self):
        shared = nn.Module()
        shared.w13 = mock.Mock(
            weight=torch.empty((512, 256), dtype=torch.float8_e4m3fn),
            weight_scales=torch.ones((4, 2), dtype=torch.float32),
        )
        shared.w2 = mock.Mock(
            weight=torch.empty((256, 256), dtype=torch.float8_e4m3fn),
            weight_scales=torch.ones((2, 2), dtype=torch.float32),
        )
        executor = SequentialSharedExpertExecutor(
            FusedSharedExpertFastPath(dim=256, inter_dim=256)
        )
        with _env("MOE_STRICT_FUSED", "0"):
            executor.prepare(shared)
        with _env("MOE_STRICT_FUSED", "1"):
            with self.assertRaisesRegex(RuntimeError, "int32"):
                executor.prepare(shared)

    def test_fp32_block_w13_shared_expert_allowed_under_strict_fused(self):
        fake_w13 = nn.Linear(256, 512, bias=False)
        fake_w13.weight = nn.Parameter(
            torch.empty((512, 256), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        fake_w13.weight_scales = torch.ones((4, 2), dtype=torch.float32)
        fake_w2 = nn.Linear(256, 256, bias=False)
        fake_w2.weight = nn.Parameter(
            torch.empty((256, 256), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        fake_w2.weight_scales = torch.ones((2, 2), dtype=torch.float32)
        with mock.patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert.create_fp8_linear",
            side_effect=[fake_w13, fake_w2],
        ):
            shared = W13SharedExpert(
                256,
                256,
                {
                    "w13_w": fake_w13.weight,
                    "w13_s": fake_w13.weight_scales,
                    "w2_w": fake_w2.weight,
                    "w2_s": fake_w2.weight_scales,
                },
            )
        executor = SequentialSharedExpertExecutor(
            FusedSharedExpertFastPath(dim=256, inter_dim=256)
        )
        with _env("MOE_STRICT_FUSED", "1"):
            executor.prepare(shared)
            expected = torch.full((3, 256), 2.5, dtype=torch.bfloat16)
            with mock.patch.object(
                shared, "forward", return_value=expected
            ) as mock_fwd:
                got = _run_shared_expert(
                    shared, torch.zeros((3, 256), dtype=torch.bfloat16), executor._fast_path
                )
        mock_fwd.assert_called_once()
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, expected.float(), rtol=0, atol=0)

    def test_workspace_views_are_shared_and_invalidated_on_growth(self):
        ws = MoeWorkspace()
        first = FusedSharedExpertFastPath(
            max_tokens_per_rank=4,
            dim=128,
            inter_dim=128,
            moe_workspace=ws,
        )
        second = FusedSharedExpertFastPath(
            max_tokens_per_rank=4,
            dim=128,
            inter_dim=128,
            moe_workspace=ws,
        )
        grower = FusedSharedExpertFastPath(
            max_tokens_per_rank=9,
            dim=128,
            inter_dim=128,
            moe_workspace=ws,
        )
        isolated = FusedSharedExpertFastPath(
            max_tokens_per_rank=4,
            dim=128,
            inter_dim=128,
            moe_workspace=MoeWorkspace(),
        )
        x3 = torch.empty((3, 128), dtype=torch.bfloat16)

        def aligned(rows: int, element_size: int) -> int:
            del element_size
            return ((rows + 7) // 8) * 8

        with mock.patch.object(
            FusedSharedExpertFastPath,
            "_tma_aligned_rows",
            side_effect=aligned,
        ):
            workspace = first._ensure_workspace(x3)
            views = first._workspace_views(workspace, 3)
            self.assertIs(first._workspace_views(workspace, 3), views)

            second_workspace = second._ensure_workspace(x3)
            self.assertIs(second_workspace, workspace)
            self.assertIs(second._workspace_views(second_workspace, 3), views)
            isolated_workspace = isolated._ensure_workspace(x3)
            self.assertIsNot(isolated_workspace, workspace)
            self.assertNotEqual(
                isolated_workspace.out_bf16.untyped_storage().data_ptr(),
                workspace.out_bf16.untyped_storage().data_ptr(),
            )
            self.assertEqual(tuple(views.x_fp8.shape), (3, 128))
            self.assertEqual(tuple(views.x_scale.shape), (3, 1))
            self.assertEqual(tuple(views.gate_up_bf16.shape), (3, 256))
            self.assertEqual(tuple(views.hidden_fp8.shape), (3, 128))
            self.assertEqual(tuple(views.hidden_scale.shape), (3, 1))
            self.assertEqual(tuple(views.out_bf16.shape), (3, 128))

            old_x_fp8 = workspace.x_fp8
            grown_workspace = grower._ensure_workspace(
                torch.empty((9, 128), dtype=torch.bfloat16)
            )
            self.assertIsNot(grown_workspace, workspace)
            self.assertEqual(grown_workspace.capacity, 9)
            self.assertEqual(grown_workspace.views, {})
            self.assertIs(workspace.x_fp8, old_x_fp8)
            self.assertEqual(workspace.capacity, 4)
            self.assertEqual(ws._retired_graph_buffers, [])

            first_grown = first._ensure_workspace(
                torch.empty((9, 128), dtype=torch.bfloat16)
            )
            self.assertIs(first_grown, grown_workspace)
            self.assertIs(first._ensure_workspace(x3), grown_workspace)
            refreshed = first._workspace_views(first_grown, 3)
            self.assertIsNot(refreshed, views)
            self.assertIs(refreshed.x_fp8._base, grown_workspace.x_fp8)

            # Variable-length traffic retains one cached shape, while callers
            # holding previous views keep their original buffer references.
            other_shape = first._workspace_views(first_grown, 5)
            self.assertEqual(set(first_grown.views), {5})
            self.assertEqual(tuple(other_shape.x_fp8.shape), (5, 128))
            self.assertEqual(tuple(refreshed.x_fp8.shape), (3, 128))

            first.prepare(self._prepare_only_shared(inter=256, dim=128))
            resized = first._ensure_workspace(x3)
            self.assertEqual(resized.hidden_fp8.size(1), 256)
            self.assertEqual(grown_workspace.hidden_fp8.size(1), 128)

    def test_combine_preserves_fp32_accumulate_semantics(self):
        routed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        shared = torch.tensor([[0.5, -0.25], [0.125, -0.5]], dtype=torch.float32)
        with _env("MOE_STRICT_FUSED", "0"):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.float() + shared.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_strict_rejects_generic_shared_path(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        with self.assertRaisesRegex(RuntimeError, "generic Expert.forward"):
            executor.start(_Shared(), x)

    def test_sequential_executor(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        with _env("MOE_STRICT_FUSED", "0"):
            executor.start(_Shared(), x)
        got = executor.finish()
        ref = _Shared()(x).float()
        self.assertTrue(torch.equal(got, ref))

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
    def test_can_run_rejects_non_contiguous_and_strict_executor_runs_nonempty(self):
        shared, split_ref = _make_shared_expert(swiglu_limit=1.0)
        x_noncontig = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16)[:, :256]
        self.assertFalse(x_noncontig.is_contiguous())
        self.assertFalse(FusedSharedExpertFastPath.can_run(shared, x_noncontig))

        x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor(
            FusedSharedExpertExecutor(
                max_tokens_per_rank=16, dim=256, inter_dim=256, swiglu_limit=1.0
            )
        )
        with _env("MOE_STRICT_FUSED", "1"):
            executor.prepare(shared)
            with mock.patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.fp8_gemm_nt",
                side_effect=_fake_fp8_gemm_nt,
            ):
                executor.start(shared, x)
                got = executor.finish()
                ref = _split_reference(split_ref, x, 1.0)
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

    def test_multi_instance_cache_serialization_shares_largest_workspace(self):
        ws = MoeWorkspace()
        first = FusedSharedExpertFastPath(
            max_tokens_per_rank=4, dim=256, inter_dim=256, moe_workspace=ws
        )
        second = FusedSharedExpertFastPath(
            max_tokens_per_rank=8, dim=256, inter_dim=256, moe_workspace=ws
        )
        third = FusedSharedExpertFastPath(
            max_tokens_per_rank=4, dim=256, inter_dim=256, moe_workspace=ws
        )
        with mock.patch.object(
            FusedSharedExpertFastPath,
            "_tma_aligned_rows",
            side_effect=lambda rows, _el: ((rows + 3) // 4) * 4,
        ):
            ws_small = first._ensure_workspace(torch.empty((4, 256), dtype=torch.bfloat16))
            ws_large = second._ensure_workspace(torch.empty((8, 256), dtype=torch.bfloat16))
            self.assertIsNot(ws_small, ws_large)
            self.assertEqual(ws_large.capacity, 8)
            self.assertIs(
                first._ensure_workspace(torch.empty((5, 256), dtype=torch.bfloat16)),
                ws_large,
            )
            self.assertIs(
                third._ensure_workspace(torch.empty((4, 256), dtype=torch.bfloat16)),
                ws_large,
            )
            with mock.patch.object(ws, "is_graph_context", return_value=True):
                ws_graph_small = first._ensure_workspace(
                    torch.empty((4, 256), dtype=torch.bfloat16)
                )
                ws_graph_large = second._ensure_workspace(
                    torch.empty((8, 256), dtype=torch.bfloat16)
                )
            self.assertIsNot(ws_graph_small, ws_large)
            self.assertIsNot(ws_graph_small, ws_graph_large)
            self.assertIn(ws_graph_small, ws._retired_graph_buffers)


if __name__ == "__main__":
    unittest.main()

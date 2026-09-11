import ast
import inspect
import os
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.hc import build_hc_head, build_hc_unit
from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import (
    FallbackHCHead,
    FallbackHCUnit,
    _hc_split_sinkhorn,
    _tp_linear_mixes,
)
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (
    HybridHCUnit,
    TileLangHCHead,
    TileLangHCUnit,
)


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _weights(hc: int, dim: int, device: str = "cpu"):
    torch.manual_seed(0)
    mix_hc = (2 + hc) * hc
    fn = torch.randn(mix_hc, hc * dim, device=device, dtype=torch.float32) * 0.02
    base = torch.zeros(mix_hc, device=device, dtype=torch.float32)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    return fn, base, scale


class TestHCImpl(unittest.TestCase):
    def test_tp_linear_mixes_materializes_x_flat_fp32_once(self) -> None:
        source = textwrap.dedent(inspect.getsource(_tp_linear_mixes))
        tree = ast.parse(source)
        x_flat_float_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "float"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "x_flat"
        ]

        self.assertEqual(len(x_flat_float_calls), 1)

    def test_fallback_unit_linear_mixes_matches_fp32_reference(self) -> None:
        hc, dim = 4, 8
        fn, base, scale = _weights(hc, dim)
        unit = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        residual = torch.randn(3, hc, dim, dtype=torch.bfloat16)
        x_flat = residual.flatten(-2)
        expected = torch.nn.functional.linear(x_flat.float(), fn)
        expected *= torch.rsqrt(
            x_flat.float().square().mean(-1, keepdim=True) + unit.norm_eps
        )
        torch.testing.assert_close(unit._linear_mixes(x_flat), expected)

    def test_fallback_unit_pre_post_accumulate_in_fp32(self) -> None:
        hc, dim = 4, 8
        fn, base, scale = _weights(hc, dim)
        unit = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        torch.manual_seed(17)
        residual = (torch.randn(5, hc, dim) * 7).to(torch.bfloat16)

        y, post, comb = unit.pre(residual)
        mixes = unit._linear_mixes(residual.flatten(-2))
        pre_ref, post_ref, comb_ref = _hc_split_sinkhorn(
            mixes,
            scale,
            base,
            hc_mult=hc,
            sinkhorn_iters=3,
            eps=1e-6,
        )
        y_ref = torch.sum(pre_ref.unsqueeze(-1) * residual.float(), dim=-2).to(
            torch.bfloat16
        )
        torch.testing.assert_close(y, y_ref, rtol=0, atol=0)

        sublayer = (torch.randn(5, dim) * 5).to(torch.bfloat16)
        actual = unit.post(sublayer, residual, post, comb)
        expected = torch.matmul(comb_ref.transpose(-1, -2), residual.float())
        expected.add_(post_ref.unsqueeze(-1) * sublayer.float().unsqueeze(-2))
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)

    def test_fallback_tp_replicated_hidden_matches_single_rank(self) -> None:
        hc, dim = 4, 8
        fn, base, scale = _weights(hc, dim)
        residual = torch.randn(3, hc, dim, dtype=torch.bfloat16)
        reference = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        replicated = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        replicated.tp_size = 2
        replicated.tp_rank = 1
        target = "rtp_llm.models_py.distributed.collective_torch.all_reduce"
        expected = reference.pre(residual)
        with mock.patch(target) as all_reduce:
            actual = replicated.pre(residual)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        all_reduce.assert_not_called()

    def test_fallback_tp_hidden_shard_matches_global_pre_and_head(self) -> None:
        hc, global_dim, tp_size, tp_rank = 4, 8, 2, 1
        local_dim = global_dim // tp_size
        fn, base, scale = _weights(hc, global_dim)
        residual = torch.randn(3, hc, global_dim, dtype=torch.bfloat16)
        local = residual[..., tp_rank * local_dim : (tp_rank + 1) * local_dim]

        with _env("DSV4_HC_IMPL", "fallback"):
            global_unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=global_dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            local_unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=local_dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            global_head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=global_dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            local_head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=local_dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

        expected_y, expected_post, expected_comb = global_unit.pre(residual)
        expected_head = global_head.head(residual)

        rank0 = residual[..., :local_dim].flatten(-2)

        def rank0_contribution(combined, *, use_fp32):
            output_count = combined.shape[-1] - 1
            weight = (
                fn[:output_count]
                .to(torch.float32 if use_fp32 else torch.bfloat16)
                .view(output_count, hc, global_dim)[..., :local_dim]
                .reshape(output_count, hc * local_dim)
            )
            rank0_input = rank0.float() if use_fp32 else rank0
            rank0_part = torch.cat(
                (
                    rank0.float().square().sum(-1, keepdim=True),
                    torch.nn.functional.linear(rank0_input, weight).float(),
                ),
                dim=-1,
            )
            return combined + rank0_part

        def all_reduce_unit(combined, _group):
            return rank0_contribution(combined, use_fp32=False)

        def all_reduce_head(combined, _group):
            return rank0_contribution(combined, use_fp32=True)

        target = "rtp_llm.models_py.distributed.collective_torch.all_reduce"
        with mock.patch(target, side_effect=all_reduce_unit):
            actual_y, actual_post, actual_comb = local_unit.pre(local)
        with mock.patch(target, side_effect=all_reduce_head):
            actual_head = local_head.head(local)

        torch.testing.assert_close(
            actual_y, expected_y[..., local_dim:], atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(actual_post, expected_post, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(actual_comb, expected_comb, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(
            actual_head, expected_head[..., local_dim:], atol=2e-2, rtol=2e-2
        )

    def test_factory_fallback_cpu_shapes(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", "fallback"):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
        self.assertIsInstance(unit, FallbackHCUnit)
        self.assertIsInstance(head, FallbackHCHead)
        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        y, post, comb = unit.pre(x)
        self.assertEqual(tuple(y.shape), (2, 5, dim))
        self.assertEqual(tuple(post.shape), (2, 5, hc, 1))
        self.assertEqual(tuple(comb.shape), (2, 5, hc, hc))
        torch.testing.assert_close(
            comb.sum(dim=-1),
            torch.ones_like(comb.sum(dim=-1)),
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            comb.sum(dim=-2),
            torch.ones_like(comb.sum(dim=-2)),
            atol=5e-3,
            rtol=5e-3,
        )
        out = unit.post(y, x, post, comb)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        reduced = head.head(x)
        self.assertEqual(tuple(reduced.shape), (2, 5, dim))

    def test_factory_default_tilelang_fails_fast_on_cpu(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", None):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
        self.assertIsInstance(unit, TileLangHCUnit)
        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_factory_hybrid_uses_tilelang_pre_and_fallback_post_head(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", "hybrid"):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
        self.assertIsInstance(unit, HybridHCUnit)
        self.assertIsInstance(head, FallbackHCHead)

        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        residual = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        pre = torch.randn(5, dim, dtype=torch.bfloat16)
        post = torch.randn(5, hc, 1, dtype=torch.float32)
        comb = torch.randn(5, hc, hc, dtype=torch.float32)
        with mock.patch.object(
            tilelang_impl,
            "tk_mhc_pre",
            return_value=(pre.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)),
        ) as tk_pre:
            actual_pre = unit.pre(residual)
        tk_pre.assert_called_once()
        for actual, expected in zip(actual_pre, (pre, post, comb)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        sublayer = torch.randn(5, dim, dtype=torch.bfloat16)
        actual_post = unit.post(sublayer, residual, post, comb)
        expected_post = FallbackHCUnit._post_impl(unit, sublayer, residual, post, comb)
        torch.testing.assert_close(actual_post, expected_post, atol=0, rtol=0)

    def test_factory_hybrid_routes_fallback_pre_tilelang_post_and_fallback_head(
        self,
    ) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        with _env("DSV4_HC_IMPL", "hybrid"), _env(
            "DSV4_MHC_PRE_GEMM_BACKEND", " fallback "
        ), _env("DSV4_MHC_POST_BACKEND", " TILELANG "):
            unit = build_hc_unit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=3,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            head = build_hc_head(
                fn[:hc],
                base[:hc],
                scale[:1],
                dim=dim,
                hc_mult=hc,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            pre_result = object()
            post_result = object()
            pre_input = object()
            post_args = tuple(object() for _ in range(4))
            with mock.patch.object(
                FallbackHCUnit, "_pre_impl", return_value=pre_result
            ) as fallback_pre, mock.patch.object(
                TileLangHCUnit, "_pre_impl"
            ) as tilelang_pre:
                actual_pre = unit._pre_impl(pre_input, dbg_tag="explicit-fallback")
            with mock.patch.object(
                TileLangHCUnit, "_post_impl", return_value=post_result
            ) as tilelang_post, mock.patch.object(
                FallbackHCUnit, "_post_impl"
            ) as fallback_post:
                actual_post = unit._post_impl(*post_args)

        self.assertIsInstance(unit, HybridHCUnit)
        self.assertIsInstance(head, FallbackHCHead)
        self.assertIs(actual_pre, pre_result)
        fallback_pre.assert_called_once_with(
            unit, pre_input, dbg_tag="explicit-fallback"
        )
        tilelang_pre.assert_not_called()
        self.assertIs(actual_post, post_result)
        tilelang_post.assert_called_once_with(*post_args)
        fallback_post.assert_not_called()

    def test_hybrid_graph_capture_routes_only_tilelang_single_to_tilelang(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = HybridHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        cuda_x = mock.Mock(is_cuda=True)
        tilelang_result = object()
        fallback_result = object()

        for backend in (None, "tilelang", "deepgemm", "TILELANG_SINGLE "):
            with self.subTest(backend=backend), _env(
                "DSV4_MHC_PRE_GEMM_BACKEND", backend
            ), mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=True
            ), mock.patch.object(
                TileLangHCUnit, "_pre_impl", return_value=tilelang_result
            ) as tilelang_pre, mock.patch.object(
                FallbackHCUnit, "_pre_impl", return_value=fallback_result
            ) as fallback_pre:
                actual = unit._pre_impl(cuda_x, dbg_tag="graph")

            if backend is not None and backend.strip().lower() in {
                "tilelang_single",
                "deepgemm",
            }:
                self.assertIs(actual, tilelang_result)
                tilelang_pre.assert_called_once_with(cuda_x, dbg_tag="graph")
                fallback_pre.assert_not_called()
            else:
                self.assertIs(actual, fallback_result)
                fallback_pre.assert_called_once_with(unit, cuda_x, dbg_tag="graph")
                tilelang_pre.assert_not_called()

    def test_hybrid_eager_always_uses_selected_tilelang_pre(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = HybridHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        cuda_x = mock.Mock(is_cuda=True)
        expected = object()
        for backend in ("deepgemm", "deepgemm_deterministic"):
            with self.subTest(backend=backend), _env(
                "DSV4_MHC_PRE_GEMM_BACKEND", backend
            ), mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=False
            ), mock.patch.object(
                TileLangHCUnit, "_pre_impl", return_value=expected
            ) as tilelang_pre, mock.patch.object(
                FallbackHCUnit, "_pre_impl"
            ) as fallback_pre:
                actual = unit._pre_impl(cuda_x, dbg_tag="eager")

            self.assertIs(actual, expected)
            tilelang_pre.assert_called_once_with(cuda_x, dbg_tag="eager")
            fallback_pre.assert_not_called()

        with _env("DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic"), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ), mock.patch.object(FallbackHCUnit, "_pre_impl") as fallback_pre:
            with self.assertRaisesRegex(RuntimeError, "capture has not been validated"):
                unit._pre_impl(cuda_x, dbg_tag="graph")
            fallback_pre.assert_not_called()

    def test_fallback_deepgemm_backend_replaces_only_linear_projection(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        x_flat = object()
        expected = object()
        with _env("DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm"), mock.patch(
            "rtp_llm.models_py.modules.dsv4.hc.fallback_impl." "_deepgemm_linear_mixes",
            return_value=expected,
        ) as deepgemm, mock.patch(
            "rtp_llm.models_py.modules.dsv4.hc.fallback_impl._tp_linear_mixes"
        ) as aten:
            actual = unit._linear_mixes(x_flat)

        self.assertIs(actual, expected)
        deepgemm.assert_called_once_with(unit, x_flat)
        aten.assert_not_called()

    def test_fallback_pre_backend_is_fail_closed(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        with _env("DSV4_MHC_PRE_GEMM_BACKEND", "tilelang"), self.assertRaisesRegex(
            ValueError, "fallback mHC supports"
        ):
            unit._linear_mixes(object())

    def test_hybrid_unknown_backends_fail_closed_before_leaf_selection(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = HybridHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        cuda_x = mock.Mock(is_cuda=True)

        for capturing in (False, True):
            with self.subTest(path="pre", capturing=capturing), _env(
                "DSV4_MHC_PRE_GEMM_BACKEND", " typo "
            ), mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ), mock.patch.object(
                TileLangHCUnit, "_pre_impl"
            ) as tilelang_pre, mock.patch.object(
                FallbackHCUnit, "_pre_impl"
            ) as fallback_pre, self.assertRaisesRegex(
                ValueError, "DSV4_MHC_PRE_GEMM_BACKEND"
            ):
                unit._pre_impl(cuda_x, dbg_tag="invalid")
            tilelang_pre.assert_not_called()
            fallback_pre.assert_not_called()

        post_args = tuple(object() for _ in range(4))
        with _env("DSV4_MHC_POST_BACKEND", " typo "), mock.patch.object(
            TileLangHCUnit, "_post_impl"
        ) as tilelang_post, mock.patch.object(
            FallbackHCUnit, "_post_impl"
        ) as fallback_post, self.assertRaisesRegex(
            ValueError, "DSV4_MHC_POST_BACKEND"
        ):
            unit._post_impl(*post_args)
        tilelang_post.assert_not_called()
        fallback_post.assert_not_called()

    def test_hybrid_post_routes_only_explicit_tilelang_backend(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = HybridHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        args = tuple(object() for _ in range(4))
        tilelang_result = object()
        fallback_result = object()

        for backend in (None, "fallback", "tilelang_single", " TILELANG "):
            with self.subTest(backend=backend), _env(
                "DSV4_MHC_POST_BACKEND", backend
            ), mock.patch.object(
                TileLangHCUnit, "_post_impl", return_value=tilelang_result
            ) as tilelang_post, mock.patch.object(
                FallbackHCUnit, "_post_impl", return_value=fallback_result
            ) as fallback_post:
                actual = unit._post_impl(*args)

            if backend is not None and backend.strip().lower() == "tilelang":
                self.assertIs(actual, tilelang_result)
                tilelang_post.assert_called_once_with(*args)
                fallback_post.assert_not_called()
            else:
                self.assertIs(actual, fallback_result)
                fallback_post.assert_called_once_with(unit, *args)
                tilelang_post.assert_not_called()

    def test_tilelang_none_result_is_not_fallback(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = lambda *args, **kwargs: None
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            unit.pre(x)

    def test_tilelang_wrap_is_view_and_requires_contiguous(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        seen = {}

        def fake_pre(residual, *args, **kwargs):
            seen["shape"] = tuple(residual.shape)
            seen["stride"] = tuple(residual.stride())
            seen["data_ptr"] = residual.data_ptr()
            seen["is_contiguous"] = residual.is_contiguous()
            return (
                torch.zeros(1, 5, dim, dtype=torch.bfloat16),
                torch.zeros(1, 5, hc, 1, dtype=torch.float32),
                torch.zeros(1, 5, hc, hc, dtype=torch.float32),
            )

        old_pre = tilelang_impl.tk_mhc_pre
        tilelang_impl.tk_mhc_pre = fake_pre
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_pre", old_pre))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y, post, comb = unit.pre(x)
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertEqual(tuple(post.shape), (5, hc, 1))
        self.assertEqual(tuple(comb.shape), (5, hc, hc))
        self.assertEqual(seen["shape"], (1, 5, hc, dim))
        self.assertTrue(seen["is_contiguous"])
        self.assertEqual(seen["data_ptr"], x.data_ptr())

        x_noncontig = torch.randn(hc, 5, dim, dtype=torch.bfloat16).transpose(0, 1)
        self.assertEqual(tuple(x_noncontig.shape), (5, hc, dim))
        self.assertFalse(x_noncontig.is_contiguous())
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            unit.pre(x_noncontig)

    def test_tilelang_head_requires_fused_when_enabled(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        head = TileLangHCHead(
            fn[:hc],
            base[:hc],
            scale[:1],
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        calls: list[str] = []

        def fake_fused(residual, *args, **kwargs):
            calls.append("fused")
            return torch.ones(1, 5, dim, dtype=torch.bfloat16)

        def fake_old(residual, *args, **kwargs):
            calls.append("old")
            return torch.zeros(1, 5, dim, dtype=torch.bfloat16)

        old_fused = tilelang_impl.tk_mhc_head_fused
        old_head = tilelang_impl.tk_mhc_head
        tilelang_impl.tk_mhc_head_fused = fake_fused
        tilelang_impl.tk_mhc_head = fake_old
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_head_fused", old_fused))
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_head", old_head))

        x = torch.randn(5, hc, dim, dtype=torch.bfloat16)
        y = head.head(x)
        self.assertEqual(calls, ["fused"])
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertTrue(torch.all(y == 1))

        calls.clear()
        tilelang_impl.tk_mhc_head_fused = (
            lambda *args, **kwargs: calls.append("fused") or None
        )
        with self.assertRaisesRegex(RuntimeError, "fused head must succeed"):
            head.head(x)
        self.assertEqual(calls, ["fused"])

        calls.clear()
        with _env("DSV4_MHC_HEAD_FUSED", "0"):
            y = head.head(x)
        self.assertEqual(calls, ["old"])
        self.assertTrue(torch.all(y == 0))

    def test_shape_contract_is_checked_before_impl(self) -> None:
        hc, dim = 4, 16
        fn, base, scale = _weights(hc, dim)
        for mode in ("fallback", "tilelang", "hybrid"):
            with self.subTest(mode=mode), _env("DSV4_HC_IMPL", mode):
                unit = build_hc_unit(
                    fn,
                    base,
                    scale,
                    dim=dim,
                    hc_mult=hc,
                    hc_sinkhorn_iters=3,
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
                with self.assertRaises(ValueError):
                    unit.pre(torch.randn(2, hc * dim))
                residual = torch.randn(2, 5, hc, dim, dtype=torch.bfloat16)
                x = torch.randn(2, 5, dim, dtype=torch.bfloat16)
                bad_post = torch.randn(2, 5, hc)
                comb = torch.randn(2, 5, hc, hc)
                with self.assertRaises(ValueError):
                    unit.post(x, residual, bad_post, comb)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_matches_fallback_cuda(self) -> None:
        hc, dim = 4, 128
        fn, base, scale = _weights(hc, dim, device="cuda")
        x = torch.randn(1, 64, hc, dim, device="cuda", dtype=torch.bfloat16)
        sublayer = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)

        fallback = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        tilelang = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        with torch.inference_mode():
            ref_y, ref_post, ref_comb = fallback.pre(x)
            try:
                tk_y, tk_post, tk_comb = tilelang.pre(x)
                ref_out = fallback.post(sublayer, x, ref_post, ref_comb)
                tk_out = tilelang.post(sublayer, x, tk_post, tk_comb)
            except RuntimeError as exc:
                self.skipTest(str(exc))
        torch.testing.assert_close(tk_y, ref_y, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            tk_post.float(), ref_post.float(), atol=5e-3, rtol=5e-3
        )
        torch.testing.assert_close(
            tk_comb.float(), ref_comb.float(), atol=5e-3, rtol=5e-3
        )
        torch.testing.assert_close(tk_out, ref_out, atol=2e-2, rtol=2e-2)

    def test_tilelang_post_reuses_residual_buffer_in_place(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "3rdparty/tile_kernels/mhc/post_kernel.py"
        ).read_text()
        loop_start = source.index("for i_mhco, i1_h in T.Parallel(mhc, h_blk):")
        loop_end = source.index("T.copy(x_local, x_shared)", loop_start)
        forward_loop = source[loop_start:loop_end]
        post = forward_loop.index("c_local[i_mhco] * d_local[i1_h]")
        comb = forward_loop.index("for i_mhci in T.serial(mhc):")
        self.assertLess(post, comb)
        self.assertNotIn("x_local[i_mhco, i1_h] = 0.0", forward_loop)

        # Pins the memory-saving wiring (no CUDA needed): _post_impl must pass
        # out=residual so the kernel writes in place instead of allocating a
        # fresh empty_like(residual). A future refactor that drops the alias
        # would silently re-introduce the per-call allocation; this catches it.
        hc, dim, T = 4, 16, 5
        fn, base, scale = _weights(hc, dim)
        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc,
            hc_sinkhorn_iters=3,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        import rtp_llm.models_py.modules.dsv4.hc.tilelang_impl as tilelang_impl

        captured: dict = {}

        def fake_post(x, residual, post, comb, hc_mult=4, out=None):
            captured["out_is_residual"] = out is residual
            captured["out_ptr"] = None if out is None else out.data_ptr()
            return residual if out is None else out

        old_post = tilelang_impl.tk_mhc_post
        tilelang_impl.tk_mhc_post = fake_post
        self.addCleanup(lambda: setattr(tilelang_impl, "tk_mhc_post", old_post))

        residual = torch.randn(T, hc, dim, dtype=torch.bfloat16)
        x = torch.randn(T, dim, dtype=torch.bfloat16)
        post = torch.randn(T, hc, 1, dtype=torch.float32)
        comb = torch.randn(T, hc, hc, dtype=torch.float32)
        out = unit.post(x, residual, post, comb)

        self.assertTrue(captured["out_is_residual"])
        # the aliased out points at the caller's residual storage
        self.assertEqual(captured["out_ptr"], residual.data_ptr())
        self.assertEqual(out.data_ptr(), residual.data_ptr())
        self.assertEqual(tuple(out.shape), (T, hc, dim))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_nonbf16_x_raises_dtype_error(self) -> None:
        # A non-bf16 sublayer output x is an upstream dtype bug, not a TileLang
        # availability miss: it must raise loudly (with the offending dtype),
        # not return None and get disguised as "TileLang unavailable". Reaches
        # the x.dtype check before any kernel import, so tilelang is not needed.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 8
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)
        x_fp32 = torch.randn(1, T, dim, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "bfloat16 sublayer output"):
                tk_mhc_post(x_fp32, residual, post, comb, hc_mult=hc)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_post_in_place_matches_fresh_buffer_cuda(self) -> None:
        # Pins the out=residual aliasing invariant: writing the post output back
        # into the residual buffer must be bit-identical to writing into a fresh
        # buffer. The vendored kernel's safety relies on reading residual[pid_n]
        # into shared memory before overwriting out[pid_n]; if a future kernel
        # change breaks that read-before-write ordering, this fails loudly.
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post

        hc, dim, T = 4, 128, 64
        torch.manual_seed(0)
        x = torch.randn(1, T, dim, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(1, T, hc, dim, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(1, T, hc, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(1, T, hc, hc, device="cuda", dtype=torch.float32)

        residual_fresh = residual.clone()
        residual_alias = residual.clone()
        with torch.inference_mode():
            try:
                out_fresh = tk_mhc_post(x, residual_fresh, post, comb, hc_mult=hc)
                out_alias = tk_mhc_post(
                    x, residual_alias, post, comb, hc_mult=hc, out=residual_alias
                )
            except RuntimeError as exc:
                self.skipTest(str(exc))

        assert out_fresh is not None and out_alias is not None
        # fresh path allocated a new buffer; aliased path wrote into residual
        self.assertNotEqual(out_fresh.data_ptr(), residual_fresh.data_ptr())
        self.assertEqual(out_alias.data_ptr(), residual_alias.data_ptr())
        # in-place and fresh-buffer results must be bit-identical
        torch.testing.assert_close(out_alias, out_fresh, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tilelang_head_matches_fallback_cuda(self) -> None:
        hc, dim = 4, 128
        fn, base, scale = _weights(hc, dim, device="cuda")
        head_fn = fn[:hc]
        head_base = base[:hc]
        head_scale = scale[:1]
        x = torch.randn(1, 64, hc, dim, device="cuda", dtype=torch.bfloat16)
        fallback = FallbackHCHead(
            head_fn,
            head_base,
            head_scale,
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        tilelang = TileLangHCHead(
            head_fn,
            head_base,
            head_scale,
            dim=dim,
            hc_mult=hc,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        with torch.inference_mode():
            ref_y = fallback.head(x)
            try:
                tk_y = tilelang.head(x)
            except RuntimeError as exc:
                self.skipTest(str(exc))
        torch.testing.assert_close(tk_y, ref_y, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()

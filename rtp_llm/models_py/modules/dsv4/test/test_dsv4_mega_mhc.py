"""Contract and SM103 parity tests for DSV4 ``deep_gemm.mega_mhc`` integration."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

import torch


def _install_cpu_native_stubs_if_needed() -> None:
    """Allow CPU-pool contract tests to import GPU Python modules when ``libcuda.so.1`` is absent."""
    if os.environ.get("GPU_COUNT", "0") != "0":
        return
    try:
        import librtp_compute_ops  # noqa: F401
    except ImportError:
        if "librtp_compute_ops" not in sys.modules:
            pkg = types.ModuleType("librtp_compute_ops")
            sub = types.ModuleType("librtp_compute_ops.rtp_llm_ops")
            pkg.rtp_llm_ops = sub
            sys.modules["librtp_compute_ops"] = pkg
            sys.modules["librtp_compute_ops.rtp_llm_ops"] = sub


_install_cpu_native_stubs_if_needed()

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm  # noqa: E402
from rtp_llm.models_py.modules.dsv4.hc import mega_mhc as _mega_mhc  # noqa: E402
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (  # noqa: E402
    TileLangHCUnit,
)


def _make_unit(
    *,
    dim: int = 4096,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    device: str | torch.device = "cpu",
    seed: int = 0,
    layer_id: int = 3,
    name: str = "ffn_hc",
) -> TileLangHCUnit:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    fn = (
        torch.randn((24, hc_mult * dim), generator=gen, dtype=torch.float32) * 0.02
    ).to(device)
    base = (torch.randn((24,), generator=gen, dtype=torch.float32) * 0.05).to(device)
    scale = (torch.rand((3,), generator=gen, dtype=torch.float32) * 0.5 + 0.5).to(
        device
    )
    return TileLangHCUnit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=1.0e-6,
        hc_eps=1.0e-6,
        layer_id=layer_id,
        name=name,
    )


def _make_norm(
    *,
    dim: int = 4096,
    device: str | torch.device = "cpu",
    seed: int = 1,
) -> RMSNorm:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    weight = (
        torch.ones((dim,), dtype=torch.bfloat16)
        + torch.randn((dim,), generator=gen, dtype=torch.bfloat16) * 0.1
    ).to(device)
    return RMSNorm(weight, eps=1.0e-6)


class TestDSV4MegaMHCContract(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_warmed = set(_mega_mhc._WARMED_STREAMS)
        self.addCleanup(
            lambda: (
                _mega_mhc._WARMED_STREAMS.clear(),
                _mega_mhc._WARMED_STREAMS.update(self._saved_warmed),
            )
        )

    def test_num_splits_and_representative_tokens(self) -> None:
        # DSV4 Flash (dim=4096) on B200/B300 (148 SMs): 4 distinct num_splits buckets.
        reps_flash = _mega_mhc.representative_tokens(4096, 148, hidden=4096)
        self.assertEqual(reps_flash, (1, 129, 257, 385))
        self.assertEqual(
            tuple(_mega_mhc._mega_num_splits(t, 148, 4096) for t in reps_flash),
            (64, 32, 22, 16),
        )
        # Every token count in [1, 4096] maps to one of the representative splits.
        expected_splits = {_mega_mhc._mega_num_splits(t, 148, 4096) for t in reps_flash}
        for t in (1, 2, 64, 65, 128, 129, 192, 193, 256, 257, 321, 384, 385, 512, 4096):
            self.assertIn(_mega_mhc._mega_num_splits(t, 148, 4096), expected_splits)

        # Generalization across DSV4 Pro (dim=7168) and small/large max_m & SM counts.
        reps_pro = _mega_mhc.representative_tokens(4096, 148, hidden=7168)
        pro_splits = {_mega_mhc._mega_num_splits(t, 148, 7168) for t in reps_pro}
        all_pro_splits = {
            _mega_mhc._mega_num_splits(t, 148, 7168) for t in range(1, 1025, 16)
        }
        self.assertEqual(pro_splits, all_pro_splits)
        self.assertEqual(_mega_mhc.representative_tokens(64, 148, hidden=4096), (1,))
        self.assertEqual(_mega_mhc.representative_tokens(0, 148, hidden=4096), ())
        self.assertEqual(_mega_mhc.representative_tokens(128, 0, hidden=4096), ())

    def test_unit_pair_and_stream_gates_reject_ineligible_configs(self) -> None:
        prev = _make_unit(dim=4096, layer_id=0, name="attn_hc")
        nxt = _make_unit(dim=4096, layer_id=0, name="ffn_hc")
        norm = _make_norm(dim=4096)

        # CPU tensors and CPU device are never supported.
        self.assertFalse(
            _mega_mhc.is_unit_pair_supported(prev, nxt, norm, torch.device("cpu"))
        )
        self.assertFalse(_mega_mhc.can_use_on_current_stream(torch.device("cpu")))

        cuda_dev = torch.device("cuda:0")
        with (
            patch.object(_mega_mhc, "_is_sm10x_device", return_value=True),
            patch.object(_mega_mhc, "_same_cuda_device", return_value=True),
        ):
            self.assertTrue(_mega_mhc.is_unit_pair_supported(prev, nxt, norm, cuda_dev))

            # Unsupported dim (not divisible by 1024).
            bad_prev = _make_unit(dim=1280)
            bad_nxt = _make_unit(dim=1280)
            bad_norm = _make_norm(dim=1280)
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(bad_prev, bad_nxt, bad_norm, cuda_dev)
            )

            # Patched TileLang units / subclasses (e.g. PPU subclass) / patched norm fail gate.
            class SubTileLang(TileLangHCUnit):
                pass

            sub_unit = SubTileLang(
                nxt.fn,
                nxt.base,
                nxt.scale,
                dim=4096,
                hc_mult=4,
                hc_sinkhorn_iters=20,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(prev, sub_unit, norm, cuda_dev)
            )

            patched_prev = _make_unit(dim=4096)
            patched_prev._post_impl_prefill_fast = lambda *a, **k: None
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(patched_prev, nxt, norm, cuda_dev)
            )

            patched_nxt = _make_unit(dim=4096)
            patched_nxt.pre = lambda *a, **k: None
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(prev, patched_nxt, norm, cuda_dev)
            )

            patched_norm = _make_norm(dim=4096)
            patched_norm.forward = lambda x: x
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(prev, nxt, patched_norm, cuda_dev)
            )

            # Requires grad or non-contiguous weights reject cleanly.
            grad_norm = RMSNorm(
                norm.weight.clone().detach().requires_grad_(True), eps=1e-6
            )
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(prev, nxt, grad_norm, cuda_dev)
            )

        # Non-SM10x (e.g. SM90 / H20) rejects cleanly.
        with (
            patch.object(_mega_mhc, "_is_sm10x_device", return_value=False),
            patch.object(_mega_mhc, "_get_mega_mhc", return_value=lambda *a, **k: None),
        ):
            self.assertFalse(
                _mega_mhc.is_unit_pair_supported(prev, nxt, norm, cuda_dev)
            )
            self.assertFalse(_mega_mhc.can_use_on_current_stream(cuda_dev))

        # Capture stream gating: unwarmed stream returns False, warmed stream returns True.
        _mega_mhc._WARMED_STREAMS.clear()
        with (
            patch.object(_mega_mhc, "_get_mega_mhc", return_value=lambda *a, **k: None),
            patch.object(_mega_mhc, "_is_sm10x_device", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(_mega_mhc, "_stream_key", return_value=(0, 123456)),
        ):
            self.assertFalse(_mega_mhc.can_use_on_current_stream(cuda_dev))
            _mega_mhc._WARMED_STREAMS.add((0, 123456))
            self.assertTrue(_mega_mhc.can_use_on_current_stream(cuda_dev))

    def test_try_fused_post_pre_mocked_contract_and_error_propagation(self) -> None:
        prev = _make_unit(dim=4096, layer_id=2, name="attn_hc")
        nxt = _make_unit(dim=4096, layer_id=2, name="ffn_hc")
        norm = _make_norm(dim=4096)

        calls = []

        def fake_mega_mhc(*args, **kwargs):
            calls.append((args, kwargs))
            y_bf16 = kwargs["y_bf16"]
            new_res = args[17]
            new_post = args[19]
            new_comb = args[20]
            y_bf16.fill_(1.5)
            new_res.fill_(2.5)
            new_post.fill_(0.25)
            new_comb.fill_(0.125)

        x_sub = torch.randn(3, 1, 4096, dtype=torch.bfloat16)
        residual = torch.randn(3, 1, 4, 4096, dtype=torch.bfloat16)
        post = torch.randn(3, 1, 4, 1, dtype=torch.float32)
        comb = torch.randn(3, 1, 4, 4, dtype=torch.float32)

        # Returns None when mega_mhc kernel is missing.
        with patch.object(_mega_mhc, "_get_mega_mhc", return_value=None):
            self.assertIsNone(
                _mega_mhc.try_fused_post_pre(
                    x_sub, residual, post, comb, previous=prev, next_hc=nxt, norm=norm
                )
            )

        fake_dev = torch.device("cpu")
        with (
            patch.object(_mega_mhc, "_get_mega_mhc", return_value=fake_mega_mhc),
            patch.object(_mega_mhc, "_normalize_cuda_device", return_value=fake_dev),
            patch.object(_mega_mhc, "_is_sm10x_device", return_value=True),
            patch.object(_mega_mhc, "_same_cuda_device", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(_mega_mhc, "_stream_key", return_value=(0, 777)),
        ):
            _mega_mhc._WARMED_STREAMS.clear()
            out = _mega_mhc.try_fused_post_pre(
                x_sub,
                residual,
                post,
                comb,
                previous=prev,
                next_hc=nxt,
                norm=norm,
            )
            self.assertIsNotNone(out)
            new_res, normalized, new_post, new_comb = out
            self.assertEqual(tuple(new_res.shape), (3, 1, 4, 4096))
            self.assertEqual(tuple(normalized.shape), (3, 1, 4096))
            self.assertEqual(tuple(new_post.shape), (3, 1, 4, 1))
            self.assertEqual(tuple(new_comb.shape), (3, 1, 4, 4))
            # Never alias input residual.
            self.assertNotEqual(new_res.data_ptr(), residual.data_ptr())
            self.assertIn((0, 777), _mega_mhc._WARMED_STREAMS)

            args, kwargs = calls[0]
            self.assertIs(args[5], nxt.fn)
            self.assertIs(args[6], nxt.scale)
            self.assertIs(args[7], nxt.base)
            self.assertEqual(args[8], 4)
            self.assertEqual(args[11], 2.0)
            self.assertEqual(args[13], nxt.hc_sinkhorn_iters)
            self.assertIs(args[14], norm.weight)
            self.assertEqual(args[16], 1.0)
            self.assertIsNone(args[2])
            self.assertIsNone(args[18])
            self.assertIn("y_bf16", kwargs)

            # Empty token batch falls back cleanly to None.
            self.assertIsNone(
                _mega_mhc.try_fused_post_pre(
                    torch.empty(0, 4096, dtype=torch.bfloat16),
                    torch.empty(0, 4, 4096, dtype=torch.bfloat16),
                    torch.empty(0, 4, 1, dtype=torch.float32),
                    torch.empty(0, 4, 4, dtype=torch.float32),
                    previous=prev,
                    next_hc=nxt,
                    norm=norm,
                )
            )

        # Kernel runtime errors propagate cleanly (never masked as fallback).
        def exploding_mega_mhc(*args, **kwargs):
            raise RuntimeError("deep_gemm mega_mhc launch failure")

        with (
            patch.object(_mega_mhc, "_get_mega_mhc", return_value=exploding_mega_mhc),
            patch.object(_mega_mhc, "_normalize_cuda_device", return_value=fake_dev),
            patch.object(_mega_mhc, "_is_sm10x_device", return_value=True),
            patch.object(_mega_mhc, "_same_cuda_device", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(_mega_mhc, "_stream_key", return_value=(0, 777)),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failure"):
                _mega_mhc.try_fused_post_pre(
                    x_sub,
                    residual,
                    post,
                    comb,
                    previous=prev,
                    next_hc=nxt,
                    norm=norm,
                )


def _gpu_run_required() -> bool:
    return os.environ.get("GPU_COUNT", "0") != "0"


@unittest.skipUnless(_gpu_run_required(), "GPU parity tests skipped when GPU_COUNT=0")
class TestDSV4MegaMHCSm103Gpu(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "CUDA required when GPU_COUNT != 0")
        major, minor = torch.cuda.get_device_capability(0)
        self.assertEqual(
            major,
            10,
            f"deep_gemm.mega_mhc requires SM100/SM103; got sm_{major}{minor}",
        )
        self.assertIsNotNone(
            _mega_mhc._get_mega_mhc(),
            "deep_gemm.mega_mhc missing in GPU environment",
        )
        self.device = torch.device("cuda:0")

    def _assert_parity_for_shape(
        self,
        shape_prefix: tuple[int, ...],
        *,
        dim: int,
        seed: int = 7,
        zero_inputs: bool = False,
    ) -> None:
        prev = _make_unit(dim=dim, device=self.device, seed=seed)
        nxt = _make_unit(dim=dim, device=self.device, seed=seed + 1)
        norm = _make_norm(dim=dim, device=self.device, seed=seed + 2)

        gen = torch.Generator(device="cpu").manual_seed(seed + 3)
        if zero_inputs:
            x_sub = torch.zeros(
                (*shape_prefix, dim), dtype=torch.bfloat16, device=self.device
            )
            residual = torch.zeros(
                (*shape_prefix, 4, dim), dtype=torch.bfloat16, device=self.device
            )
            post = torch.zeros(
                (*shape_prefix, 4, 1), dtype=torch.float32, device=self.device
            )
            comb = torch.zeros(
                (*shape_prefix, 4, 4), dtype=torch.float32, device=self.device
            )
        else:
            x_sub = (
                torch.randn((*shape_prefix, dim), generator=gen, dtype=torch.bfloat16)
                * 0.5
            ).to(self.device)
            residual = (
                torch.randn(
                    (*shape_prefix, 4, dim), generator=gen, dtype=torch.bfloat16
                )
                * 0.5
            ).to(self.device)
            post = (
                torch.randn((*shape_prefix, 4, 1), generator=gen, dtype=torch.float32)
                * 0.2
            ).to(self.device)
            comb = (
                torch.randn((*shape_prefix, 4, 4), generator=gen, dtype=torch.float32)
                * 0.2
            ).to(self.device)

        residual_before = residual.clone()
        # Baseline: prev.post clones residual internally only if not aliasing; TileLang post mutates residual!
        ref_res = prev.post(x_sub, residual.clone(), post, comb)
        ref_pre, ref_post, ref_comb = nxt.pre(ref_res)
        ref_pre_flat = ref_pre.reshape(-1, dim)
        ref_norm = norm(ref_pre_flat).reshape(*shape_prefix, dim)

        out = _mega_mhc.try_fused_post_pre(
            x_sub,
            residual,
            post,
            comb,
            previous=prev,
            next_hc=nxt,
            norm=norm,
        )
        self.assertIsNotNone(out)
        new_res, normalized, new_post, new_comb = out
        # Fused path allocates fresh new_residual and leaves input residual untouched.
        self.assertNotEqual(new_res.data_ptr(), residual.data_ptr())
        torch.testing.assert_close(residual, residual_before, atol=0.0, rtol=0.0)

        torch.testing.assert_close(new_res, ref_res, atol=3.0e-2, rtol=2.0e-2)
        torch.testing.assert_close(normalized, ref_norm, atol=3.0e-2, rtol=2.0e-2)
        torch.testing.assert_close(new_post, ref_post, atol=5.0e-3, rtol=5.0e-3)
        torch.testing.assert_close(new_comb, ref_comb, atol=5.0e-3, rtol=5.0e-3)

    def test_gpu_parity_across_splits_buckets_and_dims(self) -> None:
        # All 4 num_splits buckets on H=4096 (1, 129, 257, 385) plus batched 4D decode.
        for tokens in (1, 129, 257, 385):
            self._assert_parity_for_shape((tokens,), dim=4096, seed=tokens)
        self._assert_parity_for_shape((8, 1), dim=4096, seed=99)
        self._assert_parity_for_shape((4,), dim=4096, seed=11, zero_inputs=True)
        # DSV4 Pro width (dim=7168).
        for tokens in (1, 129):
            self._assert_parity_for_shape((tokens,), dim=7168, seed=200 + tokens)

    def test_gpu_cuda_graph_capture_guard_and_replay(self) -> None:
        prev = _make_unit(dim=4096, device=self.device, seed=31)
        nxt = _make_unit(dim=4096, device=self.device, seed=32)
        norm = _make_norm(dim=4096, device=self.device, seed=33)

        b, s, dim = 4, 1, 4096
        x_sub = torch.randn((b, s, dim), dtype=torch.bfloat16, device=self.device)
        residual = torch.randn((b, s, 4, dim), dtype=torch.bfloat16, device=self.device)
        post = torch.randn((b, s, 4, 1), dtype=torch.float32, device=self.device)
        comb = torch.randn((b, s, 4, 4), dtype=torch.float32, device=self.device)

        # 1. Unwarmed dedicated capture stream safely returns None during capture.
        unwarmed_stream = torch.cuda.Stream(device=self.device)
        captured_unwarmed = {}
        with torch.cuda.stream(unwarmed_stream):
            unwarmed_stream.synchronize()
            g_unwarmed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_unwarmed, stream=unwarmed_stream):
                captured_unwarmed["out"] = _mega_mhc.try_fused_post_pre(
                    x_sub,
                    residual,
                    post,
                    comb,
                    previous=prev,
                    next_hc=nxt,
                    norm=norm,
                )
        self.assertIsNone(captured_unwarmed["out"])

        # 2. Pre-warm dedicated stream eagerly -> captures deep_gemm.mega_mhc cleanly and replays.
        warmed_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(warmed_stream):
            eager_out = _mega_mhc.try_fused_post_pre(
                x_sub,
                residual,
                post,
                comb,
                previous=prev,
                next_hc=nxt,
                norm=norm,
            )
            self.assertIsNotNone(eager_out)
        warmed_stream.synchronize()

        g_warmed = torch.cuda.CUDAGraph()
        captured_out = {}
        with torch.cuda.graph(g_warmed, stream=warmed_stream):
            captured_out["res"] = _mega_mhc.try_fused_post_pre(
                x_sub,
                residual,
                post,
                comb,
                previous=prev,
                next_hc=nxt,
                norm=norm,
            )
        self.assertIsNotNone(captured_out["res"])

        # Update inputs twice and replay to verify updated results track inputs accurately.
        for scale_factor in (0.5, -1.25):
            new_x = torch.randn_like(x_sub) * scale_factor
            new_res_in = torch.randn_like(residual) * scale_factor
            new_post_in = torch.randn_like(post) * 0.1
            new_comb_in = torch.randn_like(comb) * 0.1
            x_sub.copy_(new_x)
            residual.copy_(new_res_in)
            post.copy_(new_post_in)
            comb.copy_(new_comb_in)

            g_warmed.replay()
            torch.cuda.synchronize(self.device)

            ref_res = prev.post(new_x, new_res_in.clone(), new_post_in, new_comb_in)
            ref_pre, ref_post, ref_comb = nxt.pre(ref_res)
            ref_norm = norm(ref_pre.reshape(-1, dim)).reshape(b, s, dim)

            g_res, g_norm, g_post, g_comb = captured_out["res"]
            torch.testing.assert_close(g_res, ref_res, atol=3.0e-2, rtol=2.0e-2)
            torch.testing.assert_close(g_norm, ref_norm, atol=3.0e-2, rtol=2.0e-2)
            torch.testing.assert_close(g_post, ref_post, atol=5.0e-3, rtol=5.0e-3)
            torch.testing.assert_close(g_comb, ref_comb, atol=5.0e-3, rtol=5.0e-3)


if __name__ == "__main__":
    unittest.main()

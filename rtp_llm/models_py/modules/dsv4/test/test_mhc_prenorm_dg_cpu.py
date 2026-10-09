"""CPU contracts for the opt-in DeepGEMM prenorm backend.

Drives the real resolver and dispatch with kernel spies. Tests flag parsing,
flag-off parity, split-K epilogue wiring, failure propagation, and override
mutants. CPU simulations do not establish GPU numerical or performance
qualification.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from unittest import mock

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
WT = HERE.parents[4]  # rtp_llm/models_py/modules/dsv4/test -> worktree root
ARCH_PATH = WT / "rtp_llm/models_py/utils/arch.py"
WRAPPER_PATH = (
    WT / "rtp_llm/models_py/3rdparty/tile_kernels/modeling/mhc/ops/pre_big_fuse.py"
)
WARMUP_PATH = WT / "rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py"
ARCH_SRC = ARCH_PATH.read_text()
WARMUP_SRC = WARMUP_PATH.read_text()

PRENORM_FLAG = "DSV4_MHC_PRENORM_DEEPGEMM"
BACKEND_ENV = "DSV4_MHC_PRE_GEMM_BACKEND"

# Tolerance for this deterministic fp32 accumulation-order simulation only.
# It is not a numerical acceptance bound for the production TF32 kernels.
SPLITK_SIM_ABS_MAX = 5e-3


def _package(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            _package(parent)
            setattr(sys.modules[parent], child, mod)
    return sys.modules[name]


def _load_file(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)  # never poison the cache on a failed load
        raise
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_package(parent), child, mod)
    return mod


def _load_real_arch():
    _package("rtp_llm")
    _package("rtp_llm.device")
    _load_file("rtp_llm.device.device_type", WT / "rtp_llm/device/device_type.py")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.utils")
    return _load_file("rtp_llm.models_py.utils.arch", ARCH_PATH)


_DGW_NAME = "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"


def _stub_dgw():
    """Register a stub DeepGEMM wrapper: pre_big_fuse's call-time
    ``from ... import tf32_hc_prenorm_gemm`` binds whatever attribute this
    module carries — in production the real DeepGEMM-backed symbol, here a
    per-test spy.  (The real wrapper module needs ``rtp_llm.utils.module_util``
    + triton; the stub keeps this suite free of non-globbed sources.)"""
    _load_real_arch()
    _package("rtp_llm.models_py.kernels")
    _package("rtp_llm.models_py.kernels.cuda")
    if _DGW_NAME in sys.modules:
        return sys.modules[_DGW_NAME]
    mod = types.ModuleType(_DGW_NAME)
    mod.tf32_hc_prenorm_gemm = None  # replaced by spies per test
    sys.modules[_DGW_NAME] = mod
    setattr(sys.modules[_DGW_NAME.rpartition(".")[0]], "deepgemm_wrapper", mod)
    return mod


def _load_wrapper():
    """Load the real resolver/dispatch with deterministic kernel doubles."""
    name = "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
    if name in sys.modules:
        return sys.modules[name]
    _stub_dgw()
    tk = "rtp_llm.models_py.3rdparty.tile_kernels"
    _package(tk)
    _package(tk + ".mhc")
    _package(tk + ".modeling")
    _package(tk + ".modeling.mhc")
    _package(tk + ".modeling.mhc.ops")

    norm_fn = types.ModuleType(tk + ".mhc.norm_fn_kernel")
    norm_fn._mhc_pre_norm_fn_fwd_mul = None  # replaced by spies per test
    norm_fn.round_to_tf32 = lambda x: (x.view(torch.int32) + 0x1000).view(torch.float32)
    sys.modules[norm_fn.__name__] = norm_fn

    big_fuse = types.ModuleType(tk + ".mhc.pre_big_fuse_kernel")
    big_fuse._mhc_pre_big_fuse = None
    sys.modules[big_fuse.__name__] = big_fuse

    return _load_file(name, WRAPPER_PATH)


class _Spies:
    """Deterministic doubles for the three possible call targets."""

    def __init__(self, wrapper):
        self.mul_calls = []
        self.fuse_calls = []
        self.dg_calls = []
        wrapper._mhc_pre_norm_fn_fwd_mul = self._mul
        wrapper._mhc_pre_big_fuse = self._fuse
        self._dgw = _stub_dgw()
        self._dg_patcher = mock.patch.object(
            self._dgw, "tf32_hc_prenorm_gemm", side_effect=self._dg
        )
        self._dg_patcher.start()

    def stop(self):
        self._dg_patcher.stop()

    def _mul(self, mhc_mult3, n_rms_group, hidden):
        def launch(x, fn, out_mul, out_sqrsum):
            self.mul_calls.append((tuple(x.shape), tuple(fn.shape)))
            out_mul.fill_(1.0)
            out_sqrsum.fill_(2.0)

        return launch

    def _fuse(
        self,
        hidden,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_mult,
        sinkhorn_repeat,
        *,
        n_splits,
        mhc_mult,
    ):
        def launch(
            gemm_out_mul,
            gemm_out_sqrsum,
            scale,
            base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
        ):
            self.fuse_calls.append(int(n_splits))
            # The outputs fold in the GEMM buffers so a test can tell which
            # GEMM fed the epilogue.
            post_mix.copy_(
                gemm_out_mul.sum(dim=(0, 2), keepdim=False).unsqueeze(-1)[:, :mhc_mult]
                if False
                else gemm_out_mul.sum(dim=(0, -1))[:, None].expand(-1, mhc_mult)
            )
            comb_mix.zero_()
            layer_input.zero_()

        return launch

    def _dg(self, x, fn, out, sqrsum, num_split):
        self.dg_calls.append((tuple(x.shape), tuple(fn.shape), int(num_split)))
        out.fill_(3.0)
        sqrsum.fill_(4.0)
        return None


def _inputs(T=6, H=64, mult=4):
    g = torch.Generator().manual_seed(11)
    residual = torch.randn(2, 3, mult, H, generator=g).to(torch.bfloat16)
    fn = torch.randn(mult * 2 + mult * mult, mult * H, generator=g) * 0.02
    scale = torch.randn(3, generator=g) * 0.5 + 1.0
    base = torch.randn(mult * 2 + mult * mult, generator=g) * 0.1
    return residual, fn, scale, base


def _call(wrapper, residual, fn, scale, base):
    return wrapper.mhc_pre_big_fuse(
        residual,
        fn,
        scale,
        base,
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )


class Resolver(unittest.TestCase):
    def setUp(self):
        self.arch = _load_real_arch()

    def test_flag_default_off_and_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PRENORM_FLAG, None)
            self.assertFalse(self.arch.mhc_prenorm_dg_forced())
        with mock.patch.dict(os.environ, {PRENORM_FLAG: "0"}):
            self.assertFalse(self.arch.mhc_prenorm_dg_forced())
        with mock.patch.dict(os.environ, {PRENORM_FLAG: "1"}):
            self.assertTrue(self.arch.mhc_prenorm_dg_forced())
        for bad in ("yes", "2", "on", ""):
            with mock.patch.dict(os.environ, {PRENORM_FLAG: bad}):
                if bad == "":
                    # empty string is not "0"/"1": fail closed
                    with self.assertRaises(ValueError):
                        self.arch.mhc_prenorm_dg_forced()
                else:
                    with self.assertRaises(ValueError):
                        self.arch.mhc_prenorm_dg_forced()

    def test_resolver_forces_deepgemm_over_recipe_pin(self):
        with mock.patch.dict(
            os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "1"}
        ):
            self.assertEqual(
                self.arch.mhc_pre_gemm_backend(torch.device("cpu")), "deepgemm"
            )
        with mock.patch.dict(
            os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "0"}
        ):
            self.assertEqual(
                self.arch.mhc_pre_gemm_backend(torch.device("cpu")),
                "tilelang_single",
            )
        with mock.patch.dict(os.environ, {BACKEND_ENV: "tilelang_single"}):
            os.environ.pop(PRENORM_FLAG, None)
            self.assertEqual(
                self.arch.mhc_pre_gemm_backend(torch.device("cpu")),
                "tilelang_single",
            )

    def test_resolver_does_not_weaken_an_explicit_deepgemm(self):
        with mock.patch.dict(os.environ, {BACKEND_ENV: "dg", PRENORM_FLAG: "0"}):
            self.assertEqual(
                self.arch.mhc_pre_gemm_backend(torch.device("cpu")), "deepgemm"
            )


class WrapperWiring(unittest.TestCase):
    def setUp(self):
        self.wrapper = _load_wrapper()
        self._nsplit = mock.patch.object(
            self.wrapper, "_compute_num_split", return_value=4
        )
        self._nsplit.start()

    def tearDown(self):
        self._nsplit.stop()

    def test_deepgemm_override_uses_expected_kernel(self):
        residual, fn, scale, base = _inputs()
        spies = _Spies(self.wrapper)
        try:
            with mock.patch.dict(
                os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "1"}
            ):
                post, comb, layer_input = _call(self.wrapper, residual, fn, scale, base)
        finally:
            spies.stop()
        # The DeepGEMM split-K path and ONLY it ran.
        self.assertEqual(len(spies.dg_calls), 1)
        x_shape, fn_shape, num_split = spies.dg_calls[0]
        self.assertEqual(num_split, 4)
        # residual_flat [num_tokens=6, mhc_hidden=4*64] and fn [24, 256].
        self.assertEqual(x_shape, (6, 256))
        self.assertEqual(fn_shape, (24, 256))
        self.assertEqual(spies.mul_calls, [], "tilelang single GEMM ran (wrong target)")
        # The production epilogue consumed the split-K partials.
        self.assertEqual(spies.fuse_calls, [4])
        # The epilogue folds in the GEMM buffers: DG stub fills 3.0 over
        # [4 splits, 6 tokens, 24] -> 4*6*24*3.0/6 = 288.0 per row; the
        # tilelang stub (1.0 over [1, 6, 24]) would give 24.0 — the value
        # discriminates which GEMM fed the epilogue.
        self.assertTrue(torch.all(post == 288.0), "epilogue not fed by the DG GEMM")
        self.assertEqual(tuple(post.shape), (2, 3, 4, 1))

    def test_repeated_calls_dispatch_deepgemm(self):
        residual, fn, scale, base = _inputs()
        spies = _Spies(self.wrapper)
        try:
            with mock.patch.dict(
                os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "1"}
            ):
                first = _call(self.wrapper, residual, fn, scale, base)
                second = _call(self.wrapper, residual, fn, scale, base)
        finally:
            spies.stop()
        self.assertEqual(len(spies.dg_calls), 2)
        self.assertEqual(spies.mul_calls, [])
        self.assertEqual(spies.fuse_calls, [4, 4])
        for got, want in zip(second, first):
            self.assertTrue(torch.equal(got, want))

    def test_flag_off_keeps_tilelang_dispatch_byte_identical(self):
        residual, fn, scale, base = _inputs()
        spies = _Spies(self.wrapper)
        try:
            with mock.patch.dict(os.environ, {BACKEND_ENV: "tilelang_single"}):
                os.environ.pop(PRENORM_FLAG, None)
                out_unset = _call(self.wrapper, residual, fn, scale, base)
            with mock.patch.dict(
                os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "0"}
            ):
                out_off = _call(self.wrapper, residual, fn, scale, base)
        finally:
            spies.stop()
        self.assertEqual(spies.dg_calls, [])
        self.assertEqual(len(spies.mul_calls), 2)  # one per flag-off call
        for a, b in zip(out_unset, out_off):
            self.assertTrue(torch.equal(a, b), "env-unset vs explicit-0 differ")

    def test_missing_deepgemm_raises_loudly(self):
        residual, fn, scale, base = _inputs()
        with mock.patch.object(
            _stub_dgw(),
            "tf32_hc_prenorm_gemm",
            side_effect=RuntimeError("DeepGEMM is not available"),
        ):
            with mock.patch.dict(
                os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "1"}
            ):
                with self.assertRaisesRegex(RuntimeError, "DeepGEMM is not available"):
                    _call(self.wrapper, residual, fn, scale, base)

    def test_dropped_override_mutant_disengages(self):
        """Dropping the opt-in override must be visible to the kernel spies."""
        residual, fn, scale, base = _inputs()
        spies = _Spies(self.wrapper)
        try:
            with mock.patch.object(
                self.wrapper,
                "mhc_pre_gemm_backend",
                lambda device=None: "tilelang_single",  # mutant resolver
            ):
                with mock.patch.dict(
                    os.environ, {BACKEND_ENV: "tilelang_single", PRENORM_FLAG: "1"}
                ):
                    _call(self.wrapper, residual, fn, scale, base)
        finally:
            spies.stop()
        self.assertEqual(spies.dg_calls, [], "mutant still engaged DG")
        self.assertEqual(len(spies.mul_calls), 1)


class SplitKDivergenceSimulation(unittest.TestCase):
    """The Class-B divergence class on CPU: single-pass fp32 GEMM (the
    TileLang single backend's accumulation order) vs split-K fp32 partial
    sums (the DeepGEMM kernel's class).  Pure-fp32 simulation — the hardware
    kernels add TF32 rounding on top, same class on both arms."""

    def _run(self, K=28672, splits=4, seed=5):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(64, K, generator=g).to(torch.bfloat16).float()
        fn = (torch.randn(24, K, generator=g) * 0.02).float()
        ref = x @ fn.t()  # single-pass fp32
        exact = (x.double() @ fn.t().double()).float()
        chunk = K // splits
        partials = [
            x[:, i * chunk : (i + 1) * chunk] @ fn[:, i * chunk : (i + 1) * chunk].t()
            for i in range(splits)
        ]
        cand = partials[0]
        for p in partials[1:]:
            cand = cand + p
        return ref, cand, exact

    def test_splitk_divergence_inside_envelope(self):
        ref, cand, exact = self._run()
        div = (cand - ref).abs().max().item()
        self.assertGreater(div, 0.0)  # anti-vacuity: reassociation is real
        self.assertLessEqual(div, SPLITK_SIM_ABS_MAX)
        # Same precision class: both arms sit at the same distance from fp64.
        err_ref = (ref - exact).abs().max().item()
        err_cand = (cand - exact).abs().max().item()
        self.assertLessEqual(err_cand, 2.0 * max(err_ref, 1e-12))

    def test_injected_noise_exceeds_envelope(self):
        ref, cand, _ = self._run()
        noisy = cand + 10.0 * SPLITK_SIM_ABS_MAX
        div = (noisy - ref).abs().max().item()
        self.assertGreater(div, SPLITK_SIM_ABS_MAX)


class SourceInvariants(unittest.TestCase):
    def test_resolver_structure(self):
        self.assertIn("DSV4_MHC_PRENORM_DEEPGEMM", ARCH_SRC)
        self.assertIn("must be 0 or 1", ARCH_SRC)
        # The opt-in override is applied LAST (after the explicit-backend branch).
        resolver = ARCH_SRC[ARCH_SRC.index("def mhc_pre_gemm_backend") :]
        self.assertLess(
            resolver.index("if requested not in"),
            resolver.index("mhc_prenorm_dg_forced()"),
        )

    def test_warmup_shares_the_single_resolver(self):
        self.assertIn('return mhc_pre_gemm_backend(device) == "deepgemm"', WARMUP_SRC)
        # No second parse of the opt-in flag in the warmup file.
        self.assertNotIn("DSV4_MHC_PRENORM_DEEPGEMM", WARMUP_SRC)


def setUpModule():
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()


if __name__ == "__main__":
    unittest.main()

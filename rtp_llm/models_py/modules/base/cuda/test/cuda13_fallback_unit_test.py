import builtins
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


_CUDA_DIR = pathlib.Path(__file__).resolve().parent.parent
_MODELS_PY_DIR = pathlib.Path(__file__).resolve().parents[4]


def _stub_module(name, *, package=False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_module(name, path, *, package=False, stubs=None):
    search_locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=search_locations
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    injected = dict(stubs or {})
    injected[name] = module
    with mock.patch.dict(sys.modules, injected, clear=False):
        spec.loader.exec_module(module)
    return module


def _fp4_module(*, is_cuda, has_registered_op):
    package_name = "fp4_init_under_test"
    compute_ops = _stub_module("rtp_llm.ops.compute_ops")
    if has_registered_op:
        compute_ops.silu_and_mul_scaled_fp4_experts_quant = object()

    cutlass_wrapper = object()
    quant_wrapper = object()
    moe_wrapper = object()
    grouped_quant = object()
    silu_grouped_quant = object()
    fp4_impl = _stub_module(
        f"{package_name}.fp4_kernel",
        cutlass_scaled_fp4_mm_wrapper=cutlass_wrapper,
        scaled_fp4_quant_wrapper=quant_wrapper,
    )
    cute_impl = _stub_module(
        f"{package_name}.flashinfer_cutedsl_moe",
        flashinfer_cutedsl_moe_masked=moe_wrapper,
        scaled_fp4_grouped_quant=grouped_quant,
        silu_and_mul_scaled_fp4_grouped_quant=silu_grouped_quant,
    )
    ops_package = _stub_module("rtp_llm.ops", package=True, compute_ops=compute_ops)
    stubs = {
        "rtp_llm.models_py.utils.arch": _stub_module(
            "rtp_llm.models_py.utils.arch", is_cuda=lambda: is_cuda
        ),
        "rtp_llm.ops": ops_package,
        "rtp_llm.ops.compute_ops": compute_ops,
        f"{package_name}.fp4_kernel": fp4_impl,
        f"{package_name}.flashinfer_cutedsl_moe": cute_impl,
    }
    module = _load_module(
        package_name,
        _MODELS_PY_DIR / "kernels/cuda/fp4_kernel/__init__.py",
        package=True,
        stubs=stubs,
    )
    return module, {
        "cutlass": cutlass_wrapper,
        "quant": quant_wrapper,
        "moe": moe_wrapper,
        "grouped": grouped_quant,
        "silu_grouped": silu_grouped_quant,
    }


def _indexer_module():
    class KVCache:
        pass

    collective = _stub_module(
        "rtp_llm.models_py.distributed.collective_torch",
        Group=object,
        all_gather=lambda *args, **kwargs: None,
        barrier=lambda *args, **kwargs: None,
    )
    fp8_kernel = _stub_module(
        "rtp_llm.models_py.kernels.cuda.fp8_kernel",
        sgl_per_token_group_quant_fp8=lambda *args, **kwargs: None,
    )
    compute_ops = _stub_module(
        "rtp_llm.ops.compute_ops", KVCache=KVCache, rtp_llm_ops=object()
    )
    rope = _stub_module("flashinfer.rope")
    stubs = {
        "rtp_llm": _stub_module("rtp_llm", package=True),
        "rtp_llm.models_py": _stub_module("rtp_llm.models_py", package=True),
        "rtp_llm.models_py.distributed": _stub_module(
            "rtp_llm.models_py.distributed", package=True
        ),
        "rtp_llm.models_py.distributed.collective_torch": collective,
        "rtp_llm.models_py.kernels": _stub_module(
            "rtp_llm.models_py.kernels", package=True
        ),
        "rtp_llm.models_py.kernels.cuda": _stub_module(
            "rtp_llm.models_py.kernels.cuda", package=True
        ),
        "rtp_llm.models_py.kernels.cuda.fp8_kernel": fp8_kernel,
        "rtp_llm.ops": _stub_module(
            "rtp_llm.ops", package=True, compute_ops=compute_ops
        ),
        "rtp_llm.ops.compute_ops": compute_ops,
        "deep_gemm": _stub_module("deep_gemm"),
        "flashinfer": _stub_module("flashinfer", package=True, rope=rope),
        "flashinfer.rope": rope,
    }
    return _load_module(
        "indexer_op_under_test", _CUDA_DIR / "indexer_op.py", stubs=stubs
    )


class Fp4FallbackTest(unittest.TestCase):
    def test_missing_registered_op_exports_callable_failures(self):
        module, _ = _fp4_module(is_cuda=True, has_registered_op=False)
        for export_name in module.__all__:
            with self.subTest(export_name=export_name):
                exported = getattr(module, export_name)
                self.assertTrue(callable(exported))
                with self.assertRaisesRegex(RuntimeError, "ENABLE_FP4"):
                    exported()

    def test_registered_op_exports_fp4_implementations(self):
        module, expected = _fp4_module(is_cuda=True, has_registered_op=True)
        self.assertIs(module.cutlass_scaled_fp4_mm_wrapper, expected["cutlass"])
        self.assertIs(module.scaled_fp4_quant_wrapper, expected["quant"])
        self.assertIs(module.flashinfer_cutedsl_moe_masked, expected["moe"])
        self.assertIs(module.scaled_fp4_grouped_quant, expected["grouped"])
        self.assertIs(
            module.silu_and_mul_scaled_fp4_grouped_quant,
            expected["silu_grouped"],
        )

    def test_non_cuda_build_preserves_none_exports(self):
        module, _ = _fp4_module(is_cuda=False, has_registered_op=False)
        for export_name in module.__all__:
            with self.subTest(export_name=export_name):
                self.assertIsNone(getattr(module, export_name))


class IndexerFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _indexer_module()

    def _construct_indexer_with_import_error(self, import_error):
        original_import = builtins.__import__

        def import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fast_hadamard_transform":
                raise import_error
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=import_with_failure):
            return self.module.IndexerOp(1, 128, 8, 64)

    def test_missing_fast_hadamard_fails_during_construction(self):
        missing = ModuleNotFoundError(
            "No module named 'fast_hadamard_transform'",
            name="fast_hadamard_transform",
        )
        with self.assertRaisesRegex(RuntimeError, "current.*build|this build") as ctx:
            self._construct_indexer_with_import_error(missing)
        self.assertIs(ctx.exception.__cause__, missing)

    def test_transitive_import_error_is_not_rewritten(self):
        missing = ModuleNotFoundError("No module named 'triton'", name="triton")
        with self.assertRaises(ModuleNotFoundError) as ctx:
            self._construct_indexer_with_import_error(missing)
        self.assertIs(ctx.exception, missing)


if __name__ == "__main__":
    unittest.main()

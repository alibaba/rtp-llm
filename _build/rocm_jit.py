"""Prepare cold AITER extensions before starting timed pytest cases."""

import inspect
import os


MODULES = (
    "module_gemm_a8w8_bpreshuffle_cktile",
    "module_hipbsolgemm",
    "module_rmsnorm_quant",
)


def prepare_modules(core):
    for name in MODULES:
        try:
            core.get_module(name)
        except ModuleNotFoundError:
            config = core.get_args_of_build(name)
            config["md_name"] = name
            build_args = {
                key: value
                for key, value in config.items()
                if key in inspect.signature(core.build_module).parameters
            }
            compiler = config.get("hip_clang_path")
            previous = os.environ.get("HIP_CLANG_PATH")
            try:
                if compiler and os.path.exists(compiler):
                    os.environ["HIP_CLANG_PATH"] = compiler
                core.build_module(**build_args)
                core.get_module(name)
            finally:
                if previous is None:
                    os.environ.pop("HIP_CLANG_PATH", None)
                else:
                    os.environ["HIP_CLANG_PATH"] = previous
        print(f"[rocm_jit] ready: {name}", flush=True)


if __name__ == "__main__":
    import pybind11
    from aiter.jit import core

    print(f"[rocm_jit] pybind11={pybind11.__version__}", flush=True)
    prepare_modules(core)

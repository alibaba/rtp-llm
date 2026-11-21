"""Check independent ownership and both import orders without allocating on a PPU."""

import argparse
import importlib.metadata as metadata
import os
import subprocess
import sys
import tempfile
from itertools import combinations, permutations
from pathlib import Path


def check_exact4_cold_jit(prefill_path):
    """Compile the exact4 general-M kernel without a device or a warm JIT cache."""
    include = Path(prefill_path) / "deep_gemm/include"
    sdk = Path(
        os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME") or "/usr/local/PPU_SDK"
    )
    compiler = (
        os.environ.get("DG_JIT_HGCC_COMPILER")
        or os.environ.get("DG_HGCC_COMPILER")
        or str(sdk / "bin/hgcc")
    )
    source = r"""
#define DG_MOE_FAST_MATH 0
#include <deep_gemm/impls/sm89_fp8_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8Fp4Gemm1Kernel<128, 256, 4096, 128, 32, 2, deep_gemm::GatedActivation(0), deep_gemm::w4fa8::FragmentPolicy::RollingK32, deep_gemm::w4fa8::ALoadPolicy::Ring2, 128, 256, 1, 117, 128, 8, 1024, 6, 2, 128, false, 79734784, 10528768, 77637632, 81307648, 752396288, 773367808, 16384, 8388608, 80783360, 1935096448, 1>;
static_assert(Builder::shared_memory_bytes() == 68608, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp4_mega_moe_l1<Builder>);
}
"""
    flags = "-std=c++20 -DNDEBUG -DUSE_HGGC -DUSE_CLANG -DUSE_ACWRAPPER -arch=ppu_15 -dopt 3 -O3 --compiler-options=-fPIC,-Wno-deprecated-declarations,-Wno-abi -ftemplate-depth=8192 --expt-relaxed-constexpr --expt-extended-lambda -x hg -Xllvm -ppu-patch-fence-ppu=false -Xllvm -wno-loop-miss-transform -Xllvm -ppu-cg-to-kp1=true -Xllvm -ppu-fix-uninit=true -Xllvm -ppu-simt-branch=false".split()
    with tempfile.TemporaryDirectory(prefix="exact4-cold-jit-") as directory:
        root = Path(directory)
        kernel, binary = root / "kernel.cu", root / "kernel.hgbin"
        kernel.write_text(source)
        command = [
            compiler,
            str(kernel),
            "-hgbin",
            "-o",
            str(binary),
            f"-I{include / 'actlize_v1.0.0'}",
            f"-I{include}",
            *flags,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(command, cwd=root, env=env, check=True, timeout=600)
        if not binary.is_file() or binary.stat().st_size == 0:
            raise RuntimeError("Exact4 cold JIT did not produce a native binary")
        print(f"Validated exact4 cold JIT: {binary.stat().st_size} bytes")


def check_hc_cold_jit(prefill_path):
    """Exercise the production Python JIT with a fresh cache, without a device."""
    with tempfile.TemporaryDirectory(prefix="hc-cold-jit-") as directory:
        env = os.environ.copy()
        env["PYTHONPATH"] = prefill_path + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["DG_CACHE_DIR"] = directory
        subprocess.run(
            [sys.executable, __file__, "--hc-cold-jit-child"],
            env=env,
            check=True,
            timeout=600,
        )


def compile_hc_fixture():
    import importlib.util

    import torch
    from deep_gemm.jit import compiler, cpp_format, generate

    import rtp_llm

    # Use the production template and geometry, not a separate test kernel.
    path = (
        Path(rtp_llm.__file__).parent
        / "models_py/modules/dsv4/ppu/hc/ppu_hc_prenorm.py"
    )
    spec = importlib.util.spec_from_file_location("hc_prenorm", path)
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    arg_defs = (
        ("a", torch.bfloat16),
        ("b", torch.float32),
        ("d", torch.float32),
        ("sqr_sum", torch.float32),
        ("m", int),
        ("stream", torch.cuda.Stream),
        ("num_sms", int),
        ("smem_size", int),
    )
    # Image builds have no device; only substitute target architecture detection.
    compiler.is_ppu1v5_device = lambda: True
    compiler.set_jit_include_dir("actlize_v1.0.0")
    for m in (1, 512, 4096):
        code = generate(
            ('"../deep_gemm/tf32_hc_prenorm_gemm.cuh"',),
            arg_defs,
            cpp_format(hc._TEMPLATE, hc.launch_geometry(m, 16384, 24)),
        )
        runtime = compiler.build("rtp_dsv4_hc_prenorm_partials", arg_defs, code)
        binary = Path(runtime.path) / "kernel.so"
        if not binary.is_file() or binary.stat().st_size == 0:
            raise RuntimeError("HC cold JIT did not produce a shared library")
        print(f"Validated HC cold JIT M={m}: {binary.stat().st_size} bytes")


def check_install(prefill_path=None):
    packages = ("deep_gemm", "deep_gemm_mega", "deep_gemm_route_a")
    distributions = [metadata.distribution(name) for name in packages]
    for left, right in combinations(distributions, 2):
        overlap = {p.locate().resolve() for p in left.files or ()} & {
            p.locate().resolve() for p in right.files or ()
        }
        if overlap:
            raise RuntimeError(f"DeepGEMM packages own overlapping files: {overlap}")
    checks = [(order, None) for order in permutations(packages)]
    if prefill_path:
        checks.extend((order, prefill_path) for order in permutations(packages[:2]))
    for order, runtime_path in checks:
        env = os.environ.copy()
        if runtime_path:
            env["PYTHONPATH"] = runtime_path + os.pathsep + env.get("PYTHONPATH", "")
        key = "RTP_PPU_MEGA_LUT_PATH"
        env.pop(key, None)
        if order[0] == "deep_gemm_mega":
            env[key] = "/custom/megamoe/lut"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import torch",
                        f"import {order[0]}",
                        f"import {order[1]}",
                        *(f"import {name}" for name in order[2:]),
                        "import deep_gemm, hashlib",
                        *(["import deep_gemm_route_a"] if runtime_path is None else []),
                        (
                            "assert hashlib.sha256(open(deep_gemm._C.__file__, 'rb').read()).hexdigest() == '6926c504e3945293c65ce78b795b3a357d8d2e22c4078ddf408fca57f7dd778d'"
                            if runtime_path
                            else "pass"
                        ),
                        (
                            "assert hashlib.sha256(open(deep_gemm_route_a._C.__file__, 'rb').read()).hexdigest() == '495872e3c2eab553d728ad54e709b3006d3a3e6818dc2e363f8cf2ba29455fa7'"
                            if runtime_path is None
                            else "pass"
                        ),
                        "from pathlib import Path",
                        "import deep_gemm_mega",
                        "import ast, os",
                        "bundled = Path(deep_gemm_mega.__file__).parent / 'configs/dsv4_flash_bm64'",
                        f"assert os.environ[{key!r}] == "
                        + (repr(env[key]) if key in env else "str(bundled)"),
                        "from deep_gemm_mega import mega",
                        "native = deep_gemm_mega.deep_gemm_mega_cpp",
                        "assert mega._C is native",
                        "tree = ast.parse(Path(mega.__file__).read_text())",
                        "required = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == '_C'}",
                        "missing = sorted(name for name in required if not callable(getattr(native, name, None)))",
                        "assert not missing, f'MegaMoE native APIs missing from {native.__file__}: {missing}'",
                        "num_bytes, slice_buffers = native.get_symm_buffer_size_for_mega_moe(8, 256, 128, 6, 4096, 2048, 'fp4xfp4', 'swiglu', 0)",
                        "assert isinstance(num_bytes, int) and num_bytes > 0 and callable(slice_buffers)",
                        "print(f'Validated MegaMoE native APIs: {sorted(required)}; buffer_bytes={num_bytes}')",
                        "for name in ('get_symm_buffer_for_mega_moe', 'transform_weights_for_mega_moe', 'mega_moe_pre_dispatch', 'fp4_fp4_mega_moe'):",
                        "    assert callable(getattr(deep_gemm_mega, name))",
                        "assert (Path(deep_gemm_mega.__file__).parent / 'configs/dsv4_flash_bm64/ZW-M890P_fp4_megamoe.lut').is_file()",
                        "import ctypes",
                        "for kind, entries in (('sparse', ('create', 'begin', 'finish', 'destroy')), ('signal', ('create', 'exchange', 'destroy'))):",
                        "    library = ctypes.CDLL(str(Path(deep_gemm_mega.__file__).parent / 'lib' / f'lib{kind}_gin.so'))",
                        "    for entry in entries:",
                        "        assert getattr(library, f'{kind}_gin_{entry}') is not None",
                    ]
                ),
            ],
            env=env,
            check=True,
        )
    if prefill_path:
        check_hc_cold_jit(prefill_path)
        check_exact4_cold_jit(prefill_path)
    print(f"Independent DeepGEMM installations: {[d.version for d in distributions]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-path")
    parser.add_argument("--hc-cold-jit-child", action="store_true")
    args = parser.parse_args()
    if args.hc_cold_jit_child:
        compile_hc_fixture()
    else:
        check_install(args.prefill_path)

"""Build one isolated CK RN precision candidate via a manual Bazel target.

Required: AITER_JIT_DIR and --isolated-jit-dir (or MKT_CASCADE_RN_JIT_DIR)
must identify the same new directory. It may contain other regular .so files,
but no previous build or target module. Run this and the precision diagnostic
in separate processes; unset AITER_AOT_IMPORT for the diagnostic.

Only BF16 / D128 / page16 / vectorized VLLM / LSE is generated. This is not a
general replacement for the identically named AITER module. No GPU kernel is
launched here; the existing JIT compiler loads its extension after linking.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODULE = "mha_batch_prefill_bf16_nlogits_nbias_nmask_lse_ndropout_nqscale_nsink"
FILTER = (
    "fmha_batch_prefill_d128_bf16_group_ps16_*_"
    "nlogits_nbias_nmask_lse_ndropout_nqscale_nsink_vectorized_vllm"
)
RN_MACRO = "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"
FAST_EXP_MACRO = "CK_TILE_FMHA_FWD_FAST_EXP2"
RECEIPT = "rn_build_receipt.json"
HOST_SOURCES = (
    "kernels/mha_common.cu",
    "py_itfs_ck/mha_batch_prefill_kernels.cu",
    "pybind/mha_batch_prefill_pybind.cu",
    "cpp_itfs/mha_fwd_batch_prefill.cu",
)
TILES = (
    "b64x128x64x128x64x128_r4x1x1_r4x1x1_w16x16x16_w16x16x16",
    "b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16",
)
EXPECTED_BLOBS = {
    "fmha_batch_prefill_api.cpp",
    *(
        f"fmha_batch_prefill_d128_bf16_group_ps16_{load}_{tile}_"
        "qr_async_vr_psskddv_nlogits_nbias_nmask_lse_ndropout_"
        "nqscale_nsink_vectorized_vllm.cpp"
        for load in ("bload", "gload")
        for tile in TILES
    ),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def safe_absolute_path(value):
    # core.build_module interpolates paths into shell commands internally.
    require(
        bool(value) and re.fullmatch(r"/[A-Za-z0-9_./-]+", value),
        f"Require a shell-safe absolute path: {value!r}",
    )
    path = Path(value)
    require(path == path.resolve(), f"Symlink or noncanonical directory: {path}")
    return path


def set_fixed_env(name, value):
    require(
        os.environ.get(name, value) == value,
        f"Conflicting environment: {name}={os.environ.get(name)!r}; require {value}",
    )
    os.environ[name] = value


def validate_macro(text, name, value):
    definitions = re.findall(r"-D" + re.escape(name) + r"=([^\s]+)", text)
    require(
        definitions and set(definitions) == {value},
        f"Require only {name}={value}; found {definitions}",
    )


def prepare_directory(explicit):
    require(
        not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules),
        "AITER was already imported; use a fresh build process",
    )
    directory = safe_absolute_path(explicit)
    require(
        safe_absolute_path(os.environ.get("AITER_JIT_DIR", "")) == directory,
        "AITER_JIT_DIR must equal the explicitly selected RN directory",
    )
    spec = importlib.util.find_spec("aiter")
    require(spec is not None and spec.origin, "Cannot locate installed AITER sources")
    package = Path(spec.origin).resolve().parent
    require(
        not directory.is_relative_to(package.parent)
        and not package.parent.is_relative_to(directory),
        "RN output must be separate from installed/site-packages directories",
    )
    require(directory.parent.is_dir(), "Create the parent experiment directory first")
    directory.mkdir(exist_ok=True)
    require(
        not list(directory.glob(MODULE + "*")), "Target native module already exists"
    )
    entries = sorted(directory.iterdir())
    require(
        all(p.is_file() and not p.is_symlink() and p.suffix == ".so" for p in entries),
        "RN directory must be new: only other regular .so dependencies are allowed",
    )
    # Exclusive creation also prevents a second helper from sharing this directory.
    with (directory / RECEIPT).open("x") as stream:
        json.dump({"status": "PREPARING", "pid": os.getpid()}, stream)
    return directory, entries


def build(directory, receipt):
    for name, value in {
        RN_MACRO: "0",
        "GPU_ARCHS": "gfx942",
        "CU_NUM": "80",
        "AITER_AOT_IMPORT": "1",
        "AITER_REBUILD": "0",
        "AITER_TRITON_ONLY": "0",
        "ENABLE_CK": "1",
        "CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT": "0",
        "CK_TILE_ATTENTION_USE_SOFTSIGN_ASM": "1",
    }.items():
        set_fixed_env(name, value)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    jobs = os.environ.setdefault("MAX_JOBS", "4")
    require(jobs.isdigit() and 1 <= int(jobs) <= 4, "MAX_JOBS must be between 1 and 4")
    receipt["environment"] = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("AITER_", "CK_TILE_"))
        or k in ("GPU_ARCHS", "CU_NUM", "MAX_JOBS", "CK_DIR", "ROCM_PATH", "HIP_PATH")
    }

    from aiter.jit import core

    require(
        Path(core.get_user_jit_dir()).resolve() == directory,
        "Imported JIT directory does not match the RN directory",
    )
    require(
        Path(core.bd_dir).resolve() == directory / "build", "Unexpected JIT build root"
    )
    require(
        core.AITER_REBUILD == 0 and core.gfxs == ["gfx942"],
        "Unexpected cached rebuild/architecture configuration",
    )
    ck = Path(core.CK_DIR).resolve()
    require(not ck.is_relative_to(directory), "CK source must be outside the RN output")
    generator = ck / "example/ck_tile/01_fmha/generate.py"
    require(generator.is_file(), "CK generator source is unavailable")
    args = core.get_args_of_build("module_mha_batch_prefill")
    require(
        args["third_party"] == [] and not args.get("hipify", False),
        "No dependency cloning or source hipification is permitted",
    )
    require(
        args["is_python_module"] and not args["is_standalone"],
        "Only a private Python extension may be built",
    )
    require(
        not args["flags_extra_cc"] and not args.get("flags_extra_hip_per_source"),
        "Unexpected host or per-source flag overrides",
    )
    flags = " ".join(args["flags_extra_hip"])
    validate_macro(flags, RN_MACRO, "0")
    validate_macro(flags, FAST_EXP_MACRO, "1")
    expected_sources = [Path(core.AITER_CSRC_DIR) / p for p in HOST_SOURCES]
    require(
        [Path(p).resolve() for p in args["srcs"]]
        == [p.resolve() for p in expected_sources],
        "Unexpected host source set",
    )
    generator_argv = [
        str(generator),
        "--targets",
        "gfx942",
        "-d",
        "batch_prefill",
        "--receipt",
        "200",
        "--filter",
        FILTER,
    ]
    args["md_name"] = MODULE
    args["blob_gen_cmd"] = [shlex.join(generator_argv) + " --output_dir {}"]
    kwargs = {
        k: args[k] for k in inspect.signature(core.build_module).parameters if k in args
    }
    receipt["build_arguments"] = kwargs
    receipt["source_hash_scope"] = (
        "Host TUs, JIT/codegen implementation, and BF16/pipeline headers"
    )
    source_paths = [
        Path(__file__),
        Path(core.__file__),
        Path(core.__file__).with_name("optCompilerConfig.json"),
        Path(core.__file__).parent / "utils/cpp_extension.py",
        Path(core.__file__).parent / "utils/chip_info.py",
        generator,
        *expected_sources,
        ck / "include/ck_tile/core/numeric/bfloat16.hpp",
        ck
        / "include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp",
    ]
    source_paths += sorted((generator.parent / "codegen").rglob("*.py"))
    receipt["sources"] = [file_record(p) for p in source_paths]
    blob_list = directory / "rn_blobs.txt"
    subprocess.run(
        [sys.executable, *generator_argv, "--list_blobs", str(blob_list)], check=True
    )
    names = [Path(p).name for p in blob_list.read_text().splitlines() if p]
    require(
        len(names) == 5 and set(names) == EXPECTED_BLOBS,
        f"Codegen must select precisely 4 kernels and 1 API: {names}",
    )
    receipt["generated_names"] = names
    receipt["blob_list"] = file_record(blob_list)
    receipt["status"] = "BUILDING"
    write_receipt(directory, receipt)
    require(
        not list(directory.glob(MODULE + "*")), "Native target appeared before build"
    )
    started = time.perf_counter()
    try:
        core.build_module(**kwargs)
    finally:
        receipt["compile_seconds"] = time.perf_counter() - started

    op_dir = directory / "build" / MODULE
    blobs = sorted((op_dir / "blob").rglob("*.cpp"))
    require(
        len(blobs) == 5 and {p.name for p in blobs} == EXPECTED_BLOBS,
        "Actual compiled blob set differs from the preflight list",
    )
    receipt["generated_sources"] = [file_record(p) for p in blobs]
    receipt["copied_sources"] = [
        file_record(p) for p in sorted((op_dir / "build/srcs").rglob("*.cu"))
    ]
    ninja = op_dir / "build/build.ninja"
    ninja_text = ninja.read_text()
    validate_macro(ninja_text, RN_MACRO, "0")
    validate_macro(ninja_text, FAST_EXP_MACRO, "1")
    archs = re.findall(r"--offload-arch=([^\s]+)", ninja_text)
    require(
        archs and set(archs) == {"gfx942"},
        f"Unexpected compiled architectures: {archs}",
    )
    receipt["ninja"] = file_record(ninja)
    receipt["ninja_flag_lines"] = [
        line for line in ninja_text.splitlines() if "flags =" in line
    ]
    target = directory / (MODULE + ".so")
    require(
        target.is_file() and not target.is_symlink(),
        "RN library was not produced locally",
    )
    with target.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "RN output is not an ELF shared library")
    receipt["library"] = file_record(target)
    receipt["linked_library"] = file_record(op_dir / "build" / target.name)
    require(
        receipt["library"]["sha256"] == receipt["linked_library"]["sha256"],
        "Copied and linked native libraries differ",
    )
    require(
        all(
            file_record(p["path"])["sha256"] == p["sha256"] for p in receipt["sources"]
        ),
        "Build source changed during compilation",
    )
    require(
        all(
            file_record(p["path"])["sha256"] == p["sha256"]
            for p in receipt["preseeded_libraries"]
        ),
        "A preseeded dependency changed during compilation",
    )
    receipt["status"] = "BUILD_PASS"
    receipt["gpu_accuracy_and_performance"] = "NOT_RUN"


def write_receipt(directory, receipt):
    (directory / RECEIPT).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated-jit-dir", default=os.environ.get("MKT_CASCADE_RN_JIT_DIR")
    )
    options = parser.parse_args()
    directory, entries = prepare_directory(options.isolated_jit_dir)
    started = time.perf_counter()
    receipt = {
        "status": "PREPARING",
        "pid": os.getpid(),
        "module": MODULE,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "isolated_jit_dir": str(directory),
        "filter": FILTER,
        "scope": {
            "dtype": "BF16",
            "dim": 128,
            "page": 16,
            "arch": "gfx942",
            "cu_num": 80,
            "layout": "vectorized_vllm",
            "return_lse": True,
        },
    }
    try:
        receipt["preseeded_libraries"] = [file_record(p) for p in entries]
        build(directory, receipt)
    except BaseException as error:
        receipt["status"] = "BUILD_FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_receipt(directory, receipt)
        print(
            json.dumps(
                {"status": receipt["status"], "receipt": str(directory / RECEIPT)}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

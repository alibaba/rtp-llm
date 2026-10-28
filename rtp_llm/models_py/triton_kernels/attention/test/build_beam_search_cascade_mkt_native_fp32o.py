"""Build an isolated BF16-QKVP / FP32-O CK RN precision candidate.

Use a new AITER_JIT_DIR equal to --isolated-jit-dir (or
MKT_CASCADE_FP32O_JIT_DIR). Other regular dependency .so files may be preseeded;
the target native module must be absent. This manual builder compiles 9 TUs,
without modifying installed sources, previous libraries, or default selection.
Run numerical diagnostics in a separate process with an explicit FP32 out
buffer; unset the build-only AITER_AOT_IMPORT environment variable there.
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
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODULE = "mha_batch_prefill_bf16_nlogits_nbias_nmask_lse_ndropout_nqscale_nsink"
FILTER = "fmha_batch_prefill_d128_bf16_group_ps16_*_nlogits_nbias_nmask_lse_ndropout_nqscale_nsink_vectorized_vllm"
RECEIPT = "fp32o_build_receipt.json"
RN_MACRO = "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"
FAST_EXP_MACRO = "CK_TILE_FMHA_FWD_FAST_EXP2"
INVALID_V_HEADER = (
    "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp"
)
INVALID_V_HEADER_SHA256 = (
    "5493558a1ded949da25663244492485762e2e84e7863dd1f5757a561f49cdc69"
)
HOST_HASHES = {
    "kernels/mha_common.cu": "b196471e307e9ae39a339248620b13321dfe8983f140ce9a4efa6c902022e682",
    "py_itfs_ck/mha_batch_prefill_kernels.cu": "1e88e36a4e0cd42523d412bb24d456eb52e5013a718ea7e90b28fbda70d59813",
    "pybind/mha_batch_prefill_pybind.cu": "2b1b64bf65574c66a6c16a26f7e038e8b192d4e3bd34491274172abea4c0ad9e",
    "cpp_itfs/mha_fwd_batch_prefill.cu": "83542ba243a47f436580e817173377a3627b4ebc72f2f0de9d8bb219e6ff6bea",
}
JIT_HASHES = {
    "core.py": "2f70aaa835086d261734d08ba3bc4c7368e53769865e2a3e2579edbedd8c0294",
    "optCompilerConfig.json": "08a9d2741eb8f2a99d2ebc3f4f14d3e3c3a7e6a31a62986364214a7c2da6947d",
    "utils/cpp_extension.py": "550dc7e8a6d48ce02bb35974561d640e6e38df9804b4f4425a7091cd24ecc776",
}
CK_HASHES = {
    "example/ck_tile/01_fmha/generate.py": "87d4f012c63597bf6929fe68b6d957ebfe6fb099edf0e385fa7badc7d107ab1a",
    "example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py": "15243e2dfaf153cc99e56f357843cc2bd125ec0d597ec2071a045f1a7c709e75",
    "example/ck_tile/01_fmha/fmha_fwd.hpp": "60027c43c3daa34888e131b950ce8fe4df58a4da8c04db038d0fee46e3e73973",
    "include/ck_tile/core/numeric/bfloat16.hpp": "c499f477c1d4744c97d8c28f0716da29c7f90469ec979d70c2e46dfc530c825b",
    "include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp": "685c481ef36cbbdeafc9ec49a912136d83cd8b7eb53b998f7823051e4003b202",
    "include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp": "f72960eba179a634e0c227ff8d0ca8dc630e47dcb8dd1f2ca2dd54d500058a35",
    "include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp": "b672f6b1e48b08cbade538ffe653103655c002f879043f980ee657017df1397d",
    "include/ck_tile/ops/epilogue/default_2d_epilogue.hpp": "6ac07022bc315f17bd017f9696bfcaae804bf49dab2856daa4193f9149c6212b",
}
TILES = (
    "b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16",
    "b64x128x64x128x64x128_r4x1x1_r4x1x1_w16x16x16_w16x16x16",
)
KERNEL_NAMES = [
    f"fmha_batch_prefill_d128_bf16_group_ps16_{load}_{tile}_"
    "qr_async_vr_psskddv_nlogits_nbias_nmask_lse_ndropout_nqscale_nsink_vectorized_vllm.cpp"
    for load in ("bload", "gload")
    for tile in TILES
]
BLOB_HASHES = dict(
    zip(
        KERNEL_NAMES,
        (
            "53998623478f662f23876a757df3c684c863f9180d01818de893696d5d7aad2a",
            "0e1264c82972e8abfe24c0df93b5463a7a9c00d752b7de7a40e2fa02d0a42fee",
            "2ba84e389d2ea8a193aa8273bfb3f299cfdde4a9b93c4bfb35b736a1e6c46820",
            "39df4ac13cd56c1b690387da3c032d749b31a4ec40a0650909d573c032691768",
        ),
    )
)
BLOB_HASHES["fmha_batch_prefill_api.cpp"] = (
    "dd7954c799203ba82a80a952732262c78c4827a0fd48f63106fc20acf49566fc"
)

PIPELINE_O = "    typename FmhaFwdTypeConfig<fmha_dtype_0>::ODataType,"
EPILOGUE_O = "                                           typename FmhaFwdTypeConfig<FmhaFwdBf16>::ODataType,"
WRAPPER_OUT = (
    '    auto out_type = dtype_str == "fp8bf16" ? at::ScalarType::BFloat16 : q_dtype;'
)
WRAPPER_MESSAGE = (
    '"For FP16/BF16 input, output must have the same dtype as inputs. For FP8 "\n'
    '                    "input, output must have dtype BF16"'
)
PRIVATE_OUT = """    // This private module only contains the four validated BF16/D128/page16 arms.
    TORCH_CHECK(q_dtype == at::ScalarType::BFloat16 &&
                    head_size_q == 128 && head_size_v == 128 &&
                    num_heads == 16 && num_heads_k == 8 && page_block_size == 16 &&
                    k.dim() == 5 && v.dim() == 5 &&
                    block_table_.has_value() && seqlen_k_.has_value() &&
                    max_seqlen_q > 0 && max_seqlen_q <= 128 &&
                    p_dropout == 0.0f && logits_soft_cap == 0.0f &&
                    !is_causal && window_size_left == -1 && window_size_right == -1 &&
                    !bias_.has_value() && !alibi_slopes_.has_value() &&
                    qscale_type == quant_scale_enum::no_scale &&
                    sink_size == 0 && !sink_ptr.has_value() &&
                    return_softmax_lse && !return_dropout_randval,
                "Private FP32-O module requires BF16 Q16/KV8 D128 page16 vectorized "
                "VLLM, LSE, and no mask/bias/dropout/quant-scale/sink");
    const auto out_type = at::ScalarType::Float;"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_record(path):
    path = Path(path)
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def safe_absolute_path(value):
    require(
        bool(value) and re.fullmatch(r"/[A-Za-z0-9_./-]+", value),
        f"Require a shell-safe absolute path: {value!r}",
    )
    path = Path(value)
    require(path == path.resolve(), f"Require canonical nonsymlink path: {path}")
    return path


def set_fixed_env(name, value):
    require(
        os.environ.get(name, value) == value,
        f"Conflicting {name}={os.environ.get(name)!r}; require {value}",
    )
    os.environ[name] = value


def validate_macro(text, name, value):
    values = re.findall(r"-D" + re.escape(name) + r"=([^\s]+)", text)
    require(
        values and set(values) == {value}, f"Require only {name}={value}; got {values}"
    )


def replace_once(text, before, after):
    require(text.count(before) == 1, f"Expected one patch anchor: {before!r}")
    return text.replace(before, after, 1)


def patch_kernel(text):
    require(text.count("using fmha_dtype_0 = FmhaFwdBf16;") == 1, "Not a BF16 instance")
    original = text
    text = replace_once(text, PIPELINE_O, "    float, // private FP32 intermediate O")
    text = replace_once(
        text,
        EPILOGUE_O,
        "                                           float, // private FP32 intermediate O",
    )
    require("::ODataType" not in text, "An unpatched output type remains")
    for dtype in (
        "QDataType",
        "KDataType",
        "VDataType",
        "PDataType",
        "OaccDataType",
        "LSEDataType",
    ):
        require(
            text.count("::" + dtype) == original.count("::" + dtype), f"Changed {dtype}"
        )
    return text


def patch_wrapper(text):
    text = replace_once(text, WRAPPER_OUT, PRIVATE_OUT)
    return replace_once(
        text, WRAPPER_MESSAGE, '"Private BF16-input prefix requires FP32 output"'
    )


def verify_sources(base, expected):
    records = []
    for relative, sha in expected.items():
        record = file_record(Path(base) / relative)
        require(
            record["sha256"] == sha, f"Original source hash mismatch: {record['path']}"
        )
        records.append(record)
    return records


def create_overlay(overlay, csrc, raw_blobs, *, inclusive_tile_boundary=False):
    """Copy nine pinned original TUs, then patch only private copies."""
    overlay = Path(overlay)
    overlay.mkdir()  # Exclusive: a previous overlay is never reused or overwritten.
    (overlay / "host").mkdir()
    (overlay / "generated").mkdir()
    originals = verify_sources(csrc, HOST_HASHES) + verify_sources(
        raw_blobs, BLOB_HASHES
    )
    patched = []
    paths = []
    for relative in HOST_HASHES:
        source = Path(csrc) / relative
        destination = overlay / "host" / source.name
        text = source.read_text()
        if relative == "py_itfs_ck/mha_batch_prefill_kernels.cu":
            text = patch_wrapper(text)
            patched.append(
                dict(
                    path=str(destination),
                    changes=[
                        "out_type=Float with private geometry guard",
                        "output dtype diagnostic",
                    ],
                )
            )
        destination.write_text(text)
        paths.append(destination)
    for name in sorted(BLOB_HASHES):
        source = Path(raw_blobs) / name
        destination = overlay / "generated" / name
        text = source.read_text()
        if name != "fmha_batch_prefill_api.cpp":
            text = patch_kernel(text)
            patched.append(
                dict(
                    path=str(destination),
                    changes=["pipeline ODataType=float", "epilogue ODataType=float"],
                )
            )
        elif inclusive_tile_boundary:
            # Test only the existing M64 recipe at the 64-CTA/80-CU boundary.
            anchor = "get_num_blocks(128) < num_cus * min_cu_util_rate"
            require(text.count(anchor) == 2, "Require two pinned M64 selector branches")
            text = text.replace(
                anchor, "get_num_blocks(128) <= num_cus * min_cu_util_rate"
            )
            patched.append(
                dict(
                    path=str(destination),
                    changes=["M64 selector < to <= in E0/E1; kernel TUs unchanged"],
                )
            )
        destination.write_text(text)
        paths.append(destination)
    require(
        len(paths) == 9 and len({p.name for p in paths}) == 9, "Require nine unique TUs"
    )
    manifest = dict(
        originals=originals,
        compiled_sources=[file_record(p) for p in paths],
        patches=patched,
        invariants="Q/K/V/P BF16; Oacc/softmax/LSE FP32; output O FP32; API signature/pybind unchanged",
        inclusive_tile_boundary=inclusive_tile_boundary,
    )
    (overlay / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths, manifest


def prepare_directory(explicit):
    require(
        not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules),
        "Use a fresh build process",
    )
    directory = safe_absolute_path(explicit)
    require(
        safe_absolute_path(os.environ.get("AITER_JIT_DIR", "")) == directory,
        "AITER_JIT_DIR must equal the explicitly selected FP32-O directory",
    )
    spec = importlib.util.find_spec("aiter")
    require(spec is not None and spec.origin, "Cannot locate AITER")
    site = Path(spec.origin).resolve().parent.parent
    require(
        not directory.is_relative_to(site) and not site.is_relative_to(directory),
        "Output must be outside installed sources",
    )
    require(directory.parent.is_dir(), "Create the parent experiment directory first")
    directory.mkdir(exist_ok=True)
    require(not list(directory.glob(MODULE + "*")), "Target module already exists")
    entries = sorted(directory.iterdir())
    require(
        all(p.is_file() and not p.is_symlink() and p.suffix == ".so" for p in entries),
        "New directory may contain only other regular dependency .so files",
    )
    with (directory / RECEIPT).open("x") as stream:
        json.dump({"status": "PREPARING", "pid": os.getpid()}, stream)
    return directory, entries


def install_invalid_v_header(directory, args, sources, receipt):
    """Use a pinned private quoted include; retain every installed CK header."""
    enabled = os.getenv("MKT_CASCADE_NATIVE_INVALID_V_MASK", "0")
    require(
        enabled in ("0", "1"), "Invalid-V mask experiment must be explicitly 0 or 1"
    )
    receipt["logical_invalid_v_mask"] = enabled == "1"
    if enabled == "0":
        return None
    source = (
        Path(__file__).resolve().parent
        / "native_partial_safe/include"
        / INVALID_V_HEADER
    )
    record = file_record(source)
    require(
        record["sha256"] == INVALID_V_HEADER_SHA256, "Private V mask header pin changed"
    )
    sources.append(record)
    include = directory / "source_overlay/private_include"
    destination = include / INVALID_V_HEADER
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    # CK fmha.hpp uses a fully qualified quoted include. -iquote precedes the
    # installed -I roots even though AITER places those roots before extra_include.
    flag = "-iquote" + str(include)
    args["flags_extra_hip"] = [*args["flags_extra_hip"], flag]
    # The pinned HIP Ninja rule does not use depfile/deps. Ask the actual four
    # HIP compiler invocations to retain separate dependency files themselves.
    dependency_dir = directory / "header_dependencies"
    dependency_dir.mkdir()
    dependencies = {
        name: dependency_dir / (Path(name).stem + ".d") for name in KERNEL_NAMES
    }
    args["flags_extra_hip_per_source"] = {
        name: ["-MD", "-MF", str(depfile), "-MT", Path(name).stem + ".cuda.o"]
        for name, depfile in dependencies.items()
    }
    receipt["invalid_v_header"] = {
        "source": record,
        "compiled_copy": file_record(destination),
        "include_flag": flag,
        "dependency_files": {name: str(path) for name, path in dependencies.items()},
        "dependency_flags": args["flags_extra_hip_per_source"],
    }
    return destination


def verify_private_header_dependencies(private_header, receipt):
    """Read compiler-emitted .d files without invoking Ninja or touching indexes."""
    verified = {}
    for name in KERNEL_NAMES:
        obj = Path(name).stem + ".cuda.o"
        depfile = Path(receipt["invalid_v_header"]["dependency_files"][name])
        require(depfile.is_file(), f"HIP compiler did not emit dependencies for {obj}")
        dependency_text = depfile.read_text().replace("\\\n", " ")
        target, separator, paths = dependency_text.partition(":")
        require(
            separator and target.strip() == obj, f"Dependency target changed: {obj}"
        )
        includes = [
            path for path in shlex.split(paths) if path.endswith(INVALID_V_HEADER)
        ]
        require(
            includes == [str(private_header)],
            f"Kernel did not consume only the private header: {obj}",
        )
        verified[obj] = {
            "private_header": includes,
            "compiler_depfile": file_record(depfile),
        }
    require(
        file_record(private_header)["sha256"] == INVALID_V_HEADER_SHA256,
        "Private header changed during compilation",
    )
    receipt["invalid_v_header_dependencies"] = {
        "four_kernel_objects": verified,
        "source": "explicit -MD/-MF/-MT on the four actual HIP compile commands",
        "ninja_dependency_tool_invoked": False,
    }


def write_receipt(directory, receipt):
    (directory / RECEIPT).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def build(directory, receipt):
    for name, value in {
        RN_MACRO: "0",
        "GPU_ARCHS": "gfx942",
        "CU_NUM": "80",
        "AITER_AOT_IMPORT": "1",
        "AITER_REBUILD": "0",
        "AITER_TRITON_ONLY": "0",
        "AITER_ASM_DEBUG": "0",
        "ENABLE_CK": "1",
        "CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT": "0",
        "CK_TILE_ATTENTION_USE_SOFTSIGN_ASM": "1",
    }.items():
        set_fixed_env(name, value)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    jobs = os.environ.setdefault("MAX_JOBS", "4")
    require(jobs.isdigit() and 1 <= int(jobs) <= 4, "MAX_JOBS must be 1..4")
    receipt["environment"] = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("AITER_", "CK_TILE_"))
        or k in ("GPU_ARCHS", "CU_NUM", "MAX_JOBS", "CK_DIR", "ROCM_PATH", "HIP_PATH")
    }

    from aiter.jit import core

    require(
        Path(core.get_user_jit_dir()).resolve() == directory
        and Path(core.bd_dir).resolve() == directory / "build",
        "Imported JIT/build directory differs from explicit private directory",
    )
    require(
        core.AITER_REBUILD == 0 and core.gfxs == ["gfx942"],
        "Unexpected rebuild/architecture configuration",
    )
    ck, csrc = Path(core.CK_DIR).resolve(), Path(core.AITER_CSRC_DIR).resolve()
    require(
        not ck.is_relative_to(directory) and not csrc.is_relative_to(directory),
        "Original sources must be outside output",
    )
    jit = Path(core.__file__).resolve().parent
    sources = (
        verify_sources(jit, JIT_HASHES)
        + verify_sources(ck, CK_HASHES)
        + verify_sources(csrc, HOST_HASHES)
    )
    compiler = inspect.getmodule(core._jit_compile)
    require(
        compiler is not None
        and file_record(compiler.__file__)["sha256"]
        == JIT_HASHES["utils/cpp_extension.py"],
        "Unexpected JIT compiler implementation",
    )
    require(compiler.IS_HIP_EXTENSION, "Require the HIP extension compiler")
    generator = ck / "example/ck_tile/01_fmha/generate.py"
    sources.extend(
        file_record(p) for p in sorted((generator.parent / "codegen").rglob("*.py"))
    )
    sources.extend(
        [
            file_record(Path(__file__)),
            file_record(csrc / "include/torch/mha_batch_prefill.h"),
        ]
    )
    receipt["original_sources"] = sources
    args = core.get_args_of_build("module_mha_batch_prefill")
    host_paths = {
        "actual": [str(Path(p).resolve()) for p in args["srcs"]],
        "expected": [str((csrc / p).resolve()) for p in HOST_HASHES],
    }
    receipt["canonical_host_source_paths"] = host_paths
    require(
        host_paths["actual"] == host_paths["expected"],
        f"Unexpected host TU set: {json.dumps(host_paths)}",
    )
    require(
        args["third_party"] == [] and not args.get("hipify", False),
        "No cloning or source hipification",
    )
    require(
        args["is_python_module"]
        and not args["is_standalone"]
        and not args["torch_exclude"],
        "Require Python torch extension",
    )
    require(
        not args["flags_extra_cc"] and not args.get("flags_extra_hip_per_source"),
        "Unexpected compiler overrides",
    )
    include_paths = {
        "actual": [str(Path(p).resolve()) for p in args["extra_include"]],
        "expected": [str(generator.parent.resolve())],
    }
    receipt["canonical_extra_include_paths"] = include_paths
    require(
        include_paths["actual"] == include_paths["expected"],
        f"Unexpected CK include roots: {json.dumps(include_paths)}",
    )
    flags = " ".join(args["flags_extra_hip"])
    validate_macro(flags, RN_MACRO, "0")
    validate_macro(flags, FAST_EXP_MACRO, "1")

    raw = directory / "original_generated"
    raw.mkdir()
    generator_argv = [
        sys.executable,
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
    blob_list = directory / "fp32o_blobs.txt"
    subprocess.run([*generator_argv, "--list_blobs", str(blob_list)], check=True)
    names = [Path(p).name for p in blob_list.read_text().splitlines() if p]
    require(
        len(names) == 5 and set(names) == set(BLOB_HASHES),
        "Require exactly four kernels and one API",
    )
    subprocess.run([*generator_argv, "--output_dir", str(raw)], check=True)
    require(
        {p.name for p in raw.iterdir()} == set(BLOB_HASHES),
        "Unexpected generated files",
    )
    tile_boundary = os.getenv("MKT_CASCADE_NATIVE_TILE_BOUNDARY_INCLUSIVE", "0")
    require(
        tile_boundary in ("0", "1"),
        "Native tile boundary experiment must be explicitly 0 or 1",
    )
    receipt["inclusive_tile_boundary"] = tile_boundary == "1"
    paths, overlay = create_overlay(
        directory / "source_overlay",
        csrc,
        raw,
        inclusive_tile_boundary=tile_boundary == "1",
    )
    require(
        all(compiler._is_cuda_file(str(p)) for p in paths),
        "Every .cpp/.cu TU must select HIP compilation",
    )
    receipt["hip_source_classification"] = {
        str(p): compiler._is_cuda_file(str(p)) for p in paths
    }
    receipt["overlay"] = overlay
    receipt["overlay_manifest"] = file_record(
        directory / "source_overlay/manifest.json"
    )
    receipt["generator_argv"] = generator_argv
    receipt["blob_list"] = file_record(blob_list)
    private_header = install_invalid_v_header(directory, args, sources, receipt)
    args.update(md_name=MODULE, srcs=[str(p) for p in paths], blob_gen_cmd=[])
    kwargs = {
        k: args[k] for k in inspect.signature(core.build_module).parameters if k in args
    }
    receipt["build_arguments"] = kwargs
    receipt["status"] = "BUILDING"
    write_receipt(directory, receipt)
    require(not list(directory.glob(MODULE + "*")), "Target appeared before build")
    started = time.perf_counter()
    try:
        core.build_module(**kwargs)
    finally:
        receipt["compile_seconds"] = time.perf_counter() - started

    op_dir = directory / "build" / MODULE
    ninja = op_dir / "build/build.ninja"
    text = ninja.read_text()
    validate_macro(text, RN_MACRO, "0")
    validate_macro(text, FAST_EXP_MACRO, "1")
    archs = re.findall(r"--offload-arch=([^\s]+)", text)
    require(archs and set(archs) == {"gfx942"}, "Unexpected compiled architecture")
    rules = re.findall(
        r"^build [^\n]+: (cuda_compile|compile) (.+)$", text, re.MULTILINE
    )
    require(
        len(rules) == 9 and all(rule == "cuda_compile" for rule, _ in rules),
        "Require nine actual HIP compile rules",
    )
    require(
        {Path(p).resolve() for _, p in rules} == set(paths),
        "Ninja inputs differ from private overlay",
    )
    receipt["ninja"] = file_record(ninja)
    receipt["ninja_compile_rules"] = rules
    receipt["ninja_flag_lines"] = [
        line for line in text.splitlines() if "flags =" in line
    ]
    if private_header is not None:
        require(
            receipt["invalid_v_header"]["include_flag"] in text,
            "Private include flag absent from Ninja",
        )
        for depfile in receipt["invalid_v_header"]["dependency_files"].values():
            require(
                depfile in text, "HIP dependency flag absent from actual Ninja rules"
            )
        verify_private_header_dependencies(private_header, receipt)
    target = directory / (MODULE + ".so")
    require(
        target.is_file() and not target.is_symlink(), "Private library was not produced"
    )
    with target.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "Output is not ELF")
    receipt["library"] = file_record(target)
    receipt["linked_library"] = file_record(op_dir / "build" / target.name)
    require(
        receipt["library"]["sha256"] == receipt["linked_library"]["sha256"],
        "Installed copy differs from linked library",
    )
    for record in (
        sources
        + overlay["originals"]
        + overlay["compiled_sources"]
        + receipt["preseeded_libraries"]
    ):
        require(
            file_record(record["path"])["sha256"] == record["sha256"],
            f"Input changed during build: {record['path']}",
        )
    receipt["status"] = "BUILD_PASS"
    receipt["gpu_accuracy_and_performance"] = "NOT_RUN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated-jit-dir", default=os.environ.get("MKT_CASCADE_FP32O_JIT_DIR")
    )
    options = parser.parse_args()
    directory, entries = prepare_directory(options.isolated_jit_dir)
    started = time.perf_counter()
    receipt = dict(
        status="PREPARING",
        pid=os.getpid(),
        module=MODULE,
        filter=FILTER,
        started_utc=datetime.now(timezone.utc).isoformat(),
        isolated_jit_dir=str(directory),
        scope=dict(
            qkvp_dtype="BF16",
            intermediate_output_dtype="FP32",
            dim=128,
            page=16,
            q_heads=16,
            kv_heads=8,
            arch="gfx942",
            cu_num=80,
            return_lse=True,
        ),
    )
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

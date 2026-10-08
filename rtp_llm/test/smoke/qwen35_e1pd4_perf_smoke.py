"""Frozen single-Encoder + PD4 performance preset; dry-run unless --execute."""

import hashlib
import importlib.util
import os
import platform
import sys
import uuid
from pathlib import Path

import qwen35_epd_fusion_4gpu_smoke as smoke

TARGET = "//rtp_llm/test/smoke:qwen35_e1pd4_perf_smoke"
REFERENCE_RUN = "br-e1-n7-c512-01"
DEFAULTS = dict(
    workload="video",
    profile="baseline",
    encoder_dp2=1,
    encoder_profile="candidate-a",
    encoder_rdma_pool_bytes=137438953472,
    moe_strategy="mega_moe_fp8",
    decode_graph=1,
    fp8_kv_cache=1,
    native_fp8_attn=1,
    seq_size_per_block=4096,
    scheduler_policy="prefill-first",
    decode_prefill_ratio="7",
    schedule_trace=1,
    coord_mode="cadence",
    coord_checks=0,
    coord_fault="none",
    rank_concurrency=256,
    kv_cache_mb=0,
    runtime_reserve_mb=57344,
    graph_batches="1,2,4,8,16,32,64,96,128,160,192,224,256",
    client_mode="measured",
    client_processes=4,
    sse_chunk_size=4096,
    client_profile_requests=0,
    quality_probe=0,
    fixed_output_tokens=0,
    steady_concurrency="512",
    steady_warmup_seconds=240,
    steady_window_seconds=300,
    steady_windows=2,
    startup_timeout=2400,
    test_timeout_seconds=21600,
    perf_repeats=0,
    graph_edge_cases=0,
    throughput_sweep=0,
    capacity_batches="",
    capacity_repeats=2,
    capacity_refine=0,
)
ENV = {
    "RTP_STEP_MEASUREMENT": "1",
    "RTP_FRONTEND_MEASUREMENT": "1",
    "DG_MEGA_MOE_FP8_GRID_SYNC_DIAG": "0",
    "MM_RDMA_READ_TIMEOUT_MS": "30000",
    # Added upstream after the reference run; keep expensive input dumps off.
    "MEGA_MOE_LOG_INPUTS": "0",
}
# Published epoch-fixed wheels, selected through the normal Bazel dependencies.
# Check the resolved worker package: gpu_lock stages packages without dist-info.
DEEP_GEMM_VERSION = "2.8.0+58a6d07.cu132"
DEEP_GEMM_SHA256 = {
    "include/deep_gemm/impls/sm100_fp8_fp8_mega_moe.cuh": "d16ae49f102a0fe382ee235600ad59e2201e4dbe40094fb34fc6b6c67a7fff3f",
}
DEEP_GEMM_BINARY_SHA256 = {
    "x86_64": "d5a151a88c046a154290292a482e1d2905e126f5b83e488aa65157ea71565a31",
    "aarch64": "c3b0d44b36f4c8590a05c90ed2a1a9e9738d5a7f23b89766d0bba2f708a0ffb7",
}
SNAPSHOT_FILES = (
    "rtp_llm/test/smoke/qwen35_e1pd4_perf_smoke.py",
    "rtp_llm/test/smoke/qwen35_e1pd4_perf_smoke.bzl",
    "rtp_llm/test/smoke/qwen35_e1pd4_perf_smoke_test.py",
    "rtp_llm/test/smoke/qwen35_e1pd4_perf_reference.json",
)


def add_arguments(parser):
    root = Path(__file__).resolve().parents[4]
    parser.description = __doc__
    parser.set_defaults(
        **DEFAULTS,
        gpus="4,5,6,7",
        encoder_gpus="0",
        trace_run_id="e1pd4-" + uuid.uuid4().hex,
    )
    parser.add_argument(
        "--dg-jit-cache",
        default=str(root / "build_logs/epd-concurrency-bottleneck-20261005/jit-cache"),
        help="private writable DeepGEMM JIT cache (reference cache by default)",
    )
    parser.add_argument(
        "--pycparser-path", default="/root/.cache/epd-fusion-deps/pycparser"
    )


def configure(a, parser):
    for name, value in DEFAULTS.items():
        if getattr(a, name) != value:
            parser.error(
                f"the performance preset fixes --{name.replace('_', '-')}={value}; "
                "use qwen35_epd_fusion_4gpu_smoke.py for tuning experiments"
            )
    if len(smoke.encoder_gpu_ids(a.encoder_gpus)) != 1:
        parser.error("the performance preset requires exactly one Encoder GPU")
    if a.bazel_option:
        parser.error("--bazel-option is disabled in the frozen performance preset")
    a.bazel_target = TARGET
    a.snapshot_files = SNAPSHOT_FILES
    a.extra_test_args = []
    for name in ("dg_jit_cache", "pycparser_path"):
        value = str(Path(getattr(a, name)).expanduser().resolve())
        setattr(a, name, value)
        a.extra_test_args.append(f"--{name.replace('_', '-')}={value}")
    env = dict(ENV, DG_JIT_CACHE_DIR=a.dg_jit_cache)
    a.bazel_option = [
        f"--override_repository=pip_gpu_cuda13_torch_pycparser={a.pycparser_path}",
        "--experimental_remote_downloader=",
        "--jobs=32",
    ] + [f"--test_env={key}={value}" for key, value in env.items()]
    a.preset_metadata = dict(
        reference_run=REFERENCE_RUN,
        settings=DEFAULTS,
        environment=env,
        deep_gemm_version=DEEP_GEMM_VERSION,
        deep_gemm_sha256=DEEP_GEMM_SHA256,
        deep_gemm_binary_sha256=DEEP_GEMM_BINARY_SHA256,
        trace_run_id=a.trace_run_id,
        encoder_gpus=a.encoder_gpus,
        pd_gpus=a.gpus,
    )


def validate_runtime(a):
    if not Path(a.dg_jit_cache).is_dir() or not os.access(a.dg_jit_cache, os.W_OK):
        raise ValueError(f"private writable JIT cache required: {a.dg_jit_cache}")
    if not Path(a.pycparser_path).is_dir():
        raise ValueError(f"pycparser Bazel repository missing: {a.pycparser_path}")
    # The launcher need not have DeepGEMM installed; Bazel resolves the wheel.
    if not a.worker:
        return
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or not spec.origin:
        raise ValueError(f"DeepGEMM {DEEP_GEMM_VERSION} missing from worker runtime")
    package = Path(spec.origin).resolve().parent
    arch = platform.machine()
    if arch not in DEEP_GEMM_BINARY_SHA256:
        raise ValueError(f"unsupported DeepGEMM architecture: {arch}")
    hashes = dict(DEEP_GEMM_SHA256)
    hashes[f"_C.cpython-310-{arch}-linux-gnu.so"] = DEEP_GEMM_BINARY_SHA256[arch]
    for relative, expected in hashes.items():
        path = package / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"DeepGEMM differs from {DEEP_GEMM_VERSION}: {path}")
    a.preset_metadata["deep_gemm_package"] = str(package)
    os.environ.update(a.preset_metadata["environment"])


def main(argv=None):
    return smoke.main(argv, preset=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark prefill attention implementations with one forward per CUDA Graph.

Usage:
    bazelisk test --config=cuda12_9 --config=sm9x \
        --run_under=//rtp_llm/test/utils:gpu_lock \
        //rtp_llm/models_py/modules/factory/attention/cuda_impl/test:bench_prefill_impls \
        --test_timeout=600 --test_output=streamed --nocache_test_results

Result statuses: PASS / MISMATCH (numeric tolerance exceeded) / FAIL (raised) /
SKIP (support() rejected) / PROFILED (profile mode, no timing stats).
An impl that SKIPs every scheduled run fails the run.
Exit codes: 0 ok, 1 FAIL / unmeasured impl / harness error, 2 MISMATCH only.
--worker-timeout (BENCH_WORKER_TIMEOUT) bounds time without writing a result,
not total worker lifetime.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import traceback

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench.core import (
    PRESETS,
    BenchOptions,
    BenchReport,
    CasePlanner,
    log,
    setup_exec_context,
    short_error,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench.prefill_benches import (
    IMPL_BENCHES,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench.scheduler import (
    WORKER_EXIT_BENCH_FAILURE,
    WORKER_EXIT_CRASHED,
    WORKER_EXIT_MISMATCH,
    WORKER_EXIT_OK,
    Coordinator,
    WorkerLifecycle,
    WorkerRuntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-dtype", default="bf16", help="base QKV dtypes: bf16, fp16")
    parser.add_argument(
        "--kv-cache-dtype", default="bf16,fp8", help="KV dtypes: bf16, fp16, fp8"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--input-len", default="4096,8192,16384", help="total sequence lengths"
    )
    parser.add_argument("--reuse-cache-ratio", default="0,0.3,0.7,0.9")
    parser.add_argument("--preset", default="qwen3-8b", choices=(*PRESETS, "custom"))
    parser.add_argument("--head-num", type=int)
    parser.add_argument("--kv-head-num", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=15, help="measured graph replays")
    parser.add_argument(
        "--run-impls", default="all", help="implementation name filters"
    )
    parser.add_argument(
        "--no-check-correctness", action="store_false", dest="check_correctness"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="/tmp/bench_traces")
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--enable-csv-dump", action="store_true")
    parser.add_argument("--csv-output-path", default="/tmp/bench_prefill_impls.csv")
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=int(os.environ.get("BENCH_WORKER_TIMEOUT", "600")),
        help=(
            "max seconds a worker may go without writing a result; "
            "also settable via BENCH_WORKER_TIMEOUT"
        ),
    )
    parser.add_argument("--_worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--_results-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_counter-file", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_parent-watch-fd", type=int, default=-1, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    try:
        options = BenchOptions.from_args(args)
    except ValueError as error:
        log(f"argument error: {error}")
        return 1

    planner = CasePlanner(options, IMPL_BENCHES)
    planned = planner.plan()
    if args._worker_id < 0:
        log(
            f"Generated {planner.generated_case_count} cases; scheduled "
            f"{planner.scheduled_run_count} implementation runs; "
            f"filtered {planner.rejected_run_count} implementation runs"
        )
        return Coordinator(options, planned).run()
    WorkerLifecycle.install_parent_guard(args._parent_watch_fd)
    if not torch.cuda.is_available():
        log("CUDA is not available")
        return 1

    try:
        setup_exec_context()
    except Exception as error:
        log(f"warning: init_exec_ctx failed: {short_error(error)}")
        log(traceback.format_exc())

    log(f"started with {len(planned)} planned cases")
    worker_failed = False
    try:
        results = WorkerRuntime(options, planned, IMPL_BENCHES).run(
            args._counter_file, args._results_file
        )
    except Exception as error:
        log(f"worker failed: {short_error(error)}")
        log(traceback.format_exc())
        worker_failed = True

    if worker_failed:
        return WORKER_EXIT_CRASHED
    # Completed-with-failures workers use dedicated codes so the coordinator
    # does not treat them as crashes and kill their peers.
    if any(result.status == "FAIL" for result in results):
        return WORKER_EXIT_BENCH_FAILURE
    if any(result.status == "MISMATCH" for result in results):
        return WORKER_EXIT_MISMATCH
    skipped = BenchReport.fully_skipped_impls(results)
    if skipped:
        log(f"worker measured nothing for: {', '.join(skipped)}")
    return WORKER_EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

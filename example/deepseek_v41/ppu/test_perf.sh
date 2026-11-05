#!/usr/bin/env bash
# Standard perf framework over a separately built and verified native runtime.
set -euo pipefail
: "${RTP_V41_TEST_RUNTIME:?checked native runtime required}"
: "${RTP_V41_TEST_CHECKPOINT:?checkpoint required}"
: "${TEST_UNDECLARED_OUTPUTS_DIR:?Bazel outputs required}"
v41_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
cd "$RTP_V41_TEST_RUNTIME"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [[ "${1:-}" == contracts ]]; then
    exec "$v41_python" -B -m unittest rtp_llm.test.perf_test.test_perf_config_and_runners
fi
if [[ "${1:-}" == graph ]]; then
    export RTP_V41_TEST_DECODE_GRAPH=1
    export RTP_V41_PERF_BATCHES=${RTP_V41_PERF_BATCHES:-1,8,16}
    export RTP_V41_PERF_STEPS=${RTP_V41_PERF_STEPS:-8}
    export RTP_V41_PERF_MEASURE_RUNS=${RTP_V41_PERF_MEASURE_RUNS:-1}
fi
export MAGA_SERVER_WORK_DIR="$TEST_UNDECLARED_OUTPUTS_DIR"
export PERF_GLOBAL_BATCH=1 PERF_DETAILED_LATENCIES=1 PERF_NATIVE_HTTP=1
export PERF_GRID_WARMUP_RUNS=1
export RTP_V41_PERF_EVIDENCE_DIR="$TEST_UNDECLARED_OUTPUTS_DIR/rank-evidence"
export RTP_PYTHON_BIN="$v41_python"
export RTP_TEST_GRACEFUL_STOP_TIMEOUT=180
unset RTP_V41_PPU_MOE_PRECISION
export RTP_V41_PPU_MOE_BACKEND=megamoe
"$v41_python" -B - "$RTP_V41_TEST_CHECKPOINT" "$TEST_UNDECLARED_OUTPUTS_DIR" <<'PY'
import os
import sys
import json
from pathlib import Path
from rtp_llm.tools.v41.launch import preset_env
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill import _load_token_map

checkpoint, output = sys.argv[1:]
os.environ.update(preset_env("decode", checkpoint))
graph = os.environ.get("RTP_V41_TEST_DECODE_GRAPH") == "1"
if graph:
    os.environ["ENABLE_CUDA_GRAPH"] = "1"
token_map = str(Path(output) / "token-map.json")
_load_token_map(str(Path(checkpoint) / "tokenizer.json"), token_map)
os.environ["RTP_V41_TOKEN_MAP_CACHE"] = token_map
# Existing BatchDecodeScheduler allocates synthetic KV and excludes prefill.
os.environ["PERF_TEST"] = "1"
os.environ["DISABLE_COMPILE_PRINT"] = "1"
os.environ["FAKE_BALANCE_EXPERT"] = "0"
from rtp_llm.test.perf_test.batch_decode_test import main

sys.argv = ["batch_decode_test", "--model_type", "deepseek_v41",
            "--checkpoint_path", checkpoint, "--tokenizer_path", checkpoint,
            # input_len includes the token consumed by the first decode.
            # An even cached prefix has no missing half-pair in synthetic KV.
            "--input_len", os.environ.get("RTP_V41_PERF_INPUT_LEN", "2049"),
            "--batch_size",
            os.environ.get("RTP_V41_PERF_BATCHES", "1,8,16,32,64,128,256"),
            "--dp_size", "8", "--partial", "1", "--decode_test_length",
            os.environ.get("RTP_V41_PERF_STEPS", "129"),
            "--num_measures", os.environ.get("RTP_V41_PERF_MEASURE_RUNS", "2"),
            "--warmup_runs", "1", "--profile_runs",
            os.environ.get("RTP_V41_PERF_PROFILE_RUNS", "0"),
            "--result_dir", output, "--max_seq_len", "2304",
            "--fake_balance_expert", "0", "--warm_up", "0", "--model_warm_up", "0",
            "--frontend_pre_stop_drain_seconds", "0",
            "--dash_sc_grpc_pre_stop_drain_seconds", "0",
            "--backend_post_frontend_drain_seconds", "0"]
if graph:
    sys.argv.extend(["--enable_cuda_graph", "1", "--decode_capture_config",
                     os.environ.get("RTP_V41_GRAPH_CAPTURE_BATCHES", "1,2,4,8")])
main()
if graph:
    evidence = Path(os.environ["RTP_V41_PERF_EVIDENCE_DIR"])
    reports = [json.loads((evidence / f"moe-rank{rank}.json").read_text())
               for rank in range(8)]
    for rank, report in enumerate(reports):
        assert report["rank"] == rank and report["world_size"] == 8, report
        assert report["model_layers"] == [40], report
        assert report["decode_graph_replays"] > 0, report
        assert report["engines"], report
        bound_engines = [engine for engine in report["engines"] if engine["layers"]]
        assert sum(engine["layers"] for engine in bound_engines) == 40, report
        for engine in report["engines"]:
            assert engine["backend"] == "megamoe" and engine["precision"] == "w4a4", engine
            assert engine["grouped_gemm_calls"] == 0, engine
        for engine in bound_engines:
            assert engine["mega_moe_calls"] > 0, engine
    assert sum(engine["mega_received_assignments"] for report in reports
               for engine in report["engines"]) > 0, reports
    print("DP8 full 40-layer decode Graph replay counts:",
          [report["decode_graph_replays"] for report in reports])
PY

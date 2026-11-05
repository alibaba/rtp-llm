#!/usr/bin/env bash
# Run the existing eight-rank drivers against an explicitly published snapshot.
set -euo pipefail

v41_role=${1:?prefill, decode or moe-device}
: "${RTP_V41_TEST_SOURCE_ROOT:?published snapshot required}"
v41_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
v41_output=${TEST_UNDECLARED_OUTPUTS_DIR:?Bazel test output directory required}
v41_source=$(cd "$RTP_V41_TEST_SOURCE_ROOT" && pwd)
cd "$v41_source"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH="$v41_source:${PYTHONPATH:-}"

"$v41_python" -B -c 'from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import runtime_identity; print(runtime_identity())'
if [[ "$v41_role" == moe-device ]]; then
    export RTP_V41_TEST_W4A8_DEVICE=1
    exec "$v41_python" -B rtp_llm/models_py/modules/dsv41/ppu/test/test_moe_w4a8.py
fi
: "${RTP_V41_TEST_CHECKPOINT:?checkpoint directory required}"
v41_args=(
    --config-path "$RTP_V41_TEST_CHECKPOINT/config.json"
    --checkpoint-path "$RTP_V41_TEST_CHECKPOINT"
    --device cuda --nproc 8 --timeout 5400
)
if [[ "$v41_role" == prefill ]]; then
    v41_driver=driver_prefill_cp8.py
    v41_json="$v41_output/cp8.json"
    v41_args+=(--tokens "${RTP_V41_TEST_TOKENS:-129}" --ep-backend deepep)
    if [[ -n "${RTP_V41_TEST_PD_PEER:-}" ]]; then
        v41_args+=(--pd-peer "$RTP_V41_TEST_PD_PEER" --pd-run-id "${RTP_V41_TEST_PD_RUN_ID:?}"
            --fixture-dir "$v41_output/pd-fixtures")
    fi
else
    [[ "$v41_role" == decode ]]
    v41_driver=driver_decode_dp8.py
    v41_json="$v41_output/dp8.json"
    v41_args+=(--tokens "${RTP_V41_TEST_TOKENS:-33}" --steps 4 --scenario "${RTP_V41_TEST_SCENARIO:-one-busy}")
    if [[ -n "${RTP_V41_TEST_PD_RUN_ID:-}" ]]; then
        v41_args+=(--pd-receive --pd-run-id "$RTP_V41_TEST_PD_RUN_ID")
    fi
fi
"$v41_python" -B "rtp_llm/models_py/modules/dsv41/ppu/tools/$v41_driver" \
    "${v41_args[@]}" --output "$v41_json"
"$v41_python" - "$v41_json" <<'PY'
import json
import os
import sys

with open(sys.argv[1]) as reader:
    result = json.load(reader)
summary = result.get("summary", result)
precision = os.environ.get("RTP_V41_PPU_MOE_PRECISION", "w4a8").strip().lower()
assert summary["ranks_reported"] == 8, summary
assert summary["moe_precisions"] == [precision], summary
assert summary["moe_grouped_gemm_calls"] > 0, summary
if "all_ranks_ok" in summary:
    assert summary["all_ranks_ok"], summary
else:
    assert summary["all_finite"] and summary["checks_all_pass"], summary
print("PPU functional check:", precision, summary["moe_grouped_gemm_calls"], "grouped GEMMs")
PY

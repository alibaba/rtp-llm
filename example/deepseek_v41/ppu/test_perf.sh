#!/usr/bin/env bash
# Use a checked native runtime for device runs; help/dry-run also work on CPU.
set -euo pipefail
v41_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
v41_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${RTP_V41_TEST_RUNTIME:-$v41_source}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [[ "${1:-}" == contracts ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.test.perf_test.test_perf_config_and_runners \
        rtp_llm.test.perf_test.test_dsv41_ppu_perf \
        rtp_llm.test.perf_test.test_dsv41_ppu_reference
fi
if [[ "${1:-}" == graph-attention-ab ]]; then
    shift
    v41_result=${RTP_V41_PERF_AB_OUTPUT_DIR:-${TEST_UNDECLARED_OUTPUTS_DIR:?}}
    v41_common=(decode --decode-graph --greedy --no-fake-balance-expert "$@")
    "$v41_python" -B -m rtp_llm.test.perf_test.dsv41_ppu_perf "${v41_common[@]}" \
        --no-graph-packed-attention --no-graph-visible-indexer --result-dir "$v41_result/off"
    exec "$v41_python" -B -m rtp_llm.test.perf_test.dsv41_ppu_perf "${v41_common[@]}" \
        --graph-packed-attention --graph-visible-indexer --result-dir "$v41_result/on"
fi
for v41_arg in "$@"; do
    case "$v41_arg" in
        --protocol|--protocol=*)
            exec "$v41_python" -B -m rtp_llm.test.perf_test.dsv41_ppu_reference "$@"
            ;;
    esac
done
exec "$v41_python" -B -m rtp_llm.test.perf_test.dsv41_ppu_perf "$@"

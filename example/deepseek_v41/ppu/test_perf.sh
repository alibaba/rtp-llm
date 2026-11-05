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
        rtp_llm.test.perf_test.test_dsv41_ppu_perf
fi
exec "$v41_python" -B -m rtp_llm.test.perf_test.dsv41_ppu_perf "$@"

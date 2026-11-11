#!/usr/bin/env bash
set -euo pipefail
cd "${RTP_V41_TEST_RUNTIME:?checked runtime required}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec "${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}" -B \
    -m rtp_llm.test.perf_test.dsv41_ppu_reference_local "$@"

#!/usr/bin/env bash
set -euo pipefail
: "${RTP_V41_TEST_RUNTIME:?checked PPU runtime required}"
runtime=$(cd "$RTP_V41_TEST_RUNTIME" && pwd)
cd "$runtime"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$runtime:${PYTHONPATH:-}"
exec "${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}" -B \
    rtp_llm/models_py/modules/dsv41/ppu/test/test_graph_cache.py "$@"

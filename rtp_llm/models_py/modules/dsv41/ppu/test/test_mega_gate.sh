#!/usr/bin/env bash
# Exercise the real Gate caller with its checked native-runtime import closure.
set -euo pipefail
: "${RTP_V41_TEST_RUNTIME:?checked native runtime required}"
v41_source=$(cd "$RTP_V41_TEST_RUNTIME" && pwd)
test -f "$v41_source/native-runtime.json"
cd "$v41_source"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$v41_source:${PYTHONPATH:-}"
exec "${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}" -B \
    rtp_llm/models_py/modules/dsv41/ppu/test/test_mega_gate.py "$@"

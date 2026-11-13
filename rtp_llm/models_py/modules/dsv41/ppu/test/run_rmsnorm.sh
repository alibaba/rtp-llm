#!/usr/bin/env bash
set -euo pipefail
task_mode=${1:?contracts or device}
: "${RTP_V41_TEST_SOURCE_ROOT:?immutable PPU runtime source required}"
task_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RTP_V41_TEST_SOURCE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
case "$task_mode" in
    contracts) export RTP_V41_TEST_RMSNORM_DEVICE=0 ;;
    device) export RTP_V41_TEST_RMSNORM_DEVICE=1 ;;
    *) exit 2 ;;
esac
cd "$RTP_V41_TEST_SOURCE_ROOT"
exec "$task_python" -B -m unittest -v \
    rtp_llm.models_py.modules.dsv41.ppu.test.test_rmsnorm

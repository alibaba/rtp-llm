#!/usr/bin/env bash
set -euo pipefail
: "${RTP_V41_TEST_SOURCE_ROOT:?immutable source or checked runtime required}"
cd "$RTP_V41_TEST_SOURCE_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fake_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
if [[ "${1:?contracts or device}" == contracts ]]; then
    exec "$fake_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.fake_balance.test_fake_balance \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_moe_megamoe \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_moe_w4a8 \
        rtp_llm.tools.v41.test_integration \
        rtp_llm.test.perf_test.test_dsv41_ppu_perf
fi
[[ "$1" == device ]]
exec "$fake_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.fake_balance.device \
    --output "${TEST_UNDECLARED_OUTPUTS_DIR:?}/balance"

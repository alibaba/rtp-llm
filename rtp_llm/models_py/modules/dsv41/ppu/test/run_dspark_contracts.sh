#!/usr/bin/env bash
set -euo pipefail
: "${RTP_V41_TEST_SOURCE_ROOT:?immutable published source root required}"
cd "$RTP_V41_TEST_SOURCE_ROOT"
export PYTHONPATH="$RTP_V41_TEST_SOURCE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
exec /opt/conda312/bin/python -B -m unittest \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence \
  rtp_llm.models_py.speculative.test.test_dspark_proposer \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_cache \
  rtp_llm.tools.v41.test_integration \
  rtp_llm.test.perf_test.test_dsv41_ppu_perf

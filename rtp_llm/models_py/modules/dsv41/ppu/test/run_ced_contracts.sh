#!/usr/bin/env bash
set -euo pipefail
: "${RTP_V41_TEST_SOURCE_ROOT:?published source required}"
cd "$RTP_V41_TEST_SOURCE_ROOT"
export PYTHONPATH="$RTP_V41_TEST_SOURCE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
exec /opt/conda312/bin/python -B -m unittest \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_prefill_runtime.SharedPrefillDevice.test_local_encoder_replay_has_no_padding \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache \
  rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_cache \
  rtp_llm.tools.v41.test_integration

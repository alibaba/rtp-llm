#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export MIMO_TEST_TARGET="//rtp_llm/test/model_test:benchmark_mimo_v25_gsm8k"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/renkun.ren/models/MiMo-V2.5}"
export GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-/home/renkun.ren/dataset/gsm8k_test.jsonl}"
export GSM8K_NUM_EXAMPLES="${GSM8K_NUM_EXAMPLES:-200}"
export GSM8K_NUM_THREADS="${GSM8K_NUM_THREADS:-8}"
export GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-4096}"
export GSM8K_MAX_SEQ_LEN="${GSM8K_MAX_SEQ_LEN:-8192}"
export MIMO_BENCHMARK_LOG_DIR="${MIMO_BENCHMARK_LOG_DIR:-/home/renkun.ren/log/mimov25}"

exec "$SCRIPT_DIR/run_mimo_v25_test.sh"

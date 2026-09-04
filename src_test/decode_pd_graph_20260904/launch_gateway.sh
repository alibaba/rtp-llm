#!/usr/bin/env bash
set -euo pipefail

# A no-GPU frontend gateway is needed to exercise the actual PD-fusion route:
# the DECODE backend does not implement ordinary GenerateStreamCall itself.
MODEL_DIR="${MODEL_DIR:-/tmp/models/DeepSeek-V4-Flash-0731}"
GATEWAY_PORT="${GATEWAY_PORT:-18730}"
DECODE_PORT="${DECODE_PORT:-18530}"
PREFILL_PORT="${PREFILL_PORT:-18630}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export SCR_ENABLE=0
export RTPLLM_ENABLE_SCR=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/opt/conda310/lib/python3.10/site-packages:/home/serina.wzq/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_SERVICE_CONFIG="{\"service_id\":\"decode-pd-graph-test\",\"role_endpoints\":[{\"group\":\"default\",\"prefill_endpoint\":{\"type\":\"Vipserver\",\"address\":\"127.0.0.1:${PREFILL_PORT}\",\"protocol\":\"http\",\"path\":\"/\"},\"decode_endpoint\":{\"type\":\"Vipserver\",\"address\":\"127.0.0.1:${DECODE_PORT}\",\"protocol\":\"http\",\"path\":\"/\"}}],\"use_local\":true}"

exec /opt/conda310/bin/python3 -m rtp_llm.start_server \
  --model_type deepseek_v4 \
  --checkpoint_path "$MODEL_DIR" \
  --tokenizer_path "$MODEL_DIR" \
  --role_type FRONTEND \
  --start_port "$GATEWAY_PORT" \
  --frontend_server_count 1 \
  --grammar_backend none

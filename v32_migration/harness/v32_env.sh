#!/usr/bin/env bash
# V3.2 A/B — 公共环境（source 此文件）
export RTP_MODEL_PATH=/home/admin/models/DeepSeek-V3.2-Exp
export RTP_MODEL_TYPE=deepseek_v32
export RTP_PREFILL_TP=2
export RTP_PREFILL_DP=4
export RTP_DECODE_TP=1
export RTP_DECODE_DP=8
export RTP_DECODE_KV_MB=12288
export RTP_PREFILL_KV_MB=16384
export RTP_MAX_SEQ_LEN=65536
export MODEL_TEMPLATE_TYPE=deepseek_v31
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATASET=/tmp/rym/dataset/trace-0715-mixed5k
export RTP_FLEXLB_ENABLE_QUEUEING=1
export RTP_SSH_KNOWN_HOSTS=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-04_rtp-llm-pd-resource-profile/operator_scripts/known_hosts
export REMOTE_JAR_OVERRIDE=/home/admin/rtp-hol/flexlb/flexlb-api-spillover-ef8e77fe.jar
export JAR_SHA256_OVERRIDE=ef8e77fe35d24d9790f2c8d964f71d0e46f8f5049375850bb6e46c0e28f07149
unset RTP_DECODE_BUCKET_SPEC RTP_DECODE_BUCKET_SEQLEN RTP_FLEXLB_QUEUE_SRPT RTP_FLEXLB_SPILLOVER
unset RTP_DECODE_LONG_OUT_THRESHOLD RTP_DECODE_LONG_SLOT_QUOTA RTP_HOST_FILTER RTP_HOST_GPU_LIMIT
export RTP_START_TIMEOUT=3600
export RTP_WORKER_EXTRA_ENV="MODEL_TEMPLATE_TYPE=deepseek_v31,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
export RTP_CACHE_TRANSPORT_TCP=1
export RTP_DECODE_REUSE_CACHE=0

#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the Flash checkpoint directory}"

export DSV4_HC_IMPL=hybrid
export DSV4_MHC_PRE_GEMM_BACKEND=deepgemm_deterministic
export DSV4_MHC_POST_BACKEND=tilelang
export DSV4_MHC_POST_PDL=0
export DSV4_PPU_SGLANG_MOE=1
export DSV4_PPU_SGLANG_WO_A=1
export DSV4_MOE_SHARED_EXPERT_OVERLAP=0
export DSV4_PPU_TP_COMM_WARMUP=1
export DSV4_PPU_SHARED_QKV_QUANT=0
export DSV4_MOE_SCALE_GATHER_FUSED=1

exec python3 -m rtp_llm.start_server \
  --checkpoint_path "$CHECKPOINT_PATH" --tokenizer_path "$CHECKPOINT_PATH" \
  --start_port "${START_PORT:-8088}" \
  --model_type deepseek_v4 \
  --act_type BF16 \
  --role_type PDFUSION \
  --tp_size 4 \
  --ep_size 1 \
  --dp_size 1 \
  --world_size 4 \
  --local_world_size 4 \
  --prefill_cp_size 1 \
  --prefill_cp_kv_cache_sharded 0 \
  --load_method scratch \
  --force_cpu_load_weights false \
  --moe_pure_tp_preshard true \
  --use_all_gather 1 \
  --use_deepep_moe 0 \
  --use_deepep_internode 0 \
  --use_deepep_low_latency 0 \
  --fp8_kv_cache 1 \
  --max_seq_len 16384 \
  --max_batch_tokens_size 8192 \
  --concurrency_limit 4 \
  --max_context_batch_size 4 \
  --kv_cache_mem_mb 4096 \
  --reserver_runtime_mem_mb 32768 \
  --warm_up 0 \
  --enable_cuda_graph 0 \
  --frontend_server_count 1 \
  --reuse_cache 0 \
  --manage_jit_cache 0 \
  --frontend_pre_stop_drain_seconds 0 \
  --dash_sc_grpc_pre_stop_drain_seconds 0 \
  --module_dispatch '{"impl_overrides": {"rtp.dsv4.attention": "ppu.dsv4.attention.fp4_indexer.v1", "rtp.dsv4.block": "ppu.dsv4.block.fp4_indexer.v1", "rtp.dsv4.model": "ppu.dsv4.model.fp4_indexer.v1", "rtp.dsv4.moe": "ppu.dsv4.moe.fp4_indexer.v1"}, "mode": "auto", "platform": "ppu"}'

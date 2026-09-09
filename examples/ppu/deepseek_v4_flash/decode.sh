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
export DSV4_PPU_TP_COMM_WARMUP=0
export DSV4_PPU_DECODE_HC_REDUCTION=fused
export DSV4_PPU_DECODE_HC_NORM=fused
export DSV4_PPU_DECODE_FP8_QUANT=v2
export DSV4_PPU_DECODE_QKV=merged
export DSV4_PPU_DECODE_INDEXER=overlap
export DSV4_PPU_DECODE_ATTN_MODE=overlap
export DSV4_SHARED_EXPERT_MODE=overlap
export DSV4_PPU_DECODE_SHARED_SCHEDULE=before_route
export DSV4_PPU_DECODE_MOE_OUTPUT=bf16
export DSV4_PPU_DECODE_METADATA=graph_fused
export DSV4_PPU_DECODE_ROPE=shared
export DSV4_PPU_DECODE_MOE_HINT=capacity

exec python3 -m rtp_llm.start_server \
  --checkpoint_path "$CHECKPOINT_PATH" --tokenizer_path "$CHECKPOINT_PATH" \
  --start_port "${START_PORT:-8088}" \
  --model_type deepseek_v4 \
  --act_type BF16 \
  --role_type DECODE \
  --tp_size 1 \
  --dp_size 8 \
  --ep_size 8 \
  --world_size 8 \
  --local_world_size 8 \
  --prefill_cp_size 1 \
  --prefill_cp_kv_cache_sharded 0 \
  --load_method scratch \
  --force_cpu_load_weights false \
  --moe_pure_tp_preshard false \
  --use_all_gather 0 \
  --use_deepep_moe 1 \
  --use_deepep_internode 0 \
  --use_deepep_low_latency 1 \
  --fp8_kv_cache 1 \
  --seq_size_per_block 256 \
  --kernel_seq_size_per_block 256 \
  --max_seq_len 16384 \
  --concurrency_limit 128 \
  --max_context_batch_size 8 \
  --kv_cache_mem_mb 8192 \
  --reserver_runtime_mem_mb 16384 \
  --warm_up 1 \
  --enable_cuda_graph 1 \
  --decode_capture_config 1,2,4,8,16,32,64,128 \
  --reuse_cache 0 \
  --manage_jit_cache 0 \
  --frontend_server_count 1 \
  --cache_store_rdma_mode 0 \
  --use_batch_decode_scheduler 1 \
  --batch_decode_scheduler_batch_size 1 \
  --module_dispatch '{"mode": "auto", "platform": "ppu", "impl_overrides": {"rtp.dsv4.model": "ppu.dsv4.model.fp4_decode.v1", "rtp.dsv4.block": "ppu.dsv4.block.fp4_decode.v1", "rtp.dsv4.attention": "ppu.dsv4.attention.fp4_decode.v1", "rtp.dsv4.moe": "ppu.dsv4.moe.fp4_decode.v1"}}'

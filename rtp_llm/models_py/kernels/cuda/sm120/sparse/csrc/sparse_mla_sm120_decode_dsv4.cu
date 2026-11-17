// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define CUDA_CHECK_BOOL(call)                                               \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return false;                                                         \
    }                                                                       \
  } while (0)

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
static bool launch_decode_dsv4_impl(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                    bf16* mid_out, float* mid_lse, const int* topk_length,
                                    bf16* output, float* out_lse, const float* attn_sink,
                                    const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                    const int* extra_topk_length, int extra_topk, int pbs_extra,
                                    size_t stride_extra_kv_block, int num_tokens, int num_splits,
                                    int chunks_per_block_override, float sm_scale,
                                    size_t stride_kv_block, cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  // Ceiling div so NUM_HEADS < HPB (small-TP configs, e.g. h=8) still get a
  // tile. The kernel internally clamps Q load + mid_out writes to
  // VALID_HPB = min(NUM_HEADS, HPB).
  constexpr int H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;

  // Stage 1: decode-dsv4 (A1.2) partial-output kernel.
  // Dynamic smem layout (FP8 XV, DSV4, DSV4_BI=64, double-buffered KV):
  //   sm_q_rope    HPB * D_ROPE * 2B                      =  2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE                    = 7.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B                  = 0.44 KB
  //   sm_kv_fp8    2 * DSV4_BI * KV_SMEM_STRIDE             = 58 KB
  //   sm_kv_sc     2 * DSV4_BI * SCALE_BYTES_PER_TOKEN      =  1 KB
  //   sm_kv_rope   2 * DSV4_BI * D_ROPE * 2B                = 16 KB
  //   sm_reduce    2 * DSV4_N_WARPS * HPB * 4               = 1 KB
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4                   = 448 B
  //   sm_w_fp8 ×2  2 * HPB * (DSV4_BI + 16)                 = 2.5 KB
  //   Total                                               ~ 88 KB
  // Static smem (kernel-side):
  //   sm_p_full    HPB * DSV4_BI * 2B (bf16)                =  2 KB
  // Grand total ~ 89 KB (under 100 KB SM120 carveout, 1 block/SM).
  constexpr int N_V_CHUNKS_LAUNCH = KV::D_NOPE / KV::QUANT_TILE;  // 7
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_ROPE * (int)sizeof(bf16)                            // sm_q_rope
      + HPB * KV::Q_NOPE_STRIDE                                       // sm_q_fp8
      + HPB * KV::NUM_SCALES * (int)sizeof(float)                     // sm_q_sc
      + DSV4_KV_BUF_COUNT * DSV4_BI * KV::KV_SMEM_STRIDE              // sm_kv_fp8 ×2
      + DSV4_KV_BUF_COUNT * DSV4_BI * KV::SCALE_BYTES_PER_TOKEN       // sm_kv_sc ×2
      + DSV4_KV_BUF_COUNT * DSV4_BI * KV::D_ROPE * (int)sizeof(bf16)  // sm_kv_rope ×2
      + 16                                                            // mbar align pad
      + 4 * (int)sizeof(uint64_t)                                     // mbar_full+empty
      + 2 * DSV4_N_WARPS * HPB * (int)sizeof(float)                   // sm_reduce
      + N_V_CHUNKS_LAUNCH * HPB * (int)sizeof(float)                  // sm_w_head_sc
      + 2 * HPB * (DSV4_BI + 16) + (RTP_FP8_Q_RESIDUAL ? HPB * KV::Q_NOPE_STRIDE : 0);

  auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK_BOOL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  // Estimate serial chunk work per wave, including one chunk-equivalent
  // of Q preparation. A wave-count cap alone can choose long serial groups
  // even when more, shorter waves finish earlier.
  // AutoTuner can override per-shape via chunks_per_block_override.
  int chunks_per_block;
  int parallel_chunks = 0;
  if (chunks_per_block_override >= 1 && chunks_per_block_override <= num_splits) {
    // AutoTuner / caller override path — used by SparseMlaDecodeRunner to
    // sweep cpb tactics and pick per-shape best.
    chunks_per_block = chunks_per_block_override;
  } else {
    int sm_count = 0;
    int device = 0;
    CUDA_CHECK_BOOL(cudaGetDevice(&device));
    CUDA_CHECK_BOOL(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    const int per_token_head = num_tokens * H_BLOCKS;
    parallel_chunks = per_token_head > 0 ? sm_count / per_token_head : 0;
    chunks_per_block = 1;
    int best_cost = INT_MAX;
    for (int cpb = 1; cpb <= num_splits; ++cpb) {
      const int eff = (num_splits + cpb - 1) / cpb;
      const int active = per_token_head * eff;
      const int ceil_w = (active + sm_count - 1) / sm_count;
      const int cost = ceil_w * (cpb + 1);
      if (cost < best_cost || (cost == best_cost && cpb > chunks_per_block)) {
        best_cost = cost;
        chunks_per_block = cpb;
      }
    }
  }
  // Group chunks for scheduling only. Keep one partial per chunk so changing
  // the batch size or tuning tactic cannot change the softmax reduction tree.
  const int scheduled_splits = (num_splits + chunks_per_block - 1) / chunks_per_block;
  dim3 grid1(num_tokens, H_BLOCKS, scheduled_splits);
  dim3 block1(DSV4_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(
      Q, KV_cache, indices, mid_out, mid_lse, topk_length, extra_KV_cache, extra_indices,
      extra_topk_length, extra_topk, pbs_extra, stride_extra_kv_block, num_tokens, num_splits,
      chunks_per_block, parallel_chunks, sm_scale, stride_kv_block);
  CUDA_CHECK_BOOL(cudaGetLastError());

  if (num_splits == 1 || num_splits == 2) {
    auto small_merge = num_splits == 1 ? sparse_mla_decode_dsv4_merge_small_kernel<NUM_HEADS, 1>
                                       : sparse_mla_decode_dsv4_merge_small_kernel<NUM_HEADS, 2>;
    small_merge<<<dim3(num_tokens, NUM_HEADS), 32, 0, stream>>>(mid_out, mid_lse, output, out_lse,
                                                                attn_sink, num_tokens);
    CUDA_CHECK_BOOL(cudaGetLastError());
    return true;
  }

  if (num_splits == 10) {
    // Two warps per head reduce value dependencies without shared staging.
    // Group two heads per CTA for the common SWA plus compressed window.
    sparse_mla_decode_dsv4_merge_small_kernel<NUM_HEADS, 10, 2, 2>
        <<<dim3(num_tokens, (NUM_HEADS + 1) / 2), 128, 0, stream>>>(mid_out, mid_lse, output,
                                                                    out_lse, attn_sink, num_tokens);
    CUDA_CHECK_BOOL(cudaGetLastError());
    return true;
  }

  // Stage 2: merge splits → final output + LSE.
  // Grid: (num_tokens, NUM_HEADS). One block (BLOCK_THREADS=64) covers the
  // full D_V=512 via uint4 vec loads (8 bf16/thread × 64 threads = 512).
  // For h=128/T=16 this is 2048 blocks vs the prior 8192 (4× fewer).
  constexpr int MERGE_BLOCK_THREADS = 64;
  constexpr int MERGE_DIMS_PER_THREAD = KV::D_V / MERGE_BLOCK_THREADS;
  auto merge_kernel =
      sparse_mla_decode_dsv4_merge_kernel<MT, NUM_HEADS, KV::D_V, MERGE_BLOCK_THREADS,
                                          MERGE_DIMS_PER_THREAD>;
  dim3 grid2(num_tokens, NUM_HEADS);
  dim3 block2(MERGE_BLOCK_THREADS);
  const size_t merge_smem_bytes = (size_t)num_splits * sizeof(float);
  merge_kernel<<<grid2, block2, merge_smem_bytes, stream>>>(mid_out, mid_lse, output, out_lse,
                                                            attn_sink, num_tokens, num_splits);
  CUDA_CHECK_BOOL(cudaGetLastError());
  return true;
}

// Public surface — explicit instantiation switch over the PR-body bench grid.
// DSV4 only, page_block_size=64 only. NUM_HEADS ∈ {8, 16, 32, 64, 128},
// TOPK ∈ {128, 512, 1024}.
bool launch_sparse_mla_decode_dsv4(ModelType mt, int num_heads, int topk, int page_block_size,
                                   int num_tokens, int num_splits, const bf16* Q,
                                   const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
                                   float* mid_lse, bf16* output, float* out_lse,
                                   const int* topk_length, const float* attn_sink,
                                   const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                   const int* extra_topk_length, int extra_topk, int pbs_extra,
                                   size_t stride_extra_kv_block, int chunks_per_block_override,
                                   float sm_scale, size_t stride_kv_block, cudaStream_t stream) {
  if (mt != ModelType::DSV4 || page_block_size != 64) return false;
  if (num_splits <= 0) return false;
#define DSV4_DISPATCH(H, K)                                                                 \
  if (num_heads == (H) && topk == (K)) {                                                    \
    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                          \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \
        stride_kv_block, stream);                                                           \
  }
  DSV4_DISPATCH(8, 128)
  DSV4_DISPATCH(8, 512)
  DSV4_DISPATCH(8, 1024)
  DSV4_DISPATCH(16, 128)
  DSV4_DISPATCH(16, 512)
  DSV4_DISPATCH(16, 1024)
  DSV4_DISPATCH(32, 128)
  DSV4_DISPATCH(32, 512)
  DSV4_DISPATCH(32, 1024)
  DSV4_DISPATCH(64, 128)
  DSV4_DISPATCH(64, 512)
  DSV4_DISPATCH(64, 1024)
  DSV4_DISPATCH(128, 128)
  DSV4_DISPATCH(128, 512)
  DSV4_DISPATCH(128, 1024)
#undef DSV4_DISPATCH
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120

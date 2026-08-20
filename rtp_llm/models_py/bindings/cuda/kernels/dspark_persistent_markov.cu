// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM's SM90 DSpARK persistent Markov kernel.

#include "rtp_llm/models_py/bindings/cuda/kernels/dspark_persistent_markov.h"

#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rtp_llm {

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kRank = 256;
constexpr int kRankStage = 32;
constexpr int kSteps = 7;
constexpr int kBatchTile = 16;
constexpr int kVocabTile = 128;
constexpr int kBatchSubtiles = kBatchTile / 16;
constexpr int kVocabWarps = kVocabTile / 16;
constexpr int kWarps = kBatchSubtiles * kVocabWarps;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxBatch = 32;
constexpr int kMaxBatchTiles = kMaxBatch / kBatchTile;
constexpr int kMaxCtasPerBatchTile = 256;

struct SharedStorage {
  __nv_bfloat16 state[kBatchTile][kRank];
  alignas(16) __nv_bfloat16
      weight_stage[2][kWarps][16 * kRankStage];
  float accumulator[kWarps][16 * 16];
  unsigned long long warp_best[kWarps][16];
  int selected_target[kBatchTile];
  int is_last_cta;
};

union Bfloat16x8 {
  uint4 packed;
  __nv_bfloat16 element[8];
};

union Float8 {
  struct {
    float4 first;
    float4 second;
  } packed;
  float element[8];
};

__device__ __forceinline__ bool take_candidate(float score, int token,
                                                float best_score,
                                                int best_token) {
  return score > best_score || (score == best_score && token < best_token);
}

__device__ __forceinline__ unsigned long long pack_partial(float score,
                                                            int token) {
  return (static_cast<unsigned long long>(__float_as_uint(score)) << 32) |
         static_cast<unsigned int>(token);
}

__device__ __forceinline__ void unpack_partial(unsigned long long packed,
                                                float& score, int& token) {
  score = __uint_as_float(static_cast<unsigned int>(packed >> 32));
  token = static_cast<int>(static_cast<unsigned int>(packed));
}

__device__ __forceinline__ float add_like_bf16(float base,
                                                float correction,
                                                float scale) {
  const __nv_bfloat16 correction_bf16 = __float2bfloat16_rn(correction);
  const __nv_bfloat16 scaled_bf16 = __float2bfloat16_rn(
      __bfloat162float(correction_bf16) * scale);
  return __bfloat162float(__float2bfloat16_rn(
      base + __bfloat162float(scaled_bf16)));
}

__device__ __forceinline__ float2 add_like_bf16_pair(
    __nv_bfloat162 base, float correction0, float correction1, float scale) {
  const __nv_bfloat162 correction_bf16 =
      __floats2bfloat162_rn(correction0, correction1);
  const float2 rounded_correction = __bfloat1622float2(correction_bf16);
  const __nv_bfloat162 scaled_bf16 = __floats2bfloat162_rn(
      rounded_correction.x * scale, rounded_correction.y * scale);
  return __bfloat1622float2(__hadd2(base, scaled_bf16));
}

__device__ __forceinline__ void copy_async_16(void* destination,
                                              const void* source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void commit_async_copies() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int PendingGroups>
__device__ __forceinline__ void wait_async_copies() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
}

__device__ __forceinline__ void prefetch_weight_stage(
    __nv_bfloat16* destination, const __nv_bfloat16* markov_w2,
    int warp_vocab_start, int rank_start, int lane_id) {
  constexpr int kElementsPerCopy = 16 / sizeof(__nv_bfloat16);
  constexpr int kCopiesPerWarpStage =
      16 * kRankStage / kElementsPerCopy;
#pragma unroll
  for (int copy = lane_id; copy < kCopiesPerWarpStage; copy += 32) {
    const int element = copy * kElementsPerCopy;
    const int token_row = element / kRankStage;
    const int rank_column = element % kRankStage;
    copy_async_16(
        destination + element,
        markov_w2 + (warp_vocab_start + token_row) * kRank + rank_start +
            rank_column);
  }
}

__global__ void __launch_bounds__(kThreads)
    dspark_persistent_markov_kernel(
        int32_t* output, const int* anchor,
        const __nv_bfloat16* base_logits, const int64_t* draft_to_target_id,
        const __nv_bfloat16* markov_w1, const __nv_bfloat16* markov_w2,
        __nv_bfloat16* current_state, float* partial_scores,
        int* partial_tokens, int* barrier_state, int batch_size,
        int anchor_stride, int vocab_size,
        int base_logits_batch_stride, int base_logits_step_stride, float scale,
        int batch_tiles, int ctas_per_batch_tile) {
  __shared__ SharedStorage shared;

  const int batch_tile = blockIdx.x % batch_tiles;
  const int cta_lane = blockIdx.x / batch_tiles;
  const int batch_start = batch_tile * kBatchTile;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int warp_batch_start = warp_id / kVocabWarps * 16;
  const int warp_vocab_index = warp_id % kVocabWarps;

#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    // The first state is an indexed W1 gather and is staged in each CTA.
    // Later rounds use the contiguous state produced by the prior round's
    // winning CTA directly from L2/global memory, avoiding six redundant
    // copies and CTA barriers per persistent launch.
    const bool use_staged_state = step == 0;
    if (use_staged_state) {
      constexpr int kStateVector = 16 / sizeof(__nv_bfloat16);
      constexpr int kStateVectorsPerRow = kRank / kStateVector;
      for (int index = threadIdx.x;
           index < kBatchTile * kStateVectorsPerRow;
           index += blockDim.x) {
        const int row = index / kStateVectorsPerRow;
        const int rank = (index % kStateVectorsPerRow) * kStateVector;
        const int batch = batch_start + row;
        if (batch < batch_size) {
          *reinterpret_cast<uint4*>(&shared.state[row][rank]) =
              *reinterpret_cast<const uint4*>(
                  markov_w1 + static_cast<int>(anchor[batch * anchor_stride]) * kRank + rank);
        } else {
          *reinterpret_cast<uint4*>(&shared.state[row][rank]) =
              make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();
    }
    const __nv_bfloat16* state_base =
        use_staged_state ? &shared.state[0][0]
                         : current_state + batch_start * kRank;

    float warp_best_score = -std::numeric_limits<float>::infinity();
    int warp_best_token = std::numeric_limits<int>::max();
    const int num_vocab_tiles =
        (vocab_size + kVocabTile - 1) / kVocabTile;

    for (int vocab_tile = cta_lane; vocab_tile < num_vocab_tiles;
         vocab_tile += ctas_per_batch_tile) {
      const int warp_vocab_start =
          vocab_tile * kVocabTile + warp_vocab_index * 16;
      // swapAB: W2 is the large, contiguous row-major A operand while the
      // small state tile is the shared-memory, column-major B operand.  The
      // accumulator is [vocab, batch] instead of [batch, vocab].
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          state_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          accumulator_fragment;
      wmma::fill_fragment(accumulator_fragment, 0.0f);

      prefetch_weight_stage(shared.weight_stage[0][warp_id], markov_w2,
                            warp_vocab_start, 0, lane_id);
      commit_async_copies();
#pragma unroll
      for (int rank_start = 0; rank_start < kRank;
           rank_start += kRankStage) {
        const int stage = (rank_start / kRankStage) % 2;
        if (rank_start + kRankStage < kRank) {
          prefetch_weight_stage(shared.weight_stage[stage ^ 1][warp_id],
                                markov_w2, warp_vocab_start,
                                rank_start + kRankStage, lane_id);
          commit_async_copies();
          wait_async_copies<1>();
        } else {
          wait_async_copies<0>();
        }
        __syncwarp();
#pragma unroll
        for (int rank_inner = 0; rank_inner < kRankStage;
             rank_inner += 16) {
          wmma::load_matrix_sync(
              weight_fragment,
              shared.weight_stage[stage][warp_id] + rank_inner, kRankStage);
          wmma::load_matrix_sync(
              state_fragment,
              state_base + warp_batch_start * kRank + rank_start + rank_inner,
              kRank);
          wmma::mma_sync(accumulator_fragment, weight_fragment,
                         state_fragment, accumulator_fragment);
        }
      }
      wmma::store_matrix_sync(shared.accumulator[warp_id],
                              accumulator_fragment, 16, wmma::mem_col_major);
      __syncwarp();

      {
        const int row_offset = lane_id & 15;
        const int row = warp_batch_start + row_offset;
        const int batch = batch_start + row;
        const int column_start = (lane_id >> 4) * 8;
        const int token_start = warp_vocab_start + column_start;
        float lane_best_score = -std::numeric_limits<float>::infinity();
        int lane_best_token = std::numeric_limits<int>::max();
        if (batch < batch_size && token_start + 8 <= vocab_size) {
          Bfloat16x8 bases;
          bases.packed = *reinterpret_cast<const uint4*>(
              base_logits + batch * base_logits_batch_stride +
              step * base_logits_step_stride + token_start);
          Float8 corrections;
          corrections.packed.first = *reinterpret_cast<const float4*>(
              shared.accumulator[warp_id] + row_offset * 16 + column_start);
          corrections.packed.second = *reinterpret_cast<const float4*>(
              shared.accumulator[warp_id] + row_offset * 16 + column_start + 4);
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const int column_in_half = pair * 2;
            const __nv_bfloat162 base_pair =
                *reinterpret_cast<const __nv_bfloat162*>(
                    bases.element + column_in_half);
            const float2 scores = add_like_bf16_pair(
                base_pair, corrections.element[column_in_half],
                corrections.element[column_in_half + 1], scale);
            const int token0 = token_start + column_in_half;
            if (take_candidate(scores.x, token0, lane_best_score,
                               lane_best_token)) {
              lane_best_score = scores.x;
              lane_best_token = token0;
            }
            const int token1 = token0 + 1;
            if (take_candidate(scores.y, token1, lane_best_score,
                               lane_best_token)) {
              lane_best_score = scores.y;
              lane_best_token = token1;
            }
          }
        } else if (batch < batch_size) {
#pragma unroll
          for (int column_in_half = 0; column_in_half < 8;
               ++column_in_half) {
            const int column = column_start + column_in_half;
            const int token = warp_vocab_start + column;
            if (token < vocab_size) {
              const float score = add_like_bf16(
                  __bfloat162float(base_logits[
                      batch * base_logits_batch_stride +
                      step * base_logits_step_stride + token]),
                  shared.accumulator[warp_id][row_offset * 16 + column],
                  scale);
              if (take_candidate(score, token, lane_best_score,
                                 lane_best_token)) {
                lane_best_score = score;
                lane_best_token = token;
              }
            }
          }
        }
        const float other_score =
            __shfl_xor_sync(0xffffffff, lane_best_score, 16);
        const int other_token =
            __shfl_xor_sync(0xffffffff, lane_best_token, 16);
        if (lane_id < 16) {
          if (take_candidate(other_score, other_token, lane_best_score,
                             lane_best_token)) {
            lane_best_score = other_score;
            lane_best_token = other_token;
          }
          if (take_candidate(lane_best_score, lane_best_token,
                             warp_best_score, warp_best_token)) {
            warp_best_score = lane_best_score;
            warp_best_token = lane_best_token;
          }
        }
      }
    }

    // A warp's token ranges are disjoint, so keep its running Top-1 in
    // registers and synchronize the CTA only once after all assigned tiles.
    if (lane_id < 16) {
      shared.warp_best[warp_id][lane_id] =
          pack_partial(warp_best_score, warp_best_token);
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 16) {
      const int row = warp_batch_start + lane_id;
      float cta_best_score = -std::numeric_limits<float>::infinity();
      int cta_best_token = std::numeric_limits<int>::max();
#pragma unroll
      for (int warp = 0; warp < kVocabWarps; ++warp) {
        float candidate_score;
        int candidate_token;
        unpack_partial(shared.warp_best[warp][lane_id], candidate_score,
                       candidate_token);
        if (take_candidate(candidate_score, candidate_token, cta_best_score,
                           cta_best_token)) {
          cta_best_score = candidate_score;
          cta_best_token = candidate_token;
        }
      }
      const int partial_index =
          (batch_tile * ctas_per_batch_tile + cta_lane) * kBatchTile + row;
      auto* packed_partials =
          reinterpret_cast<unsigned long long*>(partial_scores);
      packed_partials[partial_index] =
          pack_partial(cta_best_score, cta_best_token);
      // Every writer publishes its partial before CTA 0 can announce that
      // this block has arrived at the group barrier.
      __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int ticket = atomicAdd(&barrier_state[batch_tile], 1);
      shared.is_last_cta = ticket == ctas_per_batch_tile - 1;
    }
    __syncthreads();

    // The last CTA to arrive reduces the group and releases the next round.
    // Cooperative launch guarantees that every CTA in the software barrier
    // is resident, avoiding the classic spin-barrier scheduling deadlock.
    if (shared.is_last_cta) {
      constexpr int kReductionWays = 8;
      if (threadIdx.x < kBatchTile * kReductionWays) {
        // Each half warp owns one CTA stride and keeps row loads contiguous.
        const int reduction_lane = threadIdx.x / kBatchTile;
        const int row = threadIdx.x % kBatchTile;
        const int batch = batch_start + row;
        float best_score = -std::numeric_limits<float>::infinity();
        int best_token = std::numeric_limits<int>::max();
        if (batch < batch_size) {
          for (int cta = reduction_lane; cta < ctas_per_batch_tile;
               cta += kReductionWays) {
            const int partial_index =
                (batch_tile * ctas_per_batch_tile + cta) * kBatchTile + row;
            const auto* packed_partials =
                reinterpret_cast<const unsigned long long*>(partial_scores);
            float candidate_score;
            int candidate_token;
            unpack_partial(packed_partials[partial_index], candidate_score,
                           candidate_token);
            if (take_candidate(candidate_score, candidate_token, best_score,
                               best_token)) {
              best_score = candidate_score;
              best_token = candidate_token;
            }
          }
        }
        shared.warp_best[reduction_lane][row] =
            pack_partial(best_score, best_token);
      }
      __syncthreads();
      if (threadIdx.x < kBatchTile) {
        const int row = threadIdx.x;
        const int batch = batch_start + row;
        float best_score = -std::numeric_limits<float>::infinity();
        int best_token = std::numeric_limits<int>::max();
        if (batch < batch_size) {
#pragma unroll
          for (int reduction_lane = 0; reduction_lane < kReductionWays;
               ++reduction_lane) {
            float candidate_score;
            int candidate_token;
            unpack_partial(shared.warp_best[reduction_lane][row],
                           candidate_score, candidate_token);
            if (take_candidate(candidate_score, candidate_token, best_score,
                               best_token)) {
              best_score = candidate_score;
              best_token = candidate_token;
            }
          }
          const int target_token =
              static_cast<int>(draft_to_target_id[best_token]);
          shared.selected_target[row] = target_token;
          output[batch * kSteps + step] = target_token;
        } else {
          shared.selected_target[row] = 0;
        }
      }
      __syncthreads();
      if (step + 1 < kSteps) {
        constexpr int kStateVector = 16 / sizeof(__nv_bfloat16);
        constexpr int kStateVectorsPerRow = kRank / kStateVector;
        for (int index = threadIdx.x;
             index < kBatchTile * kStateVectorsPerRow;
             index += blockDim.x) {
          const int row = index / kStateVectorsPerRow;
          const int rank = (index % kStateVectorsPerRow) * kStateVector;
          const int batch = batch_start + row;
          if (batch < batch_size) {
            *reinterpret_cast<uint4*>(
                current_state + batch * kRank + rank) =
                *reinterpret_cast<const uint4*>(
                    markov_w1 + shared.selected_target[row] * kRank + rank);
          } else {
            *reinterpret_cast<uint4*>(
                current_state + batch * kRank + rank) =
                make_uint4(0, 0, 0, 0);
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(&barrier_state[batch_tile], 0);
        atomicExch(&barrier_state[kMaxBatchTiles + batch_tile], step + 1);
      }
    } else if (threadIdx.x == 0) {
      while (atomicAdd(&barrier_state[kMaxBatchTiles + batch_tile], 0) <
             step + 1) {
        __nanosleep(64);
      }
    }
    __syncthreads();
  }
}


}  // namespace

int64_t dsparkPersistentMarkov(torch::Tensor output,
                               const torch::Tensor& anchor,
                               const torch::Tensor& base_logits,
                               const torch::Tensor& draft_to_target_id,
                               const torch::Tensor& markov_w1,
                               const torch::Tensor& markov_w2,
                               torch::Tensor current_state,
                               torch::Tensor partial_scores,
                               torch::Tensor partial_tokens,
                               torch::Tensor barrier_state,
                               double scale,
                               int64_t requested_max_ctas_per_batch_tile) {
    TORCH_CHECK(output.is_cuda() && anchor.is_cuda() && base_logits.is_cuda()
                    && draft_to_target_id.is_cuda() && markov_w1.is_cuda() && markov_w2.is_cuda()
                    && current_state.is_cuda() && partial_scores.is_cuda() && partial_tokens.is_cuda()
                    && barrier_state.is_cuda(),
                "DSpARK persistent Markov requires CUDA tensors");
    TORCH_CHECK(base_logits.scalar_type() == torch::kBFloat16
                    && markov_w1.scalar_type() == torch::kBFloat16
                    && markov_w2.scalar_type() == torch::kBFloat16
                    && current_state.scalar_type() == torch::kBFloat16,
                "DSpARK persistent Markov logits, weights, and state must be BF16");
    TORCH_CHECK(output.scalar_type() == torch::kInt32 && anchor.scalar_type() == torch::kInt32
                    && draft_to_target_id.scalar_type() == torch::kInt64,
                "DSpARK persistent Markov output/anchor/d2t must be int32/int32/int64");
    TORCH_CHECK(partial_scores.scalar_type() == torch::kFloat32
                    && partial_tokens.scalar_type() == torch::kInt32
                    && barrier_state.scalar_type() == torch::kInt32,
                "DSpARK persistent Markov partial scores/tokens/barriers must be float32/int32/int32");

    const int device_index = base_logits.get_device();
    const auto same_device = [device_index](const torch::Tensor& tensor) {
        return tensor.get_device() == device_index;
    };
    TORCH_CHECK(same_device(output) && same_device(anchor) && same_device(draft_to_target_id)
                    && same_device(markov_w1) && same_device(markov_w2) && same_device(current_state)
                    && same_device(partial_scores) && same_device(partial_tokens) && same_device(barrier_state),
                "DSpARK persistent Markov tensors must use one CUDA device");

    int batch_size = static_cast<int>(anchor.numel());
    TORCH_CHECK(anchor.dim() == 1 && batch_size >= 1 && batch_size <= kMaxBatch,
                "DSpARK persistent Markov batch size must be in [1, 32]");
    TORCH_CHECK(anchor.stride(0) > 0 && anchor.stride(0) <= std::numeric_limits<int>::max(),
                "DSpARK persistent Markov anchor stride exceeds int32");
    int anchor_stride = static_cast<int>(anchor.stride(0));
    TORCH_CHECK(base_logits.dim() == 3 && base_logits.size(0) == batch_size
                    && base_logits.size(1) == kSteps,
                "DSpARK persistent Markov base logits must have shape [B, 7, V]");
    int vocab_size = static_cast<int>(base_logits.size(2));
    TORCH_CHECK(vocab_size > 0 && base_logits.stride(2) == 1
                    && base_logits.stride(1) >= vocab_size
                    && base_logits.stride(0) == kSteps * base_logits.stride(1)
                    && base_logits.stride(0) <= std::numeric_limits<int>::max(),
                "DSpARK persistent Markov requires contiguous vocabulary rows");
    int base_logits_batch_stride = static_cast<int>(base_logits.stride(0));
    int base_logits_step_stride = static_cast<int>(base_logits.stride(1));

    const int padded_vocab_size =
        (vocab_size + kVocabTile - 1) / kVocabTile * kVocabTile;
    TORCH_CHECK(markov_w1.dim() == 2 && markov_w1.size(1) == kRank
                    && markov_w2.dim() == 2 && markov_w2.size(0) >= padded_vocab_size
                    && markov_w2.size(1) == kRank,
                "DSpARK persistent Markov weights must be [target_vocab, 256] and "
                "[padded_draft_vocab, 256]");
    TORCH_CHECK(draft_to_target_id.dim() == 1
                    && draft_to_target_id.numel() == vocab_size,
                "DSpARK persistent Markov d2t must cover the draft vocabulary");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == batch_size
                    && output.size(1) == kSteps,
                "DSpARK persistent Markov output must have shape [B, 7]");
    TORCH_CHECK(output.is_contiguous() && draft_to_target_id.is_contiguous()
                    && markov_w1.is_contiguous() && markov_w2.is_contiguous()
                    && current_state.is_contiguous() && partial_scores.is_contiguous()
                    && partial_tokens.is_contiguous() && barrier_state.is_contiguous(),
                "DSpARK persistent Markov output, d2t, weights, and workspaces must be contiguous");

    const int padded_batch_size =
        (batch_size + kBatchTile - 1) / kBatchTile * kBatchTile;
    TORCH_CHECK(current_state.numel() >= padded_batch_size * kRank,
                "DSpARK persistent Markov current-state workspace is too small");
    TORCH_CHECK(barrier_state.numel() >= 2 * kMaxBatchTiles,
                "DSpARK persistent Markov barrier workspace is too small");
    TORCH_CHECK(requested_max_ctas_per_batch_tile >= 1
                    && requested_max_ctas_per_batch_tile <= kMaxCtasPerBatchTile,
                "DSpARK persistent Markov max CTAs must be in [1, 256]");

    const c10::cuda::CUDAGuard device_guard(base_logits.device());
    cudaDeviceProp properties{};
    check_cuda_value(cudaGetDeviceProperties(&properties, device_index));
    TORCH_CHECK(properties.major == 9,
                "DSpARK persistent Markov requires an SM90 GPU");
    TORCH_CHECK(properties.cooperativeLaunch,
                "DSpARK persistent Markov requires cooperative launch support");

    int active_blocks_per_sm = 0;
    check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, dspark_persistent_markov_kernel, kThreads, 0));
    TORCH_CHECK(active_blocks_per_sm > 0,
                "DSpARK persistent Markov kernel has zero occupancy");

    int batch_tiles = (batch_size + kBatchTile - 1) / kBatchTile;
    const int num_vocab_tiles =
        (vocab_size + kVocabTile - 1) / kVocabTile;
    const int resident_ctas =
        properties.multiProcessorCount * active_blocks_per_sm;
    int ctas_per_batch_tile = std::min(
        {num_vocab_tiles,
         resident_ctas / batch_tiles,
         static_cast<int>(requested_max_ctas_per_batch_tile)});
    TORCH_CHECK(ctas_per_batch_tile > 0,
                "DSpARK persistent Markov has no resident CTA per batch tile");

    const int64_t partial_workspace_size =
        static_cast<int64_t>(batch_tiles) * ctas_per_batch_tile * kBatchTile;
    TORCH_CHECK(partial_scores.numel() >= 2 * partial_workspace_size
                    && partial_tokens.numel() >= partial_workspace_size,
                "DSpARK persistent Markov partial workspace is too small");

    auto* output_ptr = output.data_ptr<int32_t>();
    const auto* anchor_ptr = anchor.data_ptr<int32_t>();
    const auto* base_logits_ptr =
        reinterpret_cast<const __nv_bfloat16*>(base_logits.data_ptr<at::BFloat16>());
    const auto* d2t_ptr = draft_to_target_id.data_ptr<int64_t>();
    const auto* w1_ptr =
        reinterpret_cast<const __nv_bfloat16*>(markov_w1.data_ptr<at::BFloat16>());
    const auto* w2_ptr =
        reinterpret_cast<const __nv_bfloat16*>(markov_w2.data_ptr<at::BFloat16>());
    auto* state_ptr =
        reinterpret_cast<__nv_bfloat16*>(current_state.data_ptr<at::BFloat16>());
    auto* scores_ptr = partial_scores.data_ptr<float>();
    auto* tokens_ptr = partial_tokens.data_ptr<int32_t>();
    auto* barrier_ptr = barrier_state.data_ptr<int32_t>();
    float float_scale = static_cast<float>(scale);

    void* arguments[] = {
        &output_ptr,
        &anchor_ptr,
        &base_logits_ptr,
        &d2t_ptr,
        &w1_ptr,
        &w2_ptr,
        &state_ptr,
        &scores_ptr,
        &tokens_ptr,
        &barrier_ptr,
        &batch_size,
        &anchor_stride,
        &vocab_size,
        &base_logits_batch_stride,
        &base_logits_step_stride,
        &float_scale,
        &batch_tiles,
        &ctas_per_batch_tile,
    };
    const dim3 grid(batch_tiles * ctas_per_batch_tile);
    const dim3 block(kThreads);
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device_index).stream();
    check_cuda_value(cudaMemsetAsync(
        barrier_ptr, 0, 2 * kMaxBatchTiles * sizeof(int), stream));
    check_cuda_value(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(dspark_persistent_markov_kernel),
        grid,
        block,
        arguments,
        0,
        stream));
    return grid.x;
}

}  // namespace rtp_llm

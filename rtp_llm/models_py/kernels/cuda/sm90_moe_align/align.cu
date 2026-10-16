// SPDX-License-Identifier: Apache-2.0
// Derived from vLLM contributors' csrc/moe/moe_align_sum_kernels.cu at
// de8291e24f1312da0239605e33a89fe9cc901e47 (Apache License 2.0; see LICENSE).
// Original file SHA256:
// d21e3309a45fd2439ffcf08a01ff429663f3ef684a675f13e8ea82a2e96fb495
// Modified for RTP-LLM: retain only small_batch_expert single-CTA algorithm;
// specialize away expert_map, LoRA/token masks and optional scatter output;
// fuse contiguous global-to-local ID translation, checking in int64 before
// narrowing; replace stable ABI host binding with ATen/current-stream binding.
// Count/prefix/scatter ordering and the three CTA barriers are unchanged.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ int get_local_expert_id(
    size_t idx, const scalar_t* __restrict__ topk_ids,
    int32_t num_experts, int64_t local_start) {
  const int64_t expert_id = static_cast<int64_t>(topk_ids[idx]);
  if (expert_id < local_start || expert_id >= local_start + num_experts)
    return -1;
  return static_cast<int>(expert_id - local_start);
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded, int32_t max_num_m_blocks,
    int64_t local_start) {
  // Upstream's separate fill threads share this CTA and its three barriers.
  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[it] = numel;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = shared_mem + num_experts + 1;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }
  for (size_t i = tid; i < numel; i += stride) {
    if (int expert_id = get_local_expert_id(i, topk_ids, num_experts, local_start);
        expert_id != -1) {
      tokens_cnts[(tid + 1) * num_experts + expert_id] += 1;
    }
  }
  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }
  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          ((tokens_cnts[stride * num_experts + i - 1] + block_size - 1) /
           block_size) * block_size;
    }
    total_tokens_post_pad[0] = cumsum[num_experts];
  }
  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[i] = -1;
  }
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, num_experts, local_start);
    if (expert_id != -1) {
      int32_t rank_post_pad =
          tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}
}  // namespace

std::vector<at::Tensor> sm90_moe_align_cuda(
    const at::Tensor& ids, int64_t num_experts, int64_t block_size,
    int64_t local_start) {
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.dim() == 2 &&
              ids.size(1) > 0, "Expected contiguous CUDA [rows, topk]");
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong,
              "Expected int32 or int64 IDs");
  TORCH_CHECK(num_experts >= 1 && num_experts <= 64 && ids.numel() < 1024,
              "Only E <= 64 and routes < 1024 are supported");
  TORCH_CHECK(block_size >= 1 && block_size <= 256 && local_start >= 0 &&
              local_start <= std::numeric_limits<int32_t>::max() - num_experts,
              "Unsupported block size or local expert start");
  const c10::cuda::CUDAGuard guard(ids.device());
  // Static worst-case capacity, rounded to whole blocks. No value-dependent
  // allocation/readback. Empty input deliberately still launches to set count.
  const int64_t slots = ids.numel();
  const int64_t capacity_bound = std::min(
      slots * block_size, slots + num_experts * (block_size - 1));
  const int64_t capacity =
      ((capacity_bound + block_size - 1) / block_size) * block_size;
  const int64_t blocks = capacity / block_size;
  auto options = ids.options().dtype(at::kInt);
  auto sorted = at::empty({capacity}, options);
  auto experts = at::empty({blocks}, options);
  auto padded = at::empty({1}, options);
  const int32_t threads = std::max<int32_t>(num_experts, 32);
  constexpr int32_t fill_threads = 256;
  const size_t shared =
      ((threads + 1) * num_experts + num_experts + 1) * sizeof(int32_t);
  auto stream = c10::cuda::getCurrentCUDAStream(ids.get_device());
  if (ids.scalar_type() == at::kInt) {
    moe_align_block_size_small_batch_expert_kernel<int32_t, fill_threads>
        <<<1, fill_threads + threads, shared, stream.stream()>>>(
            ids.data_ptr<int32_t>(), sorted.data_ptr<int32_t>(),
            experts.data_ptr<int32_t>(), padded.data_ptr<int32_t>(),
            num_experts, block_size, slots, capacity, blocks, local_start);
  } else {
    moe_align_block_size_small_batch_expert_kernel<int64_t, fill_threads>
        <<<1, fill_threads + threads, shared, stream.stream()>>>(
            ids.data_ptr<int64_t>(), sorted.data_ptr<int32_t>(),
            experts.data_ptr<int32_t>(), padded.data_ptr<int32_t>(),
            num_experts, block_size, slots, capacity, blocks, local_start);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted, experts, padded};
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Native vLLM c3b484463 RMSNorm arithmetic, specialized to K3 BF16 rank-2 inputs.
#include "kimi_k3_rms_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cmath>
#include <limits>

namespace rtp_k3_norm {
using CubAddOp = cuda::std::plus<>;
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};
// read-only version: iterate over the input with alignment guarantees
template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_with_alignment(const InT* in, int len,
                                                     int tid, int stride,
                                                     VecOp&& vec_op,
                                                     ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  // fast path when the whole region is already aligned
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);

    for (int i = tid; i < num_vec; i += stride) {
      vin_t tmp = v_in[i];
      vec_op(tmp);
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  // 1. handle the possibly unaligned prefix with scalar access.
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(in[i]);
  }

  in += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);

  // 2. vectorized traversal of the main aligned region.
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }

  // 3. handle remaining tail elements.
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(in[i]);
  }
}

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size] or
                                          // [num_groups, hidden_size];
                                          // null if !HasWeight
    const int64_t weight_stride,          // 0 or weight.stride(0)
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t* input_row;
  const scalar_t* weight_row;
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
    weight_row = weight + blockIdx.x * weight_stride;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  }

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  rtp_k3_norm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight_row);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2;
    if constexpr (HasWeight) {
      src2 = v_w[i];
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      if constexpr (HasWeight) {
        float w = static_cast<float>(src2.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
      } else {
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
    }
    v_out[i] = dst;
  }
}


} // namespace rtp_k3_norm
namespace rtp_llm {
at::Tensor kimi_k3_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "K3 RMSNorm requires CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "K3 RMSNorm requires BF16 input and weight");
  TORCH_CHECK(input.device() == weight.device(), "K3 RMSNorm device mismatch");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1, "K3 RMSNorm expects rank-2 input/rank-1 weight");
  const auto hidden = input.size(1);
  const auto tokens = input.size(0);
  TORCH_CHECK(hidden > 0 && hidden % 8 == 0 && hidden <= std::numeric_limits<int>::max(),
              "K3 RMSNorm requires hidden size divisible by 8");
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(), "K3 RMSNorm token count overflow");
  TORCH_CHECK(input.stride(1) == 1 && input.stride(0) % 8 == 0 && weight.is_contiguous(),
              "K3 RMSNorm requires aligned rows and contiguous weight");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "K3 RMSNorm requires 16-byte aligned storage");
  TORCH_CHECK(weight.numel() == hidden, "K3 RMSNorm weight size mismatch");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0, "K3 RMSNorm requires finite nonnegative epsilon");
  const c10::cuda::CUDAGuard guard(input.device());
  auto out = at::empty(input.sizes(), input.options());
  if (tokens == 0) return out;
  const int block = std::min<int>(hidden / 8, tokens < 256 ? 1024 : 256);
  rtp_k3_norm::rms_norm_kernel<at::BFloat16, 8, 2, true>
      <<<tokens, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<at::BFloat16>(), input.data_ptr<at::BFloat16>(), input.stride(0),
        0, 0, 0, 0, weight.data_ptr<at::BFloat16>(), 0, static_cast<float>(epsilon), tokens, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
} // namespace rtp_llm

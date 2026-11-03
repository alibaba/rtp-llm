// Copyright 2026 Tencent

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "cutlass/fast_math.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/hy4_ihc.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/hy4_ihc_utils.cuh"

namespace rtp_llm {
namespace hy4_ihc {

namespace {
int get_sm_count_for_current_device() {
  int device = 0;
  int sm_count = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}
}  // namespace

namespace kernels {

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kRowsPerBlock,
          int kThreadsPerBlock, bool kHasPost, bool kUsePDL = false, bool kFuseRmsNorm = false,
          bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_pre_or_head_kernel(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                                const __nv_bfloat16* x_ptr, const float* w_ptr,
                                const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                                float norm_eps, float hc_eps, float magnitude,
                                const __nv_bfloat16* rms_weight_ptr = nullptr,
                                float rms_eps = 0.f) {
  constexpr int kProjRows = kHasPost ? 2 * kHCMult : kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kChunks = kHiddenDim / (kThreadsPerBlock * kElementsPerThread);
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr float kInvHCDim = 1.0f / kHCDim;
  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;

  static_assert(kHiddenDim % (kThreadsPerBlock * kElementsPerThread) == 0,
                "kThreadsPerBlock * kElementsPerThread must divide kHiddenDim");
  static_assert(kThreadsPerBlock % kWarpSize == 0, "block must be whole warps");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int row_base = blockIdx.x * kRowsPerBlock;
  const int num_valid_rows = min(kRowsPerBlock, num_batch - row_base);

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kRowsPerBlock][kProjRows + 1][kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kRowsPerBlock][kHCMult];

  float acc_dot[kRowsPerBlock][kProjRows];
  float acc_sqsum[kRowsPerBlock];
#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    acc_sqsum[r] = 0.f;
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      acc_dot[r][c] = 0.f;
    }
  }

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
      const int iflat = i * kHiddenDim + icol;
      constexpr int kHalf = kElementsPerThread / 2;
      vec_t<float, kElementsPerThread> reg_w[kProjRows];
#pragma unroll
      for (int c = 0; c < kProjRows; ++c) {
        const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[c][0]) = load<float, kHalf>(w_row_ptr);
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[c][kHalf]) =
            load<float, kHalf>(w_row_ptr + kHalf);
      }

      for (int r = 0; r < num_valid_rows; ++r) {
        auto reg_x = to<float>(
            load<__nv_bfloat16, kElementsPerThread>(x_ptr + (row_base + r) * kHCDim + iflat));
        const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          acc_sqsum[r] += x2[j].x * x2[j].x;
          acc_sqsum[r] += x2[j].y * x2[j].y;
        }
#pragma unroll
        for (int c = 0; c < kProjRows; ++c) {
          const auto* w2 = reinterpret_cast<const float2*>(&reg_w[c][0]);
#pragma unroll
          for (int j = 0; j < kElementsPerThread / 2; ++j) {
            acc_dot[r][c] += x2[j].x * w2[j].x;
            acc_dot[r][c] += x2[j].y * w2[j].y;
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      float v = warp_reduce_sum_xor(acc_dot[r][c]);
      if (ilane == 0) {
        smem_partial[r][c][iwarp] = v;
      }
    }
    float v = warp_reduce_sum_xor(acc_sqsum[r]);
    if (ilane == 0) {
      smem_partial[r][kProjRows][iwarp] = v;
    }
  }
  __syncthreads();

  const cutlass::FastDivmod proj_divmod(kProjRows + 1);
  for (int idx = threadIdx.x; idx < kRowsPerBlock * (kProjRows + 1); idx += kThreadsPerBlock) {
    int r, c;
    proj_divmod(r, c, idx);  // r = idx / (kProjRows+1), c = idx % (kProjRows+1)
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[r][c][w];
    }
    smem_partial[r][c][0] = sum;
  }
  __syncthreads();

  const cutlass::FastDivmod hc_divmod(kHCMult);
  for (int idx = threadIdx.x; idx < num_valid_rows * kHCMult; idx += kThreadsPerBlock) {
    int r, i;
    hc_divmod(r, i, idx);  // r = idx / kHCMult, i = idx % kHCMult
    const float rms_scale = rsqrtf_ftz(smem_partial[r][kProjRows][0] * kInvHCDim + norm_eps);

    const float mixes_pre = smem_partial[r][i][0] * rms_scale;
    smem_H_pre[r][i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;

    if constexpr (kHasPost) {
      const float mixes_post = smem_partial[r][kHCMult + i][0] * rms_scale;
      output_H_post_ptr[(row_base + r) * kHCMult + i] =
          magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
    }
  }
  __syncthreads();

  for (int r = 0; r < num_valid_rows; ++r) {
    vec_t<float, kHCMult> reg_H_pre = load<float, kHCMult>(&smem_H_pre[r][0]);
    const __nv_bfloat16* x_row_ptr = x_ptr + (row_base + r) * kHCDim;
    __nv_bfloat16* y_row_ptr = output_y_ptr + (row_base + r) * kHiddenDim;

    vec_t<float, kElementsPerThread> reg_y[kChunks];
    float y_sqsum = 0.f;

#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;

#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) {
        reg_y[ch][j] = 0.f;
      }
      auto* y2 = reinterpret_cast<float2*>(&reg_y[ch][0]);

#pragma unroll
      for (int i = 0; i < kHCMult; ++i) {
        auto reg_x =
            to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr + i * kHiddenDim + icol));
        const float h = reg_H_pre[i];
        const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          y2[j].x += h * x2[j].x;
          y2[j].y += h * x2[j].y;
        }
      }

      if constexpr (kFuseRmsNorm) {
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          if constexpr (kCastBfloatForNorm) {
            y2[j].x = __bfloat162float(__float2bfloat16(y2[j].x));
            y2[j].y = __bfloat162float(__float2bfloat16(y2[j].y));
          }
          y_sqsum += y2[j].x * y2[j].x;
          y_sqsum += y2[j].y * y2[j].y;
        }
      } else {
        store(y_row_ptr + icol, to<__nv_bfloat16>(reg_y[ch]));
      }
    }

    if constexpr (kFuseRmsNorm) {
      float v = warp_reduce_sum_xor(y_sqsum);
      if (ilane == 0) {
        smem_partial[r][0][iwarp] = v;
      }
      __syncthreads();
      float total = 0.f;
#pragma unroll
      for (int w = 0; w < kWarpsPerBlock; ++w) {
        total += smem_partial[r][0][w];
      }
      const float rms_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);

#pragma unroll
      for (int ch = 0; ch < kChunks; ++ch) {
        const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
        auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
        auto* y2 = reinterpret_cast<float2*>(&reg_y[ch][0]);
        const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          y2[j].x = y2[j].x * rms_scale * g2[j].x;
          y2[j].y = y2[j].y * rms_scale * g2[j].y;
        }
        store(y_row_ptr + icol, to<__nv_bfloat16>(reg_y[ch]));
      }
      __syncthreads();
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, bool kUsePDL = false>
__global__ void __launch_bounds__(128)
    fuse_ihc_post_kernel(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int irow = blockIdx.x;
  auto reg_H_post = load<float, kHCMult>(H_post_ptr + irow * kHCMult);

  const int icol = (blockIdx.y * blockDim.x + threadIdx.x) * kElementsPerThread;
  const int icol_stride = gridDim.y * blockDim.x * kElementsPerThread;
  const int num_iter = (kHiddenDim - icol + icol_stride - 1) / icol_stride;

  const __nv_bfloat16* x_row_ptr = x_ptr + irow * kHiddenDim + icol;
  const __nv_bfloat16* residual_row_ptr = residual_ptr + irow * kHCMult * kHiddenDim + icol;
  __nv_bfloat16* output_row_ptr = output_ptr + irow * kHCMult * kHiddenDim + icol;

  for (int iter = 0; iter < num_iter; ++iter) {
    auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr + iter * icol_stride));

#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      auto reg_residual = to<float>(load<__nv_bfloat16, kElementsPerThread>(
          residual_row_ptr + i * kHiddenDim + iter * icol_stride));

      vec_t<float, kElementsPerThread> reg_output;
      auto* o2 = reinterpret_cast<float2*>(&reg_output[0]);
      const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
      const auto* r2 = reinterpret_cast<const float2*>(&reg_residual[0]);
      const float h = reg_H_post[i];
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        o2[j].x = h * x2[j].x + r2[j].x;
        o2[j].y = h * x2[j].y + r2[j].y;
      }

      store(output_row_ptr + i * kHiddenDim + iter * icol_stride, to<__nv_bfloat16>(reg_output));
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

namespace {

constexpr int kElementsPerThread = 8;
constexpr int kPreRowsPerBlock = 1;
constexpr int kPostThreadsPerBlock = 128;

template <int kHiddenDim, bool kHasPost, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_pre_head(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                     const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     cudaStream_t stream, bool use_pdl) {
  constexpr int kThreads = 256;
  const int num_block = (num_batch + kPreRowsPerBlock - 1) / kPreRowsPerBlock;
  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel = kernels::fuse_ihc_pre_or_head_kernel<4, kHiddenDim, kElementsPerThread,
                                                       kPreRowsPerBlock, kThreads, kHasPost,
                                                       kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_block);
    config.blockDim = dim3(kThreads);
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    config.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attr;
      config.numAttrs = 1;
    }
    cudaLaunchKernelEx(&config, kernel, output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                       hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                       rms_eps);
  };
  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <int kHiddenDim>
void launch_post(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                 const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                 cudaStream_t stream, bool use_pdl) {
  constexpr int kMaxSplit = kHiddenDim / (kPostThreadsPerBlock * kElementsPerThread);
  int block_per_row = 1;
  if (num_batch > 0) {
    block_per_row = (get_sm_count_for_current_device() + num_batch - 1) / num_batch;
    block_per_row = std::min(block_per_row, kMaxSplit);
    block_per_row = std::max(block_per_row, 1);
  }

  dim3 block(kPostThreadsPerBlock);
  dim3 grid(num_batch, block_per_row, 1);

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel = kernels::fuse_ihc_post_kernel<4, kHiddenDim, kElementsPerThread, kUsePDL>;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    config.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attr;
      config.numAttrs = 1;
    }
    cudaLaunchKernelEx(&config, kernel, output_ptr, x_ptr, residual_ptr, H_post_ptr);
  };
  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

}  // namespace

void fuse_ihc_pre_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                        const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                        const float* hc_base_ptr, int num_batch, int hc_mult, int hidden_dim,
                        float norm_eps, float hc_eps, float magnitude, cudaStream_t stream,
                        const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                        bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  const bool fuse_norm = rms_weight_ptr != nullptr;
  if (hidden_dim == 6144) {
    if (!fuse_norm) {
      launch_pre_head<6144, true, false, false>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr,
                                                hc_scale_ptr, hc_base_ptr, num_batch, norm_eps,
                                                hc_eps, magnitude, nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<6144, true, true, true>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<6144, true, true, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  } else if (hidden_dim == 4096) {
    if (!fuse_norm) {
      launch_pre_head<4096, true, false, false>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr,
                                                hc_scale_ptr, hc_base_ptr, num_batch, norm_eps,
                                                hc_eps, magnitude, nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<4096, true, true, true>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<4096, true, true, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  }
}

void fuse_ihc_post_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                         int hc_mult, int hidden_dim, cudaStream_t stream, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  if (hidden_dim == 6144) {
    launch_post<6144>(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, stream, use_pdl);
  } else if (hidden_dim == 4096) {
    launch_post<4096>(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, stream, use_pdl);
  }
}

void fuse_ihc_head_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                         const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                         int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                         cudaStream_t stream, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                         bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  const bool fuse_norm = rms_weight_ptr != nullptr;
  if (hidden_dim == 6144) {
    if (!fuse_norm) {
      launch_pre_head<6144, false, false, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                 hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                 nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<6144, false, true, true>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                               hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                               rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<6144, false, true, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  } else if (hidden_dim == 4096) {
    if (!fuse_norm) {
      launch_pre_head<4096, false, false, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                 hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                 nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<4096, false, true, true>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                               hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                               rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<4096, false, true, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  }
}

}  // namespace hy4_ihc
}  // namespace rtp_llm

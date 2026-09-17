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

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             float norm_eps, float hc_eps, float magnitude,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f) {
  constexpr int kProjRows = 2 * kHCMult;
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

  const int irow = blockIdx.x;
  if (irow >= num_batch) {
    return;
  }
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kProjRows + 1][kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kHCMult];

  const auto reg_H_post_in = load<float, kHCMult>(H_post_in_ptr + irow * kHCMult);
  const __nv_bfloat16* xa_row_ptr = xa_ptr + irow * kHiddenDim;
  const __nv_bfloat16* res_row_ptr = residual_ptr + irow * kHCDim;
  __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;

  float acc_dot[kProjRows];
  float acc_sqsum = 0.f;
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    acc_dot[c] = 0.f;
  }

#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
    const auto reg_xa = to<float>(load<__nv_bfloat16, kElementsPerThread>(xa_row_ptr + icol));

#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const int iflat = i * kHiddenDim + icol;
      const auto reg_res = to<float>(load<__nv_bfloat16, kElementsPerThread>(res_row_ptr + iflat));

      vec_t<float, kElementsPerThread> reg_y;
      auto* y2 = reinterpret_cast<float2*>(&reg_y[0]);
      const auto* xa2 = reinterpret_cast<const float2*>(&reg_xa[0]);
      const auto* res2 = reinterpret_cast<const float2*>(&reg_res[0]);
      const float h = reg_H_post_in[i];
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        y2[j].x = h * xa2[j].x + res2[j].x;
        y2[j].y = h * xa2[j].y + res2[j].y;
      }
      store(y_row_ptr + iflat, to<__nv_bfloat16>(reg_y));

      const auto reg_y_bf = to<float>(to<__nv_bfloat16>(reg_y));
      const auto* yb2 = reinterpret_cast<const float2*>(&reg_y_bf[0]);

#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_sqsum += yb2[j].x * yb2[j].x;
        acc_sqsum += yb2[j].y * yb2[j].y;
      }
      constexpr int kHalf = kElementsPerThread / 2;
#pragma unroll
      for (int c = 0; c < kProjRows; ++c) {
        const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
        vec_t<float, kElementsPerThread> reg_w;
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[0]) = load<float, kHalf>(w_row_ptr);
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[kHalf]) =
            load<float, kHalf>(w_row_ptr + kHalf);
        const auto* w2 = reinterpret_cast<const float2*>(&reg_w[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          acc_dot[c] += yb2[j].x * w2[j].x;
          acc_dot[c] += yb2[j].y * w2[j].y;
        }
      }
    }
  }

#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    const float v = warp_reduce_sum_xor(acc_dot[c]);
    if (ilane == 0) {
      smem_partial[c][iwarp] = v;
    }
  }
  {
    const float v = warp_reduce_sum_xor(acc_sqsum);
    if (ilane == 0) {
      smem_partial[kProjRows][iwarp] = v;
    }
  }
  __syncthreads();

  for (int c = threadIdx.x; c < kProjRows + 1; c += kThreadsPerBlock) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[c][w];
    }
    smem_partial[c][0] = sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kHCMult; i += kThreadsPerBlock) {
    const float rms_scale = rsqrtf_ftz(smem_partial[kProjRows][0] * kInvHCDim + norm_eps);
    const float mixes_pre = smem_partial[i][0] * rms_scale;
    smem_H_pre[i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;

    const float mixes_post = smem_partial[kHCMult + i][0] * rms_scale;
    H_post_out_ptr[irow * kHCMult + i] =
        magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
  }
  __syncthreads();

  const auto reg_H_pre = load<float, kHCMult>(&smem_H_pre[0]);
  __nv_bfloat16* z_row_ptr = z_ptr + irow * kHiddenDim;

  vec_t<float, kElementsPerThread> reg_z[kChunks];
  float z_sqsum = 0.f;

#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_z[ch][j] = 0.f;
    }
    auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const auto reg_y =
          to<float>(load<__nv_bfloat16, kElementsPerThread>(y_row_ptr + i * kHiddenDim + icol));
      const float h = reg_H_pre[i];
      const auto* y2 = reinterpret_cast<const float2*>(&reg_y[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x += h * y2[j].x;
        z2[j].y += h * y2[j].y;
      }
    }
    if constexpr (kFuseRmsNorm) {
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        if constexpr (kCastBfloatForNorm) {
          z2[j].x = __bfloat162float(__float2bfloat16(z2[j].x));
          z2[j].y = __bfloat162float(__float2bfloat16(z2[j].y));
        }
        z_sqsum += z2[j].x * z2[j].x;
        z_sqsum += z2[j].y * z2[j].y;
      }
    } else {
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kFuseRmsNorm) {
    const float v = warp_reduce_sum_xor(z_sqsum);
    if (ilane == 0) {
      smem_partial[0][iwarp] = v;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += smem_partial[0][w];
    }
    const float rms_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);
#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
      const auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
      auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
      const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x = z2[j].x * rms_scale * g2[j].x;
        z2[j].y = z2[j].y * rms_scale * g2[j].y;
      }
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_stage1_kernel(__nv_bfloat16* y_ptr, float* scratch_ptr,
                                    const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                                    const float* H_post_in_ptr, const float* w_ptr, int num_batch) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kColsPerBlock = kThreadsPerBlock * kElementsPerThread;
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr int kNumChunks = kHiddenDim / kColsPerBlock;
  static_assert(kHiddenDim % kColsPerBlock == 0, "block cols must divide hidden dim");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int ch = blockIdx.x;
  const int irow = blockIdx.y;
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int icol = ch * kColsPerBlock + threadIdx.x * kElementsPerThread;

  __shared__ float smem_partial[kProjRows + 1][kWarpsPerBlock];

  const auto reg_H_post_in = load<float, kHCMult>(H_post_in_ptr + irow * kHCMult);
  const __nv_bfloat16* xa_row_ptr = xa_ptr + irow * kHiddenDim;
  const __nv_bfloat16* res_row_ptr = residual_ptr + irow * kHCDim;
  __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;

  float acc_dot[kProjRows];
  float acc_sqsum = 0.f;
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    acc_dot[c] = 0.f;
  }

  const auto reg_xa = to<float>(load<__nv_bfloat16, kElementsPerThread>(xa_row_ptr + icol));
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    const int iflat = i * kHiddenDim + icol;
    const auto reg_res = to<float>(load<__nv_bfloat16, kElementsPerThread>(res_row_ptr + iflat));
    vec_t<float, kElementsPerThread> reg_y;
    auto* y2 = reinterpret_cast<float2*>(&reg_y[0]);
    const auto* xa2 = reinterpret_cast<const float2*>(&reg_xa[0]);
    const auto* res2 = reinterpret_cast<const float2*>(&reg_res[0]);
    const float h = reg_H_post_in[i];
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      y2[j].x = h * xa2[j].x + res2[j].x;
      y2[j].y = h * xa2[j].y + res2[j].y;
    }
    store(y_row_ptr + iflat, to<__nv_bfloat16>(reg_y));
    const auto reg_y_bf = to<float>(to<__nv_bfloat16>(reg_y));
    const auto* yb2 = reinterpret_cast<const float2*>(&reg_y_bf[0]);
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      acc_sqsum += yb2[j].x * yb2[j].x;
      acc_sqsum += yb2[j].y * yb2[j].y;
    }
    constexpr int kHalf = kElementsPerThread / 2;
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
      vec_t<float, kElementsPerThread> reg_w;
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[0]) = load<float, kHalf>(w_row_ptr);
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[kHalf]) =
          load<float, kHalf>(w_row_ptr + kHalf);
      const auto* w2 = reinterpret_cast<const float2*>(&reg_w[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_dot[c] += yb2[j].x * w2[j].x;
        acc_dot[c] += yb2[j].y * w2[j].y;
      }
    }
  }

#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    const float v = warp_reduce_sum_xor(acc_dot[c]);
    if (ilane == 0) {
      smem_partial[c][iwarp] = v;
    }
  }
  {
    const float v = warp_reduce_sum_xor(acc_sqsum);
    if (ilane == 0) {
      smem_partial[kProjRows][iwarp] = v;
    }
  }
  __syncthreads();
  float* scratch_row = scratch_ptr + (irow * kNumChunks + ch) * (kProjRows + 1);
  for (int c = threadIdx.x; c < kProjRows + 1; c += kThreadsPerBlock) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[c][w];
    }
    scratch_row[c] = sum;
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_stage2_kernel(__nv_bfloat16* z_ptr, float* H_post_out_ptr,
                                    const __nv_bfloat16* y_ptr, const float* scratch_ptr,
                                    const float* hc_scale_ptr, const float* hc_base_ptr,
                                    int num_batch, float norm_eps, float hc_eps, float magnitude,
                                    const __nv_bfloat16* rms_weight_ptr = nullptr,
                                    float rms_eps = 0.f) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kChunks = kHiddenDim / (kThreadsPerBlock * kElementsPerThread);
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr float kInvHCDim = 1.0f / kHCDim;
  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;

  const int irow = blockIdx.x;
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kHCMult];

  float full[kProjRows + 1];
#pragma unroll
  for (int c = 0; c < kProjRows + 1; ++c) {
    full[c] = 0.f;
  }
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const float* scratch_row = scratch_ptr + (irow * kChunks + ch) * (kProjRows + 1);
#pragma unroll
    for (int c = 0; c < kProjRows + 1; ++c) {
      full[c] += scratch_row[c];
    }
  }

  const float rms_scale = rsqrtf_ftz(full[kProjRows] * kInvHCDim + norm_eps);
  for (int i = threadIdx.x; i < kHCMult; i += kThreadsPerBlock) {
    const float mixes_pre = full[i] * rms_scale;
    smem_H_pre[i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;
    const float mixes_post = full[kHCMult + i] * rms_scale;
    H_post_out_ptr[irow * kHCMult + i] =
        magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
  }
  __syncthreads();

  const auto reg_H_pre = load<float, kHCMult>(&smem_H_pre[0]);
  const __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;
  __nv_bfloat16* z_row_ptr = z_ptr + irow * kHiddenDim;

  vec_t<float, kElementsPerThread> reg_z[kChunks];
  float z_sqsum = 0.f;
#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_z[ch][j] = 0.f;
    }
    auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const auto reg_y =
          to<float>(load<__nv_bfloat16, kElementsPerThread>(y_row_ptr + i * kHiddenDim + icol));
      const float h = reg_H_pre[i];
      const auto* y2 = reinterpret_cast<const float2*>(&reg_y[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x += h * y2[j].x;
        z2[j].y += h * y2[j].y;
      }
    }
    if constexpr (kFuseRmsNorm) {
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        if constexpr (kCastBfloatForNorm) {
          z2[j].x = __bfloat162float(__float2bfloat16(z2[j].x));
          z2[j].y = __bfloat162float(__float2bfloat16(z2[j].y));
        }
        z_sqsum += z2[j].x * z2[j].x;
        z_sqsum += z2[j].y * z2[j].y;
      }
    } else {
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kFuseRmsNorm) {
    const float v = warp_reduce_sum_xor(z_sqsum);
    if (ilane == 0) {
      smem_partial[iwarp] = v;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += smem_partial[w];
    }
    const float z_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);
#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
      const auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
      auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
      const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x = z2[j].x * z_scale * g2[j].x;
        z2[j].y = z2[j].y * z_scale * g2[j].y;
      }
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
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

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_post_pre(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                     const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                     const float* H_post_in_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     float* scratch_ptr, cudaStream_t stream, bool use_pdl) {
  constexpr int kThreads = 256;
  constexpr int kClusterSize = kHiddenDim / (kThreads * kElementsPerThread);

  const bool use_twostage = kClusterSize >= 2 && scratch_ptr != nullptr;

  if (use_twostage) {
    float* scratch = scratch_ptr;
    dim3 grid1(kClusterSize, num_batch, 1);
    dim3 grid2(num_batch, 1, 1);
    dim3 block(kThreads);
    auto launch = [&](auto pdl_tag) {
      constexpr bool kUsePDL = decltype(pdl_tag)::value;
      auto k1 = kernels::fuse_ihc_post_pre_stage1_kernel<4, kHiddenDim, kElementsPerThread,
                                                         kThreads, kUsePDL>;
      auto k2 =
          kernels::fuse_ihc_post_pre_stage2_kernel<4, kHiddenDim, kElementsPerThread, kThreads,
                                                   kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
      cudaLaunchConfig_t c1{};
      c1.gridDim = grid1;
      c1.blockDim = block;
      c1.stream = stream;
      cudaLaunchAttribute a1[1];
      c1.numAttrs = 0;
      if constexpr (kUsePDL) {
        a1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a1[0].val.programmaticStreamSerializationAllowed = 1;
        c1.attrs = a1;
        c1.numAttrs = 1;
      }
      cudaLaunchKernelEx(&c1, k1, y_ptr, scratch, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr,
                         num_batch);
      cudaLaunchConfig_t c2{};
      c2.gridDim = grid2;
      c2.blockDim = block;
      c2.stream = stream;
      cudaLaunchAttribute a2[1];
      c2.numAttrs = 0;
      if constexpr (kUsePDL) {
        a2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a2[0].val.programmaticStreamSerializationAllowed = 1;
        c2.attrs = a2;
        c2.numAttrs = 1;
      }
      cudaLaunchKernelEx(&c2, k2, z_ptr, H_post_out_ptr, y_ptr, scratch, hc_scale_ptr, hc_base_ptr,
                         num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    };
    if (use_pdl) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
    return;
  }

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel = kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, kElementsPerThread, kThreads,
                                                    kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_batch);
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
    cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                       H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps, hc_eps,
                       magnitude, rms_weight_ptr, rms_eps);
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

void fuse_ihc_post_pre_async(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                             float magnitude, float* scratch_ptr, cudaStream_t stream,
                             const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                             bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  const bool fuse_norm = rms_weight_ptr != nullptr;

  if (hidden_dim == 6144) {
    if (!fuse_norm) {
      launch_post_pre<6144, false, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                          H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                          num_batch, norm_eps, hc_eps, magnitude, nullptr, 0.f,
                                          scratch_ptr, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_post_pre<6144, true, true>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                        H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                        norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps,
                                        scratch_ptr, stream, use_pdl);
    } else {
      launch_post_pre<6144, true, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                         H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                         norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps,
                                         scratch_ptr, stream, use_pdl);
    }
  } else if (hidden_dim == 4096) {
    if (!fuse_norm) {
      launch_post_pre<4096, false, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                          H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                          num_batch, norm_eps, hc_eps, magnitude, nullptr, 0.f,
                                          scratch_ptr, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_post_pre<4096, true, true>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                        H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                        norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps,
                                        scratch_ptr, stream, use_pdl);
    } else {
      launch_post_pre<4096, true, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                         H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                         norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps,
                                         scratch_ptr, stream, use_pdl);
    }
  }
}

size_t ihc_post_pre_scratch_floats(int num_batch, int hc_mult, int hidden_dim) {
  if (hc_mult != 4 || (hidden_dim != 6144 && hidden_dim != 4096)) {
    return 0;
  }
  constexpr int kColsPerBlock = 256 * kElementsPerThread;
  const int chunks = hidden_dim / kColsPerBlock;
  if (chunks < 2 || num_batch <= 0 || num_batch >= get_sm_count_for_current_device() * 2) {
    return 0;
  }
  const int proj_rows = 2 * hc_mult;
  return static_cast<size_t>(num_batch) * chunks * (proj_rows + 1);
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

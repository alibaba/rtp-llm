#include "fused_rmsnorm_fp8_quant.h"

#include "fp8_ue8m0_scale_layout.cuh"
#include "util.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rtp_llm {

namespace {

constexpr int kGroupSize = 128;
constexpr int kMaxGroups = 64;
constexpr int kThreadsPerGroup = 16;
constexpr int kBf16VecElems = 8;

int64_t ceilAlign(int64_t value, int64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ float bf16ToFloat(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float roundToBf16Float(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float clampFp8Range(float v, float min_8bit, float max_8bit) {
    return fminf(fmaxf(v, min_8bit), max_8bit);
}

__device__ __forceinline__ void storeFp8x8(__nv_fp8_e4m3* __restrict__ dst,
                                           const float (&vals)[kBf16VecElems],
                                           float inv_scale,
                                           float min_8bit,
                                           float max_8bit) {
    uint2 packed_u32;
    auto* packed = reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_u32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 q_pair = make_float2(clampFp8Range(vals[2 * i] * inv_scale, min_8bit, max_8bit),
                                          clampFp8Range(vals[2 * i + 1] * inv_scale, min_8bit, max_8bit));
        packed[i] = __nv_cvt_float2_to_fp8x2(q_pair, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(dst) = packed_u32;
}

__device__ __forceinline__ unsigned subwarpMask16() {
    const int warp_lane = threadIdx.x & 31;
    return 0xffffu << (warp_lane & 16);
}

__device__ __forceinline__ float subwarpReduceMax16(float val, unsigned mask) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8, kThreadsPerGroup));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4, kThreadsPerGroup));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2, kThreadsPerGroup));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1, kThreadsPerGroup));
    return val;
}

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void dsv4_fused_rmsnorm_fp8_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ output_q,
    uint32_t* __restrict__ output_s,
    int   m,
    int   k,
    int   num_groups,
    int   scale_stride,
    float norm_eps,
    float quant_eps,
    float min_8bit,
    float max_8bit) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const __nv_bfloat16* row_input = input + static_cast<int64_t>(row) * k;
    __nv_fp8_e4m3*       row_out   = output_q + static_cast<int64_t>(row) * k;

    float local_sumsq = 0.0f;
    for (int col = threadIdx.x * kBf16VecElems; col < k; col += blockDim.x * kBf16VecElems) {
        const uint4 packed = *reinterpret_cast<const uint4*>(row_input + col);
        const auto* vals   = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < kBf16VecElems; ++i) {
            const float x = bf16ToFloat(vals[i]);
            local_sumsq += x * x;
        }
    }
    const float sumsq = blockReduceSum(local_sumsq);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(k) + norm_eps);

    constexpr int groups_per_wave = BLOCK_SIZE / kThreadsPerGroup;
    const int local_group = threadIdx.x / kThreadsPerGroup;
    const int lane_id     = threadIdx.x % kThreadsPerGroup;
    for (int base_group = 0; base_group < num_groups; base_group += groups_per_wave) {
        const int g = base_group + local_group;
        if (g >= num_groups) {
            continue;
        }

        float     local_absmax = quant_eps;
        const int begin        = g * kGroupSize;
        const int col = begin + lane_id * kBf16VecElems;
        float     y_vals[kBf16VecElems];
        const uint4 input_vec  = *reinterpret_cast<const uint4*>(row_input + col);
        const uint4 weight_vec = *reinterpret_cast<const uint4*>(weight + col);
        const auto* input_vals = reinterpret_cast<const __nv_bfloat16*>(&input_vec);
        const auto* weight_vals = reinterpret_cast<const __nv_bfloat16*>(&weight_vec);
#pragma unroll
        for (int i = 0; i < kBf16VecElems; ++i) {
            const float x = bf16ToFloat(input_vals[i]);
            const float w = bf16ToFloat(weight_vals[i]);
            const float y = roundToBf16Float(x * inv_rms * w);
            y_vals[i]     = y;
            local_absmax  = fmaxf(local_absmax, fabsf(y));
        }

        const unsigned subwarp_mask = subwarpMask16();
        const float    absmax       = subwarpReduceMax16(local_absmax, subwarp_mask);
        float       scale  = absmax / max_8bit;
        if (lane_id == 0) {
            const uint8_t packed_scale = scaleToUe8m0(absmax / max_8bit);
            writeColumnMajorUe8m0Scale(output_s, row, g, scale_stride, packed_scale);
            scale = ue8m0ToScale(packed_scale);
        }
        scale                 = __shfl_sync(subwarp_mask, scale, 0, kThreadsPerGroup);
        const float inv_scale = 1.0f / scale;
        storeFp8x8(row_out + col, y_vals, inv_scale, min_8bit, max_8bit);
    }
}

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void dsv4_fused_rmsnorm_bf16_fp8_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output_y,
    __nv_fp8_e4m3* __restrict__ output_q,
    uint32_t* __restrict__ output_s,
    int   m,
    int   k,
    int   num_groups,
    int   scale_stride,
    float norm_eps,
    float quant_eps,
    float min_8bit,
    float max_8bit) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const __nv_bfloat16* row_input = input + static_cast<int64_t>(row) * k;
    __nv_bfloat16*       row_y     = output_y + static_cast<int64_t>(row) * k;
    __nv_fp8_e4m3*       row_out   = output_q + static_cast<int64_t>(row) * k;

    float local_sumsq = 0.0f;
    for (int col = threadIdx.x * kBf16VecElems; col < k; col += blockDim.x * kBf16VecElems) {
        const uint4 packed = *reinterpret_cast<const uint4*>(row_input + col);
        const auto* vals   = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < kBf16VecElems; ++i) {
            const float x = bf16ToFloat(vals[i]);
            local_sumsq += x * x;
        }
    }
    const float sumsq   = blockReduceSum(local_sumsq);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(k) + norm_eps);

    constexpr int groups_per_wave = BLOCK_SIZE / kThreadsPerGroup;
    const int local_group         = threadIdx.x / kThreadsPerGroup;
    const int lane_id             = threadIdx.x % kThreadsPerGroup;
    for (int base_group = 0; base_group < num_groups; base_group += groups_per_wave) {
        const int g = base_group + local_group;
        if (g >= num_groups) {
            continue;
        }

        float     local_absmax = quant_eps;
        const int begin        = g * kGroupSize;
        const int col          = begin + lane_id * kBf16VecElems;
        float     y_vals[kBf16VecElems];
        __nv_bfloat16 y_bf16_vals[kBf16VecElems];
        const uint4 input_vec   = *reinterpret_cast<const uint4*>(row_input + col);
        const uint4 weight_vec  = *reinterpret_cast<const uint4*>(weight + col);
        const auto* input_vals  = reinterpret_cast<const __nv_bfloat16*>(&input_vec);
        const auto* weight_vals = reinterpret_cast<const __nv_bfloat16*>(&weight_vec);
#pragma unroll
        for (int i = 0; i < kBf16VecElems; ++i) {
            const float         x      = bf16ToFloat(input_vals[i]);
            const float         w      = bf16ToFloat(weight_vals[i]);
            const __nv_bfloat16 y_bf16 = __float2bfloat16(x * inv_rms * w);
            const float         y      = bf16ToFloat(y_bf16);
            y_bf16_vals[i]             = y_bf16;
            y_vals[i]                  = y;
            local_absmax               = fmaxf(local_absmax, fabsf(y));
        }
        *reinterpret_cast<uint4*>(row_y + col) = *reinterpret_cast<const uint4*>(y_bf16_vals);

        const unsigned subwarp_mask = subwarpMask16();
        const float    absmax       = subwarpReduceMax16(local_absmax, subwarp_mask);
        float       scale  = absmax / max_8bit;
        if (lane_id == 0) {
            const uint8_t packed_scale = scaleToUe8m0(absmax / max_8bit);
            writeColumnMajorUe8m0Scale(output_s, row, g, scale_stride, packed_scale);
            scale = ue8m0ToScale(packed_scale);
        }
        scale                 = __shfl_sync(subwarp_mask, scale, 0, kThreadsPerGroup);
        const float inv_scale = 1.0f / scale;
        storeFp8x8(row_out + col, y_vals, inv_scale, min_8bit, max_8bit);
    }
}

int blockSizeForShape(int64_t m, int64_t k) {
    if (m < 512 && k >= 7168) {
        return 1024;
    }
    if (m < 512 && k >= 4096) {
        return 512;
    }
    if (k == 1536 && m < 512) {
        return 256;
    }
    if (k <= 1536) {
        return 128;
    }
    return 256;
}

bool isSupportedK(int64_t k) {
    switch (k) {
        case 1024:
        case 1536:
        case 2048:
        case 3072:
        case 4096:
        case 7168:
            return true;
        default:
            return false;
    }
}

void checkFiniteQuantContract(double norm_eps, double quant_eps, double min_8bit, double max_8bit) {
    TORCH_CHECK(std::isfinite(norm_eps) && norm_eps > 0.0, "norm_eps must be finite and > 0, got ", norm_eps);
    TORCH_CHECK(std::isfinite(quant_eps) && quant_eps >= 0.0, "quant_eps must be finite and >= 0, got ", quant_eps);
    TORCH_CHECK(std::isfinite(min_8bit) && std::isfinite(max_8bit) && min_8bit < 0.0 && max_8bit > 0.0
                    && min_8bit < max_8bit,
                "invalid FP8 quant range: min=",
                min_8bit,
                " max=",
                max_8bit);
}

void checkLaunchRange(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
                name,
                " is too large for dsv4 fused RMSNorm+FP8 quant launch: ",
                value);
}

void checkRowMajor2d(const torch::Tensor& tensor, const char* name, int64_t expected_m, int64_t expected_k) {
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2D [M, K]");
    TORCH_CHECK(tensor.size(0) == expected_m,
                name,
                " dim0 must be M, got ",
                tensor.size(0),
                " vs ",
                expected_m);
    TORCH_CHECK(tensor.size(1) == expected_k,
                name,
                " dim1 must be K, got ",
                tensor.size(1),
                " vs ",
                expected_k);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major [M, K]");
    TORCH_CHECK(tensor.stride(1) == 1,
                name,
                " stride(1) must be 1 for row-major [M, K], got ",
                tensor.stride(1));
    if (expected_m > 0 && expected_k > 0) {
        TORCH_CHECK(tensor.stride(0) == expected_k,
                    name,
                    " stride(0) must equal K for row-major [M, K], got ",
                    tensor.stride(0),
                    " vs ",
                    expected_k);
    }
}

void checkContiguous1d(const torch::Tensor& tensor, const char* name, int64_t expected_k) {
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D [K]");
    TORCH_CHECK(tensor.size(0) == expected_k,
                name,
                " size must match K, got ",
                tensor.size(0),
                " vs ",
                expected_k);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous 1D [K]");
    TORCH_CHECK(tensor.stride(0) == 1, name, " stride(0) must be 1, got ", tensor.stride(0));
}

void checkAligned16(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                name,
                " data_ptr must be 16-byte aligned for vectorized bf16 loads");
}

void checkUe8m0ScaleLayout(const torch::Tensor& output_s, int64_t m, int64_t num_groups) {
    const int64_t aligned_m      = ceilAlign(m, 4);
    const int64_t packed_groups  = ceilAlign(num_groups, 4) / 4;
    TORCH_CHECK(output_s.dim() == 2, "output_s must be 2D [M, ceil_align(K / 128, 4) / 4]");
    TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Int, "output_s must be int32 UE8M0 scale");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output_s.data_ptr()) % alignof(uint32_t) == 0,
                "output_s data_ptr must be uint32 aligned");
    TORCH_CHECK(output_s.size(0) == m,
                "output_s dim0 must match dynamic M, got ",
                output_s.size(0),
                " vs ",
                m);
    TORCH_CHECK(output_s.size(1) == packed_groups,
                "output_s dim1 must match packed K groups, got ",
                output_s.size(1),
                " vs ",
                packed_groups);
    TORCH_CHECK(output_s.stride(0) == 1,
                "output_s must use column-major UE8M0 layout with stride(0)=1, got ",
                output_s.stride(0));
    if (m > 0 && packed_groups > 0) {
        TORCH_CHECK(output_s.stride(1) == aligned_m,
                    "output_s must use TMA-aligned UE8M0 layout with stride(1)=ceil_align(M,4), got ",
                    output_s.stride(1),
                    " vs ",
                    aligned_m);
    }
    TORCH_CHECK(output_s.stride(0) > 0 && output_s.stride(1) >= 0,
                "output_s strides must be non-negative, got stride(0)=",
                output_s.stride(0),
                " stride(1)=",
                output_s.stride(1));
}

}  // namespace

void fused_rmsnorm_fp8_quant(torch::Tensor input,
                             torch::Tensor weight,
                             torch::Tensor output_q,
                             torch::Tensor output_s,
                             double        norm_eps,
                             double        quant_eps,
                             double        min_8bit,
                             double        max_8bit) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output_q);
    CHECK_CUDA(output_s);

    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [K]");
    TORCH_CHECK(output_q.dim() == 2, "output_q must be 2D [M, K]");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16, "weight must be bfloat16");
    TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be float8_e4m3fn");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must be on the same CUDA device");
    TORCH_CHECK(input.device() == output_q.device(), "output_q must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == output_s.device(), "output_s must be on the same CUDA device as input");

    const int64_t m = input.size(0);
    const int64_t k = input.size(1);
    checkRowMajor2d(input, "input", m, k);
    checkRowMajor2d(output_q, "output_q", m, k);
    checkContiguous1d(weight, "weight", k);
    TORCH_CHECK(k % kGroupSize == 0, "K must be divisible by 128");
    TORCH_CHECK(k / kGroupSize <= kMaxGroups, "K has too many FP8 groups");
    TORCH_CHECK(isSupportedK(k), "unsupported K for dsv4 fused RMSNorm+FP8 quant: ", k);
    checkFiniteQuantContract(norm_eps, quant_eps, min_8bit, max_8bit);
    checkLaunchRange(m, "M");
    checkLaunchRange(k, "K");
    checkAligned16(input, "input");
    checkAligned16(weight, "weight");
    checkUe8m0ScaleLayout(output_s, m, k / kGroupSize);
    checkLaunchRange(output_s.stride(1), "output_s stride(1)");

    if (m == 0) {
        return;
    }

    const int scale_stride = output_s.stride(output_s.dim() - 1);
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_FUSED_RMSNORM_FP8_QUANT(BLOCK_SIZE)                                                                     \
    dsv4_fused_rmsnorm_fp8_quant_kernel<BLOCK_SIZE><<<static_cast<int>(m), BLOCK_SIZE, 0, stream>>>(                   \
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),                                                       \
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),                                                      \
        reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),                                                          \
        reinterpret_cast<uint32_t*>(output_s.data_ptr()),                                                               \
        static_cast<int>(m),                                                                                            \
        static_cast<int>(k),                                                                                            \
        static_cast<int>(k / kGroupSize),                                                                               \
        scale_stride,                                                                                                   \
        static_cast<float>(norm_eps),                                                                                   \
        static_cast<float>(quant_eps),                                                                                  \
        static_cast<float>(min_8bit),                                                                                   \
        static_cast<float>(max_8bit))

    const int block_size = blockSizeForShape(m, k);
    if (block_size == 128) {
        LAUNCH_FUSED_RMSNORM_FP8_QUANT(128);
    } else if (block_size == 512) {
        LAUNCH_FUSED_RMSNORM_FP8_QUANT(512);
    } else if (block_size == 1024) {
        LAUNCH_FUSED_RMSNORM_FP8_QUANT(1024);
    } else {
        LAUNCH_FUSED_RMSNORM_FP8_QUANT(256);
    }
#undef LAUNCH_FUSED_RMSNORM_FP8_QUANT
}

void fused_rmsnorm_bf16_fp8_quant(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor output_y,
                                  torch::Tensor output_q,
                                  torch::Tensor output_s,
                                  double        norm_eps,
                                  double        quant_eps,
                                  double        min_8bit,
                                  double        max_8bit) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output_y);
    CHECK_INPUT(output_q);
    CHECK_CUDA(output_s);

    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [K]");
    TORCH_CHECK(output_y.dim() == 2, "output_y must be 2D [M, K]");
    TORCH_CHECK(output_q.dim() == 2, "output_q must be 2D [M, K]");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16, "weight must be bfloat16");
    TORCH_CHECK(output_y.scalar_type() == at::ScalarType::BFloat16, "output_y must be bfloat16");
    TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be float8_e4m3fn");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must be on the same CUDA device");
    TORCH_CHECK(input.device() == output_y.device(), "output_y must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == output_q.device(), "output_q must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == output_s.device(), "output_s must be on the same CUDA device as input");

    const int64_t m = input.size(0);
    const int64_t k = input.size(1);
    checkRowMajor2d(input, "input", m, k);
    checkRowMajor2d(output_y, "output_y", m, k);
    checkRowMajor2d(output_q, "output_q", m, k);
    checkContiguous1d(weight, "weight", k);
    TORCH_CHECK(k % kGroupSize == 0, "K must be divisible by 128");
    TORCH_CHECK(k / kGroupSize <= kMaxGroups, "K has too many FP8 groups");
    TORCH_CHECK(isSupportedK(k), "unsupported K for dsv4 fused RMSNorm BF16+FP8 quant: ", k);
    checkFiniteQuantContract(norm_eps, quant_eps, min_8bit, max_8bit);
    checkLaunchRange(m, "M");
    checkLaunchRange(k, "K");
    checkAligned16(input, "input");
    checkAligned16(weight, "weight");
    checkAligned16(output_y, "output_y");
    checkUe8m0ScaleLayout(output_s, m, k / kGroupSize);
    checkLaunchRange(output_s.stride(1), "output_s stride(1)");

    if (m == 0) {
        return;
    }

    const int scale_stride = output_s.stride(output_s.dim() - 1);
    auto      stream       = at::cuda::getCurrentCUDAStream();
#define LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT(BLOCK_SIZE)                                                               \
    dsv4_fused_rmsnorm_bf16_fp8_quant_kernel<BLOCK_SIZE>                                                              \
        <<<static_cast<int>(m), BLOCK_SIZE, 0, stream>>>(                                                              \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),                                                  \
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),                                                 \
            reinterpret_cast<__nv_bfloat16*>(output_y.data_ptr()),                                                     \
            reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),                                                     \
            reinterpret_cast<uint32_t*>(output_s.data_ptr()),                                                          \
            static_cast<int>(m),                                                                                       \
            static_cast<int>(k),                                                                                       \
            static_cast<int>(k / kGroupSize),                                                                          \
            scale_stride,                                                                                              \
            static_cast<float>(norm_eps),                                                                              \
            static_cast<float>(quant_eps),                                                                             \
            static_cast<float>(min_8bit),                                                                              \
            static_cast<float>(max_8bit))

    const int block_size = blockSizeForShape(m, k);
    if (block_size == 128) {
        LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT(128);
    } else if (block_size == 512) {
        LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT(512);
    } else if (block_size == 1024) {
        LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT(1024);
    } else {
        LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT(256);
    }
#undef LAUNCH_FUSED_RMSNORM_BF16_FP8_QUANT
}

}  // namespace rtp_llm

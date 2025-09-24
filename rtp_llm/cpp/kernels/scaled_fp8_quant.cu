#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant_utils.h"
#include "rtp_llm/cpp/kernels/vec_dtypes.cuh"
#include "rtp_llm/cpp/cuda/launch_utils.h"
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <c10/util/Float8_e4m3fn.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>

#ifdef ENABLE_FP8
using namespace tensorrt_llm::common;
#endif

namespace rtp_llm {

#ifdef ENABLE_FP8
// Helper functions and kernels extracted from cuda_fp8_utils
constexpr int CTA_SIZE = 256;
static constexpr int kWarpSize = 32;

template<bool QUANTIZE>
__inline__ __device__ float scale(float a, float b) {
    return QUANTIZE ? a / b : a * b;
}

template<QuantizeMode QUANTIZE_MODE, bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda) {
    for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x) {
        if (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL) {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i % lda])));
        } else if (QUANTIZE_MODE == QuantizeMode::PER_TOKEN) {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i / lda])));
        } else if (QUANTIZE_MODE == QuantizeMode::PER_TENSOR) {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[0])));
        }
    }
}

template<typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(T_OUT*       output,
                          T_S const*   input_scale,
                          T_IN const*  input,
                          int64_t      numel,
                          int64_t      lda,
                          QuantizeMode quantize_mode,
                          cudaStream_t stream) {
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    if (quantize_mode == QuantizeMode::PER_CHANNEL) {
        scaleMatrix<QuantizeMode::PER_CHANNEL, true>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    } else if (quantize_mode == QuantizeMode::PER_TOKEN) {
        scaleMatrix<QuantizeMode::PER_TOKEN, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    } else if (quantize_mode == QuantizeMode::PER_TENSOR) {
        scaleMatrix<QuantizeMode::PER_TENSOR, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    check_cuda_error();
}

// Template instantiations for invokeQuantizeMatrix
#ifdef ENABLE_FP8
template void invokeQuantizeMatrix<__nv_fp8_e4m3, float, float>(__nv_fp8_e4m3* output, float const* input_scale, float const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
template void invokeQuantizeMatrix<__nv_fp8_e4m3, float, half>(__nv_fp8_e4m3* output, float const* input_scale, half const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
template void invokeQuantizeMatrix<__nv_fp8_e4m3, half, half>(__nv_fp8_e4m3* output, half const* input_scale, half const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeQuantizeMatrix<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>(__nv_fp8_e4m3* output, __nv_bfloat16 const* input_scale, __nv_bfloat16 const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
template void invokeQuantizeMatrix<__nv_fp8_e4m3, float, __nv_bfloat16>(__nv_fp8_e4m3* output, float const* input_scale, __nv_bfloat16 const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
template void invokeQuantizeMatrix<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(__nv_bfloat16* output, __nv_bfloat16 const* input_scale, __nv_fp8_e4m3 const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
template void invokeQuantizeMatrix<__nv_bfloat16, float, __nv_fp8_e4m3>(__nv_bfloat16* output, float const* input_scale, __nv_fp8_e4m3 const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);
#endif
#endif

// Helper functions for invokeComputeFP8Quantize128 and computeFP8ActivationAndQuantize
inline __device__ __nv_bfloat16 max_abs_op(bf16_4_t v) {
    return cuda_max(cuda_max<__nv_bfloat16>(cuda_abs(v.x)), cuda_max<__nv_bfloat16>(cuda_abs(v.y)));
}

inline __device__ __nv_bfloat16 max_abs_op(bf16_8_t v) {
    return cuda_max<__nv_bfloat16>(max_abs_op(bf16_4_t{v.x, v.y}), max_abs_op(bf16_4_t{v.z, v.w}));
}

inline __device__ __nv_bfloat162 mul(__nv_bfloat162 v, __nv_bfloat16 scale) {
    return bf16hmul2(v, bf162bf162(scale));
}

inline __device__ bf16_4_t mul(bf16_4_t v, __nv_bfloat16 scale) {
    bf16_4_t n;
    n.x = mul(v.x, scale);
    n.y = mul(v.y, scale);
    return n;
}

inline __device__ bf16_8_t mul(bf16_8_t v, __nv_bfloat16 scale) {
    bf16_8_t n;
    n.x = mul(v.x, scale);
    n.y = mul(v.y, scale);
    n.z = mul(v.z, scale);
    n.w = mul(v.w, scale);
    return n;
}

inline __device__ void convert_to_fp8(fp8_4_t* v, const bf16_4_t u) {
    v[0] = fp8_4_t(u.x, u.y);
}

inline __device__ void convert_to_fp8(fp8_8_t* v, const bf16_8_t u) {
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}

static __device__ __forceinline__ __nv_bfloat162 silu(const __nv_bfloat162& val) {
    return make_bfloat162((__nv_bfloat16)((float)val.x / (1.0f + __expf((float)-val.x))),
                          (__nv_bfloat16)((float)val.y / (1.0f + __expf((float)-val.y))));
}

inline __device__ bf16_8_t act_and_mul(bf16_8_t v, bf16_8_t u) {
    bf16_8_t n;
    n.x = bf16hmul2(silu(v.x), u.x);
    n.y = bf16hmul2(silu(v.y), u.y);
    n.z = bf16hmul2(silu(v.z), u.z);
    n.w = bf16hmul2(silu(v.w), u.w);
    return n;
}

template<typename T_S, bool COL_MAJOR_SCALE, int ELEM_PER_THREAD>
__global__ void computeFP8Quantize128Kernel(__nv_fp8_e4m3*       fp8_output,
                                            T_S*                 quant_ptr,
                                            const __nv_bfloat16* weights,
                                            const int64_t        dim0,
                                            const int64_t        dim1,
                                            const int64_t        size) {
    const int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    using InputElem          = typename packed_type<__nv_bfloat16, ELEM_PER_THREAD>::type;
    using OutputElem         = typename packed_type<__nv_fp8_e4m3, ELEM_PER_THREAD>::type;
    auto weights_vec         = reinterpret_cast<InputElem const*>(weights);
    auto output_vec          = reinterpret_cast<OutputElem*>(fp8_output);

    if (global_idx * ELEM_PER_THREAD >= size) {
        return;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    auto                 w8              = weights_vec[global_idx];
    float                scale           = cuda_max((float)1e-4, (float)max_abs_op(w8));
    static constexpr int THREADS_PER_ROW = 128 / ELEM_PER_THREAD;
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        scale = max(scale, __shfl_xor_sync(0xFFFFFFFF, scale, mask, THREADS_PER_ROW));
    }
      scale = scale / tensorrt_llm::common::FP8_E4M3_MAX;
    w8    = mul(w8, (__nv_bfloat16)(1 / scale));
    convert_to_fp8(output_vec + global_idx, w8);
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        if constexpr (COL_MAJOR_SCALE) {
            const int64_t now_idx               = global_idx / THREADS_PER_ROW;
            const int64_t row_idx               = now_idx / dim1;
            const int64_t col_idx               = now_idx % dim1;
            quant_ptr[col_idx * dim0 + row_idx] = scale;
        } else {
            quant_ptr[global_idx / THREADS_PER_ROW] = scale;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void invokeComputeFP8Quantize128(__nv_fp8_e4m3*       fp8_output,
                                 float*               quant_ptr,
                                 const __nv_bfloat16* weights,
                                 const int64_t        dim0,
                                 const int64_t        dim1,
                                 const int64_t        size,
                                 bool                 col_major_scale,
                                 cudaStream_t         stream) {
    RTP_LLM_CHECK(dim1 % 128 == 0);
    static constexpr int ELEM_PER_THREAD = 8;
    const int            num_per_grid    = CTA_SIZE / (128 / ELEM_PER_THREAD);
    dim3                 grid((size / 128 + num_per_grid - 1) / num_per_grid);
    dim3                 block(CTA_SIZE);
    if (col_major_scale) {
        LAUNCH_KERNEL_WITH_PDL((computeFP8Quantize128Kernel<float, true, ELEM_PER_THREAD>),
                               grid,
                               block,
                               0,
                               stream,
                               fp8_output,
                               quant_ptr,
                               weights,
                               dim0,
                               dim1 / 128,
                               size);
    } else {
        LAUNCH_KERNEL_WITH_PDL((computeFP8Quantize128Kernel<float, false, ELEM_PER_THREAD>),
                               grid,
                               block,
                               0,
                               stream,
                               fp8_output,
                               quant_ptr,
                               weights,
                               dim0,
                               dim1,
                               size);
    }
}

template<typename T_S, bool COL_MAJOR_SCALE, int ELEM_PER_THREAD>
__global__ void computeFP8ActivationAndQuantizeKernel(__nv_fp8_e4m3*       fp8_output,
                                                      T_S*                 quant_ptr,
                                                      const __nv_bfloat16* gate_up_output,
                                                      const int64_t        dim0,
                                                      const int64_t        dim1) {
    const int64_t global_idx  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t size        = dim0 * dim1;
    const int64_t padded_dim0 = (dim0 + 63) / 64 * 64;
    if (global_idx * ELEM_PER_THREAD >= size) {
        return;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    const int64_t row_idx    = global_idx * ELEM_PER_THREAD / dim1;
    const int64_t col_idx    = global_idx * ELEM_PER_THREAD % dim1;
    const int64_t row_stride = dim1 * 2;
    using InputElem          = typename packed_type<__nv_bfloat16, ELEM_PER_THREAD>::type;
    using OutputElem         = typename packed_type<__nv_fp8_e4m3, ELEM_PER_THREAD>::type;
    auto weights_vec         = reinterpret_cast<InputElem const*>(gate_up_output);
    auto output_vec          = reinterpret_cast<OutputElem*>(fp8_output);

    auto gate8 = weights_vec[(row_idx * row_stride + col_idx) / ELEM_PER_THREAD],
         up8   = weights_vec[(row_idx * row_stride + col_idx + dim1) / ELEM_PER_THREAD];

    auto                 w8              = act_and_mul(gate8, up8);
    float                scale           = cuda_max((float)1e-4, (float)max_abs_op(w8));
    static constexpr int THREADS_PER_ROW = 128 / ELEM_PER_THREAD;
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        scale = max(scale, __shfl_xor_sync(0xFFFFFFFF, scale, mask, THREADS_PER_ROW));
    }
      scale = scale / tensorrt_llm::common::FP8_E4M3_MAX;
    w8    = mul(w8, (__nv_bfloat16)(1 / scale));
    convert_to_fp8(output_vec + global_idx, w8);
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        const int64_t dim                          = dim1 / 128;
        const int64_t now_idx                      = global_idx / THREADS_PER_ROW;
        const int64_t row_idx                      = now_idx / dim;
        const int64_t col_idx                      = now_idx % dim;
        quant_ptr[col_idx * padded_dim0 + row_idx] = scale;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void computeFP8ActivationAndQuantize(__nv_fp8_e4m3*       fp8_output,
                                     float*               quant_ptr,
                                     const __nv_bfloat16* weights,
                                     const int64_t        dim0,
                                     const int64_t        dim1,
                                     cudaStream_t         stream) {
    RTP_LLM_CHECK(dim1 % 128 == 0);
    static constexpr int ELEM_PER_THREAD = 8;
    const int            num_per_grid    = CTA_SIZE / (128 / ELEM_PER_THREAD);
    const int            size            = dim0 * dim1;
    dim3                 grid((size / 128 + num_per_grid - 1) / num_per_grid);
    dim3                 block(CTA_SIZE);
    LAUNCH_KERNEL_WITH_PDL((computeFP8ActivationAndQuantizeKernel<float, true, ELEM_PER_THREAD>),
                           grid,
                           block,
                           0,
                           stream,
                           fp8_output,
                           quant_ptr,
                           weights,
                           dim0,
                           dim1);
}

template<typename T>
__global__ void
per_tensor_absmax_kernel(const T* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
    float        max_value = 0.0f;
    unsigned int tid       = threadIdx.x;
    unsigned int gid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int    grid_size = blockDim.x * gridDim.x;

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_t                 = rtp_llm::vec_t<T, vec_size>;

    const int32_t num_vec_elems = num_elements / vec_size;

    for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
        vec_t input_vec;
        input_vec.cast_load(input + i * vec_size);

#pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }

    const int32_t remaining_start = num_vec_elems * vec_size;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val = static_cast<float>(input[idx]);
        max_value = fmaxf(max_value, fabsf(val));
    }

    max_value = blockReduceMax(max_value);

    if (tid == 0) {
          atomicMaxFloat(output_s, max_value / tensorrt_llm::common::FP8_E4M3_MAX);
    }
}

template<typename T, typename DST_DTYPE>
__global__ void per_tensor_quant_fp8_kernel(const T* __restrict__ input,
                                            DST_DTYPE* __restrict__ output,
                                            const float* __restrict__ scale,
                                            const int64_t num_elements) {
    const int   gid        = blockIdx.x * blockDim.x + threadIdx.x;
    const int   grid_size  = blockDim.x * gridDim.x;
    float       safe_scale = fmax(1e-9, *scale);
    const float scale_val  = 1.0f / safe_scale;

    // We want to store 128 bits of data at a time. 16 = 128 / 8 bits
    // Load is already vectorized, so 16 elements work for T.
    const uint32_t VEC_SIZE = 16;
    using vec_t             = rtp_llm::vec_t<T, VEC_SIZE>;

    const int32_t num_vec_elems = num_elements / VEC_SIZE;

    for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
        vec_t input_vec;
        input_vec.cast_load(input + i * VEC_SIZE);

        DST_DTYPE output_arr[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            float val     = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, tensorrt_llm::common::FP8_E4M3_MAX), -tensorrt_llm::common::FP8_E4M3_MAX);
            output_arr[j] = static_cast<DST_DTYPE>(val);
        }
        *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
    }

    const int32_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val   = fmax(-tensorrt_llm::common::FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, tensorrt_llm::common::FP8_E4M3_MAX));
        output[idx] = static_cast<DST_DTYPE>(val);
    }
}

// adapted from sglang
// ---------------------------------------------------------------------------
// 1. Warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template<typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void per_token_quant_fp8_kernel(const T* __restrict__ input,
                                           DST_DTYPE* __restrict__ output_q,
                                           float* __restrict__ output_s,
                                           const int64_t hidden_dim,
                                           const int64_t num_tokens) {
    const int warp_id  = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
    const int lane_id  = threadIdx.x & (kWarpSize - 1);  // 0‑31
    const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
    if (token_id >= num_tokens)
        return;

    // Global tensors for this token
    const T*   token_input  = input + token_id * hidden_dim;
    DST_DTYPE* token_output = output_q + token_id * hidden_dim;
    float*     token_scale  = output_s + token_id;

    //
    // Pass-1: Perform a warp reduce to find the max_value of a token's hidden_dim
    //
    float max_value             = 0.f;
    using vec_t                 = rtp_llm::vec_t<T, kVecSize>;
    const int32_t num_vec_elems = hidden_dim / kVecSize;

    for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
            max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
        }
    }

    float warp_max = warpReduceMax(max_value);

    __shared__ float scale;
    scale = warp_max / tensorrt_llm::common::FP8_E4M3_MAX;
    // Broadcast scale
    if (lane_id == 0) {
        token_scale[0] = scale;
    }
    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    //
    // Pass-2: quantize and write back
    //
    for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);
        DST_DTYPE output_arr[kVecSize];
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
            float val     = static_cast<float>(input_vec[j]) * scale_inv;
            val           = fmaxf(fminf(val, tensorrt_llm::common::FP8_E4M3_MAX), -tensorrt_llm::common::FP8_E4M3_MAX);
            output_arr[j] = static_cast<DST_DTYPE>(val);
        }
        if constexpr (kVecSize == 16) {
            *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
        } else {
            // Use element-wise copy for vector size 8 to ensure correctness
            for (int k = 0; k < kVecSize; ++k) {
                token_output[i * kVecSize + k] = output_arr[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template<typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(const T* __restrict__ input,
                                                       DST_DTYPE* __restrict__ output_q,
                                                       float* __restrict__ output_s,
                                                       const int64_t hidden_dim,
                                                       const int64_t num_tokens) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    const int tid       = threadIdx.x;
    const int block_dim = blockDim.x;

    const T*   token_input  = input + token_idx * hidden_dim;
    DST_DTYPE* token_output = output_q + token_idx * hidden_dim;

    float max_value = 0.0f;

    // Use template parameter for vector size
    using vec_t                 = rtp_llm::vec_t<T, kVecSize>;
    const int32_t num_vec_elems = hidden_dim / kVecSize;

    // Find max using vectorized loads
    for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }

    max_value = blockReduceMax(max_value);

    __shared__ float scale;
    if (tid == 0) {
        scale               = max_value / tensorrt_llm::common::FP8_E4M3_MAX;
        output_s[token_idx] = scale;
    }
    __syncthreads();

    const float scale_inv = 1.0f / scale;

    // Quantize using vectorized loads
    for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

        DST_DTYPE output_arr[kVecSize];
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
            float val     = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, tensorrt_llm::common::FP8_E4M3_MAX), -tensorrt_llm::common::FP8_E4M3_MAX);
            output_arr[j] = static_cast<DST_DTYPE>(val);
        }

        if constexpr (kVecSize == 16) {
            *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
        } else {
            // Use element-wise copy for vector size 8 to ensure correctness
            for (int k = 0; k < kVecSize; ++k) {
                token_output[i * kVecSize + k] = output_arr[k];
            }
        }
    }
}

void per_tensor_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s, bool is_static) {
    CHECK_INPUT(input);
    CHECK_INPUT(output_q);
    CHECK_INPUT(output_s);
    if (is_static) {
        CHECK_EQ(output_s.numel(), 1);
    }

    const int block_size   = 256;
    const int num_elements = input.numel();
    assert(num_elements % (16 / input.element_size()) == 0);
    const int num_blocks = min((num_elements + block_size - 1) / block_size, 1024);

    dim3 grid(num_blocks);
    dim3 block(block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
        if (is_static == false) {
            per_tensor_absmax_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(input.data_ptr()), static_cast<float*>(output_s.data_ptr()), num_elements);
        }
        per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3>
            <<<grid, block, 0, stream>>>(static_cast<scalar_t*>(input.data_ptr()),
                                         static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                         static_cast<float*>(output_s.data_ptr()),
                                         num_elements);
        return true;
    });
}

void per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s) {
    CHECK_INPUT(input);
    CHECK_INPUT(output_q);
    CHECK_INPUT(output_s);
    const auto    input_sizes = input.sizes();
    const int64_t num_tokens  = input_sizes[0];
    const int64_t hidden_dim  = input_sizes[1];
    TORCH_CHECK(hidden_dim % 8 == 0, "Hidden dimension must be divisible by 8, but got ", hidden_dim);
    cudaStream_t stream          = at::cuda::getCurrentCUDAStream();
    const int    sm_count        = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int    TOKENS_PER_CTA  = 8;
    const bool   use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
    const bool   use_vec16       = (hidden_dim % 16 == 0);

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
        if (use_warp_kernel) {
            // -------- warp‑local ---------------------------------------------------
            constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;  // 256
            dim3          grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
            dim3          block(THREADS);

            if (use_vec16) {
                per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 16>
                    <<<grid, block, 0, stream>>>(static_cast<const scalar_t*>(input.data_ptr()),
                                                 static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                                 static_cast<float*>(output_s.data_ptr()),
                                                 hidden_dim,
                                                 num_tokens);
            } else {
                per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 8>
                    <<<grid, block, 0, stream>>>(static_cast<const scalar_t*>(input.data_ptr()),
                                                 static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                                 static_cast<float*>(output_s.data_ptr()),
                                                 hidden_dim,
                                                 num_tokens);
            }
        } else {
            constexpr int THREADS = 256;
            dim3          grid(num_tokens);
            dim3          block(THREADS);

            if (use_vec16) {
                per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 16>
                    <<<grid, block, 0, stream>>>(static_cast<const scalar_t*>(input.data_ptr()),
                                                 static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                                 static_cast<float*>(output_s.data_ptr()),
                                                 hidden_dim,
                                                 num_tokens);
            } else {
                per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 8>
                    <<<grid, block, 0, stream>>>(static_cast<const scalar_t*>(input.data_ptr()),
                                                 static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                                 static_cast<float*>(output_s.data_ptr()),
                                                 hidden_dim,
                                                 num_tokens);
            }
        }
        return true;
    });
}

#endif  // ENABLE_FP8

}  // namespace rtp_llm

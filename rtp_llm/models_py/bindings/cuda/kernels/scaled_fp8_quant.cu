#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant_utils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/vec_dtypes.cuh"
#include "rtp_llm/models_py/bindings/cuda/launch_utils.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_fp8_utils.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/version.h>
#if !defined(USE_PPU) || !USE_PPU
#include <cuda.h>
#endif

// The three-compute-kernel path is validated only for this dependency lane.
// Header existence alone does not establish compatibility with other builds.
#if (!defined(USE_PPU) || !USE_PPU) && defined(CUDA_VERSION) && CUDA_VERSION >= 13000 && TORCH_VERSION_MAJOR == 2      \
    && TORCH_VERSION_MINOR == 11
#define RTP_QSA_PREFILL_FP8_AVAILABLE 1
#include <ATen/ops/_scaled_mm.h>
#else
#define RTP_QSA_PREFILL_FP8_AVAILABLE 0
#endif
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#ifdef ENABLE_FP8
using namespace tensorrt_llm::common;
#endif

namespace rtp_llm {

#ifdef ENABLE_FP8
// Helper functions and kernels extracted from cuda_fp8_utils
constexpr int        CTA_SIZE  = 256;
static constexpr int kWarpSize = 32;

// Helper functions for invokeComputeFP8Quantize128
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
                                            const int64_t num_elements,
                                            const int64_t output_elements) {
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
            float val     = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, tensorrt_llm::common::FP8_E4M3_MAX),
                             -tensorrt_llm::common::FP8_E4M3_MAX);
            output_arr[j] = static_cast<DST_DTYPE>(val);
        }
        *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
    }

    const int32_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val   = fmax(-tensorrt_llm::common::FP8_E4M3_MAX,
                         fmin(static_cast<float>(input[idx]) * scale_val, tensorrt_llm::common::FP8_E4M3_MAX));
        output[idx] = static_cast<DST_DTYPE>(val);
    }
    // Padded GEMM rows share this launch; the amax above still uses only the
    // real input. Default-sized outputs do not enter the loop.
    for (int64_t idx = num_elements + gid; idx < output_elements; idx += grid_size) {
        output[idx] = static_cast<DST_DTYPE>(0.0f);
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
            float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, tensorrt_llm::common::FP8_E4M3_MAX),
                              -tensorrt_llm::common::FP8_E4M3_MAX);
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
    RTP_LLM_CHECK_WITH_INFO(output_q.numel() >= input.numel(), "FP8 quantized output must cover the input");
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
                                         num_elements,
                                         output_q.numel());
        return true;
    });
}

#if RTP_QSA_PREFILL_FP8_AVAILABLE
namespace {

constexpr int kQsaWidth        = 2560;
constexpr int kQsaPackedWidth  = 3712;
constexpr int kQsaPartialCount = 512;
constexpr int kQsaThreads      = 256;
constexpr int kQsaQuantBlocks  = 1024;
// Quantization kernel element counts are int32, including the M16 padding.
constexpr int kQsaMaxPaddedRows = (std::numeric_limits<int>::max() / kQsaWidth / 16) * 16;
static_assert(kQsaPartialCount == 2 * kQsaThreads, "QSA quant reduction reads two partials per thread");

// Reuse the existing native absmax vector load/reduction, but each CTA owns a
// separate slot instead of atomicMax into an accumulator needing a zero launch.
__global__ void qsa_fp8_partial_amax_kernel(const __nv_bfloat16* __restrict__ input,
                                            float* __restrict__ partial_scales,
                                            const int num_elements) {
    const int     gid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int     grid_size = blockDim.x * gridDim.x;
    constexpr int kVecSize  = 8;  // 16-byte BF16 load, identical to native amax.
    using Vec               = rtp_llm::vec_t<__nv_bfloat16, kVecSize>;
    float maximum           = 0.0f;
    for (int i = gid; i < num_elements / kVecSize; i += grid_size) {
        Vec values;
        values.cast_load(input + i * kVecSize);
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            maximum = fmaxf(maximum, fabsf(static_cast<float>(values[j])));
        }
    }
    maximum = blockReduceMax(maximum);
    if (threadIdx.x == 0) {
        float scale = __fdiv_rn(maximum, 448.0f);
        // Match the original CUDA build's FP32 scale FTZ, including BF16
        // subnormal inputs. Empty CTAs also overwrite their slot with zero.
        scale                      = scale < FLT_MIN ? 0.0f : scale;
        partial_scales[blockIdx.x] = scale;
    }
}

__global__ void qsa_fp8_quantize_from_partials_kernel(const __nv_bfloat16* __restrict__ input,
                                                      const float* __restrict__ partial_scales,
                                                      __nv_fp8_e4m3* __restrict__ output,
                                                      float* __restrict__ row_scales,
                                                      const int num_elements,
                                                      const int output_elements,
                                                      const int padded_rows) {
    // Fixed 512-float scratch: two loads per thread and one block reduction.
    // Broadcast the result once, then each thread converts full 16-element
    // vectors in the same grid-stride layout as per_tensor_quant_fp8_kernel.
    float scale = fmaxf(partial_scales[threadIdx.x], partial_scales[threadIdx.x + kQsaThreads]);
    scale       = blockReduceMax(scale);
    __shared__ float block_scale;
    if (threadIdx.x == 0) {
        block_scale = scale;
    }
    __syncthreads();
    scale                 = block_scale;
    const float inverse   = __fdiv_rn(1.0f, fmaxf(1.0e-9f, scale));
    const int   gid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int   grid_size = blockDim.x * gridDim.x;

    constexpr int kVecSize = 16;
    using Vec              = rtp_llm::vec_t<__nv_bfloat16, kVecSize>;
    for (int i = gid; i < num_elements / kVecSize; i += grid_size) {
        Vec values;
        values.cast_load(input + i * kVecSize);
        alignas(16) __nv_fp8_e4m3 converted[kVecSize];
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            // CUDA fmin/fmax intentionally map NaN (also Inf * 0) to +448.
            const float value = fmaxf(fminf(static_cast<float>(values[j]) * inverse, 448.0f), -448.0f);
            converted[j]      = static_cast<__nv_fp8_e4m3>(value);
        }
        reinterpret_cast<uint4*>(output)[i] = *reinterpret_cast<uint4*>(converted);
    }
    // Real/padded element counts are multiples of 16. Padding never enters
    // the absmax domain and is written in this same kernel with vector stores.
    for (int i = num_elements / kVecSize + gid; i < output_elements / kVecSize; i += grid_size) {
        reinterpret_cast<uint4*>(output)[i] = make_uint4(0, 0, 0, 0);
    }
    for (int row = gid; row < padded_rows; row += grid_size) {
        row_scales[row] = scale;
    }
}

void validate_qsa_prefill_input(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
                "QSA FP8 prefill expects contiguous CUDA BF16 input");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == kQsaWidth && input.size(0) > 0,
                "QSA FP8 prefill expects [positive T, 2560]");
    TORCH_CHECK(input.size(0) <= kQsaMaxPaddedRows,
                "QSA FP8 prefill padded input exceeds int32 quantization element counts");
    TORCH_CHECK(!input.requires_grad(), "QSA FP8 prefill is inference-only");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
                "QSA FP8 prefill input must be 16-byte aligned");
    const auto* props = at::cuda::getDeviceProperties(input.get_device());
    TORCH_CHECK(props->major == 9 && props->minor == 0, "QSA FP8 prefill requires SM90");
}

std::pair<at::Tensor, at::Tensor> qsa_prefill_fp8_quant_impl(const at::Tensor& input) {
    const int rows        = input.size(0);
    const int padded_rows = ((rows + 15) / 16) * 16;
    auto      quantized   = at::empty({padded_rows, kQsaWidth}, input.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto      scales      = at::empty({padded_rows, 1}, input.options().dtype(at::kFloat));
    auto      partials    = at::empty({kQsaPartialCount}, input.options().dtype(at::kFloat));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qsa_fp8_partial_amax_kernel<<<kQsaPartialCount, kQsaThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), partials.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    qsa_fp8_quantize_from_partials_kernel<<<kQsaQuantBlocks, kQsaThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_fp8_e4m3*>(quantized.data_ptr()),
        scales.data_ptr<float>(),
        input.numel(),
        quantized.numel(),
        padded_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scales};
}

}  // namespace
#endif  // RTP_QSA_PREFILL_FP8_AVAILABLE

bool qsa_prefill_fp8_available() {
    return RTP_QSA_PREFILL_FP8_AVAILABLE;
}

std::pair<at::Tensor, at::Tensor> qsa_prefill_fp8_quant(const at::Tensor& input) {
#if RTP_QSA_PREFILL_FP8_AVAILABLE
    validate_qsa_prefill_input(input);
    const c10::cuda::CUDAGuard device_guard(input.device());
    return qsa_prefill_fp8_quant_impl(input);
#else
    TORCH_CHECK(false, "QSA FP8 prefill requires the validated non-PPU CUDA13 / Torch2.11 build");
#endif
}

at::Tensor
qsa_prefill_fp8_project(const at::Tensor& input, const at::Tensor& packed_weight_t, const at::Tensor& column_scales) {
#if RTP_QSA_PREFILL_FP8_AVAILABLE
    validate_qsa_prefill_input(input);
    TORCH_CHECK(packed_weight_t.device() == input.device()
                    && packed_weight_t.scalar_type() == at::ScalarType::Float8_e4m3fn && packed_weight_t.dim() == 2
                    && packed_weight_t.size(0) == kQsaWidth && packed_weight_t.size(1) == kQsaPackedWidth
                    && packed_weight_t.stride(0) == 1 && packed_weight_t.stride(1) == kQsaWidth
                    && !packed_weight_t.requires_grad(),
                "QSA FP8 packed weight must be column-major FP8 [2560,3712] on input device");
    TORCH_CHECK(column_scales.device() == input.device() && column_scales.scalar_type() == at::kFloat
                    && column_scales.dim() == 2 && column_scales.size(0) == 1
                    && column_scales.size(1) == kQsaPackedWidth && column_scales.is_contiguous()
                    && !column_scales.requires_grad(),
                "QSA FP8 column scales must be contiguous float32 [1,3712] on input device");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       quantized = qsa_prefill_fp8_quant_impl(input);
    // The validated CUDA13 / Torch2.11 SM90 dispatcher uses cuBLASLt rowwise
    // scaling. Activation row scales repeat the original global scale; packed
    // weight columns retain their two independent scales. Accurate accumulation
    // is fixed for the full supported domain. Library workspace operations are
    // included in profiler accounting, separately from the three compute kernels.
    return at::_scaled_mm(quantized.first,
                          packed_weight_t,
                          quantized.second,
                          column_scales,
                          std::nullopt,
                          std::nullopt,
                          at::kBFloat16,
                          false);
#else
    // Unsupported builds must reject before allocation or any CUDA launch.
    TORCH_CHECK(false, "QSA FP8 projection requires the validated non-PPU CUDA13 / Torch2.11 build");
#endif
}

QsaFp8ProjectionGuard::QsaFp8ProjectionGuard(const std::array<at::Tensor, 4>& sources) {
#if RTP_QSA_PREFILL_FP8_AVAILABLE
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto& tensor = sources[i];
        TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.layout() == at::kStrided && tensor.is_contiguous()
                        && !tensor.requires_grad(),
                    "QSA FP8 source guard requires contiguous inference-only CUDA tensors");
        TORCH_CHECK(tensor.device() == sources[0].device(), "QSA FP8 sources must share one CUDA device");
        const bool weight = i == 0 || i == 2;
        TORCH_CHECK(weight ? (tensor.scalar_type() == at::ScalarType::Float8_e4m3fn && tensor.dim() == 2
                              && tensor.size(0) == (i == 0 ? 3072 : 640) && tensor.size(1) == kQsaWidth) :
                             (tensor.scalar_type() == at::kFloat && tensor.numel() == 1),
                    "QSA FP8 source guard received unsupported weight or scale metadata");
        auto& snapshot          = sources_[i];
        snapshot.tensor         = tensor;
        snapshot.data_ptr       = tensor.const_data_ptr();
        snapshot.sizes          = tensor.sizes().vec();
        snapshot.strides        = tensor.strides().vec();
        snapshot.storage_offset = tensor.storage_offset();
        snapshot.dtype          = tensor.scalar_type();
        snapshot.device         = tensor.device();
        snapshot.requires_grad  = tensor.requires_grad();
        snapshot.inference      = tensor.is_inference();
        snapshot.version = snapshot.inference ? 0 : tensor.unsafeGetTensorImpl()->version_counter().current_version();
    }
#else
    TORCH_CHECK(false, "QSA FP8 source guard requires the validated non-PPU CUDA13 / Torch2.11 build");
#endif
}

std::optional<at::Tensor> QsaFp8ProjectionGuard::project(const at::Tensor&                input,
                                                         const at::Tensor&                packed_weight_t,
                                                         const at::Tensor&                column_scales,
                                                         const std::array<at::Tensor, 4>& current_sources) const {
#if RTP_QSA_PREFILL_FP8_AVAILABLE
    // Preserve the Python fallback contract without dispatching each input
    // metadata accessor through Python. Shape-count overflow remains a hard
    // error in qsa_prefill_fp8_project, before any allocation or CUDA launch.
    if (!input.defined() || !input.is_cuda() || input.layout() != at::kStrided || input.device() != sources_[0].device
        || input.scalar_type() != at::kBFloat16 || input.dim() != 2 || input.size(1) != kQsaWidth || input.size(0) <= 0
        || !input.is_contiguous() || input.requires_grad()
        || reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 != 0) {
        return std::nullopt;
    }
    // Metadata reads do not synchronize CUDA. Inference tensors still require
    // immutable values, but their mutable metadata is checked on every call.
    // Do not inspect only the retained Tensor handles: Python can swap the
    // TensorImpl of an otherwise unchanged source object.
    for (size_t i = 0; i < current_sources.size(); ++i) {
        const auto& tensor   = current_sources[i];
        const auto& snapshot = sources_[i];
        if (!tensor.defined() || tensor.unsafeGetTensorImpl() != snapshot.tensor.unsafeGetTensorImpl()
            || tensor.layout() != at::kStrided || tensor.scalar_type() != snapshot.dtype
            || tensor.device() != snapshot.device || tensor.requires_grad() != snapshot.requires_grad
            || tensor.is_inference() != snapshot.inference || tensor.storage_offset() != snapshot.storage_offset
            || tensor.sizes() != at::IntArrayRef(snapshot.sizes)
            || tensor.strides() != at::IntArrayRef(snapshot.strides) || tensor.const_data_ptr() != snapshot.data_ptr
            || (!snapshot.inference
                && tensor.unsafeGetTensorImpl()->version_counter().current_version() != snapshot.version)) {
            return std::nullopt;
        }
    }
    // No allocation, quantization or GEMM is allowed before source validation.
    return qsa_prefill_fp8_project(input, packed_weight_t, column_scales);
#else
    TORCH_CHECK(false, "QSA FP8 source guard requires the validated non-PPU CUDA13 / Torch2.11 build");
#endif
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

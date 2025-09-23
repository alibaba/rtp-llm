#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant_utils.h"
#include "rtp_llm/cpp/kernels/vec_dtypes.cuh"

#include <c10/util/Float8_e4m3fn.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>

namespace rtp_llm {

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
        atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
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
            float val     = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
            output_arr[j] = static_cast<DST_DTYPE>(val);
        }
        *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
    }

    const int32_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val   = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
        output[idx] = static_cast<DST_DTYPE>(val);
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

}  // namespace rtp_llm

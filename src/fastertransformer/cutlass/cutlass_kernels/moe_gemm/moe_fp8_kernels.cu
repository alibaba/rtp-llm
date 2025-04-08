#include "src/fastertransformer/cutlass/cutlass_kernels/moe_gemm/moe_fp8_kernels.h"
// using namespace tensorrt_llm::common;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "src/fastertransformer/cuda/trt_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/reduce_kernel_utils.cuh"


#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

namespace fastertransformer {

const int EXPAND_THREADS_PER_BLOCK = 256;
template<class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++) {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

template<typename T>
__global__ void expandInputRowsContiguousKernel(T const*      unpermuted_input,
                                                float const*  fp8_scales,
                                                T*            permuted_output,
                                                float*        permuted_output_fp8_scales,
                                                float*        unpermuted_scales,
                                                float*        permuted_scales,
                                                int const*    source_rows,
                                                int const*    permuted_src_row_to_dst,
                                                int *         src_row_to_dst,
                                                int64_t const num_rows,
                                                int64_t const cols,
                                                int64_t       k) {
    int const permuted_source_row = blockIdx.x;
    int const expanded_source_row = source_rows[permuted_source_row];
    int const expanded_dest_row   = permuted_src_row_to_dst[permuted_source_row];
    // int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
    using DataElem                    = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    int const source_k_rank = expanded_source_row / num_rows;
    int const source_row    = expanded_source_row % num_rows;

    if (threadIdx.x == 0)
    {
        assert(expanded_dest_row <= INT32_MAX);
        src_row_to_dst[expanded_source_row] = static_cast<int>(expanded_dest_row);
    }

    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + (int64_t)source_row * cols);
    auto*       dest_row_ptr   = reinterpret_cast<DataElem*>(permuted_output + (int64_t)expanded_dest_row * cols);

    int64_t const start_offset     = threadIdx.x;
    int64_t const stride           = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }

    using ScaleDataElem                    = cutlass::Array<float, 4>;
    int const scale_cols = cols / 128;
    auto const* source_fp8_scales_row_ptr = reinterpret_cast<ScaleDataElem const*>(fp8_scales + (int64_t)source_row * scale_cols);
    auto*       dest_fp8_scales_row_ptr   = reinterpret_cast<ScaleDataElem*>(permuted_output_fp8_scales + (int64_t)expanded_dest_row * scale_cols);

    int const scale_num_elems_in_col = scale_cols / 4;

    for (int elem_index = start_offset; elem_index < scale_num_elems_in_col; elem_index += stride) {
        dest_fp8_scales_row_ptr[elem_index] = source_fp8_scales_row_ptr[elem_index];
    }
    if (permuted_scales && threadIdx.x == 0) {
        int64_t const source_k_idx         = source_row * k + source_k_rank;
        permuted_scales[expanded_dest_row] = unpermuted_scales[source_k_idx];
    }
}

template<typename T>
void expandInputRowsKernelLauncherContiguous(T const*      unpermuted_input,
                                             float const*  fp8_scales,
                                             T*            permuted_output,
                                             float*        permuted_output_fp8_scales,
                                             float*        unpermuted_scales,
                                             float*        permuted_scales,
                                             int const*    source_rows,
                                             int const*    permuted_src_row_to_dst,
                                             int *         src_row_to_dst,
                                             int64_t const num_rows,
                                             int64_t const dest_num_rows,
                                             int64_t const cols,
                                             int const     k,
                                             cudaStream_t  stream) {
    int64_t const blocks  = dest_num_rows;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;
    FT_CHECK(cols % 512 == 0);
    expandInputRowsContiguousKernel<T><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                       fp8_scales,
                                                                       permuted_output,
                                                                       permuted_output_fp8_scales,
                                                                       unpermuted_scales,
                                                                       permuted_scales,
                                                                       source_rows,
                                                                       permuted_src_row_to_dst,
                                                                       src_row_to_dst,
                                                                       num_rows,
                                                                       cols,
                                                                       k);
}

template void expandInputRowsKernelLauncherContiguous<__nv_bfloat16>(__nv_bfloat16 const* unpermuted_input,
                                                                     float const*         fp8_scales,
                                                                     __nv_bfloat16*       permuted_output,
                                                                     float*               permuted_output_fp8_scales,
                                                                     float*               unpermuted_scales,
                                                                     float*               permuted_scales,
                                                                     int const*           source_rows,
                                                                     int const*           permuted_src_row_to_dst,
                                                                     int*                 src_row_to_dst,
                                                                     int64_t const        num_rows,
                                                                     int64_t const        dest_num_rows,
                                                                     int64_t const        cols,
                                                                     int const            k,
                                                                     cudaStream_t         stream);

#ifdef ENABLE_FP8
template void expandInputRowsKernelLauncherContiguous<__nv_fp8_e4m3>(__nv_fp8_e4m3 const* unpermuted_input,
                                                                     float const*         fp8_scales,
                                                                     __nv_fp8_e4m3*       permuted_output,
                                                                     float*               permuted_output_fp8_scales,
                                                                     float*               unpermuted_scales,
                                                                     float*               permuted_scales,
                                                                     int const*           source_rows,
                                                                     int const*           permuted_src_row_to_dst,
                                                                     int*                 src_row_to_dst,
                                                                     int64_t const        num_rows,
                                                                     int64_t const        dest_num_rows,
                                                                     int64_t const        cols,
                                                                     int const            k,
                                                                     cudaStream_t         stream);
#endif

const int FINALIZE_THREADS_PER_BLOCK = 256;

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class GemmOutputType, class ScaleBiasType, template <class> class ActFn>
__global__ void doActivationContiguousKernel(__nv_fp8_e4m3* output_fp8,
                                             float *output_fp8_scale,
                                             GemmOutputType const* gemm_result,
                                             ScaleBiasType const* bias_ptr,
                                             bool bias_is_broadcast,
                                             int const* src_row_to_dst,
                                             int64_t inter_size,
                                             bool gated,
                                             int const* permuted_experts)
{
    int64_t const tid = threadIdx.x;
    int64_t const expanded_source_row = blockIdx.x;
    int const expert = permuted_experts[expanded_source_row];
    int64_t const token = src_row_to_dst[expanded_source_row];
    if (token < 0)
    {
        return;
    }

    size_t gated_size_mul = gated ? 2 : 1;
    size_t gated_off = gated ? inter_size : 0;

    gemm_result = gemm_result + token * inter_size * gated_size_mul;
    output_fp8 = output_fp8 + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases
    output_fp8_scale = output_fp8_scale + token * inter_size / 128;
    // int64_t expert = source_k_rank;
    // if (bias_ptr)
    // {
    //     // TODO this is almost certainly faster as a linear scan
    //     expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts, (int64_t) token + 1) - 1;
    // }

    if (bias_ptr)
    {
        size_t bias_offset
            = (bias_is_broadcast ? expert * inter_size * gated_size_mul : token * inter_size * gated_size_mul);
        bias_ptr = bias_ptr + bias_offset;
    }

    // Load 128-bits per thread, according to the smallest data type we read/write
    // constexpr int64_t ACTIVATION_ELEM_PER_THREAD
    //     = 128 / std::min(cutlass::sizeof_bits<T>::value, cutlass::sizeof_bits<GemmOutputType>::value);
    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 4;

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<__nv_fp8_e4m3, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output_fp8);
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
        if (bias_ptr)
        {
            fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
        }

        auto gate_act = fn(fc1_value);

        if (gated)
        {
            auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
            if (bias_ptr_vec)
            {
                gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
            }
            gate_act = gate_act * gate_mul;
        }
        // plus<Array<T, N>> op;
        cutlass::maximum_absolute_value_reduction<ComputeElem, false> max_abs_op;
        float scale = max_abs_op((float)1e-4, gate_act);
        scale = warpReduceMax<float>(scale);
        scale = scale / FP8_E4M3_MAX;
        gate_act = gate_act * (1 / scale);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act);
        if (tid % 32 == 0) {
            output_fp8_scale[elem_index / 32] = scale;
        }
    }
}

template <class GemmOutputType, class ScaleBiasType, template <class> class ActFn>
__global__ void doActivationMaskedKernel(__nv_fp8_e4m3* output_fp8,
                                         float *output_fp8_scale,
                                         GemmOutputType const* gemm_result,
                                         ScaleBiasType const* bias_ptr,
                                         bool bias_is_broadcast,
                                         int token_num,
                                         int64_t inter_size,
                                         bool gated,
                                         int const* masked_m)
{
    int64_t const tid = threadIdx.x;
    int const token = blockIdx.x;
    int const token_idx = blockIdx.x % token_num;
    int const expert = blockIdx.x / token_num;
    if (token_idx >= masked_m[expert])
    {
        return;
    }

    size_t gated_size_mul = gated ? 2 : 1;
    size_t gated_off = gated ? inter_size : 0;

    gemm_result = gemm_result + token * inter_size * gated_size_mul;
    output_fp8 = output_fp8 + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases
    output_fp8_scale = output_fp8_scale + token * inter_size / 128;
    // int64_t expert = source_k_rank;
    // if (bias_ptr)
    // {
    //     // TODO this is almost certainly faster as a linear scan
    //     expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts, (int64_t) token + 1) - 1;
    // }

    if (bias_ptr)
    {
        size_t bias_offset
            = (bias_is_broadcast ? expert * inter_size * gated_size_mul : token * inter_size * gated_size_mul);
        bias_ptr = bias_ptr + bias_offset;
    }

    // Load 128-bits per thread, according to the smallest data type we read/write
    // constexpr int64_t ACTIVATION_ELEM_PER_THREAD
    //     = 128 / std::min(cutlass::sizeof_bits<T>::value, cutlass::sizeof_bits<GemmOutputType>::value);
    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 4;

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<__nv_fp8_e4m3, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output_fp8);
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
        if (bias_ptr)
        {
            fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
        }

        auto gate_act = fn(fc1_value);

        if (gated)
        {
            auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
            if (bias_ptr_vec)
            {
                gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
            }
            gate_act = gate_act * gate_mul;
        }
        // plus<Array<T, N>> op;
        cutlass::maximum_absolute_value_reduction<ComputeElem, false> max_abs_op;
        float scale = max_abs_op((float)1e-4, gate_act);
        scale = warpReduceMax<float>(scale);
        scale = scale / FP8_E4M3_MAX;
        gate_act = gate_act * (1 / scale);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act);
        if (tid % 32 == 0) {
            output_fp8_scale[elem_index / 32] = scale;
        }
    }
}

template<class GemmOutputType, class ScaleBiasType>
void doActivationContiguous(__nv_fp8_e4m3*        output_fp8,
                            float*                fp8_scale,
                            GemmOutputType const* gemm_result,
                            ScaleBiasType const*  bias,
                            bool                  bias_is_broadcast,
                            int const*            src_row_to_dst,
                            int                   num_rows,
                            int64_t               inter_size,
                            ActivationType        activation_type,
                            int const*            permuted_experts,
                            cudaStream_t          stream) {
    int64_t const blocks  = num_rows;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;
    FT_CHECK(inter_size % 128 == 0);

    auto fn_list = std::array{
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,  // Gelu
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu>,  // Relu
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,  // Silu
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,  // Swiglu
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,  // Geglu
        &doActivationContiguousKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity>  // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];
    fn<<<blocks, threads, 0, stream>>>(output_fp8,
                                       fp8_scale,
                                       gemm_result,
                                       bias,
                                       bias_is_broadcast,
                                       src_row_to_dst,
                                       inter_size,
                                       isGatedActivation(activation_type),
                                       permuted_experts);
}

template void doActivationContiguous<__nv_bfloat16, __nv_bfloat16>(__nv_fp8_e4m3*       output_fp8,
                                                                   float*               fp8_scale,
                                                                   __nv_bfloat16 const* gemm_result,
                                                                   __nv_bfloat16 const* bias,
                                                                   bool                 bias_is_broadcast,
                                                                   int const*           src_row_to_dst,
                                                                   int                  num_rows,
                                                                   int64_t              inter_size,
                                                                   ActivationType       activation_type,
                                                                   int const*           permuted_experts,
                                                                   cudaStream_t         stream);

template<class GemmOutputType, class ScaleBiasType>
void doActivationMasked(__nv_fp8_e4m3*        output_fp8,
                        float*                fp8_scale,
                        GemmOutputType const* gemm_result,
                        ScaleBiasType const*  bias,
                        bool                  bias_is_broadcast,
                        int                   expert_num,
                        int                   token_num,
                        int64_t               inter_size,
                        ActivationType        activation_type,
                        int const*            masked_m,
                        cudaStream_t          stream) {
    int64_t const blocks  = expert_num * token_num;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;
    FT_CHECK(inter_size % 128 == 0);

    auto fn_list = std::array{
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,  // Gelu
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu>,  // Relu
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,  // Silu
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,  // Swiglu
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,  // Geglu
        &doActivationMaskedKernel<GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity>  // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];
    fn<<<blocks, threads, 0, stream>>>(output_fp8,
                                       fp8_scale,
                                       gemm_result,
                                       bias,
                                       bias_is_broadcast,
                                       token_num,
                                       inter_size,
                                       isGatedActivation(activation_type),
                                       masked_m);
}

template void doActivationMasked<__nv_bfloat16, __nv_bfloat16>(__nv_fp8_e4m3*       output_fp8,
                                                               float*               fp8_scale,
                                                               __nv_bfloat16 const* gemm_result,
                                                               __nv_bfloat16 const* bias,
                                                               bool                 bias_is_broadcast,
                                                               int                  expert_num,
                                                               int                  token_num,
                                                               int64_t              inter_size,
                                                               ActivationType       activation_type,
                                                               int const*           masked_m,
                                                               cudaStream_t         stream);

}  // namespace fastertransformer

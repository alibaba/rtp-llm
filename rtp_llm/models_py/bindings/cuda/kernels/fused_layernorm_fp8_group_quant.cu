// Fused AddBiasRes + LayerNorm + FP8 Per-Group Quantization kernel.
//
// Warp-per-group design: each warp handles exactly one group of `group_size`
// elements. Group absmax reduction uses warp shuffle only — zero extra
// __syncthreads() beyond the LayerNorm mean/variance block reduce.
// Normed values are cached in registers and quantized in-place.

#include "fused_layernorm_fp8_group_quant.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

namespace rtp_llm {

// T = scalar type (__nv_bfloat16 or half)
// Each warp handles one group of group_size elements.
// block = num_groups * 32 threads.
// Each thread handles group_size/32 = 4 elements (for group_size=128).
template<typename T, int ELEMS_PER_THREAD, bool HAS_BIAS, bool HAS_BETA>
__global__ void fusedAddBiasResLayerNormFP8GroupQuantKernel(
    T*                  normed_output,
    T*                  residual_output,
    __nv_fp8_e4m3*      fp8_output,
    float*              group_scales_out,
    const T*            input,
    const T*            residual,
    const T*            bias,
    const T*            gamma,
    const T*            beta,
    const float         eps,
    const int           hidden_dim,
    const int           group_size,
    const int           scale_stride) {

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x;
    const int num_groups = hidden_dim / group_size;

    // Each thread handles ELEMS_PER_THREAD consecutive elements within its group.
    const int group_elem_start = warp_id * group_size + lane_id * ELEMS_PER_THREAD;
    const int global_elem_start = row * hidden_dim + group_elem_start;

    // ========== Pass 1: Load input + residual + bias, cache pre-norm in registers ==========
    float prenorm_f[ELEMS_PER_THREAD];
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        float inp_f = static_cast<float>(input[global_elem_start + j]);
        float res_f = static_cast<float>(residual[global_elem_start + j]);
        float val = inp_f + res_f;
        if (HAS_BIAS) {
            val += static_cast<float>(bias[group_elem_start + j]);
        }
        prenorm_f[j] = val;
        local_sum += val;
        local_var_sum += val * val;

        // Write pre-norm residual output
        residual_output[global_elem_start + j] = static_cast<T>(val);
    }

    // ========== Block reduce for mean & variance (diff-of-squares) ==========
    float packed[2] = {local_sum, local_var_sum};
    blockReduceSumV2<float, 2>(packed);

    __shared__ float s_mean, s_variance;
    if (threadIdx.x == 0) {
        float mean = packed[0] / hidden_dim;
        float variance = (packed[1] / hidden_dim) - (mean * mean);
        s_mean = mean;
        s_variance = rsqrtf(variance + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float s_inv = s_variance;

    // ========== Pass 2: Normalize, write BF16, find group absmax, quantize ==========
    // All within warp — no __syncthreads() needed!
    constexpr float kQuantEps = 1e-4f;
    float group_max = kQuantEps;
    T normed_cached[ELEMS_PER_THREAD];

    #pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        float g_f = static_cast<float>(gamma[group_elem_start + j]);
        float n = (prenorm_f[j] - mean) * s_inv * g_f;
        if (HAS_BETA) {
            n += static_cast<float>(beta[group_elem_start + j]);
        }

        // Convert to BF16 (matching reference precision for absmax)
        T n_t = static_cast<T>(n);
        normed_cached[j] = n_t;

        // Write BF16 normed output (for residual in next layer)
        normed_output[global_elem_start + j] = n_t;

        // Group absmax from BF16-rounded values (matches reference quant kernel)
        group_max = fmaxf(group_max, fabsf(static_cast<float>(n_t)));
    }

    // Warp-level absmax reduce — NO __syncthreads()!
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffff, group_max, mask));
    }

    float scale = group_max / 448.0f;

    // Quantize and write FP8
    #pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        float val = static_cast<float>(normed_cached[j]);
        float q = fmaxf(-448.0f, fminf(448.0f, val / scale));
        fp8_output[global_elem_start + j] = __nv_fp8_e4m3(q);
    }

    // Write scale (column-major)
    if (lane_id == 0) {
        group_scales_out[warp_id * scale_stride + row] = scale;
    }
}

template<typename T>
void launchFusedAddBiasResLayerNormFP8GroupQuant(
    T*                  normed_output,
    T*                  residual_output,
    __nv_fp8_e4m3*      fp8_output,
    float*              group_scales,
    const T*            input,
    const T*            residual,
    const T*            bias,
    const T*            gamma,
    const T*            beta,
    float               eps,
    int                 tokens,
    int                 hidden_dim,
    int                 group_size,
    int                 scale_stride,
    cudaStream_t        stream) {

    const int num_groups = hidden_dim / group_size;
    const int elems_per_thread = group_size / 32;
    const int num_threads = num_groups * 32;

    dim3 grid(tokens);
    dim3 block(num_threads);

    // blockReduceSumV2 uses a small fixed amount of shared memory
    size_t shmem_size = 0;

    auto select_kernel = [&](bool has_bias, bool has_beta) {
        switch (elems_per_thread) {
        case 4:  // group_size=128
            if (has_bias && has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 4, true, true>;
            if (has_bias)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 4, true, false>;
            if (has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 4, false, true>;
            return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 4, false, false>;
        case 2:  // group_size=64
            if (has_bias && has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 2, true, true>;
            if (has_bias)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 2, true, false>;
            if (has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 2, false, true>;
            return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 2, false, false>;
        case 8:  // group_size=256
            if (has_bias && has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 8, true, true>;
            if (has_bias)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 8, true, false>;
            if (has_beta)
                return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 8, false, true>;
            return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 8, false, false>;
        default:
            TORCH_CHECK(false, "Unsupported group_size/32 = ", elems_per_thread);
            return fusedAddBiasResLayerNormFP8GroupQuantKernel<T, 4, false, false>;
        }
    };

    auto kernel_fn = select_kernel(bias != nullptr, beta != nullptr);

    kernel_fn<<<grid, block, shmem_size, stream>>>(
        normed_output,
        residual_output,
        fp8_output,
        group_scales,
        input,
        residual,
        bias,
        gamma,
        beta,
        eps,
        hidden_dim,
        group_size,
        scale_stride);
}

void invokeAddBiasResLayerNormFP8GroupQuant(
    void*        normed_output,
    void*        residual_output,
    void*        fp8_output,
    float*       group_scales,
    const void*  input,
    const void*  residual,
    const void*  bias,
    const void*  gamma,
    const void*  beta,
    float        eps,
    int          tokens,
    int          hidden_dim,
    int          group_size,
    int          scale_stride,
    int          dtype_flag,
    cudaStream_t stream) {

    TORCH_CHECK(hidden_dim % group_size == 0,
                "hidden_dim must be divisible by group_size");
    TORCH_CHECK(group_size % 32 == 0,
                "group_size must be divisible by 32");

    if (dtype_flag == 0) {  // BFloat16
        launchFusedAddBiasResLayerNormFP8GroupQuant(
            static_cast<__nv_bfloat16*>(normed_output),
            static_cast<__nv_bfloat16*>(residual_output),
            static_cast<__nv_fp8_e4m3*>(fp8_output),
            group_scales,
            static_cast<const __nv_bfloat16*>(input),
            static_cast<const __nv_bfloat16*>(residual),
            static_cast<const __nv_bfloat16*>(bias),
            static_cast<const __nv_bfloat16*>(gamma),
            static_cast<const __nv_bfloat16*>(beta),
            eps, tokens, hidden_dim, group_size, scale_stride, stream);
    } else if (dtype_flag == 1) {  // Float16
        launchFusedAddBiasResLayerNormFP8GroupQuant(
            static_cast<half*>(normed_output),
            static_cast<half*>(residual_output),
            static_cast<__nv_fp8_e4m3*>(fp8_output),
            group_scales,
            static_cast<const half*>(input),
            static_cast<const half*>(residual),
            static_cast<const half*>(bias),
            static_cast<const half*>(gamma),
            static_cast<const half*>(beta),
            eps, tokens, hidden_dim, group_size, scale_stride, stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype_flag: ", dtype_flag);
    }
}

}  // namespace rtp_llm

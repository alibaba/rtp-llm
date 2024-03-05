#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"


namespace fastertransformer{
template<typename T, int N>
__global__ void alphaAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* residual1, const T* bias, const T* gamma, const T* beta, T alpha, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out =
            (float)(input[blockIdx.x * n + idx] + residual1[blockIdx.x * n + idx] * alpha + __ldg_func(&bias[idx]));
        mean += local_out;
        // save local_out to local_out_cache to save some recompute
        local_out_cache[i] = local_out;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out           = local_out_cache[i];
        out[blockIdx.x * n + idx] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg_func(&gamma[idx]))
                                        + (float)(__ldg_func(&beta[idx])));
        idx += blockDim.x;
    }
}

template<typename T>
__global__ void generalAlphaAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* residual1, const T* bias, const T* gamma, const T* beta, T alpha, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out =
            (float)(input[blockIdx.x * n + idx] + residual1[blockIdx.x * n + idx] * alpha + __ldg_func(&bias[idx]));
        mean += local_out;
        // save local_out to out to save some recompute
        out[blockIdx.x * n + idx] = local_out;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out           = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg_func(&gamma[idx]))
                                        + (float)(__ldg_func(&beta[idx])));
    }
}

template<>
__global__ void generalAlphaAddBiasResidualPostLayerNorm(half*       out,
                                                         const half* input,
                                                         const half* residual1,
                                                         const half* bias,
                                                         const half* gamma,
                                                         const half* beta,
                                                         half        alpha,
                                                         int         m,
                                                         int         n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    half2        alpha2       = make_half2(alpha, alpha);
    half2*       out_ptr      = (half2*)out;
    const half2* input_ptr    = (const half2*)input;
    const half2* residual_ptr = (const half2*)residual1;
    const half2* bias_ptr     = (const half2*)bias;
    const half2* gamma_ptr    = (const half2*)gamma;
    const half2* beta_ptr     = (const half2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id  = blockIdx.x * n / 2 + idx;
        half2  tmp = __hadd2(__hadd2(input_ptr[id], __hmul2(residual_ptr[id], alpha2)), __ldg_func(&bias_ptr[idx]));
        float2 local_out_fp2 = __half22float2(tmp);
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save tmp to out_ptr to save some recomputation
        out_ptr[id] = tmp;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        float2 gamma_val     = __half22float2(__ldg_func(&gamma_ptr[idx]));
        float2 beta_val      = __half22float2(__ldg_func(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = __float22half2_rn(local_out_fp2);
    }
}

template<typename T>
__global__ void alphaAddBiasResidualPostLayerNormV2(T* out,
                                                    const T* __restrict input,
                                                    const T* __restrict residual1,
                                                    const T* __restrict bias,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    T   alpha,
                                                    int n) {
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
        int id       = bid * n + col_id;
        local_out[i] = (float)(input[id] + __ldg_func(&residual1[id]) * alpha + __ldg_func(&bias[col_id]));
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id]    = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg_func(&gamma[col_id])
                      + (float)__ldg_func(&beta[col_id]));
    }
}

template<>
__global__ void alphaAddBiasResidualPostLayerNormV2(half* out,
                                                    const half* __restrict input,
                                                    const half* __restrict residual1,
                                                    const half* __restrict bias,
                                                    const half* __restrict gamma,
                                                    const half* __restrict beta,
                                                    half alpha,
                                                    int  n) {
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    half2            local_out_half2[ite];

    half2        alpha2       = make_half2(alpha, alpha);
    half2*       out_ptr      = (half2*)out;
    const half2* input_ptr    = (const half2*)input;
    const half2* residual_ptr = (const half2*)residual1;
    const half2* bias_ptr     = (const half2*)bias;
    const half2* gamma_ptr    = (const half2*)gamma;
    const half2* beta_ptr     = (const half2*)beta;

    // float sum = 0.0f;
    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = input_ptr[id] + __ldg_func(&residual_ptr[id]) * alpha2 + __ldg_func(&bias_ptr[col_id]);
        sum += local_out_half2[i];
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = local_out_half2[i] - s_mean_2;
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg_func(&gamma_ptr[col_id]) + __ldg_func(&beta_ptr[col_id]);
    }
}

template<typename T>
void invokeAlphaAddBiasResidualLayerNorm(T*           out,
                                         const T*     input,
                                         const T*     residual1,
                                         const T*     bias,
                                         const T*     gamma,
                                         const T*     beta,
                                         T            alpha,
                                         int          m,
                                         int          n,
                                         cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        alphaAddBiasResidualPostLayerNormV2<T>
            <<<grid, n / 4, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            alphaAddBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else if (num_trips == 2) {
            alphaAddBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else {
            generalAlphaAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        }
    }
}

template<>
void invokeAlphaAddBiasResidualLayerNorm(half*        out,
                                         const half*  input,
                                         const half*  residual1,
                                         const half*  bias,
                                         const half*  gamma,
                                         const half*  beta,
                                         half         alpha,
                                         int          m,
                                         int          n,
                                         cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        alphaAddBiasResidualPostLayerNormV2<half>
            <<<grid, n / 8, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            alphaAddBiasResidualPostLayerNorm<half, 1>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else if (num_trips == 2) {
            alphaAddBiasResidualPostLayerNorm<half, 2>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else {
            generalAlphaAddBiasResidualPostLayerNorm<half>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        }
    }
}

#define INVOKE_ALPHA_ADD_BIAS_RES_LN(T)                                                                                \
    template void invokeAlphaAddBiasResidualLayerNorm(T*           out,                                                \
                                                      const T*     input,                                              \
                                                      const T*     residual1,                                          \
                                                      const T*     bias,                                               \
                                                      const T*     gamma,                                              \
                                                      const T*     beta,                                               \
                                                      T            alpha,                                              \
                                                      int          m,                                                  \
                                                      int          n,                                                  \
                                                      cudaStream_t stream);
INVOKE_ALPHA_ADD_BIAS_RES_LN(float)
INVOKE_ALPHA_ADD_BIAS_RES_LN(half)
#ifdef ENABLE_BF16
INVOKE_ALPHA_ADD_BIAS_RES_LN(__nv_bfloat16)
#endif

}
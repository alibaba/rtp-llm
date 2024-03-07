#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

// wont't support new features
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

template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = (float)(add(out[blockIdx.x * n + idx], input[blockIdx.x * n + idx], ldg(&bias[idx])));
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
        s_variance = variance / n + layernorm_eps;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[idx])) + (float)(ldg(&beta[idx])));
        idx += blockDim.x;
    }
}

template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n) {
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        T2     tmp           = hadd2(hadd2(out_ptr[id], input_ptr[id]), ldg(&bias_ptr[idx]));
        float2 local_out_fp2 = cuda_cast<float2>(tmp);
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
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        float2 gamma_val     = cuda_cast<float2>(ldg(&gamma_ptr[idx]));
        float2 beta_val      = cuda_cast<float2>(ldg(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = cuda_cast<T2>(local_out_fp2);
    }
}

template<>
__global__ void generalAddBiasResidualPostLayerNorm(float*       out,
                                                    const float* input,
                                                    const float* bias,
                                                    const float* gamma,
                                                    const float* beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg_func(&bias[idx]));
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out           = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = (float)(((local_out - s_mean) * s_variance) * (float)(__ldg_func(&gamma[idx]))
                                            + (float)(__ldg_func(&beta[idx])));
    }
}

template<typename T>
__global__ void addBiasResidualPostLayerNormV2(T* out,
                                               const T* __restrict input,
                                               const T* __restrict bias,
                                               const T* __restrict gamma,
                                               const T* __restrict beta,
                                               const float layernorm_eps,
                                               int         n) {
    using T2             = typename TypeConverter<T>::Type;
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    T2               local_out_half2[ite];

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    // float sum = 0.0f;
    T2 sum = cuda_cast<T2>(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = add(out_ptr[id], ldg(&input_ptr[id]), ldg(&bias_ptr[col_id]));
        sum                = add(sum, local_out_half2[i]);
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    T2    s_mean_2 = cuda_cast<T2>(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = hsub2(local_out_half2[i], s_mean_2);
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    T2 s_var_2 = cuda_cast<T2>(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = fma(local_out_half2[i], s_var_2, ldg(&gamma_ptr[col_id]), ldg(&beta_ptr[col_id]));
    }
}

template<>
__global__ void addBiasResidualPostLayerNormV2(float* out,
                                               const float* __restrict input,
                                               const float* __restrict bias,
                                               const float* __restrict gamma,
                                               const float* __restrict beta,
                                               const float layernorm_eps,
                                               int         n) {
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
        local_out[i] = (float)(out[id] + __ldg_func(&input[id]) + __ldg_func(&bias[col_id]));
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id]    = (float)((local_out[i] - s_mean) * s_variance * (float)__ldg_func(&gamma[col_id])
                          + (float)__ldg_func(&beta[col_id]));
    }
}

template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        addBiasResidualPostLayerNormV2<T><<<grid, n / 8, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else {
            generalAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

template<>
void invokeAddBiasResidualLayerNorm(float*       out,
                                    const float* input,
                                    const float* bias,
                                    const float* gamma,
                                    const float* beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2<float>
            <<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<float, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<float, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else {
            generalAddBiasResidualPostLayerNorm<float>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

#define INVOKE_ADD_BIAS_RES_LN(T)                                                                                      \
    template void invokeAddBiasResidualLayerNorm(T*           out,                                                     \
                                                 const T*     input,                                                   \
                                                 const T*     bias,                                                    \
                                                 const T*     gamma,                                                   \
                                                 const T*     beta,                                                    \
                                                 const float  layernorm_eps,                                           \
                                                 int          m,                                                       \
                                                 int          n,                                                       \
                                                 cudaStream_t stream);
INVOKE_ADD_BIAS_RES_LN(float)
INVOKE_ADD_BIAS_RES_LN(half)
#ifdef ENABLE_BF16
INVOKE_ADD_BIAS_RES_LN(__nv_bfloat16)
#endif

template<typename T, bool DYNAMIC_SCALING = false>
__global__ void generalLayerNormWithPadding(const T* __restrict input,
                                            const T* __restrict gamma,
                                            const T* __restrict beta,
                                            T*          normed_output,
                                            const float layernorm_eps,
                                            int         m,
                                            int         real_n,
                                            int         padding_n,
                                            float*      scale,
                                            float*      dynamic_scale,
                                            const int   int8_mode) {
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const Float_Packed_T scale_to_int = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

    float local_sum = 0.0f;
    for (int i = tid; i < real_n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * padding_n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / real_n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < real_n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * padding_n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / real_n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    for (int i = tid; i < real_n; i += blockDim.x) {
        const int index    = blockIdx.x * padding_n + i;
        float     beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        T         val      = (T)((((float)input[index] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (DYNAMIC_SCALING) {
            abs_max  = cuda_max(cuda_max<Scalar_T, T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        } else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        } else {
            normed_output[index] = val;
        }
    }

    if (DYNAMIC_SCALING) {
        float          abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const Scalar_T dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < real_n; i += blockDim.x) {
            const int index                                        = blockIdx.x * padding_n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T>
void invokeGeneralLayerNormWithPadding(T*           out,
                                       const T*     input,
                                       const T*     gamma,
                                       const T*     beta,
                                       const float  layernorm_eps,
                                       const int    m,
                                       const int    real_n,
                                       const int    padding_n,
                                       float*       scale,
                                       float*       dynamic_scale,
                                       const int    int8_mode,
                                       cudaStream_t stream,
                                       int          opt_version) {
    dim3       grid(m);
    const bool dynamic_quant = dynamic_scale != nullptr;

    dim3 block(min(real_n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    block.x = 32 * ((block.x + 31) / 32);

    /* should pay attention to the rsqrt precision*/
    if (dynamic_quant) {
        size_t maxbytes = real_n * sizeof(T);
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalLayerNormWithPadding<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
        }
        generalLayerNormWithPadding<T, true><<<grid, block, maxbytes, stream>>>(
            input, gamma, beta, out, layernorm_eps, m, real_n, padding_n, scale, dynamic_scale, int8_mode);
    } else {
        generalLayerNormWithPadding<T, false><<<grid, block, 0, stream>>>(
            input, gamma, beta, out, layernorm_eps, m, real_n, padding_n, scale, dynamic_scale, int8_mode);
    }
}

#define INVOKE_GENERAL_LN_WITH_PADDING(T)                                                                              \
    template void invokeGeneralLayerNormWithPadding(T*           out,                                                  \
                                                    const T*     input,                                                \
                                                    const T*     gamma,                                                \
                                                    const T*     beta,                                                 \
                                                    const float  layernorm_eps,                                        \
                                                    const int    m,                                                    \
                                                    const int    real_n,                                               \
                                                    const int    padding_n,                                            \
                                                    float*       scale,                                                \
                                                    float*       dynamic_scale,                                        \
                                                    const int    int8_mode,                                            \
                                                    cudaStream_t stream,                                               \
                                                    int          opt_version);
INVOKE_GENERAL_LN_WITH_PADDING(float)
INVOKE_GENERAL_LN_WITH_PADDING(half)
#ifdef ENABLE_BF16
INVOKE_GENERAL_LN_WITH_PADDING(__nv_bfloat16)
#endif

}
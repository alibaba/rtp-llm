#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/kernels/fused_qk_rmsnorm.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

template<typename Tf, typename T, bool IS_BETA>
__inline__ __device__ Tf compute_rmsnorm(Tf val, float s_variance, const T* gamma, const T* beta, int i) {
    Tf ret = val * s_variance * cuda_cast<Tf>(gamma[i]);
    if (IS_BETA) {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

template<typename T, bool IS_BIAS>
__global__ void fusedQkRmsNorm(T* __restrict input,
                               const T* __restrict q_gamma,
                               const T* __restrict q_bias,
                               const T* __restrict k_gamma,
                               const T* __restrict k_bias,
                               const int   q_group_num,
                               const int   k_group_num,
                               const float eps,
                               const int   n,
                               const int   norm_size) {
    constexpr auto num_elems_T        = num_elems<T>::value;
    using float_packed_t              = typename packed_as<float, num_elems_T>::type;
    constexpr int vec_size            = num_elems<T>::value;
    constexpr int warp_size           = 32;
    const int     elements_per_thread = norm_size / (warp_size * vec_size);

    const int sample_idx  = blockIdx.x / (q_group_num + k_group_num);
    const int group_idx   = blockIdx.x % (q_group_num + k_group_num);
    T*        group_start = input + sample_idx * (n / vec_size) + group_idx * (norm_size / vec_size);

    const T* gamma = group_idx < q_group_num ? q_gamma : k_gamma;
    const T* bias  = group_idx < q_group_num ? q_bias : k_bias;

    __shared__ float smem_scale;

    float square_sum = 0.0f;
    for (int i = 0; i < elements_per_thread; ++i) {
        const int elem_idx   = i * warp_size + threadIdx.x;
        T         packed_val = group_start[elem_idx];
        auto      val        = cuda_cast<float_packed_t>(packed_val);

        square_sum += cuda_sum<float>(val * val);
    }

    float variance = warpReduceSum(square_sum) / norm_size;

    if (threadIdx.x == 0) {
        smem_scale = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = 0; i < elements_per_thread; ++i) {
        const int elem_idx   = i * warp_size + threadIdx.x;
        T         packed_val = group_start[elem_idx];

        const float_packed_t val_f = cuda_cast<float_packed_t>(packed_val);
        const T              val =
            cuda_cast<T>(compute_rmsnorm<float_packed_t, T, IS_BIAS>(val_f, smem_scale, gamma, bias, elem_idx));
        group_start[elem_idx] = cuda_cast<T>(val);
    }
}

template<typename T>
void invokeFusedQkRmsNorm(T* __restrict input,
                          const T* __restrict q_gamma,
                          const T* __restrict q_bias,
                          const T* __restrict k_gamma,
                          const T* __restrict k_bias,
                          const float  layernorm_eps,
                          const int    q_group_num,
                          const int    k_group_num,
                          const int    m,
                          const int    n,
                          const int    norm_size,
                          cudaStream_t stream) {
    constexpr size_t vec_size  = 2;
    constexpr size_t warp_size = 32;

    // 参数校验
    if (n % norm_size != 0) {
        throw std::invalid_argument("n must be divisible by norm_size");
    }
    if (norm_size % (warp_size * vec_size) != 0) {
        throw std::invalid_argument("norm_size must be multiple of " + std::to_string(warp_size * vec_size));
    }

    dim3 grid(m * (q_group_num + k_group_num));  // 每个block处理一个样本的一个头
    dim3 block(warp_size);

    using Tp     = typename packed_as<T, vec_size>::type;
    bool is_bias = k_bias != nullptr && q_bias != nullptr;
    if (is_bias) {
        fusedQkRmsNorm<Tp, true><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(input),
                                                             reinterpret_cast<const Tp*>(q_gamma),
                                                             reinterpret_cast<const Tp*>(q_bias),
                                                             reinterpret_cast<const Tp*>(k_gamma),
                                                             reinterpret_cast<const Tp*>(k_bias),
                                                             q_group_num,
                                                             k_group_num,
                                                             layernorm_eps,
                                                             n,
                                                             norm_size);
    } else {
        fusedQkRmsNorm<Tp, false><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(input),
                                                              reinterpret_cast<const Tp*>(q_gamma),
                                                              nullptr,
                                                              reinterpret_cast<const Tp*>(k_gamma),
                                                              nullptr,
                                                              q_group_num,
                                                              k_group_num,
                                                              layernorm_eps,
                                                              n,
                                                              norm_size);
    }
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
}

#define INSTANTIATE_FUSED_QK_RMSNORM(T)                                                                                \
    template void invokeFusedQkRmsNorm(T* __restrict input,                                                            \
                                       const T* __restrict q_gamma,                                                    \
                                       const T* __restrict q_bias,                                                     \
                                       const T* __restrict k_gamma,                                                    \
                                       const T* __restrict k_bias,                                                     \
                                       const float  layernorm_eps,                                                     \
                                       const int    q_group_num,                                                       \
                                       const int    k_group_num,                                                       \
                                       const int    m,                                                                 \
                                       const int    n,                                                                 \
                                       const int    norm_size,                                                         \
                                       cudaStream_t stream);
INSTANTIATE_FUSED_QK_RMSNORM(float);
INSTANTIATE_FUSED_QK_RMSNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_FUSED_QK_RMSNORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_FUSED_QK_RMSNORM

}  // namespace rtp_llm
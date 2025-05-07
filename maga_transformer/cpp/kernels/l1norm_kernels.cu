#include "maga_transformer/cpp/cuda/cuda_type_utils.cuh"
#include "maga_transformer/cpp/cuda/reduce_kernel_utils.cuh"
#include "maga_transformer/cpp/kernels/l1norm_kernels.h"

namespace rtp_llm
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_l1norm(Tf val, float r_sum)
{
    Tf ret = val * r_sum;
    return ret;
}

template <typename T>
__global__ void generall1Norm(T* output, const T* input, const float eps, int tokens, int hidden_dim)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float local_sum = 0.0f;
    float r_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        T val = cuda_cast<T>(input[index]);

        shmem[i] = val;

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);

        local_sum += cuda_sum<float>(cuda_abs(val_f));
    }

    float packed[1] = {local_sum};
    blockReduceSumV2<float, 1>(packed);
    r_sum = 1.0f / (packed[0] + eps);

    __syncthreads();

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
        const T val = cuda_cast<T>(compute_l1norm<float_packed_t, T>(val_f, r_sum));

        output[index] = val;
    }
}

template <typename T>
void dispatch_l1norm_type_square_method(T* output, const T* input, const float eps, int tokens, int hidden_dim,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream)
{
    if (shmem_size >= (48 << 10))
    {
        cudaError_t ret = cudaFuncSetAttribute(generall1Norm<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    }
    generall1Norm<T><<<grid, block, shmem_size, stream>>>(output, input, eps, tokens, hidden_dim);
}

template <typename T>
void invokeGeneralL1Norm(T* out, const T* input, const float eps, const int tokens, const int hidden_dim, cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    const size_t shmem_size = hidden_dim * sizeof(T);
    const bool use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_l1norm_type_square_method(reinterpret_cast<Tp*>(out), reinterpret_cast<const Tp*>(input), eps, tokens,
            hidden_dim, grid, block, shmem_size, stream);
    }
    else
    {
        dispatch_l1norm_type_square_method(out, input, eps, tokens, hidden_dim, grid, block, shmem_size, stream);
    }
}

#define INSTANTIATE_GENERAL_L1NORM(T)                                                                                  \
    template void invokeGeneralL1Norm(T* out, const T* input, const float eps, const int tokens, const int hidden_dim, \
        cudaStream_t stream);

INSTANTIATE_GENERAL_L1NORM(float);
INSTANTIATE_GENERAL_L1NORM(half);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_L1NORM(__nv_bfloat16);
#endif

} // namespace rtp_llm

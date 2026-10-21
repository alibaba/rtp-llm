#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/fused_qk_rmsnorm.h"

#if USING_CUDA && defined(ENABLE_BF16)
#include "rtp_llm/models_py/bindings/cuda/kernels/vec_dtypes.cuh"
#include <algorithm>
#include <unordered_map>
#endif

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
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
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
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


#if USING_CUDA && defined(ENABLE_BF16)
/*
 * The per-head algorithm below is adapted from FlashInfer, Copyright (c) 2024
 * by FlashInfer team, licensed under the Apache License, Version 2.0:
 * http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * See the License for the specific language governing permissions and limitations.
 * Source: flashinfer/data/include/flashinfer/norm.cuh, QKRMSNormKernel,
 * revision 8c4f4dcf; norm.cuh SHA256:
 * f32e86ca9e3a9caf0163ac619464f784f26cddaa8f19ef47a706ca7a602b0987.
 * Changes: one job domain for Q16/K8, separate weights, fixed packed layout;
 * explicit FP32 FTZ instructions preserve the original CUDA fast-math binary.
 * The legacy vec2 implementation above is unchanged.
 */
namespace {

__device__ __forceinline__ float qkVec8Fma(float a, float b, float c) {
    float out;
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float qkVec8Add(float a, float b) {
    float out;
    asm volatile("add.rn.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float qkVec8Mul(float a, float b) {
    float out;
    asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float qkVec8Rcp(float x) {
    float out;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ float qkVec8Rsqrt(float x) {
    float out;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ float qkVec8Shuffle(float x, unsigned offset) {
    float out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(out) : "f"(x), "r"(offset));
    return out;
}

__global__ void fusedQkRmsNormCudaVec8(__nv_bfloat16* __restrict__ input,
                                      const __nv_bfloat16* __restrict__ q_weight,
                                      const __nv_bfloat16* __restrict__ k_weight,
                                      uint32_t m, uint32_t d, float weight_bias, float eps) {
    const uint32_t lane = threadIdx.x;
    const uint32_t worker = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t workers = gridDim.x * blockDim.y;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    for (uint32_t job = worker; job < m * 24; job += workers) {
        const uint32_t token = job / 24;
        const uint32_t head = job % 24;
        __nv_bfloat16* const row = input + token * 4096 + head * 128;
        const __nv_bfloat16* const weight = head < 16 ? q_weight : k_weight;
        vec_t<__nv_bfloat16, 8> input_vec;
        input_vec.fill(__float2bfloat16_rn(0.f));
        if (lane < 16) {
            input_vec.load(row + lane * 8);
        }
        float sum_sq = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            const float value = __bfloat162float(input_vec[j]);
            sum_sq = qkVec8Fma(value, value, sum_sq);
        }
#pragma unroll
        for (uint32_t offset = 16; offset > 0; offset /= 2) {
            sum_sq = qkVec8Add(sum_sq, qkVec8Shuffle(sum_sq, offset));
        }
        // The reference's use_fast_math contracts sum/d + eps into FFMA.FTZ.
        // Keep runtime d and weight_bias, including weight + (+0), explicit.
        const float inv_rms = qkVec8Rsqrt(qkVec8Fma(sum_sq, qkVec8Rcp(float(d)), eps));
        vec_t<__nv_bfloat16, 8> weight_vec;
        vec_t<__nv_bfloat16, 8> output_vec;
        input_vec.fill(__float2bfloat16_rn(0.f));
        weight_vec.fill(__float2bfloat16_rn(0.f));
        if (lane < 16) {
            input_vec.load(row + lane * 8);
            weight_vec.load(weight + lane * 8);
        }
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            const float biased_weight = qkVec8Add(weight_bias, __bfloat162float(weight_vec[j]));
            const float normalized = qkVec8Mul(__bfloat162float(input_vec[j]), inv_rms);
            output_vec[j] = __float2bfloat16_rn(qkVec8Mul(normalized, biased_weight));
        }
        if (lane < 16) {
            output_vec.store(row + lane * 8);
        }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace

cudaError_t invokeFusedQkRmsNormCudaVec8(__nv_bfloat16* input,
                                        const __nv_bfloat16* q_weight,
                                        const __nv_bfloat16* k_weight,
                                        uint32_t m, float eps, int device_id,
                                        int sm_count, cudaStream_t stream) {
    // Per-device/per-thread metadata only; no GPU synchronization or allocation.
    // Warmup populates this before graph capture and avoids repeated occupancy calls.
    static thread_local std::unordered_map<int, int> grid_limits;
    auto found = grid_limits.find(device_id);
    if (found == grid_limits.end()) {
        int active_blocks = 0;
        const cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, fusedQkRmsNormCudaVec8, 128, 0);
        if (status != cudaSuccess) {
            return status;
        }
        found = grid_limits.emplace(device_id, active_blocks * sm_count).first;
    }
    const uint32_t blocks = std::min(uint32_t(found->second), (m * 24 + 3) / 4);
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = false;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(32, 4);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, fusedQkRmsNormCudaVec8, input, q_weight, k_weight,
                              m, uint32_t(128), 0.f, eps);
}
#endif  // USING_CUDA && ENABLE_BF16

}  // namespace rtp_llm
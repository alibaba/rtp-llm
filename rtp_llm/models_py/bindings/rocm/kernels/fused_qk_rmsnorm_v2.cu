// Warp-per-(token, head) fused QK RMSNorm — ROCm wave64 single-pass kernel.
// Design adapted from SGL PR #21440 (CUDA `qknorm_rope.cuh`), without RoPE.
// In-place over the fused QKV tensor, drop-in replacement for invokeFusedQkRmsNorm.

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_qk_rmsnorm.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_qk_rmsnorm_v2.h"

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm {
namespace fused_qk_rmsnorm_v2 {

constexpr int WAVE_SIZE       = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE      = WAVE_SIZE * WARPS_PER_BLOCK;  // 256

template<int N>
__device__ __forceinline__ float wave64_reduce_sum(float v) {
#pragma unroll
    for (int offset = N / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, N);
    }
    return v;
}

// ============================================================================
// BF16 specialization — head_dim=256, wave=64, 4 elems/thread (bf16_4_t)
// ============================================================================
#ifdef ENABLE_BF16
template<int HEAD_DIM>
__global__ __launch_bounds__(BLOCK_SIZE) void fusedQkRmsNormV2BF16Kernel(
    __nv_bfloat16* __restrict       input,
    const __nv_bfloat16* __restrict q_gamma,
    const __nv_bfloat16* __restrict k_gamma,
    const float                     eps,
    const int                       q_group_num,
    const int                       k_group_num,
    const int                       n,
    const int                       num_qk_heads,
    const int                       num_works) {
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / WAVE_SIZE;  // 4 for HEAD_DIM=256
    static_assert(ELEMS_PER_THREAD == 4, "this kernel hard-codes bf16_4_t (8B/thread)");

    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int work_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (work_id >= num_works) {
        return;
    }

    const int token_id = work_id / num_qk_heads;
    const int head_id  = work_id % num_qk_heads;
    const bool is_q    = head_id < q_group_num;

    // Each row of `input` has stride `n` (cols = q_size + 2*kv_size + maybe_v).
    // Q heads: [0, q_group_num) → column offset head_id * HEAD_DIM
    // K heads: [q_group_num, q_group_num + k_group_num) → column offset head_id * HEAD_DIM
    // (head_id is contiguous across q+k thanks to fused QKV layout)
    __nv_bfloat16*       row_ptr  = input + token_id * n + head_id * HEAD_DIM;
    const __nv_bfloat16* gamma    = is_q ? q_gamma : k_gamma;

    // Vectorized 8-byte load: each thread owns 4 contiguous elements at offset lane_id*4
    bf16_4_t* vec_ptr        = reinterpret_cast<bf16_4_t*>(row_ptr);
    const bf16_4_t* gamma_v  = reinterpret_cast<const bf16_4_t*>(gamma);

    bf16_4_t packed = vec_ptr[lane_id];
    bf16_4_t gpack  = gamma_v[lane_id];

    float2 v_x = __bfloat1622float2(packed.x);  // {x0, x1}
    float2 v_y = __bfloat1622float2(packed.y);  // {x2, x3}

    float local_sq = v_x.x * v_x.x + v_x.y * v_x.y + v_y.x * v_y.x + v_y.y * v_y.y;
    float wave_sq  = wave64_reduce_sum<WAVE_SIZE>(local_sq);

    const float scale = rsqrtf(wave_sq / static_cast<float>(HEAD_DIM) + eps);

    float2 g_x = __bfloat1622float2(gpack.x);
    float2 g_y = __bfloat1622float2(gpack.y);

    bf16_4_t out;
    out.x = __floats2bfloat162_rn(v_x.x * scale * g_x.x, v_x.y * scale * g_x.y);
    out.y = __floats2bfloat162_rn(v_y.x * scale * g_y.x, v_y.y * scale * g_y.y);

    vec_ptr[lane_id] = out;
}
#endif  // ENABLE_BF16

// ============================================================================
// Generic fallback — float / half. Single-pass, wave-level reduce.
// Each thread owns ELEMS_PER_THREAD scalar elements (no vector load).
// ============================================================================
template<typename T, int HEAD_DIM>
__global__ __launch_bounds__(BLOCK_SIZE) void fusedQkRmsNormV2GenericKernel(
    T* __restrict       input,
    const T* __restrict q_gamma,
    const T* __restrict k_gamma,
    const float         eps,
    const int           q_group_num,
    const int           k_group_num,
    const int           n,
    const int           num_qk_heads,
    const int           num_works) {
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / WAVE_SIZE;

    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int work_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (work_id >= num_works) {
        return;
    }

    const int  token_id = work_id / num_qk_heads;
    const int  head_id  = work_id % num_qk_heads;
    const bool is_q     = head_id < q_group_num;

    T*       row_ptr = input + token_id * n + head_id * HEAD_DIM;
    const T* gamma   = is_q ? q_gamma : k_gamma;

    float vals[ELEMS_PER_THREAD];
    float local_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int idx = i * WAVE_SIZE + lane_id;
        vals[i]       = static_cast<float>(row_ptr[idx]);
        local_sq += vals[i] * vals[i];
    }

    const float wave_sq = wave64_reduce_sum<WAVE_SIZE>(local_sq);
    const float scale   = rsqrtf(wave_sq / static_cast<float>(HEAD_DIM) + eps);

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int idx = i * WAVE_SIZE + lane_id;
        const float g = static_cast<float>(gamma[idx]);
        row_ptr[idx]  = static_cast<T>(vals[i] * scale * g);
    }
}

}  // namespace fused_qk_rmsnorm_v2

template<typename T>
void invokeFusedQkRmsNormV2(T* __restrict       input,
                            const T* __restrict q_gamma,
                            const T* __restrict q_bias,
                            const T* __restrict k_gamma,
                            const T* __restrict k_bias,
                            const float         layernorm_eps,
                            const int           q_group_num,
                            const int           k_group_num,
                            const int           m,
                            const int           n,
                            const int           norm_size,
                            cudaStream_t        stream) {
    using namespace fused_qk_rmsnorm_v2;

    if (q_bias != nullptr || k_bias != nullptr) {
        // V2 currently does not support bias path; fall back.
        invokeFusedQkRmsNorm<T>(input, q_gamma, q_bias, k_gamma, k_bias, layernorm_eps,
                                q_group_num, k_group_num, m, n, norm_size, stream);
        return;
    }
    if (norm_size != 256) {
        // V2 currently specializes on HEAD_DIM=256 (Qwen3.5-9B Full-Attn); fall back.
        invokeFusedQkRmsNorm<T>(input, q_gamma, q_bias, k_gamma, k_bias, layernorm_eps,
                                q_group_num, k_group_num, m, n, norm_size, stream);
        return;
    }

    constexpr int HEAD_DIM = 256;
    const int     num_qk_heads = q_group_num + k_group_num;
    const int     num_works    = m * num_qk_heads;
    const int     num_blocks   = (num_works + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

#ifdef ENABLE_BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        fusedQkRmsNormV2BF16Kernel<HEAD_DIM><<<grid, block, 0, stream>>>(
            input, q_gamma, k_gamma, layernorm_eps,
            q_group_num, k_group_num, n, num_qk_heads, num_works);
        return;
    }
#endif
    fusedQkRmsNormV2GenericKernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        input, q_gamma, k_gamma, layernorm_eps,
        q_group_num, k_group_num, n, num_qk_heads, num_works);
}

#define INSTANTIATE_FUSED_QK_RMSNORM_V2(T)                                    \
    template void invokeFusedQkRmsNormV2(T* __restrict       input,           \
                                         const T* __restrict q_gamma,         \
                                         const T* __restrict q_bias,          \
                                         const T* __restrict k_gamma,         \
                                         const T* __restrict k_bias,          \
                                         const float         layernorm_eps,   \
                                         const int           q_group_num,     \
                                         const int           k_group_num,     \
                                         const int           m,               \
                                         const int           n,               \
                                         const int           norm_size,       \
                                         cudaStream_t        stream);
// Note: only Half + BF16 are instantiated. The lone caller in
// FusedQKRmsNorm.cc uses DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16, which
// dispatches Half / BFloat16 only and TORCH_CHECKs on float — so an
// invokeFusedQkRmsNormV2<float> instantiation would be dead code.
INSTANTIATE_FUSED_QK_RMSNORM_V2(half);
#ifdef ENABLE_BF16
INSTANTIATE_FUSED_QK_RMSNORM_V2(__nv_bfloat16);
#endif
#undef INSTANTIATE_FUSED_QK_RMSNORM_V2

}  // namespace rtp_llm

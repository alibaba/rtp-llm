#include "rtp_llm/cpp/kernels/rocm/masked_silu_and_mul/mask_kernel.h"

namespace rtp_llm {
using utils::fp8_e4m3_t;

#define HIP_CHECK(cmd)                                                  \
  do {                                                                  \
    hipError_t e = cmd;                                                 \
    if (e != hipSuccess) {                                              \
      printf("HIP error: %s (%d) at %s:%d\n",                           \
             hipGetErrorString(e), static_cast<int>(e),                 \
             __FILE__, __LINE__);                                       \
      __builtin_trap();                                                 \
    }                                                                   \
  } while (0)

template<class GemmOutputType, class ScaleBiasType, int warpSize = 64>
__global__ void doActivationMaskedKernelHIP(fp8_e4m3_t*              output,
                                            float*                output_fp8_scale, // [num_experts, num_tokens, 1]
                                            GemmOutputType const* __restrict__ gemm_result,
                                            int64_t               token_num,
                                            int64_t               inter_size,
                                            bool                  gated,
                                            int const*            __restrict__ masked_m) {
  using utils::Arr;
  using utils::SiLU;
  using utils::arrayConvert;
  using utils::max_abs;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 8;
  using GemmResultElem = Arr<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using OutputElem     = Arr<fp8_e4m3_t,    ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem    = Arr<float,           ACTIVATION_ELEM_PER_THREAD>;

  const int64_t tid          = threadIdx.x;
  const int     batch_idx    = blockIdx.x;     // expert index
  const int64_t batch_stride = gridDim.y;      // token striping across blocks in y
  const int     max_token    = masked_m[batch_idx];

  size_t gated_size_mul = gated ? 2 : 1;
  size_t gated_off      = gated ? inter_size : 0;

  // ===== base pointers =====
  gemm_result        += (int64_t)batch_idx * token_num * inter_size * gated_size_mul;
  output             += (int64_t)batch_idx * token_num * inter_size;
  output_fp8_scale   += (int64_t)batch_idx * token_num;

  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);

  const int64_t gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

  SiLU<ACTIVATION_ELEM_PER_THREAD> silu{};

  for (int token_idx = blockIdx.y; token_idx < max_token; token_idx += batch_stride) {
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result + (int64_t)token_idx * inter_size * gated_size_mul);
    auto output_vec = reinterpret_cast<OutputElem*>(output + (int64_t)token_idx * inter_size);

    float thread_max = 0.0f;
    ComputeElem act_cached;

    // fc1 (activation input)
    ComputeElem fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[tid]);
    act_cached = silu(fc1_value);

    if (gated) {
      ComputeElem gate = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[tid + gated_off_vec]);
      act_cached = act_cached * gate;
    }

    float local_max = max_abs(act_cached);
    thread_max = fmaxf(thread_max, local_max);

    float block_max = utils::blockReduceMax<float>(thread_max);

    float scale = fmaxf(block_max, 1e-4f) / utils::FP8_E4M3_MAX;
    if (tid == 0) {
      output_fp8_scale[token_idx] = scale; // [num_experts, num_tokens, 1]
    }
    float inv_scale = 1.0f / scale;

    output_vec[tid] = utils::pack_fp8_scaled<ACTIVATION_ELEM_PER_THREAD>(act_cached, inv_scale);
  }
}

extern "C"
void launch_doActivationMaskedKernelHIP(
    fp8_e4m3_t*       output,
    float*            output_fp8_scale,
    const hip_bfloat16* gemm_result,
    int64_t           expert_num,
    int64_t           token_num,
    int64_t           inter_size,
    bool              gated,
    const int*        masked_m,
    hipStream_t       stream)
{
  assert(inter_size <= 8192);
  assert(inter_size % 8 == 0);
  int gy = static_cast<int>(std::min<int64_t>(token_num, 64));
  dim3 grid(static_cast<unsigned>(expert_num), static_cast<unsigned>(gy));
  dim3 block(inter_size / 8);

  hipLaunchKernelGGL(
      (doActivationMaskedKernelHIP<hip_bfloat16,hip_bfloat16>),
      grid, block, 0, stream,
      output,
      output_fp8_scale,
      gemm_result,
      token_num,
      inter_size,
      gated,
      masked_m);
  HIP_CHECK(hipGetLastError());
}
}  // namespace rtp_llm


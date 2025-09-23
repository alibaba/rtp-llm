#include "rtp_llm/cpp/cuda/deep_gemm/JITRuntime.h"

using namespace std;

namespace rtp_llm {

string KernelParams::getShortParamsStr() const {
    return to_string(n) + "_" + to_string(k) + "_" + to_string(num_groups) + "_" + to_string(uint32_t(gemm_type));
}

string KernelParams::getParamsStr() const {
    return to_string(n) + "_" + to_string(k) + "_" + to_string(bm) + "_" + to_string(bn) + "_" + to_string(bk) + "_"
           + to_string(num_groups) + "_" + to_string(num_stages) + "_" + to_string(num_tma_multicast) + "_"
           + to_string(uint32_t(gemm_type)) + "_" + to_string(uint32_t(swap_ab));
}

string KernelParams::getKernelStr() const {
    const string template_str = to_string(n) + ", " + to_string(k) + ", " + to_string(bm) + ", " + to_string(bn) + ", "
                                + to_string(bk) + ", " + to_string(num_groups) + ", " + to_string(num_stages) + ", "
                                + to_string(num_tma_multicast) + ", " + getDeepGemmTypeStr(gemm_type);
    const string func_name_str = getParamsStr();
    string       code          = R"delimiter(
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#include "rtp_llm/cpp/cuda/deep_gemm/include/fp8_gemm.cuh"

extern "C"{
void runDeepGemm_)delimiter"
                  + func_name_str + R"delimiter((__nv_bfloat16*         output,
                             __nv_fp8_e4m3*         lhs,
                             float*                 lhs_scale,
                             __nv_fp8_e4m3*         rhs,
                             float*                 rhs_scale,
                             int*                   grouped_layout,
                             uint32_t               m,
                             cudaStream_t           stream,
                             uint32_t               num_sms,
                             uint32_t               smem_size) {
)delimiter";
    if (!swap_ab) {
        code += R"delimiter(
    using gemm_runner = deep_gemm::Gemm<)delimiter"
                + template_str + R"delimiter(>;
    auto tma_a_desc = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(lhs, m);
    auto tma_b_desc = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(rhs);
    auto tma_scales_a_desc = gemm_runner::template make_2d_tma_scales_a_desc<float>(lhs_scale, m);
    auto tma_d_desc = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
    gemm_runner::run(output, rhs_scale, grouped_layout, m, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, stream, num_sms, smem_size);
    return;
}
}
#endif
)delimiter";
    } else {
        if (gemm_type != DeepGemmType::Normal && gemm_type != DeepGemmType::GroupedMasked) {
            RTP_LLM_FAIL("deep_gemm::GemmSwapAB does not support gemm type other than Normal and GroupedMasked yet");
        }
        code += R"delimiter(
    using gemm_runner = deep_gemm::GemmSwapAB<)delimiter"
                + template_str + R"delimiter(>;
    auto tma_a_desc = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(rhs);
    auto tma_b_desc = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(lhs, m);
    auto tma_scales_b_desc = gemm_runner::template make_2d_tma_scales_b_desc<float>(lhs_scale, m);
    auto tma_d_desc = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
    gemm_runner::run(output, rhs_scale, grouped_layout, m, tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc, stream, num_sms, smem_size);
    return;
}
}
#endif
)delimiter";
    }
    return code;
}

vector<uint32_t> KernelParams::getKey() const {
    return {n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};
}

string KernelParams::getCommandStr(const string& hdrs_path) const {
    return " -std=c++17 -shared -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_90a,code=sm_90a --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi -diag-suppress 177 -DENABLE_FP8 -I"
           + hdrs_path + "/../ -I" + hdrs_path + "/cpp/cuda/deep_gemm/cutlass_hdr/cutlass/include";
}

#ifdef ENABLE_FP8
void runKernel(__nv_bfloat16* output,
               __nv_fp8_e4m3* lhs,
               float*         lhs_scale,
               __nv_fp8_e4m3* rhs,
               float*         rhs_scale,
               int*           grouped_layout,
               uint32_t       m,
               uint32_t       n,
               uint32_t       k,
               uint32_t       bm,
               uint32_t       bn,
               uint32_t       bk,
               uint32_t       num_groups,
               uint32_t       num_stages,
               uint32_t       num_tma_multicast,
               DeepGemmType   gemm_type,
               cudaStream_t   stream,
               uint32_t       num_sms,
               uint32_t       smem_size,
               bool           swap_ab) {
    KernelParams now_params(n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab);
    static JIT<KernelParams, runDeepGemmFunc> jit;
    auto                                      kernel_ptr = jit.getKernelPtr(now_params);
    if (kernel_ptr) {
        kernel_ptr(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, stream, num_sms, smem_size);
        return;
    } else {
        RTP_LLM_FAIL("Failed to get kernel ptr for %u %u %u %u", n, k, num_groups, uint32_t(gemm_type));
    }
}
#endif
}  // namespace rtp_llm

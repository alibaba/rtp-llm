#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>
#include <filesystem>

#include "rtp_llm/cpp/deep_gemm/JIT.h"

using namespace std;

namespace rtp_llm {

#ifdef ENABLE_FP8

JITRuntimeMap JIT::jit_kernels_;

string JIT::getParamsStr(KernelParams& params) {
    auto& [n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab] = params;
    return to_string(n) + "_" + to_string(k) + "_" + to_string(bm) + "_" + to_string(bn) + "_" + to_string(bk) + "_"
           + to_string(num_groups) + "_" + to_string(num_stages) + "_" + to_string(num_tma_multicast) + "_"
           + to_string(uint32_t(gemm_type)) + "_" + to_string(uint32_t(swap_ab));
}

string JIT::getKernelStr(KernelParams& params) {
    auto& [n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab] = params;
    const string template_str = to_string(n) + ", " + to_string(k) + ", " + to_string(bm) + ", " + to_string(bn) + ", "
                                + to_string(bk) + ", " + to_string(num_groups) + ", " + to_string(num_stages) + ", "
                                + to_string(num_tma_multicast) + ", " + getDeepGemmTypeStr(gemm_type);
    const string func_name_str = getParamsStr(params);
    string       code          = R"delimiter(
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#include "rtp_llm/cpp/deep_gemm/include/fp8_gemm.cuh"

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

bool JIT::loadFromCache(KernelParams& params) {
    auto& [n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab] = params;
    const string short_params_str =
        to_string(n) + "_" + to_string(k) + "_" + to_string(num_groups) + "_" + to_string(uint32_t(gemm_type));

    filesystem::path local_dir_path, remote_dir_path;

    const string params_str = getParamsStr(params);
    const string func_name  = "runDeepGemm_" + params_str;

    const string file_hash = getFilesHash("/cpp/deep_gemm");
    local_dir_path         = string("./deep_gemm_runtime/" + file_hash + "/" + short_params_str + "/" + params_str);
    remote_dir_path =
        string(remote_jit_dir + "/deep_gemm_runtime/" + file_hash + "/" + short_params_str + "/" + params_str);

    auto cached_kernel = findCachedKernel(remote_dir_path, local_dir_path, params_str);
    if (cached_kernel) {
        auto kernel_key = vector<uint32_t>{
            n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};
        jit_kernels_.insert(kernel_key, (runDeepGemmFunc)cached_kernel);
        return true;
    }

    return false;
}

void JIT::compileAndSave(KernelParams& params) {
    auto& [n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab] = params;
    const string short_params_str =
        to_string(n) + "_" + to_string(k) + "_" + to_string(num_groups) + "_" + to_string(uint32_t(gemm_type));

    filesystem::path local_dir_path, remote_dir_path;

    const string params_str = getParamsStr(params);
    const string func_name  = "runDeepGemm_" + params_str;

    const string file_hash = getFilesHash("/cpp/deep_gemm");
    local_dir_path         = string("./deep_gemm_runtime/" + file_hash + "/" + short_params_str + "/" + params_str);
    remote_dir_path =
        string(remote_jit_dir + "/deep_gemm_runtime/" + file_hash + "/" + short_params_str + "/" + params_str);

    RTP_LLM_LOG_INFO("JIT compilation " + params_str + " begin");

    string command;
    command =
        " -std=c++17 -shared -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_90a,code=sm_90a --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi -diag-suppress 177 -DENABLE_FP8 -I"
        + jit_hdrs_path + "/../ -I" + jit_hdrs_path + "/cpp/deep_gemm/cutlass_hdr/cutlass/include";

    string so_filename_final =
        compileAndSaveKernel(local_dir_path, remote_dir_path, getKernelStr(params), command, true);
    if (so_filename_final == "") {
        RTP_LLM_FAIL("Failed to compile and save kernel for " + getParamsStr(params));
    }

    RTP_LLM_LOG_INFO("JIT compilation " + so_filename_final + " finished");
    auto kernel = loadKernel(so_filename_final, params_str);
    if (!kernel) {
        RTP_LLM_FAIL("Failed to load kernel from " + so_filename_final);
    }
    auto kernel_key = vector<uint32_t>{
        n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};
    jit_kernels_.insert(kernel_key, (runDeepGemmFunc)kernel);
}

runDeepGemmFunc JIT::searchKernel(KernelParams& params) {
    auto& [n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab] = params;

    auto kernel_key = vector<uint32_t>{
        n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};

    // find in runtime cache first
    auto kernel_value = jit_kernels_.find(kernel_key);
    if (kernel_value) {
        return kernel_value;
    }

    RTP_LLM_LOG_INFO("Start compile and load deepgemm kernel for %u %u %u %u", n, k, num_groups, uint32_t(gemm_type));

    KernelParams now_params{n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab};
    if (!loadFromCache(now_params)) {
        compileAndSave(now_params);
    }

    RTP_LLM_LOG_INFO("Finish compile and load deepgemm kernel for %u %u %u %u", n, k, num_groups, uint32_t(gemm_type));

    kernel_value = jit_kernels_.find(kernel_key);
    if (kernel_value) {
        return kernel_value;
    } else {
        string params_str = getParamsStr(params);
        RTP_LLM_FAIL("Not find matched kernel for params " + params_str);
    }
}

void JIT::runKernel(__nv_bfloat16* output,
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
    auto kernel_params = KernelParams{n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab};
    auto kernel        = searchKernel(kernel_params);
    kernel(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, stream, num_sms, smem_size);
}
#endif

}  // namespace rtp_llm

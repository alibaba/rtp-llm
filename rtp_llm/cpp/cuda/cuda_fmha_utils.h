#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace rtp_llm {

class CudaFmhaUtils {
public:

template<typename T>
static bool UseTrtFMHA(const rtp_llm::GptInitParameter& gpt_init_parameter) {
    bool use_trt_fmha = CheckUseFMHA<T>(gpt_init_parameter) && CheckQKVLengthEqual<T>(gpt_init_parameter);
    if (!(is_sm8x() || is_sm90() || is_sm70())) {
        RTP_LLM_LOG_INFO("TRT FMHA is disabled for sm %d", get_sm());
        use_trt_fmha = false;
    }
    if (gpt_init_parameter.is_sparse_head_){
        RTP_LLM_LOG_INFO("TRT FMHA is disabled for sparse");
        use_trt_fmha = false;
    }
    char* fmha_env = std::getenv("ENABLE_TRT_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        RTP_LLM_LOG_INFO("TRT FMHA is disabled for by env");
        use_trt_fmha = false;
    }
    if (!tensorrt_llm::kernels::MHARunner::fmha_supported(gpt_init_parameter.size_per_head_, rtp_llm::get_sm())) {
        RTP_LLM_LOG_INFO("TRT FMHA is disabled for by check fmha_supported");
        use_trt_fmha = false;
    }
    return use_trt_fmha;
}

template<typename T>
static bool UseOldTrtFMHA(const rtp_llm::GptInitParameter& gpt_init_parameter) {
#ifdef USE_OLD_TRT_FMHA
    bool use_old_trt_fmha = CheckUseFMHA<T>(gpt_init_parameter) && CheckQKVLengthEqual<T>(gpt_init_parameter);
    if (!use_old_trt_fmha) {
        return false;
    }
    if(!std::is_same<T, half>::value){
        RTP_LLM_LOG_INFO("OLD TRT FMHA only support half");
        return false;
    }
    if (gpt_init_parameter.head_num_ != gpt_init_parameter.head_num_kv_) {
        RTP_LLM_LOG_INFO("OLD TRT not support head_num != head_num_kv");
        return false;
    }
    auto testRunner = FusedMHARunnerFP16v2(gpt_init_parameter.head_num_, gpt_init_parameter.size_per_head_, get_sm(), gpt_init_parameter.q_scaling_);
    if (!testRunner.fmha_supported(gpt_init_parameter.is_causal_)) {
        RTP_LLM_LOG_INFO("OLD TRT disabled by call fmha_supported");
        return false;
    }
    return true;
#else
    RTP_LLM_LOG_INFO("USE_OLD_TRT_FMHA not enabled by define");
    return false;
#endif

}

template<typename T>
static bool UsePagedTrtFMHA(const rtp_llm::GptInitParameter& gpt_init_parameter) {
    bool use_paged_trt_fmha = CheckUseFMHA<T>(gpt_init_parameter);
    if (!(is_sm8x() || is_sm90())) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for sm %d", get_sm());
        use_paged_trt_fmha = false;
    }
    if (!gpt_init_parameter.use_kvcache_) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled when not use kvcache");
        use_paged_trt_fmha = false;
    }
    if (gpt_init_parameter.is_sparse_head_) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for sparse");
        use_paged_trt_fmha = false;
    }
    if (gpt_init_parameter.isKvCacheQuant()) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for int8 kvcache");
        use_paged_trt_fmha = false;
    }
    char* paged_fmha_env = std::getenv("ENABLE_PAGED_TRT_FMHA");
    if (paged_fmha_env && std::string(paged_fmha_env) == "OFF") {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for by ENABLE_PAGED_TRT_FMHA=OFF env");
        use_paged_trt_fmha = false;
    }
    return use_paged_trt_fmha;
}

template<typename T>
static bool UseOpenSourceFMHA(const rtp_llm::GptInitParameter& gpt_init_parameter) {
    bool use_open_source_fmha = CheckUseFMHA<T>(gpt_init_parameter) && CheckQKVLengthEqual<T>(gpt_init_parameter);
    if (!(is_sm8x() || is_sm90())) {
        RTP_LLM_LOG_INFO("opensource FMHA is disabled for sm %d", get_sm());
        use_open_source_fmha = false;
    }
    char* fmha_env = std::getenv("ENABLE_OPENSOURCE_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        RTP_LLM_LOG_INFO("opensource FMHA is disabled for by env");
        use_open_source_fmha = false;
    }
    return use_open_source_fmha;
}

protected:
template<typename T>
static bool CheckUseFMHA(const rtp_llm::GptInitParameter& params) {
    char* fmha_env        = std::getenv("ENABLE_FMHA");
    bool  fmha_enable     = (fmha_env == nullptr || std::string(fmha_env) != "OFF");
    if (!fmha_enable){
        RTP_LLM_LOG_INFO("FMHA is not enbaled");
        return false;
    }
    if(std::is_same<T, float>::value){
        RTP_LLM_LOG_INFO("FMHA not support float");
        return false;
    }
    return true;
}

template<typename T>
static bool CheckQKVLengthEqual(const rtp_llm::GptInitParameter& params)  {
    char* reuse_cache_env = std::getenv("REUSE_CACHE");
    bool  not_prefix =
        params.pre_seq_len_ == 0 && (reuse_cache_env == nullptr || std::string(reuse_cache_env) != "1");
    char* multi_task_prompt_env = std::getenv("MULTI_TASK_PROMPT");
    char* multi_task_prompt_str_env = std::getenv("MULTI_TASK_PROMPT_STR");
    char* sp_model_env = std::getenv("SP_MODEL_TYPE");

    char* enable_partial_fallback_env = std::getenv("ENABLE_PARTIAL_FALLBACK");
    if (enable_partial_fallback_env != nullptr && std::string(enable_partial_fallback_env) == "1") {
        RTP_LLM_LOG_INFO("QKV length not equal: enable part fallback");
        return false;
    }

    char* enable_fast_gen_env = std::getenv("ENABLE_FAST_GEN");
    if (enable_fast_gen_env != nullptr && std::string(enable_fast_gen_env) == "1") {
        RTP_LLM_LOG_INFO("QKV length not equal: enable fast gen");
        return false;
    }

    if (!not_prefix){
        RTP_LLM_LOG_INFO("QKV length not equal: use kv cache reuse");
        return false;
    }
    if (sp_model_env != nullptr){
        RTP_LLM_LOG_INFO("QKV length not equal: use sp_model");
        return false;
    }
    if (multi_task_prompt_env && strcmp(multi_task_prompt_env, "") != 0) {
        RTP_LLM_LOG_INFO("QKV length not equal: use multi_task_prompt");
        return false;
    }
    if (multi_task_prompt_str_env && strcmp(multi_task_prompt_str_env, "") != 0) {
        RTP_LLM_LOG_INFO("QKV length not equal: use multi_task_prompt_str");
        return false;
    }
    return true;
}

};

}
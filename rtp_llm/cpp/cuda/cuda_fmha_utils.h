#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
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
    static bool UseTrtFMHA(const FMHAConfig& fmha_config,
                          const ModelConfig& model_config,
                          const RuntimeConfig& runtime_config,
                          const KVCacheConfig& kv_cache_config,
                          const SpeculativeExecutionConfig& sp_config) {
        bool use_trt_fmha = CheckUseFMHA<T>(fmha_config) && CheckQKVLengthEqual<T>(model_config, runtime_config, kv_cache_config, sp_config);
        if (!(is_sm8x() || is_sm90() || is_sm70())) {
            RTP_LLM_LOG_INFO("TRT FMHA is disabled for sm %d", get_sm());
            use_trt_fmha = false;
        }

        bool fmha_env = fmha_config.enable_trt_fmha;
        if (!fmha_env) {
            RTP_LLM_LOG_INFO("TRT FMHA is disabled for by env");
            use_trt_fmha = false;
        }
        if (!tensorrt_llm::kernels::MHARunner::fmha_supported(model_config.attn_config.size_per_head, rtp_llm::get_sm())) {
            RTP_LLM_LOG_INFO("TRT FMHA is disabled for by check fmha_supported");
            use_trt_fmha = false;
        }
        return use_trt_fmha;
    }

    template<typename T>
    static bool UseOldTrtFMHA(const FMHAConfig& fmha_config,
                             const ModelConfig& model_config,
                             const RuntimeConfig& runtime_config,
                             const KVCacheConfig& kv_cache_config,
                             const SpeculativeExecutionConfig& sp_config) {
#ifdef USE_OLD_TRT_FMHA
        bool use_old_trt_fmha = CheckUseFMHA<T>(fmha_config) && CheckQKVLengthEqual<T>(model_config, runtime_config, kv_cache_config, sp_config);
        if (!use_old_trt_fmha) {
            return false;
        }
        if (!std::is_same<T, half>::value) {
            RTP_LLM_LOG_INFO("OLD TRT FMHA only support half");
            return false;
        }
        if (model_config.attn_config.head_num != model_config.attn_config.kv_head_num) {
            RTP_LLM_LOG_INFO("OLD TRT not support head_num != head_num_kv");
            return false;
        }
        auto testRunner = FusedMHARunnerFP16v2(
            model_config.attn_config.head_num, model_config.attn_config.size_per_head, get_sm(), model_config.attn_config.q_scaling);
        if (!testRunner.fmha_supported(model_config.attn_config.is_causal)) {
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
    static bool UsePagedTrtFMHA(const FMHAConfig& fmha_config,
                               const ModelConfig& model_config) {
        bool use_paged_trt_fmha = CheckUseFMHA<T>(fmha_config);
        if (!(is_sm8x() || is_sm90())) {
            RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for sm %d", get_sm());
            use_paged_trt_fmha = false;
        }
        if (!model_config.use_kvcache) {
            RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled when not use kvcache");
            use_paged_trt_fmha = false;
        }
        if (model_config.isKvCacheQuant()) {
            RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for int8 kvcache");
            use_paged_trt_fmha = false;
        }
        bool paged_fmha_env = fmha_config.enable_paged_trt_fmha;
        if (!paged_fmha_env) {
            RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for by ENABLE_PAGED_TRT_FMHA=OFF env");
            use_paged_trt_fmha = false;
        }
        return use_paged_trt_fmha;
    }

    template<typename T>
    static bool UseOpenSourceFMHA(const FMHAConfig& fmha_config,
                                  const ModelConfig& model_config,
                                  const RuntimeConfig& runtime_config,
                                  const KVCacheConfig& kv_cache_config,
                                  const SpeculativeExecutionConfig& sp_config) {
        bool use_open_source_fmha = CheckUseFMHA<T>(fmha_config) && CheckQKVLengthEqual<T>(model_config, runtime_config, kv_cache_config, sp_config);
        if (!(is_sm8x() || is_sm90())) {
            RTP_LLM_LOG_INFO("opensource FMHA is disabled for sm %d", get_sm());
            use_open_source_fmha = false;
        }
        bool fmha_env = fmha_config.enable_open_source_fmha;
        if (!fmha_env) {
            RTP_LLM_LOG_INFO("opensource FMHA is disabled for by env");
            use_open_source_fmha = false;
        }
        return use_open_source_fmha;
    }

protected:
    template<typename T>
    static bool CheckUseFMHA(const FMHAConfig& fmha_config) {
        bool fmha_enable = fmha_config.enable_fmha;
        if (!fmha_enable) {
            RTP_LLM_LOG_INFO("FMHA is not enbaled");
            return false;
        }
        if (std::is_same<T, float>::value) {
            RTP_LLM_LOG_INFO("FMHA not support float");
            return false;
        }
        return true;
    }

    template<typename T>
    static bool CheckQKVLengthEqual(const ModelConfig& model_config,
                                    const RuntimeConfig& runtime_config,
                                    const KVCacheConfig& kv_cache_config,
                                    const SpeculativeExecutionConfig& sp_config) {
        bool               reuse_cache_env           = kv_cache_config.reuse_cache;
        bool               not_prefix                = model_config.pre_seq_len == 0 && !reuse_cache_env;
        const std::string& multi_task_prompt_env     = kv_cache_config.multi_task_prompt;
        const std::string& multi_task_prompt_str_env = kv_cache_config.multi_task_prompt_str;

        bool enable_partial_fallback_env = runtime_config.fifo_scheduler_config.enable_partial_fallback;
        if (enable_partial_fallback_env) {
            RTP_LLM_LOG_INFO("QKV length not equal: enable part fallback");
            return false;
        }

        if (runtime_config.fifo_scheduler_config.enable_fast_gen) {
            RTP_LLM_LOG_INFO("QKV length not equal: enable fast gen");
            return false;
        }

        if (!not_prefix) {
            RTP_LLM_LOG_INFO("QKV length not equal: use kv cache reuse");
            return false;
        }
        if (!sp_config.model_type.empty()) {
            RTP_LLM_LOG_INFO("QKV length not equal: use sp_model");
            return false;
        }
        if (multi_task_prompt_env != "") {
            RTP_LLM_LOG_INFO("QKV length not equal: use multi_task_prompt");
            return false;
        }
        if (multi_task_prompt_str_env != "") {
            RTP_LLM_LOG_INFO("QKV length not equal: use multi_task_prompt_str");
            return false;
        }
        return true;
    }
};

}  // namespace rtp_llm
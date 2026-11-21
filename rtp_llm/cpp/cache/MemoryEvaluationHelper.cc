#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

#include <numeric>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

// Helper function to update memory size if below minimum requirement
void MemoryEvaluationHelper::updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario) {
    if (current_size < min_required) {
        current_size = min_required;
        RTP_LLM_LOG_INFO("%s needs at least %ld MiB memory for runtime by default, "
                         "but only %ld MiB memory reserved. adjust to minimal value.",
                         scenario,
                         min_required / 1024 / 1024,
                         current_size / 1024 / 1024);
    }
}

rtp_llm::DataType MemoryEvaluationHelper::getDataTypeForCache(const ModelConfig& model_config) {
#if defined(BUILDING_ARM_ONLY)
    auto dtype = rtp_llm::TYPE_FP32;
#else
    auto dtype = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 :
                                                                                   model_config.data_type;
#endif
    return dtype;
}

size_t MemoryEvaluationHelper::getDefaultRuntimeMemorySize(const RuntimeConfig&     runtime_config,
                                                           const ParallelismConfig& parallelism_config,
                                                           const ModelConfig&       model_config,
                                                           const std::optional<SpeculativeExecutionConfig>& sp_config) {
    size_t reserve_runtime_mem_bytes = runtime_config.reserve_runtime_mem_mb * 1024 * 1024;
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

    // Reserve at least 5% of total GPU memory for runtime (forward pass intermediates, cublas workspace, etc.)
    // with a minimum floor of 2048 MiB. This is needed because KV cache is pre-allocated as a fixed block,
    // and the remaining GPU memory must be sufficient for model forward passes.
    size_t                  total_gpu_bytes = 0;
    [[maybe_unused]] size_t free_gpu_bytes  = 0;
#if USING_CUDA
    check_cuda_value(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ROCM
    ROCM_CHECK(hipMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#endif
    const auto minimal_runtime_bytes = std::max(2048L * 1024 * 1024, (long)(total_gpu_bytes * 0.05));
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        RTP_LLM_LOG_INFO("tp_size %d needs at least %ld MiB memory for runtime by default, "
                         "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                         parallelism_config.get_attn_tp_size(),
                         minimal_runtime_bytes / 1024 / 1024,
                         reserve_runtime_mem_bytes / 1024 / 1024);
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
    }

    if (model_config.mm_model_config.is_multimodal) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        updateMemoryIfNeeded(reserve_runtime_mem_bytes, minimal_runtime_required, "multimodal");
    }

    if (sp_config && sp_config->type != SP_TYPE_NONE) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        updateMemoryIfNeeded(reserve_runtime_mem_bytes, minimal_runtime_required, "speculative decoding");
    }

    return reserve_runtime_mem_bytes;
}

size_t MemoryEvaluationHelper::getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                                    const KVCacheConfig&                             kv_cache_config,
                                                    const ModelConfig&                               model_config,
                                                    const ParallelismConfig&                         parallelism_config,
                                                    const std::optional<WarmUpResult>&               warm_up_result,
                                                    const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const size_t device_reserved_memory_bytes = getGpuExecStatus().device_memory_status.available_bytes;
    const size_t env_runtime_required_bytes = kv_cache_config.kv_cache_mem_mb > 0 ? 0 :
        getDefaultRuntimeMemorySize(runtime_config, parallelism_config, model_config, sp_config);
    return getKVCacheMemorySizeForBudget(runtime_config,
                                        kv_cache_config,
                                        model_config,
                                        parallelism_config,
                                        device_reserved_memory_bytes,
                                        env_runtime_required_bytes,
                                        warm_up_result);
}

size_t MemoryEvaluationHelper::getKVCacheMemorySizeForBudget(
    const RuntimeConfig&               runtime_config,
    const KVCacheConfig&               kv_cache_config,
    const ModelConfig&                 model_config,
    const ParallelismConfig&           parallelism_config,
    size_t                             device_reserved_memory_bytes,
    size_t                             env_runtime_required_bytes,
    const std::optional<WarmUpResult>& warm_up_result) {
    size_t runtime_required_bytes       = 0;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        // Explicit KV sizing leaves runtime and capture headroom to the caller.
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        runtime_required_bytes = std::max(env_runtime_required_bytes, warm_up_result->max_used_memory);

        RTP_LLM_LOG_INFO(
            "devices reserved %ld MiB memory, warm up consumed %ld MiB max memory, env runtime memory %ld MiB, final runtime memory %ld MiB",
            device_reserved_memory_bytes / 1024 / 1024,
            warm_up_result->max_used_memory / 1024 / 1024,
            env_runtime_required_bytes / 1024 / 1024,
            runtime_required_bytes / 1024 / 1024);
    } else {
        runtime_required_bytes = env_runtime_required_bytes;
        RTP_LLM_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                         runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
    RTP_LLM_LOG_INFO("sampler needs %ld MiB memory, model runtime needs %ld MiB memory, take max value.",
                     sample_need_mem / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024);
    runtime_required_bytes = std::max(sample_need_mem, runtime_required_bytes);

    // Match target-role selection, publisher ownership and stream chunking without a cache -> engine dependency.
    // Python canonical-hook capability is validated later; has_post_decoder_layernorm is not that capability.
    const bool capture_owner = !model_config.hidden_state_capture_layer_ids.empty()
                               && (parallelism_config.role_type == RoleType::PREFILL
                                   || parallelism_config.role_type == RoleType::PDFUSION)
                               && parallelism_config.tp_rank == 0 && parallelism_config.pp_size == 1
                               && runtime_config.fifo_scheduler_config.prefill_chunk_size <= 0;
    const size_t capture_headroom_bytes = capture_owner && model_config.hidden_state_capture_max_inflight_bytes > 0 ?
        static_cast<size_t>(model_config.hidden_state_capture_max_inflight_bytes) : 0;
    // This CPU/GPU logical quota excludes transient capture/CP/quantizer/staging allocations, not a GPU peak bound.
    if (capture_headroom_bytes > 0) {
        RTP_LLM_LOG_INFO("capture logical headroom %zu MiB is additional to runtime memory %zu MiB",
                         capture_headroom_bytes / 1024 / 1024,
                         runtime_required_bytes / 1024 / 1024);
    }
    // Check by subtraction so runtime + capture cannot overflow or exhaust the KV budget.
    RTP_LLM_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes
                                && device_reserved_memory_bytes - runtime_required_bytes > capture_headroom_bytes,
                            "device reserved memory %zu MiB must exceed runtime required memory %zu MiB plus "
                            "capture logical headroom %zu MiB",
                            device_reserved_memory_bytes / 1024 / 1024,
                            runtime_required_bytes / 1024 / 1024,
                            capture_headroom_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes - capture_headroom_bytes;
    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %ld MiB", kv_cache_mem_size / 1024 / 1024);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm

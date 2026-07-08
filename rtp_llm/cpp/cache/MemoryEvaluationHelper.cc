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
#include "autil/EnvUtil.h"
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

    // NOTE: the old "max(2048 MiB, 5% of total)" floor has been removed. Runtime headroom is now an
    // additive safety margin on top of the measured forward peak (see RUNTIME_MEM_SAFETY_RATIO in
    // getKVCacheMemorySize), so it no longer vanishes once the warmup peak exceeds the floor.
    // reserve_runtime_mem_mb is kept only as an optional manual lower bound.
    (void)parallelism_config;

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
    const auto   gpu_mem                      = getGpuExecStatus().device_memory_status;
    size_t       device_reserved_memory_bytes = gpu_mem.available_bytes;
    const size_t total_gpu_bytes              = gpu_mem.used_bytes + gpu_mem.free_bytes;
    size_t       runtime_required_bytes       = 0;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    size_t env_runtime_required_bytes = MemoryEvaluationHelper::getDefaultRuntimeMemorySize(
        runtime_config, parallelism_config, model_config, sp_config);

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

    // NOTE: the MoE worst-case headroom is no longer an analytic term here. During warmup the MoE
    // module (FusedMoe.forward, gated on is_trace_memory()) forces worst-case routing -- every rank
    // dispatches ALL its tokens to a single rank's experts, so that rank receives the absolute
    // per-rank max all_tokens = ep_size * T_local * top_k -- and its measured peak becomes the
    // binding (min block_num) constraint. The warmup peak (warm_up_result->max_used_memory) thus
    // ALREADY contains the skewed MoE activation. No model/kernel-specific per-token formula is
    // needed; what the dummy can't model is left to the additive safety headroom below.

    // Additive safety headroom for everything still un-modeled — mainly sustained NCCL/DeepEP
    // dispatch buffers that a single warmup forward under-measures (warmup non_torch ~70 MiB vs
    // runtime ~3.4 GiB per rank observed under long-context concurrent load), plus cuda-graph
    // pool + allocator fragmentation. Empirically 5% is too tight for high-EP long-context;
    // 10% clears real LongBench 40×63k stress on 4×H20 (see doc §8). Override via env.
    const double safety_ratio          = autil::EnvUtil::getEnv("RUNTIME_MEM_SAFETY_RATIO", 0.10);
    const size_t safety_headroom_bytes = static_cast<size_t>(total_gpu_bytes * safety_ratio);
    runtime_required_bytes += safety_headroom_bytes;

    RTP_LLM_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes,
                            "device reserved memory %ld  MiB is less than runtime required memory %ld MiB",
                            device_reserved_memory_bytes / 1024 / 1024,
                            runtime_required_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %ld MiB", kv_cache_mem_size / 1024 / 1024);
    RTP_LLM_LOG_WARNING(
        "[KV_ALLOC] warm_up=%d device_reserved=%ld MiB | runtime_required parts: torch=%ld MiB non_torch=%ld MiB "
        "(warmup peak incl. worst-case MoE skew) safety_%.0f%%=%ld MiB | runtime_required=%ld MiB => "
        "kv_cache_free=%ld MiB (%.2f GiB)",
        warm_up_result.has_value(),
        device_reserved_memory_bytes / 1024 / 1024,
        warm_up_result ? warm_up_result->torch_peak_increase / 1024 / 1024 : 0,
        warm_up_result ? warm_up_result->non_torch_increase / 1024 / 1024 : 0,
        safety_ratio * 100,
        safety_headroom_bytes / 1024 / 1024,
        runtime_required_bytes / 1024 / 1024,
        kv_cache_mem_size / 1024 / 1024,
        kv_cache_mem_size / 1024.0 / 1024.0 / 1024.0);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm

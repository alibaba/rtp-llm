#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

#include <algorithm>
#include <cstdint>
#include <limits>

// No platform headers here on purpose: this file queries device memory only through
// getGpuExecStatus(), which owns the CUDA/ROCm calls.
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
constexpr size_t kBytesPerMiB = 1024 * 1024;

size_t checkedMiBToBytes(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value >= 0, "%s must be non-negative, got %ld", name, value);
    RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(value) <= std::numeric_limits<size_t>::max() / kBytesPerMiB,
                            "%s is too large: %ld MiB",
                            name,
                            value);
    return static_cast<size_t>(value) * kBytesPerMiB;
}

}  // namespace

// Helper function to update memory size if below minimum requirement
void MemoryEvaluationHelper::updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario) {
    if (current_size < min_required) {
        const size_t original_size = current_size;
        current_size               = min_required;
        RTP_LLM_LOG_INFO("%s runtime memory reserve adjusted from %ld MiB to %ld MiB",
                         scenario,
                         original_size / 1024 / 1024,
                         min_required / 1024 / 1024);
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

size_t MemoryEvaluationHelper::getDefaultRuntimeMemorySize(const RuntimeConfig& runtime_config,
                                                           const ModelConfig&   model_config,
                                                           const std::optional<SpeculativeExecutionConfig>& sp_config) {
    size_t reserve_runtime_mem_bytes =
        checkedMiBToBytes(runtime_config.reserve_runtime_mem_mb, "reserve_runtime_mem_mb");
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

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
                                                    const std::optional<WarmUpResult>&               warm_up_result,
                                                    const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto   gpu_mem                      = getGpuExecStatus().device_memory_status;
    size_t       device_reserved_memory_bytes = gpu_mem.available_bytes;
    const size_t total_gpu_bytes              = gpu_mem.total_bytes;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return checkedMiBToBytes(kv_cache_config.kv_cache_mem_mb, "kv_cache_mem_mb");
    }

    size_t configured_runtime_required_bytes =
        MemoryEvaluationHelper::getDefaultRuntimeMemorySize(runtime_config, model_config, sp_config);

    size_t warmup_required_bytes      = 0;
    bool   warmup_measurement_valid   = false;
    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        // measured_peak_growth_bytes is torch_peak + non_torch_transient. It deliberately omits
        // non_torch_resident: device_reserved_memory_bytes is cudaMemGetInfo free memory sampled
        // *after* the warmup teardown, so it already excludes every allocation still alive then,
        // and reserving those bytes here would subtract them a second time. Which half a given
        // allocation falls into is measured rather than assumed (see makeWarmUpResult in
        // NormalEngine.cc), because it differs by role: PREFILL growth is mostly process-global
        // lazy initialisation that survives teardown, while DECODE also allocates per-captured-
        // graph driver state that is released with the executor and does have to be reserved.
        warmup_required_bytes    = warm_up_result->measured_peak_growth_bytes;
        warmup_measurement_valid = warmup_required_bytes > 0;
        if (!warmup_measurement_valid) {
            // A real warmup forward always grows the torch allocator peak above the baseline
            // snapshotted in setTraceMemory(true), so 0 means the measurement pipeline broke
            // (trace window not active, baselines not captured, peak stats reset elsewhere).
            // Sizing with 0 would make the additive warmup formula reserve *less* than the
            // untraced path (it has no no_warmup_floor), so fall back to the no-warmup formula.
            RTP_LLM_LOG_WARNING(
                "warmup ran but measured_peak_growth_bytes is 0, which a real forward cannot produce: "
                "the memory-trace measurement is broken. Falling back to the no-warmup sizing formula "
                "(including runtime_mem_no_warmup_floor_mb) so the runtime reserve cannot drop below "
                "the untraced path. [KV_ALLOC] below reports warm_up=0 for this reason.");
        }

        // non_torch_transient assumes the warmup rank had the GPU to itself: another
        // process allocating during warmup inflates it and every inflated byte is
        // silently reserved away from the KV cache. Flag it instead of relying on
        // someone grepping the [KV_ALLOC] split.
        const size_t transient_warn_bytes = std::max<size_t>(1024 * kBytesPerMiB, total_gpu_bytes / 50);
        if (warm_up_result->non_torch_transient > transient_warn_bytes) {
            RTP_LLM_LOG_WARNING(
                "warmup non_torch_transient=%ld MiB exceeds %ld MiB (max(1 GiB, 2%% of total GPU)): either an "
                "external process allocated on this GPU during warmup, or the warmup teardown returned an "
                "unexpectedly large amount of driver memory. The full amount is reserved away from the KV "
                "cache; if it is misattributed, correct it via runtime_mem_safety_ratio or pin the cache size "
                "with an explicit kv_cache_mem_mb.",
                warm_up_result->non_torch_transient / 1024 / 1024,
                transient_warn_bytes / 1024 / 1024);
        }
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
    const double safety_ratio = kv_cache_config.runtime_mem_safety_ratio;
    const size_t no_warmup_floor_bytes =
        checkedMiBToBytes(kv_cache_config.runtime_mem_no_warmup_floor_mb, "runtime_mem_no_warmup_floor_mb");
    // Named assignment on purpose: four of these fields are adjacent size_t, so a
    // positional aggregate would compile fine with any two of them swapped.
    RuntimeMemorySizingInput sizing_input;
    sizing_input.has_warmup               = warmup_measurement_valid;
    sizing_input.configured_reserve_bytes = configured_runtime_required_bytes;
    sizing_input.warmup_required_bytes    = warmup_required_bytes;
    sizing_input.sampler_required_bytes   = sample_need_mem;
    sizing_input.total_gpu_bytes          = total_gpu_bytes;
    sizing_input.safety_ratio             = safety_ratio;
    sizing_input.no_warmup_floor_bytes    = no_warmup_floor_bytes;
    // The sizing layer stays dependency-free and reports configuration errors as
    // std exceptions; convert them here so they go through myAssert's ERROR log
    // and core-dump switch like every other startup configuration failure.
    RuntimeMemorySizingResult sizing;
    try {
        sizing = calculateRuntimeMemorySizing(sizing_input);
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("%s", e.what());
    }
    const size_t runtime_required_bytes = sizing.runtime_required_bytes;

    // Name every knob that feeds runtime_required and its live value: this aborts startup, so
    // the message has to say what to turn down without the operator reading the sizing code.
    RTP_LLM_CHECK_WITH_INFO(
        device_reserved_memory_bytes > runtime_required_bytes,
        "device reserved memory %ld MiB is less than runtime required memory %ld MiB. "
        "runtime_required = %s (configured=%ld MiB, safety_term=%ld MiB), currently configured as "
        "reserver_runtime_mem_mb=%ld, runtime_mem_safety_ratio=%.4f, runtime_mem_no_warmup_floor_mb=%ld "
        "(sampler estimate=%ld MiB, total GPU=%ld MiB). Reduce reserver_runtime_mem_mb / "
        "runtime_mem_safety_ratio%s, or bypass this sizing entirely with an explicit kv_cache_mem_mb.",
        device_reserved_memory_bytes / 1024 / 1024,
        runtime_required_bytes / 1024 / 1024,
        warmup_measurement_valid ? "max(configured, measured_peak, sampler) + safety_term" :
                                   "max(configured, sampler, no_warmup_floor, safety_term)",
        configured_runtime_required_bytes / 1024 / 1024,
        sizing.safety_ratio_bytes / 1024 / 1024,
        runtime_config.reserve_runtime_mem_mb,
        safety_ratio,
        kv_cache_config.runtime_mem_no_warmup_floor_mb,
        sample_need_mem / 1024 / 1024,
        total_gpu_bytes / 1024 / 1024,
        warmup_measurement_valid ? "" : " / runtime_mem_no_warmup_floor_mb");

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    // Every input of the sizing decision is logged so total is derivable (see
    // calculateRuntimeMemorySizing):
    //   warm_up=1: total = max(configured, measured_peak, sampler) + safety (additive headroom)
    //   warm_up=0: total = max(configured, sampler, no_warmup_floor, safety) -- the pre-warmup
    //              formula, where the ratio term is a floor, so upgrades without a traced warmup
    //              keep their old KV cache size.
    // warm_up reflects the formula actually used: a warmup whose measurement came back 0 is
    // degraded to the no-warmup formula (WARNING above) and reports warm_up=0 here.
    // measured_peak = torch_peak + non_torch_transient. non_torch_resident is logged but is not a
    // term: it is already absent from device_reserved. Watch the split -- non_torch_transient
    // growing into the GiB range means the warmup teardown is handing back far more driver memory
    // than expected and this sizing needs revisiting.
    RTP_LLM_LOG_INFO("[KV_ALLOC] warm_up=%d device_reserved=%ld MiB | runtime_required: measured_peak=%ld MiB "
                     "configured=%ld MiB sampler=%ld MiB "
                     "no_warmup_floor=%ld MiB safety_%.0f%%=%ld MiB total=%ld MiB | "
                     "non_torch_transient=%ld MiB (in measured_peak) non_torch_resident=%ld MiB "
                     "(already excluded from device_reserved) | kv_cache_free=%ld MiB (%.2f GiB)",
                     warmup_measurement_valid,
                     device_reserved_memory_bytes / 1024 / 1024,
                     warmup_required_bytes / 1024 / 1024,
                     configured_runtime_required_bytes / 1024 / 1024,
                     sample_need_mem / 1024 / 1024,
                     no_warmup_floor_bytes / 1024 / 1024,
                     safety_ratio * 100,
                     sizing.safety_ratio_bytes / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024,
                     warm_up_result ? warm_up_result->non_torch_transient / 1024 / 1024 : 0,
                     warm_up_result ? warm_up_result->non_torch_resident / 1024 / 1024 : 0,
                     kv_cache_mem_size / 1024 / 1024,
                     kv_cache_mem_size / 1024.0 / 1024.0 / 1024.0);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm

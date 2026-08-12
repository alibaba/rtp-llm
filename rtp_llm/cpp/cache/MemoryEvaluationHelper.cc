#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"
#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

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

size_t
MemoryEvaluationHelper::getConfiguredRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                                       const ModelConfig&                               model_config,
                                                       const std::optional<SpeculativeExecutionConfig>& sp_config) {
    // The C++ field drops the trailing "r" of the CLI flag; operator-facing text carries both
    // spellings so either one greps.
    static constexpr const char* kReserveKnobName = "reserve_runtime_mem_mb (--reserver_runtime_mem_mb)";
    size_t reserve_runtime_mem_bytes = checkedMiBToBytes(runtime_config.reserve_runtime_mem_mb, kReserveKnobName);
    RTP_LLM_LOG_INFO("RuntimeConfig has %s=%ld", kReserveKnobName, runtime_config.reserve_runtime_mem_mb);

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
    return getKVCacheMemorySize(runtime_config,
                                kv_cache_config,
                                model_config,
                                getGpuExecStatus().device_memory_status,
                                warm_up_result,
                                sp_config);
}

size_t MemoryEvaluationHelper::getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                                    const KVCacheConfig&                             kv_cache_config,
                                                    const ModelConfig&                               model_config,
                                                    const MemoryStatus&                              gpu_memory_status,
                                                    const std::optional<WarmUpResult>&               warm_up_result,
                                                    const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto&  gpu_mem                      = gpu_memory_status;
    size_t       device_reserved_memory_bytes = gpu_mem.available_bytes;
    const size_t total_gpu_bytes              = gpu_mem.total_bytes;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        RTP_LLM_LOG_INFO("[KV_ALLOC] warm_up=0 source=explicit total=%ld MiB", kv_cache_config.kv_cache_mem_mb);
        return checkedMiBToBytes(kv_cache_config.kv_cache_mem_mb, "kv_cache_mem_mb");
    }

    size_t configured_runtime_required_bytes =
        MemoryEvaluationHelper::getConfiguredRuntimeMemorySize(runtime_config, model_config, sp_config);

    size_t warmup_required_bytes    = 0;
    bool   warmup_measurement_valid = false;
    if (warm_up_result) {
        warmup_required_bytes    = warm_up_result->measured_total_growth_bytes;
        warmup_measurement_valid = warm_up_result->measurement_trusted && warmup_required_bytes > 0;

        if (warmup_measurement_valid) {
            // Base and growth term are read at the same instant: the pool as it was before the
            // warmup allocated anything, against the warmup's *total* growth. Serving needs all of
            // that growth on top of the weights, so nothing here has to know which share the
            // teardown returned.
            //
            // Deliberately NOT min()'d against the config-time sample below. That sample is taken
            // after the teardown, so it is smaller by exactly what the warmup left resident -- and
            // those bytes are already a term in measured_total_growth_bytes. Taking the min would
            // subtract them twice, which is the over-reservation this pairing exists to avoid.
            device_reserved_memory_bytes = warm_up_result->available_bytes_pre_warmup;

            // What the warmup permanently cost the device must not exceed what it was measured to
            // have grown. After allocation the leftover is (growth - pool_shrink) + safety, so any
            // excess of pool_shrink over growth comes straight out of the safety margin. It means
            // memory vanished during the window without being measured as growth: allocated after
            // the peak sample (teardown-side driver work) or by another process on this GPU.
            const size_t pool_shrink = poolShrinkBytes(*warm_up_result);
            if (pool_shrink > warmup_required_bytes) {
                RTP_LLM_LOG_WARNING(
                    "[KV_ALLOC_POOL_SHRINK] warmup cost the device %ld MiB permanently but was only measured to grow "
                    "%ld MiB: the %ld MiB "
                    "difference was allocated after the peak sample (teardown-side driver work) or by another process "
                    "on this GPU. It is not part of the reserve and is therefore taken out of the safety margin. "
                    "Raise runtime_mem_safety_ratio or reserver_runtime_mem_mb to cover it, or pin the cache size "
                    "with an explicit kv_cache_mem_mb.",
                    pool_shrink / 1024 / 1024,
                    warmup_required_bytes / 1024 / 1024,
                    (pool_shrink - warmup_required_bytes) / 1024 / 1024);
            }
        } else {
            // Measurement unused: keep the inherited base, the pool as it was after the teardown.
            // A static reserve does not account for what the warmup left resident, so the base has
            // to be the one that already excludes it.
            if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
                RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                    "the amount when warm up %ld. take min value.",
                                    device_reserved_memory_bytes,
                                    warm_up_result->device_reserved_bytes);
                device_reserved_memory_bytes =
                    std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
            }

            if (!warm_up_result->measurement_trusted) {
                // Deliberate, not broken: PDFUSION runs the warmup forward only for lazy init and
                // the post-forward device_reserved sample, and always sizes with the pre-upgrade
                // no-warmup formula. [KV_ALLOC] below reports warm_up=0 for this reason.
                RTP_LLM_LOG_INFO("warmup measurement is deliberately discarded for this role; "
                                 "sizing with the no-warmup formula against the post-teardown pool.");
            } else {
                // A real forward always grows the torch allocator above its baseline, so a total
                // growth of 0 means the measurement pipeline broke (trace window not active,
                // baselines not captured, peak stats reset elsewhere) rather than anything a
                // forward can produce. Sizing with 0 would make the additive warmup formula reserve
                // *less* than the untraced path (it has no no_warmup_floor), so fall back.
                RTP_LLM_LOG_WARNING(
                    "warmup ran but measured_total_growth_bytes is 0, which a real forward cannot produce: "
                    "the memory-trace measurement is broken. Falling back to the no-warmup sizing formula "
                    "(including runtime_mem_no_warmup_floor_mb) so the runtime reserve cannot drop below "
                    "the untraced path. [KV_ALLOC] below reports warm_up=0 for this reason.");
            }
        }
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
    const double safety_ratio = kv_cache_config.runtime_mem_safety_ratio;
    const size_t no_warmup_floor_bytes =
        checkedMiBToBytes(kv_cache_config.runtime_mem_no_warmup_floor_mb, "runtime_mem_no_warmup_floor_mb");
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

    // Observable entry point for the one behaviour change on the warmup path: the
    // pre-feature max(2048 MiB, 5% x total) hard minimum no longer applies, so a small
    // model on a small GPU can end up reserving less than it did before the upgrade.
    // The formula is unchanged on purpose (a measurement is meant to replace the
    // guesswork floors); this WARNING just makes affected instances greppable during a
    // rollout instead of surfacing later as a runtime OOM.
    if (sizing.warmup_below_no_warmup_floor) {
        RTP_LLM_LOG_WARNING(
            "[KV_ALLOC_BELOW_FLOOR] warmup runtime reserve %ld MiB is below the no-warmup floor %ld MiB "
            "(runtime_mem_no_warmup_floor_mb), which this deployment would have reserved before the "
            "forward-warmup feature: the warmup path sizes from the measured peak and deliberately "
            "does not apply that floor. This is expected for small models on small GPUs. To restore "
            "an absolute floor on this path set --reserver_runtime_mem_mb (currently %ld MiB); "
            "measured_growth=%ld MiB, safety_term=%ld MiB.",
            runtime_required_bytes / 1024 / 1024,
            no_warmup_floor_bytes / 1024 / 1024,
            runtime_config.reserve_runtime_mem_mb,
            warmup_required_bytes / 1024 / 1024,
            sizing.safety_ratio_bytes / 1024 / 1024);
    }

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
        warmup_measurement_valid ? "max(configured, measured_growth, sampler) + safety_term" :
                                   "max(configured, sampler, no_warmup_floor, safety_term)",
        configured_runtime_required_bytes / 1024 / 1024,
        sizing.safety_ratio_bytes / 1024 / 1024,
        runtime_config.reserve_runtime_mem_mb,
        safety_ratio,
        kv_cache_config.runtime_mem_no_warmup_floor_mb,
        sample_need_mem / 1024 / 1024,
        total_gpu_bytes / 1024 / 1024,
        warmup_measurement_valid ? "" : " / runtime_mem_no_warmup_floor_mb");

    auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;

    // Hard upper bound on the trusted path. The base above is available_bytes_pre_warmup, sampled
    // before the warmup allocated anything, while the pool physically free at this point is
    // gpu_mem.available_bytes (smaller by what the warmup left resident). The additive formula
    // already covers that resident share via measured_growth, so normally the budget fits and
    // nothing below binds.
    //
    // The cap is against available_bytes - safety_ratio_bytes, NOT
    // available_bytes - runtime_required_bytes: runtime_required contains the measured growth,
    // which is a pre-warmup-frame quantity, whereas available_bytes is a post-teardown sample.
    // Comparing them would mix frames and make the cap bind even on healthy layouts. The safety
    // term is the one component of runtime_required derived from total GPU memory rather than from
    // any growth measurement, so it is frame-independent -- and it is exactly the margin that
    // exists to absorb this kind of measurement uncertainty.
    //
    // Consequence: the cap binds precisely when pool_shrink exceeds max(configured, growth), i.e.
    // the same abnormal condition the pool-shrink WARNING above reports. When it binds, the safety
    // margin is what survives; capping to available_bytes alone would leave nothing and turn this
    // startup guard into a serving-time OOM on the first forward.
    if (warmup_measurement_valid) {
        RTP_LLM_CHECK_WITH_INFO(gpu_mem.available_bytes > sizing.safety_ratio_bytes,
                                "only %ld MiB is actually free at cache-config time, which cannot even hold the "
                                "%ld MiB safety margin (runtime_mem_safety_ratio=%.4f of %ld MiB total). The warmup "
                                "left %ld MiB resident but was only measured to grow %ld MiB, so the measurement "
                                "cannot be trusted to size anything. Reduce runtime_mem_safety_ratio, or pin the "
                                "cache size with an explicit kv_cache_mem_mb.",
                                gpu_mem.available_bytes / 1024 / 1024,
                                sizing.safety_ratio_bytes / 1024 / 1024,
                                safety_ratio,
                                total_gpu_bytes / 1024 / 1024,
                                poolShrinkBytes(*warm_up_result) / 1024 / 1024,
                                warmup_required_bytes / 1024 / 1024);
        const size_t max_kv_bytes = gpu_mem.available_bytes - sizing.safety_ratio_bytes;
        if (kv_cache_mem_size > max_kv_bytes) {
            const size_t config_time_pool_shrink =
                warm_up_result->available_bytes_pre_warmup > gpu_mem.available_bytes ?
                    warm_up_result->available_bytes_pre_warmup - gpu_mem.available_bytes :
                    0;
            RTP_LLM_LOG_WARNING(
                "[KV_ALLOC_CAPPED] KV cache budget %ld MiB would leave less than the %ld MiB safety margin free out "
                "of the %ld MiB "
                "actually available at cache-config time; capping to %ld MiB. Config-time free memory is %ld MiB "
                "below the pre-warmup sample while measured growth is %ld MiB. Raise runtime_mem_safety_ratio or "
                "reserver_runtime_mem_mb to widen the margin, or pin the cache size with an explicit "
                "kv_cache_mem_mb.",
                kv_cache_mem_size / 1024 / 1024,
                sizing.safety_ratio_bytes / 1024 / 1024,
                gpu_mem.available_bytes / 1024 / 1024,
                max_kv_bytes / 1024 / 1024,
                config_time_pool_shrink / 1024 / 1024,
                warmup_required_bytes / 1024 / 1024);
            kv_cache_mem_size = max_kv_bytes;
        }
    }

    // Every input of the sizing decision is logged so total is derivable (see
    // calculateRuntimeMemorySizing):
    //   warm_up=1: total = max(configured, measured_growth, sampler) + safety (additive headroom),
    //              divided out of the pool as it was *before* the warmup allocated anything.
    //   warm_up=0: total = max(configured, sampler, no_warmup_floor, safety) -- the pre-warmup
    //              formula, divided out of the pool *after* the warmup teardown, so upgrades
    //              without a trusted warmup keep their old KV cache size.
    // base names which pool that is, because the two paths do not use the same one: the measured
    // path pairs the pre-warmup pool with the total growth (same instant on both sides), while a
    // discarded or broken measurement has no growth term to account for what the warmup left
    // resident and therefore has to divide the pool that already excludes it.
    // warm_up reflects the formula actually used: a warmup whose measurement came back 0 is
    // degraded to the no-warmup formula (WARNING above) and reports warm_up=0 here, as does a
    // deliberately discarded measurement (PDFUSION, measurement_trusted=false, INFO above) --
    // measured_growth is still printed for diagnostics in both cases but is not a term.
    // The per-sample memory readings behind measured_growth are logged by the [*_WARMUP] result
    // line in NormalEngine.cc.
    RTP_LLM_LOG_INFO("[KV_ALLOC] warm_up=%d base=%ld MiB (%s) | runtime_required: measured_growth=%ld MiB "
                     "configured=%ld MiB sampler=%ld MiB "
                     "no_warmup_floor=%ld MiB safety_%.0f%%=%ld MiB total=%ld MiB "
                     "| kv_cache_free=%ld MiB (%.2f GiB)",
                     warmup_measurement_valid,
                     device_reserved_memory_bytes / 1024 / 1024,
                     warmup_measurement_valid ? "pre_warmup" : "post_teardown",
                     warmup_required_bytes / 1024 / 1024,
                     configured_runtime_required_bytes / 1024 / 1024,
                     sample_need_mem / 1024 / 1024,
                     no_warmup_floor_bytes / 1024 / 1024,
                     safety_ratio * 100,
                     sizing.safety_ratio_bytes / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024,
                     kv_cache_mem_size / 1024 / 1024,
                     kv_cache_mem_size / 1024.0 / 1024.0 / 1024.0);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm

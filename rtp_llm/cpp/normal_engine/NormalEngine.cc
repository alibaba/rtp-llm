#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/WarmupRoleGate.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <thread>
#include <random>

#if USING_CUDA
#include "c10/cuda/CUDACachingAllocator.h"
#endif

#ifdef __linux__
#include <malloc.h>
#endif

using namespace std;
namespace rtp_llm {

namespace {
// 释放glibc缓存的host内存，将其归还给操作系统
// 在模型加载完成后调用，可以显著减少常驻内存占用
void releaseHostMemoryCache() {
#ifdef __linux__
    // malloc_trim(0) 会释放所有可以释放的内存回操作系统
    // 这对于checkpoint加载后释放临时分配的大量CPU内存很重要
    int result = malloc_trim(0);
    RTP_LLM_LOG_INFO("Released host memory cache to OS (malloc_trim returned %d)", result);
#else
    RTP_LLM_LOG_DEBUG("malloc_trim not available on this platform");
#endif
}

bool cacheStatusSnapshotEnabled() {
    const char* env = std::getenv("RTP_LLM_CACHE_STATUS_SNAPSHOT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

// Must keep the same parsing semantics as NormalExecutor::useDeviceInput()
// (RTP_LLM_DEVICE_INPUT == "1"); the guard below rejects combinations that
// the executor's device-input path cannot handle.
bool deviceInputEnabled() {
    const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

bool shouldRefreshCacheStatusSnapshot(RoleType role_type, const std::list<GenerateStreamPtr>& streams) {
    if (!cacheStatusSnapshotEnabled() || (role_type != RoleType::PREFILL && role_type != RoleType::PDFUSION)) {
        return false;
    }
    return std::any_of(streams.begin(), streams.end(), [](const GenerateStreamPtr& stream) {
        return stream && !stream->isFakeStream() && stream->isContextStream();
    });
}

#if USING_CUDA
// Starts tracing on construction and, on every exit path including a throwing preRun, destroys
// the warmup executor before closing the trace phase.
class WarmupTraceScope {
public:
    explicit WarmupTraceScope(std::unique_ptr<Executor>& executor): executor_(executor) {
        rtp_llm::setTraceMemory(true);
        // Sampled after setTraceMemory on purpose: that call runs emptyCache() and snapshots the
        // growth baselines, so free memory here is the same steady state (weights only) the growth
        // deltas are measured against. Sampling before it would include loader-cached free blocks
        // the warmup is free to reuse, overstating the pool relative to the growth.
        pre_warmup_available_bytes_ = getGpuExecStatus().device_memory_status.available_bytes;
    }

    ~WarmupTraceScope() {
        executor_.reset();
        rtp_llm::setTraceMemory(false);
    }

    // Releases the traced executor and samples memory again while the trace window is still open.
    // Two consumers: the raw post-teardown readings, logged to show what the warmup left behind,
    // and available_bytes, which is the sizing base for the paths that discard the measurement.
    // Sampling after setTraceMemory(false) would not work: closing the phase makes isTraceMemory()
    // false, so getGpuExecStatus() no longer fills the traced growth fields and they read back as 0.
    MemoryStatus teardownAndSample() {
        executor_.reset();
        cudaDeviceSynchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
        return getGpuExecStatus().device_memory_status;
    }

    // Free memory at the trace baseline: weights loaded, warmup has allocated nothing.
    size_t preWarmupAvailableBytes() const {
        return pre_warmup_available_bytes_;
    }

    WarmupTraceScope(const WarmupTraceScope&)            = delete;
    WarmupTraceScope& operator=(const WarmupTraceScope&) = delete;

private:
    std::unique_ptr<Executor>& executor_;
    size_t                     pre_warmup_available_bytes_ = 0;
};

// Thin wrapper over assembleWarmUpResult: it owns the logging and turns the pure layer's
// std::exception into myAssert, exactly like the sizing layer's caller does.
WarmUpResult makeWarmUpResult(size_t              pre_warmup_available_bytes,
                              const MemoryStatus& peak_status,
                              const MemoryStatus& post_teardown_status,
                              bool                measurement_trusted,
                              const char*         phase) {
    WarmUpResult result;
    try {
        result =
            assembleWarmUpResult(pre_warmup_available_bytes, peak_status, post_teardown_status, measurement_trusted);
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("%s %s", phase, e.what());
    }

    // Raw samples, not derived halves: the growth term is torch_peak + non_torch_at_peak, and
    // pool_shrink (how much the warmup permanently cost the device) is the difference between the
    // two free readings. torch_current_after_teardown is what the warmup left behind -- kept
    // because nothing else in the log answers that, and a large value there is worth investigating.
    const size_t pool_shrink = poolShrinkBytes(result);

    RTP_LLM_LOG_INFO("[%s] result: available_pre_warmup=%ld available_post_teardown=%ld pool_shrink=%ld | "
                     "total_growth=%ld = torch_peak=%ld + non_torch_at_peak=%ld | "
                     "torch_current_after_teardown=%ld non_torch_after_teardown=%ld",
                     phase,
                     (long)result.available_bytes_pre_warmup,
                     (long)result.device_reserved_bytes,
                     (long)pool_shrink,
                     (long)result.measured_total_growth_bytes,
                     (long)peak_status.max_consumed_bytes,
                     (long)peak_status.non_torch_increase_bytes,
                     (long)post_teardown_status.torch_current_increase_bytes,
                     (long)post_teardown_status.non_torch_increase_bytes);

    // The non-torch growth assumes the warmup rank had the GPU to itself: it is a residual (device
    // used minus torch reserved), so another process allocating during the window lands in it and is
    // then reserved away from the KV cache. Warned here rather than at the sizing layer because this
    // is where the raw sample lives.
    const size_t non_torch_warn_bytes = std::max<size_t>(1024UL * 1024 * 1024, peak_status.total_bytes / 50);
    if (peak_status.non_torch_increase_bytes > non_torch_warn_bytes) {
        RTP_LLM_LOG_WARNING("[%s] non_torch growth %ld MiB exceeds %ld MiB (max(1 GiB, 2%% of total GPU)): either an "
                            "external process allocated on this GPU during warmup, or the warmup allocated an "
                            "unexpectedly large amount of driver memory. It is part of the growth reserved away "
                            "from the KV cache; if it is misattributed, correct it via runtime_mem_safety_ratio or "
                            "pin the cache size with an explicit kv_cache_mem_mb.",
                            phase,
                            (long)(peak_status.non_torch_increase_bytes / 1024 / 1024),
                            (long)(non_torch_warn_bytes / 1024 / 1024));
    }
    return result;
}
#endif

}  // anonymous namespace

NormalEngine::NormalEngine(const EngineInitParams&                       params,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params):
    EngineBase(params),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    runtime_config(params.runtime_config),
    eplb_config(params.eplb_config),
    pd_sep_config(params.pd_sep_config),
    profiling_debug_logging_config(params.profiling_debug_logging_config),
    kv_cache_config(params.kv_cache_config),
    ffn_disaggregate_config(params.ffn_disaggregate_config),
    model_specific_config(params.model_specific_config),
    sp_config(params.sp_config),
    metrics_reporter_(params.metrics_reporter),
    propose_params_(std::move(propose_params)),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank) {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    if (!model_config_.output_vocab_ids.empty()) {
        RTP_LLM_CHECK_WITH_INFO(sp_config.type == SP_TYPE_NONE && !propose_params_,
                                "output vocabulary pruning does not support speculative, MTP, or EAGLE engines");
        RTP_LLM_CHECK_WITH_INFO(!runtime_config.warm_up_with_loss,
                                "output vocabulary pruning does not support warm_up_with_loss");
        // forwardPostLayersLastHidden (prefill CP) is a second lm_head exit that does not
        // narrow P-wide logits down to the output vocabulary width, so padded zero columns
        // would reach sampling. Reject the combination until that path narrows as well.
        RTP_LLM_CHECK_WITH_INFO(!parallelism_config.prefill_cp_config.is_enabled(),
                                "output vocabulary pruning does not support prefill context parallelism");
        // publishNormalDeviceState stores sampler token ids as the next step's device
        // input without restoration; under pruning those are compact ids, which would
        // be fed to the embedding lookup as-is. Reject until that path restores them.
        RTP_LLM_CHECK_WITH_INFO(!deviceInputEnabled(),
                                "output vocabulary pruning does not support device-input mode (RTP_LLM_DEVICE_INPUT)");
        const auto& output_vocab_ids = model_config_.output_vocab_ids;
        RTP_LLM_CHECK_WITH_INFO(std::is_sorted(output_vocab_ids.begin(), output_vocab_ids.end())
                                    && std::adjacent_find(output_vocab_ids.begin(), output_vocab_ids.end())
                                           == output_vocab_ids.end(),
                                "output_vocab_ids must be strictly ascending and deduplicated");
        RTP_LLM_CHECK_WITH_INFO(output_vocab_ids.front() >= 0 && output_vocab_ids.back() < model_config_.vocab_size,
                                "output_vocab_ids must be within [0, vocab_size)");
        RTP_LLM_CHECK_WITH_INFO(model_config_.output_vocab_padded_size >= static_cast<int64_t>(output_vocab_ids.size()),
                                "output_vocab_padded_size must be >= output_vocab_ids.size()");
    }
    if (propose_params_) {
        reserve_step_ = propose_params_->gen_num_per_circle + 1;
    } else {
        reserve_step_ = 0;
    }
    RTP_LLM_LOG_INFO("normal engine speculative reserve_step is %d", reserve_step_);
#if !USING_CUDA
    // On ROCm, this constructor runs on a gRPC handler thread that defaults to
    // GPU 0. Set the correct device so all GPU allocations (KV cache, etc.) go
    // to the right device.  The guard is scoped to the constructor body.
    c10::DeviceGuard ctor_device_guard(
        c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(parallelism_config.local_rank)));
    RTP_LLM_LOG_INFO("ROCm NormalEngine ctor: set device to %d", parallelism_config.local_rank);
#endif

    std::optional<WarmUpResult> warm_up_result = std::nullopt;
#if USING_CUDA
    // The user-visible --warm_up help describes the PD measurement behavior and rollback switch;
    // the complete role/multimodal/FFN-disaggregation eligibility conditions live here.
    const bool is_pd_separated_role = isPdSeparatedRole(pd_sep_config.role_type);
    const bool is_warmup_role       = isWarmupRole(pd_sep_config.role_type);
    if (runtime_config.warm_up && is_warmup_role && (!model_config_.mm_model_config.is_multimodal)
        && !ffn_disaggregate_config.enable_ffn_disaggregate) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         runtime_config.fifo_scheduler_config.max_context_batch_size,
                         model_config_.max_seq_len,
                         int(runtime_config.warm_up_with_loss));
        warm_up_result = warmUp(params);
        if (is_pd_separated_role) {
            // Machine-greppable summary; the values are growth deltas over the warmup, not absolute
            // peaks. Smoke contract (multi_inst_case_runner._assert_warmup_sizing_logs): it regex-
            // matches "[WARMUP_DONE] measured_total_growth_bytes=<int>" and requires the value > 0,
            // so the tag, the field name, and the plain-integer formatting must change together.
            RTP_LLM_LOG_INFO("[WARMUP_DONE] measured_total_growth_bytes=%ld sizing_base=%ld "
                             "(available_bytes_pre_warmup) available_post_teardown=%ld",
                             warm_up_result->measured_total_growth_bytes,
                             warm_up_result->available_bytes_pre_warmup,
                             warm_up_result->device_reserved_bytes);
        } else {
            // PDFUSION: the forward runs only to keep the pre-warmup-feature behavior -- lazy init
            // happens at startup and device_reserved_bytes is sampled *after* a forward, so its
            // resident allocations are already excluded from the KV budget. The measurement itself
            // is deliberately discarded and sizing stays on the no-warmup formula against that
            // post-teardown sample, bit-for-bit the pre-upgrade sizing. This is also why the base
            // must stay the post-teardown pool for this role: with the measurement discarded,
            // nothing would account for what the warmup left resident.
            // Deliberately NOT [WARMUP_DONE]: that tag is a smoke contract requiring
            // measured_total_growth_bytes > 0, which never applies to this role.
            //
            // Coverage: the role -> trusted mapping is gated by WarmupRoleGateTest (host-side), and
            // its consequence (no-warmup formula, warm_up=0, post-teardown base) by
            // MemoryEvaluationHelperTest.KvAllocWarmUpFlagReflectsTheFormulaActuallyUsed.
            RTP_LLM_LOG_INFO("[PDFUSION_WARMUP] forward warmup executed for lazy init and post-forward "
                             "memory sampling; measurement discarded (measured_total_growth_bytes=%ld), "
                             "sizing uses the no-warmup formula against the post-teardown pool. "
                             "sizing_base=%ld (device_reserved_bytes) available_pre_warmup=%ld",
                             warm_up_result->measured_total_growth_bytes,
                             warm_up_result->device_reserved_bytes,
                             warm_up_result->available_bytes_pre_warmup);
        }
    } else {
        RTP_LLM_LOG_INFO("skip forward warm up: enabled=%d role=%d pd_separation=%d multimodal=%d "
                         "ffn_disaggregate=%d.",
                         runtime_config.warm_up,
                         static_cast<int>(pd_sep_config.role_type),
                         is_pd_separated_role,
                         model_config_.mm_model_config.is_multimodal,
                         ffn_disaggregate_config.enable_ffn_disaggregate);
    }
#else
    RTP_LLM_LOG_INFO("skip warm up on non-CUDA platform.");
#endif
    initCacheManager(warm_up_result);
    RTP_LLM_LOG_INFO("create cache manager done");

    initExecutor(params, propose_params_);

    RTP_LLM_LOG_INFO("create normal executor done");

    // 释放模型加载过程中使用的临时host内存
    // 此时checkpoint已加载完成，可以将glibc缓存的内存归还给操作系统
    releaseHostMemoryCache();

    initScheduler();
    step_profiler_.configureFromConfig(profiling_debug_logging_config);
    (void)startLoop();
}

void NormalEngine::initExecutor(const EngineInitParams&                        params,
                                std::unique_ptr<ProposeModelEngineInitParams>& propose_params) {
    if (propose_params_) {
        executor_.reset(new MtpExecutor(
            params, propose_params, resource_context_.cache_manager, mla_ops_type_, kv_cache_group_num_));
    } else {
        executor_.reset(new NormalExecutor(
            params,
            resource_context_.cache_manager,
            false,
            false,
            0,
            mla_ops_type_,
            [this]() { step_profiler_.startStep(); },
            [this]() { step_profiler_.finishStep(); }));
    }
}

void NormalEngine::initScheduler() {
    const auto pdfusion_scheduler_mode =
        parsePDFusionSchedulerMode(runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode);
    if (pdfusion_scheduler_mode == PDFusionSchedulerMode::UNKNOWN) {
        RTP_LLM_LOG_WARNING("unknown pdfusion_scheduler_mode [%s], expected '' or 'ratio'; mode will be ignored",
                            runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode.c_str());
    }
    if (runtime_config.use_batch_decode_scheduler) {
        scheduler_.reset(new BatchDecodeScheduler(
            runtime_config, resource_context_.cache_manager, metrics_reporter_, parallelism_config.dp_rank));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else if (pdfusion_scheduler_mode == PDFusionSchedulerMode::RATIO
               && pd_sep_config.role_type == RoleType::PDFUSION) {
        scheduler_.reset(new PDFusionRatioScheduler(runtime_config,
                                                    model_config_,
                                                    pd_sep_config,
                                                    parallelism_config,
                                                    model_specific_config,
                                                    resource_context_.cache_manager,
                                                    metrics_reporter_));
        RTP_LLM_LOG_INFO("create pdfusion ratio scheduler done");
    } else {
        if (pdfusion_scheduler_mode == PDFusionSchedulerMode::RATIO) {
            RTP_LLM_LOG_WARNING("pdfusion_scheduler_mode [ratio] is ignored because role_type [%d] is not PDFUSION",
                                static_cast<int>(pd_sep_config.role_type));
        }
        scheduler_.reset(new FIFOScheduler(runtime_config,
                                           model_config_,
                                           pd_sep_config,
                                           parallelism_config,
                                           model_specific_config,
                                           resource_context_.cache_manager,
                                           metrics_reporter_));
        RTP_LLM_LOG_INFO("create fifo scheduler done");
    }
}

NormalEngine::~NormalEngine() {
    RTP_LLM_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                       preRunMode                            mode) {
    c10::InferenceMode inference_guard(true);

    auto stream = std::make_shared<NormalGenerateStream>(generate_input,
                                                         model_config_,
                                                         runtime_config,
                                                         resource_context_,
                                                         nullptr,
                                                         0,
                                                         mode == preRunMode::prefill_warm_up);
    stream->setReserveStep(reserve_step_);
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        size_t seq_size_per_block = model_config_.attn_config.tokens_per_block;
        size_t reserved_blocks    = (stream->seqLength() + seq_size_per_block - 1) / seq_size_per_block + reserve_step_;
        stream->fakeInitKVBlock(reserved_blocks);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUS_ERROR(stream->initKVBlock());
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
#if USING_CUDA
    if (mode == preRunMode::build_system_prompt) {
        // Keep the stream and its execution buffers alive until the resident KV writes finish.
        cudaDeviceSynchronize();
    }
#endif
    return stream;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    // Baseline dispatch order preserved: the batch-decode-scheduler branch takes precedence over
    // the role mapping. PDFUSION maps to prefillWarmUp like before the warmup feature; its
    // measurement is discarded by the constructor gate (measurement_trusted = false).
    if (runtime_config.use_batch_decode_scheduler) {
        if (runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0) {
            return decodeWarmUp(params);
        } else {
            return prefillWarmUp(params);
        }
    }
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::PREFILL) {
        return prefillWarmUp(params);
    } else if (pd_sep_config.role_type == RoleType::DECODE) {
        return decodeWarmUp(params);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
        return {};
    }
}

std::shared_ptr<GenerateInput> NormalEngine::makeFakeInput(size_t seq_len) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->generate_config               = make_shared<GenerateConfig>();
    size_t token_size                         = model_config_.embedding_size ?
                                                    std::min(model_config_.embedding_size, model_config_.vocab_size) :
                                                    model_config_.vocab_size;
    fake_input->input_ids              = torch::randint(0, (int64_t)token_size, {(int64_t)seq_len}, torch::kInt32);
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->generate_config->top_k = 1;

    return fake_input;
}

size_t NormalEngine::getWarmUpInputLength() const {
    const auto max_seq_len  = static_cast<size_t>(model_config_.max_seq_len);
    const auto reserve_step = reserve_step_ > 0 ? static_cast<size_t>(reserve_step_) : 0;
    if (reserve_step > 0) {
        RTP_LLM_CHECK_WITH_INFO(max_seq_len > reserve_step,
                                "max_seq_len [%zu] should be greater than speculative reserve_step [%zu]",
                                max_seq_len,
                                reserve_step);
        const auto input_len = max_seq_len - reserve_step;
        RTP_LLM_LOG_INFO("framework warm up input len adjusted by speculative reserve_step, "
                         "max_seq_len=%zu, reserve_step=%zu, input_len=%zu",
                         max_seq_len,
                         reserve_step,
                         input_len);
        return input_len;
    }
    RTP_LLM_CHECK_WITH_INFO(max_seq_len > 1, "max_seq_len [%zu] should be greater than 1", max_seq_len);
    return max_seq_len - 1;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("prefillWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    const size_t max_seq_len = (size_t)model_config_.max_seq_len;
    // Reject negatives before the cast: this is an int64_t knob, and a C-style cast to size_t
    // would turn -1 into SIZE_MAX, which surfaces later as an unrelated overflow error. Name the
    // knob instead, like checkedMiBToBytes does.
    const int64_t configured_context_batch = runtime_config.fifo_scheduler_config.max_context_batch_size;
    RTP_LLM_CHECK_WITH_INFO(configured_context_batch >= 0,
                            "max_context_batch_size must be non-negative, got %ld",
                            configured_context_batch);
    // Pre-feature shape, deliberately unchanged: one sequence per max_context_batch_size, each at
    // the framework warmup input length. getWarmUpInputLength() owns that length -- it derives it
    // from max_seq_len and the speculative reserve_step, and rejects max_seq_len <= 1.
    const size_t num_seqs       = (size_t)configured_context_batch;
    const size_t tokens_per_seq = getWarmUpInputLength();
    RTP_LLM_CHECK_WITH_INFO(tokens_per_seq == 0 || num_seqs <= std::numeric_limits<size_t>::max() / tokens_per_seq,
                            "prefill warmup actual input token count overflow");
    const size_t actual_input_tokens = num_seqs * tokens_per_seq;

    RTP_LLM_LOG_INFO("[PREFILL_WARMUP] max_seq_len=%ld num_seqs=%ld tokens_per_seq=%ld actual_input_tokens=%ld",
                     max_seq_len,
                     num_seqs,
                     tokens_per_seq,
                     actual_input_tokens);
    // getWarmUpInputLength() already rejects max_seq_len <= 1, so what is left to catch here is
    // max_context_batch_size == 0, which would make this a zero-sequence forward.
    RTP_LLM_CHECK_WITH_INFO(actual_input_tokens > 0,
                            "prefill warmup would run a zero-token forward (num_seqs=%ld); "
                            "max_context_batch_size must be at least 1",
                            (long)num_seqs);

    auto fake_input                                   = makeFakeInput(tokens_per_seq);
    fake_input->generate_config->num_return_sequences = num_seqs;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    MemoryStatus peak_status;
    MemoryStatus post_teardown_status;
    size_t       pre_warmup_available = 0;
    {
        WarmupTraceScope trace_scope(executor_);
        executor_.reset(new NormalExecutor(params, nullptr, true, false, 0, mla_ops_type_));
        THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
        peak_status          = getGpuExecStatus().device_memory_status;
        post_teardown_status = trace_scope.teardownAndSample();
        pre_warmup_available = trace_scope.preWarmupAvailableBytes();
    }
    return makeWarmUpResult(pre_warmup_available,
                            peak_status,
                            post_teardown_status,
                            warmupMeasurementTrustedForRole(pd_sep_config.role_type),
                            "PREFILL_WARMUP");
#endif
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("decodeWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    const size_t max_seq_len                          = (size_t)model_config_.max_seq_len;
    const size_t num_return_sequences                 = (size_t)runtime_config.max_generate_batch_size;
    const size_t kv_seq_len                           = getWarmUpInputLength();
    const size_t decode_tokens                        = num_return_sequences;
    auto         fake_input                           = makeFakeInput(kv_seq_len);
    fake_input->generate_config->num_return_sequences = num_return_sequences;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);

    RTP_LLM_LOG_INFO("[DECODE_WARMUP] max_seq_len=%ld kv_seq_len=%ld num_return_sequences=%ld decode_tokens=%ld",
                     max_seq_len,
                     kv_seq_len,
                     num_return_sequences,
                     decode_tokens);

    auto cache_config = CacheConfigCreator::createBasicConfig(model_config_, parallelism_config, false, 0);
    cache_config.seq_size_per_block        = model_config_.attn_config.tokens_per_block;
    cache_config.kernel_seq_size_per_block = model_config_.attn_config.tokens_per_block;
    cache_config.block_num                 = 5;
    ParallelismConfig temp_parallelism_config;
    RuntimeConfig     temp_runtime_config;

    // cache manager for warmup
    auto cache_manager = make_shared<KVCacheManager>(
        cache_config, true, nullptr, KVCacheConfig{}, temp_parallelism_config, temp_runtime_config);
    if (!cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed in decodeWarmUp");
    }

    MemoryStatus peak_status;
    MemoryStatus post_teardown_status;
    size_t       pre_warmup_available = 0;
    {
        WarmupTraceScope trace_scope(executor_);
        executor_.reset(new NormalExecutor(params, cache_manager, true, false, 0, mla_ops_type_));
        THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
        peak_status          = getGpuExecStatus().device_memory_status;
        post_teardown_status = trace_scope.teardownAndSample();
        pre_warmup_available = trace_scope.preWarmupAvailableBytes();
    }
    return makeWarmUpResult(pre_warmup_available,
                            peak_status,
                            post_teardown_status,
                            warmupMeasurementTrustedForRole(pd_sep_config.role_type),
                            "DECODE_WARMUP");
#endif
}

std::shared_ptr<GenerateStream> NormalEngine::createMinFakeStream(int32_t max_new_tokens) {
    RTP_LLM_LOG_DEBUG("create min fake query");
    auto fake_input                             = makeFakeInput(1);
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->fake_query                      = true;
    auto stream                                 = makeStream(fake_input);
    stream->setIsFakeStream(true);
    stream->setMetricsReporter(nullptr);
    stream->fakeInitKVBlock();
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::DECODE) {
        auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

        StreamUpdateInfo update_info{new_tokens,
                                     1,
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     false};
        stream->update(update_info);
        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .epoch                 = 0,
            .last_sample_token_gpu = torch::zeros({1}, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32),
            .last_real_seq_len     = stream->seqLength(),
            .next_real_seq_len     = stream->seqLength(),
        });
    }
    return stream;
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    if (propose_params_ && propose_params_->draftModel()) {
        auto config = CacheConfigCreator::createSpConfig(model_config_,
                                                         propose_params_->getEngineInitParams().model_config_,
                                                         parallelism_config,
                                                         runtime_config,
                                                         kv_cache_config,
                                                         sp_config,
                                                         warm_up_result,
                                                         isMTPEagle(),
                                                         isEagle());

        resource_context_.cache_manager = make_shared<KVCacheManager>(
            config, false, metrics_reporter_, kv_cache_config, parallelism_config, runtime_config, sp_config);
        resource_context_.role_type = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }

        const auto& cache_cfg = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_   = cache_cfg.groupNums();
    } else {
        auto result = CacheConfigCreator::createConfig(
            model_config_, parallelism_config, runtime_config, kv_cache_config, warm_up_result);
        RTP_LLM_LOG_INFO("create cache manager with config %s", result.debugString().c_str());
        RTP_LLM_LOG_INFO("create cache manager with block nums %d, block size %ld KB",
                         result.block_num,
                         result.block_size_bytes / 1024);
        RTP_LLM_LOG_INFO("create cache manager with linear step %d", result.linear_step);
        resource_context_.cache_manager = make_shared<KVCacheManager>(
            result, false, metrics_reporter_, kv_cache_config, parallelism_config, runtime_config);
        resource_context_.role_type = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
        const auto& cache_cfg = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_   = cache_cfg.groupNums();
    }
}

absl::Status NormalEngine::initSystemPrompt() {
    resource_context_.initCacheConfig(kv_cache_config, runtime_config.fifo_scheduler_config, model_config_.max_seq_len);

    if (!kv_cache_config.multi_task_prompt_tokens.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(
            system_prompt_param,
            SystemPromptConstructor::construct(
                kv_cache_config, this, resource_context_.cache_manager.get(), parallelism_config.tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }

    return absl::OkStatus();
}

KVCacheInfo NormalEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

absl::Status NormalEngine::startLoop() {
    if (parallelism_config.tp_rank == 0) {
        RTP_LLM_LOG_INFO("start init system prompt");
        THROW_IF_STATUS_ERROR(initSystemPrompt());
        RTP_LLM_LOG_INFO("init system prompt done");
    }
    RTP_LLM_LOG_INFO("start normal engine loop");
    running_     = true;
    loop_thread_ = autil::Thread::createThread(std::bind(&NormalEngine::loop, this), "normal_engine_loop");
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    RTP_LLM_LOG_INFO("stop normal engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    loop_thread_->join();
    return absl::OkStatus();
}

void NormalEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    c10::InferenceMode inference_guard(true);
    setCurrentThreadDevice(getDeviceId());
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    stream->setReserveStep(reserve_step_);
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    stream->setReserveStep(reserve_step_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

std::vector<std::shared_ptr<GenerateStream>>
NormalEngine::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    std::vector<std::shared_ptr<GenerateStream>> streams;
    streams.reserve(inputs.size());
    for (auto& inp : inputs) {
        auto stream = std::make_shared<NormalGenerateStream>(
            inp, model_config_, runtime_config, resource_context_, metrics_reporter_);
        stream->setReserveStep(reserve_step_);
        streams.push_back(stream);
    }
    return scheduler_->batchEnqueue(streams);
}

absl::Status NormalEngine::step() {
    RTP_LLM_PROFILE_SCOPE("engine.normal.step_work");
    while (pause_) {
        // wait 50ms if system paused.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    int64_t                 tps_schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.schedule(reserve_step=%d)", reserve_step_);
            CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        }
        if (parallelism_config.dp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("engine.normal.may_add_fake_stream_work");
            mayAddFakeStream(streams);
        }
        // When TP > 1, all ranks must enter process() together so that
        // tpSyncModelInputs (collective broadcast) does not deadlock.
        // The skip_run flag inside process() handles the "no work" case.
        if (streams.empty() && parallelism_config.tp_size <= 1) {
            return absl::OkStatus();
        }
    }

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = absl::OkStatus();

    // Per-request timeline: if any stream requested gen_timeline and no session is
    // active yet, configure the profiler so the executor-driven step window
    // captures THIS step.
    if (!step_profiler_.enabled()) {
        for (const auto& stream : streams) {
            if (stream && stream->genTimeline()) {
                const auto& cfg = stream->generateConfig();
                step_profiler_.configure(true, cfg->profile_trace_name, 0, cfg->profile_step);
                break;
            }
        }
    }

    {
        // NormalExecutor drives startStep/finishStep via callbacks; MtpExecutor
        // has no callbacks yet, so bracket the propose path here on the engine
        // loop thread (Kineto requires enable/disable on the same thread).
        if (propose_params_) {
            step_profiler_.startStep();
        }
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        const bool refresh_cache_status_snapshot =
            resource_context_.cache_manager && shouldRefreshCacheStatusSnapshot(pd_sep_config.role_type, streams);
        status = executor_->process(streams, tps_schedule_time_us);
        if (status.ok() && refresh_cache_status_snapshot) {
            RTP_LLM_PROFILE_SCOPE("engine.normal.refresh_cache_status_snapshot");
            resource_context_.cache_manager->refreshKVCacheInfoSnapshot();
        }
        if (propose_params_) {
            step_profiler_.finishStep();
        }
    }

    // report step metrics
    if (parallelism_config.tp_rank == 0) {
        RTP_LLM_PROFILE_SCOPE("engine.normal.report_metrics_work");
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({step_latency});
    }

    return status;
}

bool NormalEngine::updateEplbConfig(const EPLBConfig& config) {
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
}

void NormalEngine::startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {
    step_profiler_.configure(true, trace_name, start_step, num_steps);
}

bool NormalEngine::isMTPEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_MTP || propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

bool NormalEngine::isEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

void NormalEngine::mayAddFakeStream(std::list<GenerateStreamPtr>& streams) {
    if (isMTPEagle()) {
        int propose_step   = sp_config.gen_num_per_cycle;
        int mtp_vocab_size = propose_params_->getEngineInitParams().model_config_.vocab_size;
        switch (pd_sep_config.role_type) {
            case RoleType::PREFILL:
                if (streams.empty()) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                break;
            case RoleType::DECODE:
                if (streams.empty()) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, mtp_vocab_size));
                }
                break;
            case RoleType::PDFUSION: {
                bool has_prefill = false;
                bool has_decode  = false;
                for (auto& stream : streams) {
                    if (stream->isContextStream()) {
                        has_prefill = true;
                    } else {
                        has_decode = true;
                    }
                }
                if (!has_prefill) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                if (!has_decode) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, mtp_vocab_size));
                }
                break;
            }
            default:
                RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
                break;
        }
    } else {
        if (streams.empty()) {
            streams.emplace_back(createMinFakeStream(1));
        }
    }
}

}  // namespace rtp_llm

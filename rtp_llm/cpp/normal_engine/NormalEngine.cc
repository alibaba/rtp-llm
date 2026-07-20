#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
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

std::vector<int32_t> flattenLayerToGroup(const CacheConfig& cache_config) {
    auto layer_to_group_ids = cache_config.layerGroupIdsSnapshot();
    std::vector<int32_t> layer_to_group;
    layer_to_group.reserve(layer_to_group_ids.size());
    for (size_t layer = 0; layer < layer_to_group_ids.size(); ++layer) {
        RTP_LLM_CHECK_WITH_INFO(layer_to_group_ids[layer].size() == 1,
                                "layer %zu owns %zu cache groups; expected exactly one group",
                                layer,
                                layer_to_group_ids[layer].size());
        layer_to_group.push_back(static_cast<int32_t>(layer_to_group_ids[layer].front()));
    }
    return layer_to_group;
}
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
    if (runtime_config.warm_up && (!model_config_.mm_model_config.is_multimodal)
        && !ffn_disaggregate_config.enable_ffn_disaggregate) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         runtime_config.fifo_scheduler_config.max_context_batch_size,
                         model_config_.max_seq_len,
                         int(runtime_config.warm_up_with_loss));
        warm_up_result = warmUp(params);
        RTP_LLM_LOG_INFO(
            "warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
            warm_up_result->max_used_memory,
            warm_up_result->max_used_memory / 1024 / 1024,
            warm_up_result->device_reserved_bytes,
            warm_up_result->device_reserved_bytes / 1024 / 1024);
    } else {
        RTP_LLM_LOG_INFO("skip warm up.");
    }
#else
    RTP_LLM_LOG_INFO("skip warm up on non-CUDA platform.");
#endif
    initCacheManager(warm_up_result);
    RTP_LLM_LOG_INFO("create cache manager done");

    initExecutor(params, propose_params_);
    if (propose_params_) {
        reserve_step_ = propose_params_->gen_num_per_circle + 1;
    } else {
        reserve_step_ = 0;
    }

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
        executor_.reset(new MtpExecutor(params,
                                        propose_params,
                                        resource_context_.cache_manager,
                                        mla_ops_type_,
                                        kv_cache_group_num_,
                                        kv_cache_layer_to_group_));
    } else {
        executor_.reset(new NormalExecutor(params,
                                           resource_context_.cache_manager,
                                           false,
                                           false,
                                           0,
                                           mla_ops_type_,
                                           kv_cache_group_num_,
                                           kv_cache_layer_to_group_));
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
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        size_t seq_size_per_block = model_config_.attn_config.tokens_per_block;
        size_t reserved_blocks    = (stream->seqLength() + seq_size_per_block - 1) / seq_size_per_block + reserve_step_;
        stream->fakeInitKVBlock(reserved_blocks);
    } else if (mode == preRunMode::build_system_prompt) {
        stream->setReserveStep(reserve_step_);
        THROW_IF_STATUS_ERROR(stream->initKVBlock());
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return stream;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
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

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("prefillWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.fifo_scheduler_config.max_context_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);
    executor_.reset(new NormalExecutor(
        params, nullptr, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("decodeWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.max_generate_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);

    auto cache_config               = CacheConfigCreator::createBasicConfig(model_config_, parallelism_config, false, 0);
    cache_config.seq_size_per_block = model_config_.attn_config.tokens_per_block;
    cache_config.block_num          = 5;
    ParallelismConfig temp_parallelism_config;
    RuntimeConfig     temp_runtime_config;
    auto              cache_manager = make_shared<KVCacheManager>(
        cache_config, true, nullptr, KVCacheConfig{}, temp_parallelism_config, temp_runtime_config);
    if (!cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed in decodeWarmUp");
    }
    const auto& temp_cache_config = cache_manager->cacheConfig();
    auto        temp_layer_to_group = flattenLayerToGroup(temp_cache_config);
    executor_.reset(new NormalExecutor(params,
                                       cache_manager,
                                       true,
                                       false,
                                       0,
                                       mla_ops_type_,
                                       static_cast<int32_t>(temp_cache_config.groupNums()),
                                       temp_layer_to_group));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
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

        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = flattenLayerToGroup(cache_cfg);
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
        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = flattenLayerToGroup(cache_cfg);
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
    restart();
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    loop_thread_->join();
    return absl::OkStatus();
}

void NormalEngine::pause() {
    {
        // Bump the epoch under pause_mutex_ so it is published together with pause_=true.
        // A quiesce reader (enterPausedState) that observes pause_=true then loads the epoch
        // under the same lock is guaranteed to see the bumped value, so it can never
        // acknowledge a stale epoch and strand pauseAndWaitQuiesced() until timeout.
        // Bumping the epoch is also the sole "reset" of the quiesce ack: quiesced_pause_epoch_
        // stays below this new epoch until a real quiesce records it, with no separate reset
        // step that a concurrent acknowledgement could clobber.
        std::lock_guard<std::mutex> lock(pause_mutex_);
        bool                        expected = false;
        if (pause_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            auto epoch = pause_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
            RTP_LLM_LOG_INFO("normal engine pause requested, epoch=%lu", epoch);
        }
    }
    if (scheduler_) {
        scheduler_->wake();
    }
}

void NormalEngine::restart() {
    // Clear pause_ under pause_mutex_ so the store cannot slip into the window
    // between a waiter's predicate check and its wait: holding the lock forces
    // the waiter to either observe pause_=false before parking or receive the
    // notify while parked. A bare store+notify here could be lost.
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        pause_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();
}

void NormalEngine::markPauseQuiesced(uint64_t pause_epoch) {
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        // Monotonic: only ever advance. A stale/late caller for an older epoch cannot
        // pull the ack backwards and strand a waiter that captured a newer epoch.
        if (pause_epoch > quiesced_pause_epoch_) {
            quiesced_pause_epoch_ = pause_epoch;
        }
    }
    pause_cv_.notify_all();
}

void NormalEngine::enterPausedState() {
    std::unique_lock<std::mutex> lock(pause_mutex_);
    // Loop, re-reading the epoch under the lock on every wake, so that a pause
    // epoch published *while we are parked here* is still acknowledged. A rapid
    // restart() (pause_ cleared) immediately followed by a new pause() (pause_
    // re-set, epoch bumped) would otherwise leave us waiting in a call that only
    // ever recorded the previous epoch, stranding the new epoch's coordinator
    // until its deadline. Loading the epoch under the lock also rules out a stale
    // pre-bump value, since pause() bumps it under the same lock.
    while (running_.load()) {
        if (!pause_.load(std::memory_order_acquire)) {
            break;
        }
        const uint64_t epoch = pause_epoch_.load(std::memory_order_acquire);
        if (epoch > quiesced_pause_epoch_) {
            quiesced_pause_epoch_ = epoch;
            pause_cv_.notify_all();
        }
        pause_cv_.wait(lock, [this, epoch] {
            return !pause_.load(std::memory_order_acquire) || pause_epoch_.load(std::memory_order_acquire) > epoch
                   || !running_.load();
        });
    }
}

absl::Status NormalEngine::runExecutorProcess(const std::list<GenerateStreamPtr>& streams) {
    std::lock_guard<std::mutex> lock(process_mutex_);
    auto                        status = executor_->process(streams);
    if (status.ok() && executor_->consumeLastPauseSignal()) {
        pause();
    }
    if (pause_.load(std::memory_order_acquire)) {
        processed_pause_epoch_.store(pause_epoch_.load(std::memory_order_acquire), std::memory_order_release);
    }
    return status;
}

bool NormalEngine::collectiveSleepQuiesceEnabled() const {
    // The decision whether step() performs the extra pause-wave all-reduce MUST be
    // rank-symmetric: if it diverges across ranks, some ranks call execAllReduce
    // while others don't and the collective deadlocks during *normal* serving, not
    // just during sleep. So gate it on enabled() (the static enable_sleep_mode launch
    // config, set in initRuntime() before startLoop() and identical on every rank),
    // NOT on effective(): effective() folds in runtimeSupported(), a per-rank VMM
    // availability flag that is set only after the loop thread has started
    // (LocalRpcServer::installSleepHooks) and may legitimately differ across ranks.
    // A rank without VMM support still participates in the quiesce collective (safe,
    // it just no-ops the local memory release); a sleep request there fails cleanly
    // with DISABLED and the coordinator aborts, rather than hanging the fleet.
    return sleep_controller_.enabled() && parallelism_config.world_size > 1
           && (parallelism_config.dp_size > 1 || parallelism_config.ep_size > 1);
}

// Why the sleep-quiesce consensus needs a collective at all, and why it now costs the steady
// serving path nothing:
//
// Under DP/EP (world_size > 1 with dp_size > 1 or ep_size > 1) sleep cannot be a rank-local
// decision. The pause request is broadcast to every rank's engine independently
// (grpc_client_wrapper.sleep_serving fans out to every control rank), but arrives
// asynchronously: rank A may observe pause_ several steps before rank B. If a rank simply
// stopped as soon as it saw pause_, its peers' next forward-pass collective (TP all-reduce /
// EP all-to-all) would block forever on the departed rank -- a fleet-wide hang, not a clean
// sleep. The ranks must instead agree, from a value every rank observes identically, that
// "everyone has received the pause AND everyone is drained" before any of them stops.
//
// That agreement is an all-reduce, but it is issued ASYNCHRONOUSLY and ONLY WHILE A SLEEP IS
// ARMED, on a DEDICATED communicator (ParallelMode::SLEEP_QUIESCE) that carries no other
// traffic:
//   * Steady state (no pause armed): this function returns after a single relaxed atomic
//     load. It issues NO collective, acquires NO GIL, does NO device work -- the decode hot
//     path pays nothing. This is what the earlier every-step blocking all-reduce could not do.
//   * Armed: the rank latches "engaged" and drives one async all-reduce at a time. Because it
//     is async_op=True the rank KEEPS participating in its forward collectives while the
//     consensus is in flight, so an early-arming rank never strands its peers -- the deadlock
//     that forces a *blocking* consensus to run every step cannot occur.
//   * Dedicated comm: an async round can complete a variable number of engine steps after it
//     was issued. On the shared DP_AND_TP communicator that lag would let the consensus
//     interleave with EPLB's periodic all-reduce/broadcast (ExpertBalancer, also on
//     DP_AND_TP) and desync the collective stream. A private communicator carrying only the
//     consensus makes the k-th round unambiguous on every rank.
//
// Round matching / termination: a rank issues round k+1 only after round k completes, so all
// ranks run the same 1..K rounds. The verdict is derived from the shared summed result, so
// every rank reaches the same terminal round K simultaneously (all armed + all drained ->
// REACHED; or, after a cancel/wake, the pending count returns to zero -> CANCELLED) and stops
// issuing together -- no rank is ever left with an unmatched collective. (The single-rank /
// pure-TP deployment escapes all of this -- see releasePendingTpCollectiveForPause() /
// enterPausedState(), which need no collective.)
absl::Status NormalEngine::maybeReachCollectiveSleepQuiesce() {
    if (!collectiveSleepQuiesceEnabled()) {
        return absl::OkStatus();
    }

    const bool pending = pause_.load(std::memory_order_acquire);

    // Steady serving fast path: nothing armed and no round in flight -> issue no collective.
    // engaged_ and handle_ are touched only on the engine loop thread (this function runs
    // from step()), so they need no synchronization.
    if (!collective_quiesce_engaged_ && collective_quiesce_handle_ == 0 && !pending) {
        return absl::OkStatus();
    }

    // First observation of pause_ arms this rank. Once engaged we stay engaged (still issuing
    // rounds) even if pause_ later clears via wake, so a cancel converges to a shared CANCELLED
    // verdict rather than leaving peers waiting on a round this rank stopped issuing.
    if (pending) {
        collective_quiesce_engaged_ = true;
    }

    bool reached = false;
    {
        // Guard the reusable buffer against a concurrent execute-path caller. The async round
        // reduces IN-PLACE on collective_quiesce_state_; because we never refill or read the
        // buffer until the in-flight round completes, its contents are stable while in flight.
        std::lock_guard<std::mutex> lock(collective_quiesce_state_mutex_);
        c10::DeviceGuard            device_guard(c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(getDeviceId())));
        if (!collective_quiesce_state_.defined()) {
            const auto options = torch::TensorOptions()
                                     .dtype(torch::kInt64)
                                     .device(torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(getDeviceId())));
            collective_quiesce_state_ = torch::empty({2}, options);
        }

        if (collective_quiesce_handle_ == 0) {
            // No round in flight: fill this rank's contribution and enqueue an async round.
            //   data[0] = 1 if this rank currently holds the pause request, else 0.
            //   data[1] = unfinished streams on this rank (only tp_rank 0 can observe them).
            const int64_t pending_flag = pending ? 1 : 0;
            const int64_t unfinished   = (pending && parallelism_config.tp_rank == 0 && scheduler_) ?
                                             std::max<int64_t>(0, scheduler_->onflightStreams()) :
                                             0;
            collective_quiesce_state_.select(0, 0).fill_(pending_flag);
            collective_quiesce_state_.select(0, 1).fill_(unfinished);
            collective_quiesce_handle_ =
                execAllReduceAsync({collective_quiesce_state_, ReduceOp::Sum, false, ParallelMode::SLEEP_QUIESCE});
            // handle 0 => async path unavailable (no callback / degenerate group). Nothing to
            // wait for; retry next step. (Should not happen once the group is built.)
            return absl::OkStatus();
        }

        // A round is in flight. Poll without blocking the loop; if not done, keep serving and
        // participating in forward collectives (the whole point of async arm-on-demand).
        if (!pollAsyncComm(collective_quiesce_handle_)) {
            return absl::OkStatus();
        }
        collective_quiesce_handle_ = 0;

        // Round complete: read the summed verdict. The device->host copy only ever runs on the
        // armed path (never in steady serving). pollAsyncComm() already waited on the collective,
        // so the reduced buffer is safe to read here.
        auto reduced = collective_quiesce_state_;
        if (reduced.device().is_cuda()) {
            reduced = reduced.cpu();
        }
        auto          reduced_data    = reduced.data_ptr<int64_t>();
        const int64_t pending_sum     = reduced_data[0];
        const int64_t not_ready_count = reduced_data[1];

        if (pending_sum == parallelism_config.world_size && not_ready_count == 0) {
            reached = true;  // all ranks armed AND drained on the same round -> sleep now
        } else if (pending_sum == 0) {
            // Every rank has left the pause window (sleep cancelled / woken before quiesce):
            // disengage together on this shared verdict and resume normal serving at zero cost.
            collective_quiesce_engaged_ = false;
            RTP_LLM_LOG_INFO("normal engine collective sleep quiesce cancelled before reaching, world_size=%ld",
                             parallelism_config.world_size);
        }
        // else: partial (some ranks still arming, or still draining) -> stay engaged and issue
        // the next round on a later step.
    }

    if (reached) {
        // Disengage before parking so a subsequent wake resumes at the steady zero-cost path.
        collective_quiesce_engaged_ = false;
        const auto pause_epoch      = pause_epoch_.load(std::memory_order_acquire);
        processed_pause_epoch_.store(pause_epoch, std::memory_order_release);
        RTP_LLM_LOG_INFO("normal engine collective sleep quiesce reached, epoch=%lu, world_size=%ld",
                         pause_epoch,
                         parallelism_config.world_size);
        // CPU consensus above only proves no rank will enqueue new work; it does NOT prove the
        // GPU is idle. This step's runExecutorProcess() launched a (fake-MoE) forward whose
        // EP dispatch/combine is still async-retiring; for internode DeepEP the cross-node
        // combine has a real RDMA tail. All ranks reach this branch in the same step (consensus
        // is derived from the shared all-reduce), so draining here is symmetric and cannot hang.
        // Combined with the coordinator's two-phase protocol (commit is sent only after every
        // rank's prepare/quiesce returns, i.e. after this sync), it guarantees no rank tears down
        // its MR / unmaps weight pages while a peer's collective still references them -- which
        // is what otherwise poisons the CUDA context and makes torch_memory_saver's cuMemUnmap
        // return a sticky CUDA 999 and abort the whole fleet.
#if USING_CUDA
        cudaDeviceSynchronize();
#endif
        enterPausedState();
    }
    return absl::OkStatus();
}

absl::Status NormalEngine::releasePendingTpCollectiveForPause(uint64_t pause_epoch) {
    if (parallelism_config.tp_size <= 1 || parallelism_config.tp_rank != 0) {
        return absl::OkStatus();
    }
    if (processed_pause_epoch_.load(std::memory_order_acquire) >= pause_epoch) {
        return absl::OkStatus();
    }

    std::lock_guard<std::mutex> lock(process_mutex_);
    if (processed_pause_epoch_.load(std::memory_order_acquire) >= pause_epoch) {
        return absl::OkStatus();
    }

    RTP_LLM_LOG_INFO("normal engine pause: run one empty TP sync step, epoch=%lu", pause_epoch);
    auto status = executor_->processForPause();
    (void)executor_->consumeLastPauseSignal();
    if (!status.ok()) {
        return status;
    }
    processed_pause_epoch_.store(pause_epoch, std::memory_order_release);
    // Rank0 may be blocked in scheduler_->schedule() while worker ranks are
    // waiting in tpSyncModelInputs. The RPC thread's empty sync step releases
    // those workers, and rank0 itself is not touching GPU while scheduler-blocked.
    markPauseQuiesced(pause_epoch);
    return absl::OkStatus();
}

absl::Status NormalEngine::pauseAndWaitQuiesced(int64_t timeout_ms) {
    constexpr int64_t kDefaultPauseQuiesceTimeoutMs = 60000;
    const int64_t     effective_timeout_ms          = timeout_ms > 0 ? timeout_ms : kDefaultPauseQuiesceTimeoutMs;

    pause();
    const auto pause_epoch = pause_epoch_.load(std::memory_order_acquire);
    if (!running_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }

    if (!collectiveSleepQuiesceEnabled()) {
        auto status = releasePendingTpCollectiveForPause(pause_epoch);
        if (!status.ok()) {
            return status;
        }
    }

    std::unique_lock<std::mutex> lock(pause_mutex_);
    if (quiesced_pause_epoch_ >= pause_epoch || !pause_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(effective_timeout_ms);
    if (!pause_cv_.wait_until(lock, deadline, [this, pause_epoch] {
            return quiesced_pause_epoch_ >= pause_epoch || !pause_.load(std::memory_order_acquire) || !running_.load();
        })) {
        return absl::Status(absl::StatusCode::kDeadlineExceeded,
                            "normal engine pause quiesce timeout after " + std::to_string(effective_timeout_ms)
                                + " ms");
    }
    return absl::OkStatus();
}

void NormalEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    c10::InferenceMode inference_guard(true);
    cudaPreRun(getDeviceId());
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
    const bool collective_sleep_quiesce = collectiveSleepQuiesceEnabled();
    if (pause_.load(std::memory_order_acquire) && !collective_sleep_quiesce
        && (parallelism_config.tp_size <= 1 || parallelism_config.tp_rank == 0)) {
        enterPausedState();
    }

    // stop() wakes a paused loop by clearing pause_ (via restart()) with running_
    // already false. Without this guard the woken loop would fall through to a
    // real schedule()/execute step while the KV backing is still released
    // (sleeping) -- for TP>1 that empty step also blocks in tpSyncModelInputs
    // against ranks that have already torn down. Bail out promptly on shutdown.
    if (!running_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }

    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.schedule(reserve_step=%d)", reserve_step_);
            CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        }
        if (parallelism_config.dp_size > 1 || (collective_sleep_quiesce && parallelism_config.ep_size > 1)) {
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

    if (pause_.load(std::memory_order_acquire) && !collective_sleep_quiesce && parallelism_config.tp_rank == 0) {
        enterPausedState();
        return absl::OkStatus();
    }

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = absl::OkStatus();

    // Per-request timeline: if any stream requested gen_timeline and no session is
    // active yet, configure the profiler so the next stepScope() captures THIS step.
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
        [[maybe_unused]] auto profile_step = step_profiler_.stepScope();
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        status = runExecutorProcess(streams);
    }

    if (status.ok() && collective_sleep_quiesce) {
        status = maybeReachCollectiveSleepQuiesce();
    } else if (status.ok() && pause_.load(std::memory_order_acquire)) {
        enterPausedState();
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
        int propose_step = sp_config.gen_num_per_cycle;
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
                        propose_step, model_config_, runtime_config, resource_context_));
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
                        propose_step, model_config_, runtime_config, resource_context_));
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

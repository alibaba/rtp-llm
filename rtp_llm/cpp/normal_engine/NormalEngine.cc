#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/engine_base/SpeculativeConfigValidator.h"
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

    reserve_step_ = propose_params_ ? propose_params_->gen_num_per_circle + 1 : 0;

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

    initExecutor(params);

    RTP_LLM_LOG_INFO("create normal executor done");

    // 释放模型加载过程中使用的临时host内存
    // 此时checkpoint已加载完成，可以将glibc缓存的内存归还给操作系统
    releaseHostMemoryCache();

    initScheduler();
    (void)startLoop();
}

std::unique_ptr<Executor> NormalEngine::createExecutor(const EngineInitParams&                params,
                                                       const std::shared_ptr<KVCacheManager>& cache_manager,
                                                       bool                                   warm_up,
                                                       std::optional<RoleType>                role_override) {
    int32_t              cache_group_num = kv_cache_group_num_;
    std::vector<int32_t> layer_to_group  = kv_cache_layer_to_group_;
    if (cache_manager) {
        const auto& cache_config = cache_manager->cacheConfig();
        cache_group_num          = cache_config.groupNums();
        layer_to_group           = cache_config.layer_to_group_id;
    }

    if (propose_params_) {
        RTP_LLM_CHECK_WITH_INFO(cache_manager != nullptr, "speculative executor requires a cache manager");
        return std::make_unique<MtpExecutor>(params,
                                             propose_params_,
                                             cache_manager,
                                             mla_ops_type_,
                                             cache_group_num,
                                             layer_to_group,
                                             warm_up,
                                             role_override);
    }

    return std::make_unique<NormalExecutor>(params,
                                            cache_manager,
                                            warm_up,
                                            false,
                                            0,
                                            mla_ops_type_,
                                            cache_group_num,
                                            layer_to_group);
}

void NormalEngine::initExecutor(const EngineInitParams& params) {
    executor_ = createExecutor(params, resource_context_.cache_manager, false, std::nullopt);
}

void NormalEngine::initScheduler() {
    if (runtime_config.use_batch_decode_scheduler) {
        scheduler_.reset(new BatchDecodeScheduler(
            runtime_config, resource_context_.cache_manager, metrics_reporter_, parallelism_config.dp_rank));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else {
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
    auto streams = preRunWithResourceContext({generate_input}, mode, resource_context_);
    THROW_IF_STATUSOR_ERROR(streams);
    RTP_LLM_CHECK_WITH_INFO(streams.value().size() == 1, "single preRun must create exactly one stream");
    return streams.value().front();
}

std::vector<GenerateStreamPtr> NormalEngine::createPreRunStreams(
    const std::vector<std::shared_ptr<GenerateInput>>& generate_inputs,
    preRunMode                                          mode,
    const ResourceContext&                              resource_context) {
    RTP_LLM_CHECK_WITH_INFO(!generate_inputs.empty(), "preRun requires at least one input");
    const bool speculative_warm_up = propose_params_ != nullptr
                                     && (mode == preRunMode::prefill_warm_up
                                         || mode == preRunMode::decode_warm_up);
    std::vector<GenerateStreamPtr> streams;
    streams.reserve(generate_inputs.size());
    for (const auto& generate_input : generate_inputs) {
        auto stream = std::make_shared<NormalGenerateStream>(generate_input,
                                                             model_config_,
                                                             runtime_config,
                                                             resource_context,
                                                             nullptr,
                                                             0,
                                                             speculative_warm_up);
        stream->setReserveStep(reserve_step_);
        if (speculative_warm_up) {
            stream->setIsFakeStream(true);
        }
        if (mode == preRunMode::decode_warm_up || speculative_warm_up) {
            const size_t seq_size_per_block = model_config_.attn_config.tokens_per_block;
            RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "tokens_per_block must be greater than zero");
            const size_t reserved_tokens = static_cast<size_t>(stream->seqLength()) + reserve_step_;
            const size_t reserved_blocks = (reserved_tokens + seq_size_per_block - 1) / seq_size_per_block;
            stream->fakeInitKVBlock(reserved_blocks);
        }
        if (mode == preRunMode::decode_warm_up) {
            stream->setIsContextStream(false);
        } else if (mode == preRunMode::build_system_prompt) {
            THROW_IF_STATUS_ERROR(stream->initKVBlock());
        }
        streams.push_back(std::move(stream));
    }
    return streams;
}

absl::StatusOr<std::vector<GenerateStreamPtr>> NormalEngine::preRunWithResourceContext(
    const std::vector<std::shared_ptr<GenerateInput>>& generate_inputs,
    preRunMode                                          mode,
    const ResourceContext&                              resource_context) {
    c10::InferenceMode inference_guard(true);
    auto streams = createPreRunStreams(generate_inputs, mode, resource_context);
    std::list<GenerateStreamPtr> stream_list(streams.begin(), streams.end());
    RETURN_IF_STATUS_ERROR(executor_->process(stream_list));
    return streams;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    if (pd_sep_config.role_type == RoleType::PDFUSION) {
        const auto prefill_result = prefillWarmUp(params);
        const auto decode_result  = decodeWarmUp(params);
        return WarmUpResult({std::min(prefill_result.device_reserved_bytes, decode_result.device_reserved_bytes),
                             std::max(prefill_result.max_used_memory, decode_result.max_used_memory)});
    }
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
    size_t token_size = effectiveInputVocabSize(model_config_);
    if (propose_params_ && propose_params_->draftModel()) {
        token_size =
            std::min(token_size, effectiveInputVocabSize(propose_params_->getEngineInitParams().model_config_));
    }
    fake_input->input_ids              = torch::randint(0, (int64_t)token_size, {(int64_t)seq_len}, torch::kInt32);
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->generate_config->top_k = 1;

    return fake_input;
}

std::vector<std::shared_ptr<GenerateInput>>
NormalEngine::makeWarmUpInputs(size_t seq_len, size_t batch_size, bool independent_streams) {
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0, "warmup batch size must be greater than zero");
    const size_t input_count = independent_streams ? batch_size : 1;
    std::vector<std::shared_ptr<GenerateInput>> inputs;
    inputs.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        auto input                                   = makeFakeInput(seq_len);
        input->request_id                            = static_cast<int64_t>(i);
        input->generate_config->num_return_sequences = independent_streams ? 1 : static_cast<int>(batch_size);
        input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
        input->generate_config->force_sp_accept      = propose_params_ != nullptr;
        inputs.push_back(std::move(input));
    }
    return inputs;
}

std::shared_ptr<KVCacheManager> NormalEngine::createWarmUpCacheManager(const EngineInitParams& params) {
    CacheConfig cache_config;
    if (propose_params_ && propose_params_->draftModel()) {
        auto warm_up_kv_config           = params.kv_cache_config;
        warm_up_kv_config.test_block_num = 1;
        cache_config = CacheConfigCreator::createSpConfig(params.model_config_,
                                                          propose_params_->getEngineInitParams().model_config_,
                                                          params.parallelism_config,
                                                          params.runtime_config,
                                                          warm_up_kv_config,
                                                          params.sp_config,
                                                          std::nullopt,
                                                          isMTPEagle(),
                                                          isEagle());
    } else {
        cache_config             = CacheConfigCreator::createBasicConfig(params.model_config_, params.parallelism_config);
        cache_config.block_num   = 1;
        cache_config.linear_step = 1;
    }
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, true);
    RTP_LLM_CHECK_WITH_INFO(cache_manager->init(), "init warmup cache manager failed");
    return cache_manager;
}

WarmUpResult NormalEngine::runWarmUp(const EngineInitParams&                            params,
                                     const std::vector<std::shared_ptr<GenerateInput>>& generate_inputs,
                                     preRunMode                                          mode) {
#if !USING_CUDA
    RTP_LLM_FAIL("warmup is not supported on non-CUDA platforms");
    return {};
#else
    RTP_LLM_CHECK_WITH_INFO(executor_ == nullptr, "warmup must run before executor initialization");
    const RoleType warm_up_role =
        mode == preRunMode::prefill_warm_up ? RoleType::PREFILL : RoleType::DECODE;
    size_t max_consumed = 0;
    rtp_llm::setTraceMemory(true);
    try {
        const bool needs_cache = propose_params_ != nullptr || mode == preRunMode::decode_warm_up;
        auto cache_manager = needs_cache ? createWarmUpCacheManager(params) : nullptr;
        ResourceContext warm_up_context;
        warm_up_context.cache_manager = cache_manager;
        warm_up_context.role_type     = warm_up_role;
        executor_ = createExecutor(params, cache_manager, true, warm_up_role);
        THROW_IF_STATUSOR_ERROR(preRunWithResourceContext(generate_inputs, mode, warm_up_context));
        cudaSyncAndCheck();
        max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
        executor_.reset();
        rtp_llm::setTraceMemory(false);
    } catch (...) {
        executor_.reset();
        rtp_llm::setTraceMemory(false);
        throw;
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("prefillWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    const int64_t reserved_tokens = std::max<int64_t>(reserve_step_, 1);
    RTP_LLM_CHECK_WITH_INFO(model_config_.max_seq_len > reserved_tokens,
                            "warmup sequence length must exceed reserved speculative window");
    const size_t batch_size = runtime_config.fifo_scheduler_config.max_context_batch_size;
    auto fake_inputs = makeWarmUpInputs(static_cast<size_t>(model_config_.max_seq_len - reserved_tokens),
                                        batch_size,
                                        propose_params_ != nullptr);
    return runWarmUp(params, fake_inputs, preRunMode::prefill_warm_up);
#endif
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("decodeWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    const int64_t reserved_tokens = std::max<int64_t>(reserve_step_, 1);
    RTP_LLM_CHECK_WITH_INFO(model_config_.max_seq_len > reserved_tokens,
                            "warmup sequence length must exceed reserved speculative window");
    const size_t batch_size = runtime_config.max_generate_batch_size;
    auto fake_inputs = makeWarmUpInputs(static_cast<size_t>(model_config_.max_seq_len - reserved_tokens),
                                        batch_size,
                                        propose_params_ != nullptr);
    return runWarmUp(params, fake_inputs, preRunMode::decode_warm_up);
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
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
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
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
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
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (stopped_) {
        return absl::OkStatus();
    }
    RTP_LLM_LOG_INFO("stop normal engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_) {
        loop_thread_->join();
        loop_thread_.reset();
    }
    stopped_ = true;
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
    while (pause_) {
        // wait 50ms if system paused.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

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

    // If any stream in this batch requested gen_timeline AND no profiling session is
    // already active, configure + tick BEFORE process() so the profiler is up before
    // the actual work runs. This guarantees the trace captures THIS step's work, not
    // the next step's. If a session is already active (e.g. external StartProfile RPC),
    // skip — first-come-first-served.
    if (!step_profiler_.enabled()) {
        for (const auto& stream : streams) {
            if (stream && stream->genTimeline()) {
                const auto& cfg = stream->generateConfig();
                step_profiler_.configure(true, cfg->profile_trace_name, 0, cfg->profile_step);
                step_profiler_.tick();  // start profiler now (start_step=0)
                break;
            }
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        status = executor_->process(streams);
    }

    // tick profiler after process() to count this step (and stop when num_steps reached).
    // All TP ranks synchronize inside process() via NCCL, so stop happens at aligned points.
    step_profiler_.tick();

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
                        propose_step,
                        model_config_,
                        propose_params_->getEngineInitParams().model_config_,
                        runtime_config,
                        resource_context_));
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
                        propose_step,
                        model_config_,
                        propose_params_->getEngineInitParams().model_config_,
                        runtime_config,
                        resource_context_));
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

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include <memory>
#include <thread>
#include <random>

using namespace std;
namespace rtp_llm {

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
    profiler_step_(0),
    gen_timeline_sync_(params.profiling_debug_logging_config.gen_timeline_sync) {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
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
    initCacheManager(warm_up_result);
    RTP_LLM_LOG_INFO("create cache manager done");

    initExecutor(params, propose_params_);
    if (propose_params_) {
        reserve_step_ = propose_params_->gen_num_per_circle + 1;
    } else {
        reserve_step_ = 0;
    }

    RTP_LLM_LOG_INFO("create normal executor done");
    initScheduler();
    (void)startLoop();
}

void NormalEngine::initExecutor(const EngineInitParams&                        params,
                                std::unique_ptr<ProposeModelEngineInitParams>& propose_params) {
    if (propose_params_) {
        executor_.reset(
            new MtpExecutor(params, propose_params, resource_context_.cache_manager, device_, getLoraManager()));
    } else {
        executor_.reset(new NormalExecutor(params, resource_context_.cache_manager, device_, getLoraManager()));
    }
}

void NormalEngine::initScheduler() {
    if (runtime_config.use_batch_decode_scheduler) {
        scheduler_.reset(
            new BatchDecodeScheduler(runtime_config, resource_context_.cache_manager, metrics_reporter_, device_));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else if (runtime_config.use_gather_batch_scheduler) {
        scheduler_.reset(new GatherBatchScheduler(runtime_config,
                                                  model_config_,
                                                  pd_sep_config,
                                                  parallelism_config,
                                                  model_specific_config,
                                                  resource_context_.cache_manager,
                                                  metrics_reporter_));
        RTP_LLM_LOG_INFO("create gather batch scheduler done");
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
    auto stream = std::make_shared<NormalGenerateStream>(generate_input,
                                                         model_config_,
                                                         runtime_config,
                                                         resource_context_,
                                                         nullptr,
                                                         0,
                                                         mode == preRunMode::prefill_warm_up);
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        stream->fakeInitKVBlock();
    } else if (mode == preRunMode::build_system_prompt) {
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
    fake_input->input_ids =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {seq_len}, rtp_llm::AllocationType::HOST});
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->generate_config->top_k = 1;
    std::default_random_engine         generator;
    size_t                             token_size = model_config_.embedding_size ?
                                                        std::min(model_config_.embedding_size, model_config_.vocab_size) :
                                                        model_config_.vocab_size;
    std::uniform_int_distribution<int> distribution(0, token_size - 1);
    for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
        *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    }

    return fake_input;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.fifo_scheduler_config.max_context_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    device_->setTraceMemory(true);
    executor_.reset(new NormalExecutor(params, nullptr, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return WarmUpResult(
        {device_status.device_memory_status.preserved_bytes, device_status.device_memory_status.max_consumed_bytes});
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.max_generate_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    device_->setTraceMemory(true);

    auto cache_config               = CacheConfigCreator::createBasicConfig(model_config_, parallelism_config);
    cache_config.seq_size_per_block = model_config_.attn_config.tokens_per_block;
    cache_config.block_num          = 5;
    ParallelismConfig temp_parallelism_config;
    RuntimeConfig     temp_runtime_config;
    auto              cache_manager = make_shared<KVCacheManager>(
        cache_config, device_, true, nullptr, KVCacheConfig{}, temp_parallelism_config, temp_runtime_config);
    if (!cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed in decodeWarmUp");
    }
    executor_.reset(new NormalExecutor(params, cache_manager, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return WarmUpResult(
        {device_status.device_memory_status.preserved_bytes, device_status.device_memory_status.max_consumed_bytes});
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
            config, device_, false, metrics_reporter_, kv_cache_config, parallelism_config, runtime_config);
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }

    } else {
        auto result = CacheConfigCreator::createConfig(
            model_config_, parallelism_config, runtime_config, kv_cache_config, warm_up_result);
        RTP_LLM_LOG_INFO(
            "create cache manager with block nums %d, block size %ld KB", result.block_num, result.block_size / 1024);
        resource_context_.cache_manager = make_shared<KVCacheManager>(
            result, device_, false, metrics_reporter_, kv_cache_config, parallelism_config, runtime_config);
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
    }
}

absl::Status NormalEngine::initSystemPrompt() {
    resource_context_.reuse_cache         = kv_cache_config.reuse_cache;
    resource_context_.enable_3fs          = kv_cache_config.enable_3fs;
    resource_context_.enable_device_cache = kv_cache_config.enable_device_cache;
    resource_context_.enable_memory_cache = kv_cache_config.enable_memory_cache;
    resource_context_.write_cache_sync    = kv_cache_config.write_cache_sync;

    if (!kv_cache_config.multi_task_prompt_tokens.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(system_prompt_param,
                             SystemPromptConstructor::construct(kv_cache_config,
                                                                this,
                                                                resource_context_.cache_manager.get(),
                                                                device_->getDeviceProperties().tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }

    return absl::OkStatus();
}

KVCacheInfo NormalEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

absl::Status NormalEngine::startLoop() {
    RTP_LLM_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt());
    RTP_LLM_LOG_INFO("init system prompt done");
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
    RTP_LLM_LOG_INFO("loop begin");
    device_->preRun();
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
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
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
        streams.push_back(stream);
    }
    (void)scheduler_->batchEnqueue(streams);
    return streams;
}

absl::Status NormalEngine::step() {
    while (pause_) {
        // wait 50ms if system paused.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step_));
        if (parallelism_config.dp_size > 1) {
            mayAddFakeStream(streams);
        }
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    bool gen_timeline = !streams.empty() && std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
        return stream->genTimeline();
    });
    if (gen_timeline && !streams.empty()) {
        auto it        = std::max_element(streams.begin(), streams.end(), [](const auto& a, const auto& b) {
            return a->profileStep() < b->profileStep();
        });
        profiler_step_ = (*it)->profileStep();
    }
    if (gen_timeline_sync_) {
        auto world_size = device_->getDeviceProperties().dp_size * device_->getDeviceProperties().tp_size;
        auto world_rank = device_->getDeviceProperties().dp_rank * device_->getDeviceProperties().tp_size
                          + device_->getDeviceProperties().tp_rank;
        auto gen_timeline_buffer = device_->allocateBuffer({DataType::TYPE_UINT8, {world_size}, AllocationType::HOST});
        *(gen_timeline_buffer->dataWithOffset<uint8_t>(world_rank)) = static_cast<uint8_t>(profiler_step_);
        device_->allGather({{gen_timeline_buffer}, ParallelMode::DP_AND_TP});
        device_->syncCommunication(false);
        auto it        = std::max_element(gen_timeline_buffer->data<uint8_t>(),
                                   gen_timeline_buffer->dataWithOffset<uint8_t>(world_size),
                                   [](const uint8_t a, const uint8_t b) { return a < b; });
        profiler_step_ = *it;
        gen_timeline   = profiler_step_ > 0;
    }
    if (gen_timeline && nullptr == profiler_) {
        auto stream_group = StreamGroups(streams);
        auto world_rank   = device_->getDeviceProperties().dp_rank * device_->getDeviceProperties().tp_size
                          + device_->getDeviceProperties().tp_rank;
        auto profiler_prefix = autil::StringUtil::formatString("normal_profiler_wr%d_b%d_s%d_prefill%d_",
                                                               world_rank,
                                                               stream_group.totalModelBatchSize(),
                                                               stream_group.maxSeqLen(),
                                                               int(stream_group.totalContextBatchSize() > 0));
        profiler_ =
            std::make_shared<CudaProfiler>(profiler_prefix, profiling_debug_logging_config.torch_cuda_profiler_dir);
        profiler_->start();
    }
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = executor_->process(streams);

    if (nullptr != profiler_) {
        profiler_step_--;
        if (profiler_step_ <= 0) {
            profiler_.reset();
            profiler_step_ = 0;
        }
    }

    // report step metrics
    if (device_->getDeviceProperties().tp_rank == 0) {
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({false, false, step_latency});
    }

    return status;
}

bool NormalEngine::updateEplbConfig(const EPLBConfig& config) {
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
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
                    streams.emplace_back(MtpExecutor::createMinFakePrefillStream(
                        propose_step, model_config_, runtime_config, resource_context_, device_));
                }
                break;
            case RoleType::DECODE:
                if (streams.empty()) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, device_));
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
                    streams.emplace_back(MtpExecutor::createMinFakePrefillStream(
                        propose_step, model_config_, runtime_config, resource_context_, device_));
                }
                if (!has_decode) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, device_));
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

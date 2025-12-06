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
#include "rtp_llm/cpp/cache_new/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams& params):
    EngineBase(params),
    params_(params.gpt_init_parameter),
    metrics_reporter_(params.metrics_reporter),
    profiler_step_(0),
    gen_timeline_sync_(params.gpt_init_parameter.profiling_debug_logging_config.gen_timeline_sync) {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
    if (params_.warm_up_ && (!params_.is_multimodal_) && !params_.ffn_disaggregate_config.enable_ffn_disaggregate) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         params_.max_context_batch_size_,
                         params_.max_seq_len_,
                         int(params_.warm_up_with_loss_));
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
    executor_.reset(new NormalExecutor(params, resource_context_.cache_manager, device_, getLoraManager()));
    RTP_LLM_LOG_INFO("create normal executor done");
    initScheduler();
    (void)startLoop();
}

void NormalEngine::initScheduler() {
    if (params_.scheduler_config.use_batch_decode_scheduler) {
        scheduler_.reset(
            new BatchDecodeScheduler(params_, resource_context_.cache_manager, metrics_reporter_, device_));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else if (params_.scheduler_config.use_gather_batch_scheduler) {
        scheduler_.reset(new GatherBatchScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
        RTP_LLM_LOG_INFO("create gather batch scheduler done");
    } else {
        scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
        RTP_LLM_LOG_INFO("create fifo scheduler done");
    }
}

NormalEngine::~NormalEngine() {
    RTP_LLM_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                       preRunMode                            mode) {
    auto stream = std::make_shared<NormalGenerateStream>(
        generate_input, params_, resource_context_, nullptr, 0, mode == preRunMode::prefill_warm_up);
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        stream->fakeInitKVBlock();
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUSOR_ERROR(stream->initKVBlock(0, 0));
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return stream;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    if (params_.scheduler_config.use_batch_decode_scheduler) {
        if (params_.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0) {
            return decodeWarmUp(params);
        } else {
            return prefillWarmUp(params);
        }
    }
    if (params_.role_type_ == RoleType::PDFUSION || params_.role_type_ == RoleType::PREFILL) {
        return prefillWarmUp(params);
    } else if (params_.role_type_ == RoleType::DECODE) {
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
    std::default_random_engine generator;
    size_t                     token_size =
        params_.embedding_size_ ? std::min(params_.embedding_size_, params_.vocab_size_) : params_.vocab_size_;
    std::uniform_int_distribution<int> distribution(0, token_size - 1);
    for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
        *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    }

    return fake_input;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
    auto fake_input                                   = makeFakeInput((size_t)params_.max_seq_len_ - 1);
    fake_input->generate_config->num_return_sequences = params_.max_context_batch_size_;
    fake_input->generate_config->calculate_loss       = int(params_.warm_up_with_loss_);
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
    auto fake_input                                   = makeFakeInput((size_t)params_.max_seq_len_ - 1);
    fake_input->generate_config->num_return_sequences = params_.max_generate_batch_size_;
    fake_input->generate_config->calculate_loss       = int(params_.warm_up_with_loss_);
    device_->setTraceMemory(true);

    auto cache_config               = rtp_llm::CacheConfigCreator::createBasicConfig(params_);
    cache_config.seq_size_per_block = params_.seq_size_per_block_;
    cache_config.block_num          = 5;
    auto cache_manager              = make_shared<KVCacheManager>(cache_config, device_, false, nullptr, params_);
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
    stream->setIsDummyStream(true);
    stream->setMetricsReporter(nullptr);
    stream->fakeInitKVBlock();
    return stream;
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    auto result = rtp_llm::CacheConfigCreator::createConfig(params_, warm_up_result);
    RTP_LLM_LOG_INFO(
        "create kv cache manager with block nums %d, block size %ld KB", result.block_num, result.block_size / 1024);
    resource_context_.cache_manager = make_shared<KVCacheManager>(result, device_, false, nullptr, params_);
    if (!resource_context_.cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed");
    }
}

absl::Status NormalEngine::initSystemPrompt() {
    resource_context_.reuse_cache               = params_.reuse_cache_;
    resource_context_.enable_3fs                = params_.kv_cache_config.enable_3fs;
    resource_context_.enable_memory_block_cache = params_.kv_cache_config.memory_block_cache_size_mb > 0;

    if (!params_.multi_task_prompt_tokens_.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(
            system_prompt_param,
            SystemPromptConstructor::construct(
                params_, this, resource_context_.cache_manager.get(), device_->getDeviceProperties().tp_rank == 0));
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
    std::shared_ptr<GenerateStream> stream =
        std::make_shared<NormalGenerateStream>(input, params_, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream =
        std::make_shared<NormalGenerateStream>(input, params_, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

std::vector<std::shared_ptr<GenerateStream>>
NormalEngine::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    std::vector<std::shared_ptr<GenerateStream>> streams;
    streams.reserve(inputs.size());
    for (auto& inp : inputs) {
        auto stream = std::make_shared<NormalGenerateStream>(inp, params_, resource_context_, metrics_reporter_);
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
    if (device_->getDeviceProperties().tp_rank == 0 && !params_.ffn_disaggregate_config.is_ffn_service()) {
        CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        if (streams.empty()) {
            if (params_.dp_size_ > 1) {
                CHECK_AND_ASSIGN(streams, scheduler_->schedule());
                if (streams.empty()) {
                    streams.emplace_back(createMinFakeStream(1));
                }
            } else {
                return absl::OkStatus();
            }
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
        profiler_            = std::make_shared<CudaProfiler>(profiler_prefix,
                                                   params_.profiling_debug_logging_config.torch_cuda_profiler_dir);
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

const rtp_llm::GptInitParameter NormalEngine::gptInitParameter() const {
    return params_;
}

bool NormalEngine::updateEplbConfig(const EplbConfig& config) {
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm

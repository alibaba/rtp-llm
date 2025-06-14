#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>
#include <random>

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams& params) :
    EngineBase(params),
    params_(params.gpt_init_parameter),
    metrics_reporter_(params.metrics_reporter),
    profiler_step_(0),
    gen_timeline_sync_(autil::EnvUtil::getEnv("GEN_TIMELINE_SYNC", 0L))
{
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
    if (params_.warm_up_ && (!params_.is_multimodal_)) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                params_.max_context_batch_size_, params_.max_seq_len_, int(params_.warm_up_with_loss_));
        warm_up_result = warmUp(params);
        RTP_LLM_LOG_INFO("warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
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
    if (device_->getDeviceProperties().tp_rank == 0 && scheduler_->canLoadBalance()) {
        initLoadBalance();
    }
}

void NormalEngine::initScheduler() {
    if (getenv("USE_BATCH_DECODE_SCHEDULER") && std::string(getenv("USE_BATCH_DECODE_SCHEDULER")) == "1") {
        scheduler_.reset(new BatchDecodeScheduler(params_, resource_context_.cache_manager, metrics_reporter_, device_));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else {
        scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
        RTP_LLM_LOG_INFO("create fifo scheduler done");
    }
}

NormalEngine::~NormalEngine() {
    RTP_LLM_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(
    const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) {
    auto stream = std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context_, nullptr);
    if (mode == preRunMode::prefill_warm_up) {
        stream->setPerfTest(true);
    } else if (mode == preRunMode::decode_warm_up) {
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
    if (params_.isPDFusion() || params_.isPrefillRole()) {
        return prefillWarmUp(params);
    } else {
        return decodeWarmUp(params);
    }
}

std::shared_ptr<GenerateInput> NormalEngine::makeFakeInput(size_t seq_len) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->input_ids                     = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {seq_len}, rtp_llm::AllocationType::HOST});

    std::default_random_engine generator;
    size_t token_size = params_.embedding_size_ ? std::min(params_.embedding_size_, params_.vocab_size_) : params_.vocab_size_;
    std::uniform_int_distribution<int> distribution(0, token_size - 1);
    for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
        *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    }

    return fake_input;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
    auto fake_input = makeFakeInput((size_t)params_.max_seq_len_ - 1);
    fake_input->generate_config->num_return_sequences = params_.max_context_batch_size_;
    fake_input->generate_config->calculate_loss = int(params_.warm_up_with_loss_);
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);
    executor_.reset(new NormalExecutor(params, nullptr, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return WarmUpResult({
        device_status.device_memory_status.preserved_bytes,
        device_status.device_memory_status.max_consumed_bytes});
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
    auto fake_input = makeFakeInput((size_t)params_.max_seq_len_ - 1);
    fake_input->generate_config->num_return_sequences = params_.max_generate_batch_size_;
    fake_input->generate_config->calculate_loss = int(params_.warm_up_with_loss_);
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);

    auto cache_config = CacheConfigCreator::createBasicConfig(params_);
    cache_config.seq_size_per_block = params_.seq_size_per_block_;
    cache_config.block_nums = 5;
    auto cache_manager = make_shared<CacheManager>(cache_config, device_, true);
    executor_.reset(new NormalExecutor(params, cache_manager, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return WarmUpResult({
        device_status.device_memory_status.preserved_bytes,
            device_status.device_memory_status.max_consumed_bytes});
}

std::shared_ptr<GenerateStream> NormalEngine::enqueueMinFakeQuery(int32_t max_new_tokens) {
    RTP_LLM_LOG_DEBUG("enqueue min fake query");
    auto fake_input = makeFakeInput(1);
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->generate_config->top_k = 1;
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query = true;
    auto stream = makeStream(fake_input);
    stream->setMetricsReporter(nullptr);
    enqueue(stream);
    return stream;
}

void NormalEngine::initLoadBalance() {
    RTP_LLM_LOG_INFO("init load balance start");
    auto stream = enqueueMinFakeQuery(3);
    while(!stream->finished() && !stream->stopped()) {
        RTP_LLM_LOG_INFO("wait load balance init run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    RTP_LLM_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)",
            step_recorder_.getStepPerMin(), step_recorder_.getStepLatency());
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    auto result = CacheConfigCreator::createConfig(params_, warm_up_result);
    RTP_LLM_LOG_INFO("create cache manager with block nums %d, block size %ld KB",
                result.block_nums, result.block_size / 1024);
    resource_context_.cache_manager = make_shared<CacheManager>(result, device_, false, metrics_reporter_);
}

absl::Status NormalEngine::initSystemPrompt() {
    resource_context_.reuse_cache = params_.reuse_cache_;

    if (!params_.multi_task_prompt_tokens_.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(system_prompt_param,
                SystemPromptConstructor::construct(params_, this, resource_context_.cache_manager.get(),
                device_->getDeviceProperties().tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }

    return absl::OkStatus();
}

LoadBalanceInfo NormalEngine::getLoadBalanceInfo() {
    auto kv_cache_info = resource_context_.cache_manager->getKVCacheInfo();
    return LoadBalanceInfo{
        (int64_t)step_recorder_.getStepLatency(),
        (int64_t)step_recorder_.getStepCount(),
        (int64_t)step_recorder_.getStepPerMin(),
        (int64_t)kv_cache_info.available_kv_cache,
        (int64_t)kv_cache_info.total_kv_cache,
        (int64_t)scheduler_->onflightStreams()
    };
}

absl::Status NormalEngine::startLoop() {
    RTP_LLM_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt());
    RTP_LLM_LOG_INFO("init system prompt done");
    RTP_LLM_LOG_INFO("start normal engine loop");
    running_ = true;
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
        input, params_, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, params_, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

absl::Status NormalEngine::step() {
    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        if (scheduler_->empty() || step_recorder_.empty()) {
            step_recorder_.reset();
            step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds());
        }
        CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        if (streams.empty()) {
            if (params_.dp_size_ > 1) {
                CHECK_AND_ASSIGN(streams, scheduler_->schedule());
                if (streams.empty()) {
                    enqueueMinFakeQuery(1);
                    return absl::OkStatus();
                }
            } else {
                return absl::OkStatus();
            }
        }
    }
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    bool gen_timeline = !streams.empty() && std::any_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->genTimeline(); });
    if (gen_timeline_sync_) {
        auto world_size = device_->getDeviceProperties().dp_size * device_->getDeviceProperties().tp_size;
        auto world_rank = device_->getDeviceProperties().dp_rank * device_->getDeviceProperties().tp_size + device_->getDeviceProperties().tp_rank;
        auto gen_timeline_buffer = device_->allocateBuffer(
                {DataType::TYPE_BOOL, {world_size}, AllocationType::HOST});
        *(gen_timeline_buffer->dataWithOffset<bool>(world_rank)) = gen_timeline;
        device_->allGather({{gen_timeline_buffer}, ParallelMode::DP_AND_TP});
        device_->syncCommunication();
        gen_timeline = std::any_of(gen_timeline_buffer->data<bool>(), gen_timeline_buffer->dataWithOffset<bool>(world_size), [](auto s) { return s;});
    }
    profiler_step_--;
    if (profiler_step_ <= 0) {
        profiler_.reset();
        profiler_step_ = 0;
    }
    if (gen_timeline && profiler_step_ <= 0) {
        auto stream_group = StreamGroups(streams);        
        auto world_rank = device_->getDeviceProperties().dp_rank * device_->getDeviceProperties().tp_size + device_->getDeviceProperties().tp_rank;
        auto profiler_prefix = autil::StringUtil::formatString("normal_profiler_wr%d_b%d_s%d_prefill%d_",
                                                               world_rank,
                                                               stream_group.totalModelBatchSize(),
                                                               stream_group.maxSeqLen(),
                                                               int(stream_group.totalContextBatchSize() > 0));
        profiler_            = std::make_shared<CudaProfiler>(profiler_prefix);
        profiler_->start();
        profiler_step_ = 3;
    }
    int64_t step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status;
    if (params_.world_size_ > 1) {
        status = executor_->process(streams);
    } else {
        try {
            status = executor_->process(streams);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("step running error: %s", e.what());
            for (auto& stream: streams) {
                stream->stopAndRelease(ErrorCode::EXECUTION_EXCEPTION, e.what());
            }
        }
    }

    // report step metrics
    if (device_->getDeviceProperties().tp_rank == 0) {
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({false, false, step_latency});
        for (auto& stream: streams) {
            if (stream->finished()) {
                step_recorder_.addStepCount(stream->iterCount());
            }
        }
        step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds());
    }

    return status;
}

const rtp_llm::GptInitParameter NormalEngine::gptInitParameter() const {
    return params_;
}

bool NormalEngine::updateEplbConfig(const EplbConfig& config)
{
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm

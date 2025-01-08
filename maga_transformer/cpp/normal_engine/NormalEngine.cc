#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams& params) :
    EngineBase(params),
    params_(params.gpt_init_parameter),
    metrics_reporter_(params.metrics_reporter)
{
    FT_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
    if (params_.warm_up_ && (!params_.is_multimodal_)) {
        // warm up
        FT_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin", params_.max_context_batch_size_, params_.max_seq_len_, int(params_.warm_up_with_loss_));
        warm_up_result = warmUp(params);
        FT_LOG_INFO("warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
                    warm_up_result->max_used_memory,
                    warm_up_result->max_used_memory / 1024 / 1024,
                    warm_up_result->device_reserved_bytes,
                    warm_up_result->device_reserved_bytes / 1024 / 1024);
    } else {
        FT_LOG_INFO("skip warm up.");
    }
    initCacheManager(warm_up_result);
    FT_LOG_INFO("create cache manager done");
    executor_.reset(new NormalExecutor(params, resource_context_.cache_manager, device_, getLoraManager()));
    FT_LOG_INFO("create normal executor done");
    scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
    FT_LOG_INFO("create fifo scheduler done");
    (void)startLoop();
    if (device_->getDeviceProperties().tp_rank == 0) {
        initLoadBalance();
    }
}

NormalEngine::~NormalEngine() {
    FT_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) {
    auto stream = std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context_, nullptr);
    if (mode == preRunMode::warm_up) {
        stream->setPerfTest(true);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUSOR_ERROR(stream->initKVBlock(0, 0));
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return stream;
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)params_.max_seq_len_ - 1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = params_.max_context_batch_size_;
    fake_input->generate_config->calculate_loss = int(params_.warm_up_with_loss_);
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);
    executor_.reset(new NormalExecutor(params, nullptr, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return WarmUpResult({
        device_status.device_memory_status.preserved_bytes,
        device_status.device_memory_status.max_consumed_bytes});
}

void NormalEngine::initLoadBalance() {
    FT_LOG_INFO("init load balance start");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = 3;
    fake_input->generate_config->top_k = 1;
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto stream = enqueue(fake_input);
    while(!stream->finished() && !stream->stopped()) {
        FT_LOG_INFO("wait load balance init run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    FT_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)", step_recorder_.getStepPerMin(), step_recorder_.getStepLatency());
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    auto result = CacheConfigCreator::createConfig(params_, warm_up_result);
    FT_LOG_INFO("create cache manager with block nums %d, block size %ld MiB",
                result.block_nums, result.block_size / 1024 / 1024);
    resource_context_.cache_manager = make_shared<CacheManager>(result, device_, metrics_reporter_);
}

absl::Status NormalEngine::initSystemPrompt() {
    if (device_->getDeviceProperties().tp_rank == 0) {
        resource_context_.reuse_cache = params_.reuse_cache_;
    }

    if (!params_.multi_task_prompt_tokens_.empty()) {
        if (device_->getDeviceProperties().tp_rank == 0) {
            resource_context_.reuse_cache = true;
            CHECK_AND_RETURN_REF(system_prompt_param, SystemPromptConstructor::construct(params_, this, resource_context_.cache_manager.get()));
            resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
        } else {
            std::list<GenerateStreamPtr> streams;
            THROW_IF_STATUS_ERROR(executor_->process(streams));
        }
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
        (int64_t)kv_cache_info.total_kv_cache
    };
}

absl::Status NormalEngine::startLoop() {
    FT_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt());
    FT_LOG_INFO("init system prompt done");
    FT_LOG_INFO("start normal engine loop");
    running_ = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    FT_LOG_INFO("stop normal engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void NormalEngine::loop() {
    FT_LOG_INFO("loop begin");
    device_->preRun();
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(input, params_, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(input, params_, resource_context_, metrics_reporter_);
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
            return absl::OkStatus();
        }
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    absl::Status status;
    try {
        status = executor_->process(streams);
    } catch (const std::exception& e) {
        FT_LOG_ERROR("step running error: %s", e.what());
        for (auto& stream: streams) {
            stream->stopAndRelease(ErrorCode::EXECUTION_EXCEPTION, e.what());
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

const ft::GptInitParameter NormalEngine::gptInitParameter() const {
    return params_;
}

}  // namespace rtp_llm

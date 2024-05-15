#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/utils/logger.h"
#include "maga_transformer/cpp/utils/TimeUtility.h"

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const MagaInitParams&                                                   params,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                           const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights,
                           const kmonitor::MetricsReporterPtr                                      metrics_reporter) :
    params_(params),
    metrics_reporter_(metrics_reporter)
{
    auto global_params = ft::DeviceFactory::getDefaultGlobalDeviceParams();
    auto& default_device_params = global_params.device_params[0].second;
    const auto rank       = stoi(string(getenv("WORLD_RANK") ? getenv("WORLD_RANK") : "0"));
    const auto world_size = stoi(string(getenv("WORLD_SIZE") ? getenv("WORLD_SIZE") : "1"));
    default_device_params.tp_size = rank;
    default_device_params.tp_rank = world_size;
    default_device_params.master_ip = params.gpt_init_parameter->nccl_ip_;
    default_device_params.master_port = params.gpt_init_parameter->nccl_port_;
    ft::DeviceFactory::initDevices(global_params);
    device_ = ft::DeviceFactory::getDefaultDevice();;
    executor_.reset(new NormalExecutor(params, layer_weights, weights, metrics_reporter_));
    initCacheManager();
    scheduler_.reset(new FIFOScheduler(params, resource_context_.cache_manager, metrics_reporter));
    (void)startLoop();
}

NormalEngine::~NormalEngine() {
    FT_LOG_INFO("destory normal engine");
    (void)stop();
}

void NormalEngine::initCacheManager() {
    auto result = CacheConfigCreator::createConfig(*params_.gpt_init_parameter);
    THROW_IF_STATUS_ERROR(result.status());
    resource_context_.cache_manager = make_shared<CacheManager>(result.value(), device_, metrics_reporter_);
}

void NormalEngine::initSystemPrompt() {
    resource_context_.reuse_cache = params_.gpt_init_parameter->reuse_cache_;
    auto ptuning_param = SystemPromptConstructor::construct(*params_.gpt_init_parameter, this, resource_context_.cache_manager.get());
    if (!ptuning_param.empty()) {
        resource_context_.reuse_cache = true;
        resource_context_.system_prompt.reset(new SystemPrompt(ptuning_param));
    }
}

absl::Status NormalEngine::addLoRA(const int64_t                                                   lora_id,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    auto status = executor_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
    reportMetrics({status.ok(), !status.ok(), 0});
    return status;
}

absl::Status NormalEngine::removeLoRA(const int64_t lora_id) {
    auto status = executor_->removeLoRA(lora_id);
    reportMetrics({status.ok(), !status.ok(), 0});
    return status;
}

absl::Status NormalEngine::startLoop() {
    FT_LOG_INFO("start normal engine");
    running_ = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    initSystemPrompt(); // system prompt constructor depends on engine startup
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
    while (running_) {
        int64_t step_begin_time_us = TimeUtility::currentTimeInMicroSeconds();
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
        reportMetrics({false, false, TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us});
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    stream->setMetricsReporter(metrics_reporter_);
    stream->setSpecialTokens(*(params_.gpt_init_parameter->special_tokens_));
    FT_LOG_DEBUG("enqueue stream: %s %d", stream->debugString().c_str(), device_->getDeviceProperties().tp_rank);
    return scheduler_->enqueue(stream);
}

absl::Status NormalEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        const auto streams_status = scheduler_->schedule();
        RETURN_IF_STATUS_OR_ERROR(streams_status);
        streams = streams_status.value();
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }
    return executor_->process(streams);
}

}  // namespace rtp_llm

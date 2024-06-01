#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/utils/logger.h"
#include "autil/TimeUtility.h"

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams& params) :
    EngineBase(params),
    params_(params.gpt_init_parameter),
    metrics_reporter_(params.metrics_reporter)
{
    executor_.reset(new NormalExecutor(params, device_));
    initCacheManager();
    scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
    (void)startLoop();
}

NormalEngine::~NormalEngine() {
    FT_LOG_INFO("destory normal engine");
    (void)stop();
}

void NormalEngine::initCacheManager() {
    auto result = CacheConfigCreator::createConfig(params_);
    // TODO(xinfei.sxf) test create cache config exception
    THROW_IF_STATUS_ERROR(result.status());
    resource_context_.cache_manager = make_shared<CacheManager>(result.value(), device_, metrics_reporter_);
}

void NormalEngine::initSystemPrompt() {
    resource_context_.reuse_cache = params_.reuse_cache_;
    auto system_prompt_param = SystemPromptConstructor::construct(params_, this, resource_context_.cache_manager.get());
    if (!system_prompt_param.empty()) {
        resource_context_.reuse_cache = true;
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
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
        int64_t step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
        reportMetrics({false, false, autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us});
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<GenerateStream>(input, params_, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
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

#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/utils/logger.h"

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const MagaInitParams&                                                   params,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                           const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights) : params_(params) {
    executor_.reset(new NormalExecutor(params, layer_weights, weights));
    // TODO(xinfei.sxf) cache config from where, new cache Manager
    int   block_num     = 100;
    char* block_num_env = std::getenv("BLOCK_NUM");
    if (block_num_env) {
        block_num = std::stoi(block_num_env);
    }
    CacheConfig                   cache_config(params.gpt_init_parameter->num_layers_,
                                               block_num,
                                               params.gpt_init_parameter->head_num_kv_,
                                               params.gpt_init_parameter->size_per_head_,
                                               params.gpt_init_parameter->seq_size_per_block_,
                                               ft::DataType::TYPE_FP16);
    ncclComm_t                    nccl_op;
    ft::DeviceBase*               device        = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    cache_manager_ = make_shared<CacheManager>(cache_config, device);
    scheduler_.reset(new FIFOScheduler(params, cache_manager_));
    (void)startLoop();
}


NormalEngine::~NormalEngine() {
    FT_LOG_DEBUG("destory Engine");
    (void)stop();
}

void NormalEngine::addLoRA(const int64_t                                                   lora_id,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    executor_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void NormalEngine::removeLoRA(const int64_t lora_id) {
    executor_->removeLoRA(lora_id);
}

absl::Status NormalEngine::startLoop() {
    running_ = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    FT_LOG_DEBUG("stop Engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void NormalEngine::loop() {
    FT_LOG_DEBUG("loop begin");
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

absl::Status NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    FT_LOG_DEBUG("enqueue stream: %s", stream->debugString().c_str());
    return scheduler_->enqueue(stream);
}

absl::Status NormalEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto streams_status = scheduler_->schedule();
    RETURN_IF_STATUS_OR_ERROR(streams_status);
    const auto& streams = streams_status.value();
    // FT_LOG_DEBUG("schedule res: %s", streams.debugString().c_str());
    if (streams.empty()) {
        FT_LOG_WARNING("no query run and sleep");
        return absl::OkStatus();
    }
    return executor_->process(streams);
}

}  // namespace rtp_llm

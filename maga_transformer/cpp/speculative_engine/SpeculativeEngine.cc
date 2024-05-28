#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeEngine.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeStream.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/dataclass/StreamGroups.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeExecutor.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include <memory>

using namespace std;

namespace rtp_llm {

SpeculativeEngine::SpeculativeEngine(
    const MagaInitParams&                                                   params,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights)
    : EngineBase(params)
{
    draft_executor_.reset(new NormalExecutor(params, layer_weights, weights));
    target_executor_.reset(new SpeculativeExecutor(params, layer_weights, weights));
    // TODO(xinfei.sxf) deal with sp cache config and sp system prompt
    int   block_num     = 100;
    char* block_num_env = std::getenv("BLOCK_NUM");
    if (block_num_env) {
        block_num = std::stoi(block_num_env);
    }
    CacheConfig     cache_config(params.gpt_init_parameter.num_layers_,
                             block_num,
                             params.gpt_init_parameter.head_num_kv_,
                             params.gpt_init_parameter.size_per_head_,
                             params.gpt_init_parameter.seq_size_per_block_,
                             ft::DataType::TYPE_FP16);
    ft::DeviceBase* device = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    draft_cache_manager_   = make_shared<CacheManager>(cache_config, device);
    target_cache_manager_  = make_shared<CacheManager>(cache_config, device);
    scheduler_.reset(new FIFOScheduler(params, target_cache_manager_));
}

SpeculativeEngine::~SpeculativeEngine() {
    (void)stop();
}

absl::Status SpeculativeEngine::startLoop() {
    running_     = true;
    loop_thread_ = std::thread(&SpeculativeEngine::loop, this);
    return absl::OkStatus();
}

absl::Status SpeculativeEngine::stop() {
    running_ = false;
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::loop() {
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
        }
    }
}

absl::Status SpeculativeEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    FT_LOG_DEBUG("enqueue stream: %s", stream->debugString().c_str());
    std::shared_ptr<SpeculativeStream> sp_stream = make_shared<SpeculativeStream>(stream,
            params_.gpt_init_parameter.gen_num_per_circle_, params_.gpt_init_parameter.vocab_size_);
    return scheduler_->enqueue(stream);
}

absl::Status SpeculativeEngine::updateDraftProb(const list<GenerateStreamPtr>& streams, uint index)
{
    for (auto &stream: streams) {
        SpeculativeStream *stream_ = dynamic_cast<SpeculativeStream*>(stream.get());
        auto output = stream_->nextOutput();
        stream_->updateDraftProb(*(output.value().generate_outputs[0].logits.value()), index);
    }
    return absl::OkStatus();
}


absl::Status SpeculativeEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto streams_status = scheduler_->schedule();
    RETURN_IF_STATUS_OR_ERROR(streams_status);
    const auto& streams = streams_status.value();
    for (uint i = 0; i < params_.gpt_init_parameter.gen_num_per_circle_; ++i) {
        RETURN_IF_STATUS_ERROR(draft_executor_->process(streams));
        (void)updateDraftProb(streams, i);
    }
    RETURN_IF_STATUS_ERROR(target_executor_->process(streams_status.value()));
    return absl::OkStatus();
}

}  // namespace rtp_llm

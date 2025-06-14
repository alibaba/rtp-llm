#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <exception>

using namespace std;
namespace rtp_llm {

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    params_(params.gpt_init_parameter), metrics_reporter_(params.metrics_reporter) {
    rtp_llm::DeviceFactory::initDevices(params.gpt_init_parameter);
    executor_.reset(new EmbeddingExecutor(params, rtp_llm::DeviceFactory::getDefaultDevice(), handler));
    scheduler_.reset(new EmbeddingScheduler(params_, metrics_reporter_));

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
}

absl::Status EmbeddingEngine::startLoop() {
    RTP_LLM_LOG_INFO("start embedding engine");
    running_ = true;
    loop_thread_ = std::thread(&EmbeddingEngine::loop, this);
    return absl::OkStatus();
}

absl::Status EmbeddingEngine::stop() {
    RTP_LLM_LOG_INFO("stop embedding engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void EmbeddingEngine::loop() {
    RTP_LLM_LOG_INFO("loop begin");
    while (running_) {
        auto status = step();
        if (!status.ok()) {
             RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id, std::optional<MultimodalFeature> multimodal_features) {
    auto input =
        std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
    return decode(input);
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(std::shared_ptr<EmbeddingInput> input) {
    auto embedding_stream = std::make_shared<EmbeddingStream>(input);
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return embedding_stream->embeddingOutput();
}

absl::Status EmbeddingEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status EmbeddingEngine::enqueue(EmbeddingStreamPtr streams) {
    return scheduler_->enqueue(streams);
}

absl::Status EmbeddingEngine::step() {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
    try {
        auto status = executor_->process(streams);
        if (!status.ok()) {
            for (auto& stream: streams) {
                stream->setError(status.ToString());
                RTP_LLM_LOG_WARNING("error_stream_info: length: %d, exception: %s", stream->inputLength(), status.ToString().c_str());
            }
        }
    } catch (const exception& e) {
        RTP_LLM_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), e.what());
        for (auto& stream: streams) {
            stream->setError(e.what());
            RTP_LLM_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
        }
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm

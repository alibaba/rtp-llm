#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <exception>

using namespace std;
namespace rtp_llm {

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    params_(params.gpt_init_parameter), metrics_reporter_(params.metrics_reporter) {
    rtp_llm::DeviceFactory::initDevices(params.gpt_init_parameter);
    executor_.reset(new EmbeddingExecutor(params, rtp_llm::DeviceFactory::getDefaultDevice(), handler));
    scheduler_.reset(new EmbeddingScheduler(params_, metrics_reporter_));
    gen_timeline_ = (autil::EnvUtil::getEnv("GEN_TIMELINE", "False") == "True");

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
}

const rtp_llm::GptInitParameter& EmbeddingEngine::GetGptInitParameter() {
    return params_;
}

absl::Status EmbeddingEngine::startLoop() {
    RTP_LLM_LOG_INFO("start embedding engine");
    running_     = true;
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

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(th::Tensor                       token_ids,
                                                         th::Tensor                       token_type_ids,
                                                         th::Tensor                       input_lengths,
                                                         int64_t                          request_id,
                                                         std::optional<MultimodalFeature> multimodal_features,
                                                         std::optional<th::Tensor>        input_embeddings) {
    auto input = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_features, input_embeddings);
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
    executor_->device_->syncAndCheck();
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
    if (gen_timeline_ && nullptr == profiler_) {
        profiler_ = std::make_shared<CudaProfiler>("embedding_profiler_");
        profiler_->start();
    }
    try {
        auto status = executor_->process(streams);
        if (!status.ok()) {
            for (auto& stream : streams) {
                stream->setError(status.ToString());
                RTP_LLM_LOG_WARNING(
                    "error_stream_info: length: %d, exception: %s", stream->inputLength(), status.ToString().c_str());
            }
        }
    } catch (const exception& e) {
        std::string error_msg = e.what();
        RTP_LLM_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), error_msg.c_str());
        for (auto& stream : streams) {
            stream->setError(error_msg);
            RTP_LLM_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
        }
        if (error_msg.find("CUDA Driver error") != string::npos || error_msg.find("CUDA error") != string::npos) {
            RTP_LLM_LOG_ERROR("detect CUDA error, do abort");
            abort();
        }
    }
    executor_->device_->syncAndCheck();
    profiler_.reset();
    return absl::OkStatus();
}

}  // namespace rtp_llm

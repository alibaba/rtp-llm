#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "src/fastertransformer/utils/logger.h"

using namespace std;
namespace rtp_llm {

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    params_(params.gpt_init_parameter), metrics_reporter_(params.metrics_reporter) {
    EngineBase::initDevices(params);
    executor_.reset(new EmbeddingExecutor(params, ft::DeviceFactory::getDefaultDevice(), handler));
    scheduler_.reset(new EmbeddingScheduler(params_, metrics_reporter_));

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    FT_LOG_INFO("destory embedding engine");
    (void)stop();
}

absl::Status EmbeddingEngine::startLoop() {
    FT_LOG_INFO("start embedding engine");
    running_ = true;
    loop_thread_ = std::thread(&EmbeddingEngine::loop, this);
    return absl::OkStatus();
}

absl::Status EmbeddingEngine::stop() {
    FT_LOG_INFO("stop embedding engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void EmbeddingEngine::loop() {
    FT_LOG_INFO("loop begin");
    while (running_) {
        auto status = step();
        if (!status.ok()) {
             FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status EmbeddingEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status EmbeddingEngine::enqueue(EmbeddingStreamPtr streams) {
    return scheduler_->enqueue(streams);
}

absl::Status EmbeddingEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto streams_status = scheduler_->scheduleNew();
    RETURN_IF_STATUS_OR_ERROR(streams_status);
    auto& streams = streams_status.value();
    if (streams.empty()) {
        FT_LOG_WARNING("no query run and sleep");
        return absl::OkStatus();
    }
    try {
        RETURN_IF_STATUS_ERROR(executor_->process(streams));
        RETURN_IF_STATUS_OR_ERROR(streams_status);
        // RETURN_IF_STATUS_ERROR(update_streams(streams));
    } catch (const exception& e) {
        FT_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), e.what());
        for (auto& stream: streams) {
            stream->setError(e.what());
            FT_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
        }
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm

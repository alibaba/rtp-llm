#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <mutex>

using namespace std;
namespace rtp_llm {

EmbeddingScheduler::EmbeddingScheduler(const ModelConfig& model_config,
                                       const ConcurrencyConfig& concurrency_config,
                                       const RuntimeConfig& runtime_config,
                                       const kmonitor::MetricsReporterPtr metrics_reporter):
    model_config_(model_config), concurrency_config_(concurrency_config), runtime_config_(runtime_config), metrics_reporter_(metrics_reporter) {}

EmbeddingScheduler::~EmbeddingScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory EmbeddingScheduler");
}

absl::Status EmbeddingScheduler::stop() {
    RTP_LLM_LOG_INFO("stop EmbeddingScheduler");
    lock_guard<mutex> lock(lock_);
    stop_ = true;
    cond_.notify_all();
    return absl::OkStatus();
}

absl::Status EmbeddingScheduler::enqueue(EmbeddingStreamPtr stream) {
    lock_guard<mutex> lock(lock_);
    waiting_streams_.emplace_back(stream);
    cond_.notify_all();
    return absl::OkStatus();
}

absl::StatusOr<list<EmbeddingStreamPtr>> EmbeddingScheduler::scheduleNew() {
    unique_lock<mutex> lock(lock_);
    cond_.wait(lock, [this] { return stop_ || !waiting_streams_.empty(); });
    std::list<EmbeddingStreamPtr> new_streams;
    int                           total_len = 0;
    auto                          it        = waiting_streams_.begin();
    while (it != waiting_streams_.end()) {
        const auto& stream = *it;
        if (total_len + stream->inputLength() > runtime_config_.fifo_scheduler_config.max_context_batch_size * model_config_.max_seq_len) {
            break;
        }
        stream->setStart();
        new_streams.push_back(stream);
        total_len += stream->inputLength();
        it = waiting_streams_.erase(it);
    }
    // if new streams is empty, meaning that first stream is too big
    if (waiting_streams_.size() != 0 && new_streams.size() == 0) {
        it                 = waiting_streams_.begin();
        const auto& stream = *it;
        stream->setError("long prompt error, not scheduled");
        it = waiting_streams_.erase(it);
    }
    reportMetrics(new_streams.size());
    return new_streams;
}

void EmbeddingScheduler::reportMetrics(size_t new_stream_size) {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size    = waitingStreamsSize();
        collector.running_stream_size = new_stream_size;
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

int EmbeddingScheduler::waitingStreamsSize() {
    return waiting_streams_.size();
}

}  // namespace rtp_llm

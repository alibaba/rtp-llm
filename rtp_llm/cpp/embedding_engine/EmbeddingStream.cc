#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <algorithm>
#include <memory>
#include <utility>

using namespace std;

namespace rtp_llm {

namespace {
std::chrono::milliseconds checkedTimeout(std::chrono::milliseconds timeout) {
    if (timeout <= std::chrono::milliseconds::zero() || timeout > std::chrono::hours(24)) {
        throw std::invalid_argument("embedding request timeout must be in [1, 86400000] ms");
    }
    return timeout;
}
}  // namespace

EmbeddingStream::EmbeddingStream(const shared_ptr<rtp_llm::EmbeddingInput>& query,
                                 EmbeddingProfileConfig                     profile_config,
                                 std::chrono::milliseconds                  timeout):
    embedding_input_(query),
    profile_config_(std::move(profile_config)),
    timeout_(checkedTimeout(timeout)),
    deadline_(std::chrono::steady_clock::now() + timeout_) {
    if (!query.get()) {
        throw std::invalid_argument("embedding input must not be null");
    }
    begin_time_       = autil::TimeUtility::currentTimeInMilliSeconds();
    embedding_output_ = make_shared<EmbeddingOutput>();
    stream_state_     = StreamState::WAITING;
    begin_time_us_    = autil::TimeUtility::currentTimeInMicroSeconds();
}

std::chrono::milliseconds EmbeddingStream::timeoutForDeadline(std::chrono::system_clock::time_point deadline,
                                                              std::chrono::system_clock::time_point now) {
    if (deadline == std::chrono::system_clock::time_point::max() || deadline - now >= kRequestTimeout) {
        return kRequestTimeout;
    }
    return std::max(std::chrono::milliseconds(1),
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
}

int64_t EmbeddingStream::streamId() const {
    return embedding_input_->request_id;
}

const std::optional<MultimodalFeature>& EmbeddingStream::multimodalFeature() const {
    return embedding_input_->multimodal_features;
}

int64_t EmbeddingStream::batchSize() const {
    return embedding_input_->input_lengths.size(0);
}

void EmbeddingStream::setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter) {
    metrics_reporter_ = metric_reporter;
}

std::shared_ptr<EmbeddingInput> EmbeddingStream::embeddingInput() const {
    return embedding_input_;
}

std::shared_ptr<EmbeddingOutput> EmbeddingStream::embeddingOutput() const {
    return embedding_output_;
}

int64_t EmbeddingStream::inputLength() const {
    return embedding_input_->total_length;
}

void EmbeddingStream::waitFinish(const std::function<bool()>& is_cancelled) {
    unique_lock<mutex> lock(lock_);
    while (stream_state_ != StreamState::FINISHED) {
        if (is_cancelled && is_cancelled()) {
            embedding_output_->setError(ErrorCode::CANCELLED, "embedding request cancelled");
        } else {
            // Cancellation has no stream notification; poll only callers that supply it.
            const auto wakeup =
                is_cancelled ? std::min(deadline_, std::chrono::steady_clock::now() + std::chrono::milliseconds(50)) :
                               deadline_;
            if (cond_.wait_until(lock, wakeup, [this] { return stream_state_ == StreamState::FINISHED; })) {
                break;
            }
            if (std::chrono::steady_clock::now() < deadline_) {
                continue;
            }
            embedding_output_->setError(ErrorCode::DEADLINE_EXCEEDED,
                                        "DEADLINE_EXCEEDED (8204): embedding request timeout after "
                                            + std::to_string(timeout_.count()) + " ms");
        }
        stream_state_ = StreamState::FINISHED;
        reportMetrics();
        cond_.notify_all();
    }
    if (embedding_output_->error_info.code() == ErrorCode::DEADLINE_EXCEEDED) {
        throw EmbeddingStreamTimeout("run stream failed: " + embedding_output_->error_info.ToString());
    }
    if (unavailable_) {
        throw EmbeddingStreamUnavailable("run stream failed: " + embedding_output_->error_info.ToString());
    }
    if (embedding_output_->error_info.code() == ErrorCode::CANCELLED) {
        throw EmbeddingStreamCancelled("run stream failed: " + embedding_output_->error_info.ToString());
    }
    if (!embedding_output_->error_info.ok()) {
        throw std::runtime_error("run stream failed: " + embedding_output_->error_info.ToString());
    }
}

void EmbeddingStream::reportMetrics() {
    if (metrics_reporter_) {
        RtpEmbeddingStreamMetricsCollector collector;
        collector.input_token_length = inputLength();
        collector.wait_latency_us    = wait_time_us_;
        collector.total_latency_us   = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        metrics_reporter_->report<RtpEmbeddingStreamMetrics, RtpEmbeddingStreamMetricsCollector>(nullptr, &collector);
    }
}

void EmbeddingStream::setError(const std::string& error_info, ErrorCode code) {
    lock_guard<mutex> lock(lock_);
    if (stream_state_ == StreamState::FINISHED) {
        return;
    }
    embedding_output_->setError(code, error_info);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::setUnavailable(const std::string& error_info) {
    lock_guard<mutex> lock(lock_);
    if (stream_state_ == StreamState::FINISHED) {
        return;
    }
    // No dedicated ErrorCode exists; CANCELLED keeps the output retry-oriented, unavailable_ picks the status.
    embedding_output_->setError(ErrorCode::CANCELLED, error_info);
    unavailable_  = true;
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::setEnqueueError(const absl::Status& status) {
    if (absl::IsUnavailable(status)) {
        setUnavailable(std::string(status.message()));
    } else {
        setError("enqueue embedding request failed: " + status.ToString());
    }
}

bool EmbeddingStream::setStart() {
    lock_guard<mutex> lock(lock_);
    if (stream_state_ == StreamState::FINISHED) {
        return false;
    }
    wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    stream_state_ = StreamState::RUNNING;
    return true;
}

void EmbeddingStream::updateTensorOutput(torch::Tensor t) {
    lock_guard<mutex> lock(lock_);
    if (stream_state_ == StreamState::FINISHED) {
        return;
    }
    embedding_output_->setTensorOutput(t);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map) {
    lock_guard<mutex> lock(lock_);
    if (stream_state_ == StreamState::FINISHED) {
        return;
    }
    embedding_output_->setMapOutput(map);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

}  // namespace rtp_llm

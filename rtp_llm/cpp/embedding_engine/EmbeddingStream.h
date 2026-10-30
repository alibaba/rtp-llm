#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <stdexcept>
#include <optional>
#include <queue>
#include <condition_variable>
#include "autil/TimeUtility.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "absl/status/statusor.h"

namespace rtp_llm {

struct EmbeddingProfileConfig {
    bool        gen_timeline = false;
    int         profile_step = 1;
    std::string profile_trace_name;
};

class EmbeddingStreamTimeout: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class EmbeddingStreamCancelled: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The server is stopping (or rejected the request); clients may retry elsewhere.
class EmbeddingStreamUnavailable: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class EmbeddingStream {
public:
    // Fixed backend queue + execution wait budget per embedding request.
    static constexpr std::chrono::milliseconds kRequestTimeout{60000};
    // kRequestTimeout, shortened to the remaining time of a finite client deadline (min 1 ms).
    static std::chrono::milliseconds timeoutForDeadline(std::chrono::system_clock::time_point deadline,
                                                        std::chrono::system_clock::time_point now);
    EmbeddingStream(const std::shared_ptr<EmbeddingInput>& query,
                    EmbeddingProfileConfig                 profile_config = {},
                    std::chrono::milliseconds              timeout        = kRequestTimeout);
    ~EmbeddingStream() {}

public:
    // Exported to python world.
    std::shared_ptr<EmbeddingInput>  embeddingInput() const;
    std::shared_ptr<EmbeddingOutput> embeddingOutput() const;

    const std::optional<MultimodalFeature>& multimodalFeature() const;

    bool genTimeline() const {
        return profile_config_.gen_timeline;
    }

    const EmbeddingProfileConfig& profileConfig() const {
        return profile_config_;
    }

    int64_t inputLength() const;

    int64_t streamId() const;

    int64_t batchSize() const;

    void updateTensorOutput(torch::Tensor t);
    void updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map);

    void setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter);

    void waitFinish(const std::function<bool()>& is_cancelled = {});

    bool setStart();

    void setError(const std::string& error_info, ErrorCode code = ErrorCode::UNKNOWN_ERROR);

    // Fails the stream because the server is stopping; waitFinish throws EmbeddingStreamUnavailable.
    void setUnavailable(const std::string& error_info);

    // Finishes a stream whose enqueue failed so waitFinish throws a typed error immediately.
    void setEnqueueError(const absl::Status& status);

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "EmbeddingStream {"
                     //  << "generate_input:" << generate_input_->debugString()
                     //  << ", max_seq_len:" << max_seq_len_
                     //  << ", input_length:" << inputLength()
                     //  << ", seq_length:" << seq_length_
                     //  << ", reuse_length:" << reuse_length_
                     //  << ", batch_size:" << batch_size_
                     << "}";
        return debug_string.str();
    }

protected:
    std::shared_ptr<EmbeddingInput>             embedding_input_;
    std::shared_ptr<EmbeddingOutput>            embedding_output_;
    EmbeddingProfileConfig                      profile_config_;
    const std::chrono::milliseconds             timeout_;
    const std::chrono::steady_clock::time_point deadline_;
    int64_t                                     begin_time_;
    std::condition_variable                     cond_;
    std::mutex                                  lock_;
    StreamState                                 stream_state_;
    bool                                        unavailable_      = false;
    size_t                                      begin_time_us_    = 0;
    size_t                                      wait_time_us_     = 0;
    kmonitor::MetricsReporterPtr                metrics_reporter_ = nullptr;
    std::optional<torch::Tensor>                context_position_ids_;

    void reportMetrics();
};

typedef std::shared_ptr<EmbeddingStream> EmbeddingStreamPtr;
}  // namespace rtp_llm

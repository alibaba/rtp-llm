#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <queue>
#include <condition_variable>
#include "rtp_llm/cpp/core/Buffer.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "absl/status/statusor.h"

namespace rtp_llm {

class EmbeddingStream {
public:
    EmbeddingStream(const std::shared_ptr<EmbeddingInput>& query);
    ~EmbeddingStream() {}

public:
    // Exported to python world.
    std::shared_ptr<EmbeddingInput>  embeddingInput() const;
    std::shared_ptr<EmbeddingOutput> embeddingOutput() const;

    const std::optional<MultimodalFeature>& multimodalFeature() const;

    int64_t inputLength() const;

    int64_t streamId() const;

    int64_t batchSize() const;

    void updateTensorOutput(torch::Tensor t);
    void updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map);

    void setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter);

    void waitFinish();

    void setStart();

    void setError(const std::string& error_info);

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
    rtp_llm::DeviceBase*             device_;
    std::shared_ptr<EmbeddingInput>  embedding_input_;
    std::shared_ptr<EmbeddingOutput> embedding_output_;
    int64_t                          begin_time_;
    std::condition_variable          cond_;
    std::mutex                       lock_;
    StreamState                      stream_state_;
    size_t                           begin_time_us_    = 0;
    size_t                           wait_time_us_     = 0;
    kmonitor::MetricsReporterPtr     metrics_reporter_ = nullptr;

    void reportMetrics();
};

typedef std::shared_ptr<EmbeddingStream> EmbeddingStreamPtr;
}  // namespace rtp_llm

#pragma once

#include <queue>
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class EmbeddingScheduler {
public:
    explicit EmbeddingScheduler(const ModelConfig& model_config,
                                const ConcurrencyConfig& concurrency_config,
                                const RuntimeConfig& runtime_config,
                                const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    ~EmbeddingScheduler();

    absl::Status enqueue(EmbeddingStreamPtr stream);

    absl::StatusOr<std::list<EmbeddingStreamPtr>> scheduleNew();

    absl::Status stop();

public:
    // for test
    int waitingStreamsSize();

private:
    void reportMetrics(size_t new_stream_size);

    const ModelConfig model_config_;
    const ConcurrencyConfig concurrency_config_;
    const RuntimeConfig runtime_config_;
    std::list<EmbeddingStreamPtr>   waiting_streams_;
    std::atomic<bool>               stop_ = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;

    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm

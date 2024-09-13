#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include "absl/status/status.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingExecutor.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingScheduler.h"

namespace rtp_llm {

class EmbeddingEngine {
public:
    EmbeddingEngine(const EngineInitParams& params, py::object handler);
    ~EmbeddingEngine();

    absl::Status enqueue(EmbeddingStreamPtr stream);
    th::Tensor decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id, std::optional<MultimodalFeature> multimodal_features = std::nullopt);
    // absl::Status update_streams(std::list<EmbeddingStreamPtr>& streams);
    absl::Status stop();

    absl::Status step();
    absl::Status startLoop();

public:
    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

private:
    absl::Status    trySaveStepError() const;
    void            loop();
private:
    const fastertransformer::GptInitParameter params_;
    std::thread                           loop_thread_;
    std::atomic<bool>                     running_{false};
    std::unique_ptr<EmbeddingExecutor>    executor_;
    std::unique_ptr<EmbeddingScheduler>   scheduler_;
    ResourceContext                       resource_context_;
    kmonitor::MetricsReporterPtr          metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm

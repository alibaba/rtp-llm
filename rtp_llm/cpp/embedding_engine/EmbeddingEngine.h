#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include "absl/status/status.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"

namespace rtp_llm {

class EmbeddingEngine {
public:
    EmbeddingEngine(const EngineInitParams& params, py::object handler);
    ~EmbeddingEngine();

    absl::Status                     enqueue(EmbeddingStreamPtr stream);
    std::shared_ptr<EmbeddingOutput> decode(th::Tensor                       token_ids,
                                            th::Tensor                       token_type_ids,
                                            th::Tensor                       input_lengths,
                                            int64_t                          request_id,
                                            std::optional<MultimodalFeature> multimodal_features = std::nullopt,
                                            std::optional<th::Tensor>        input_embeddings    = std::nullopt);
    std::shared_ptr<EmbeddingOutput> decode(std::shared_ptr<EmbeddingInput> input);

    // absl::Status update_streams(std::list<EmbeddingStreamPtr>& streams);
    absl::Status stop();

    absl::Status step();
    absl::Status startLoop();

public:
    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    const rtp_llm::GptInitParameter& GetGptInitParameter();

private:
    absl::Status trySaveStepError() const;
    void         loop();

private:
    const rtp_llm::GptInitParameter     params_;
    std::thread                         loop_thread_;
    std::atomic<bool>                   running_{false};
    std::unique_ptr<EmbeddingExecutor>  executor_;
    std::unique_ptr<EmbeddingScheduler> scheduler_;
    ResourceContext                     resource_context_;
    kmonitor::MetricsReporterPtr        metrics_reporter_ = nullptr;
    std::shared_ptr<CudaProfiler>       profiler_;
    bool                                gen_timeline_ = false;
};

}  // namespace rtp_llm

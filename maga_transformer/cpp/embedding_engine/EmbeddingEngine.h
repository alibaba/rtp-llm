#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingExecutor.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingScheduler.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace rtp_llm {

class EmbeddingEngine {
public:
    EmbeddingEngine(const MagaInitParams&                                                   params,
                    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights,
                    const HandlerBase&                                                      handler);
    ~EmbeddingEngine();

    absl::Status enqueue(EmbeddingStreamPtr stream);    
    // absl::Status update_streams(std::list<EmbeddingStreamPtr>& streams);
    absl::Status stop();

    absl::Status step();
    absl::Status startLoop();

public:
    const ResourceContext& resourceContext() const {
        return resource_context_;
    }
    const MagaInitParams magaInitParams() const {
        return params_;
    }

private:
    absl::Status    trySaveStepError() const;
    void            loop();
private:
    std::thread                           loop_thread_;
    std::atomic<bool>                     running_{false};
    std::unique_ptr<EmbeddingExecutor>    executor_;
    std::unique_ptr<EmbeddingScheduler>   scheduler_;
    MagaInitParams                        params_;
    ResourceContext                       resource_context_;
    ft::NcclParam                         tensor_para_;
    ft::NcclParam                         pipeline_para_;
    kmonitor::MetricsReporterPtr          metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm

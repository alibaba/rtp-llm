#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/batch_stream_processor/BatchStreamProcessor.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/executors/Executor.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "torch/all.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace rtp_llm {

class NormalEngine {
public:
    NormalEngine(const MagaInitParams&                                                   params,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                 const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    ~NormalEngine();

    absl::Status step();
    absl::Status stop();
    absl::Status startLoop();
    absl::Status enqueue(std::shared_ptr<GenerateStream>& stream);

private:
    absl::Status trySaveStepError() const;
    void         loop();

private:
    std::thread                           loop_thread_;
    std::atomic<bool>                     running_;
    std::unique_ptr<Executor>             executor_;
    std::unique_ptr<SchedulerBase>        scheduler_;
    std::unique_ptr<BatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<CacheManager>         cache_manager_;
};

}  // namespace rtp_llm

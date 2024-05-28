#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "torch/all.h"
#include "absl/status/status.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/common/fatal_util.h"

namespace rtp_llm {

class SpeculativeEngine: public EngineBase {
public:
    SpeculativeEngine(const MagaInitParams&                                                   params,
                      const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                      const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    ~SpeculativeEngine();
    absl::Status step();
    absl::Status stop() override;
    absl::Status startLoop();
    absl::Status enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::Status addLoRA(const int64_t                                                   lora_id,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override {
        return absl::UnimplementedError("speculative not support lora yet");
    }
    absl::Status removeLoRA(const int64_t lora_id) override {
        return absl::UnimplementedError("speculative not support lora yet");
    }
    const ResourceContext& resourceContext() const override {
        return resource_context_;
    }

private:
    absl::Status updateDraftProb(const std::list<GenerateStreamPtr>& streams, uint index);
    void loop();

private:
    std::thread                           loop_thread_;
    std::atomic<bool>                     running_;
    std::unique_ptr<Executor>             draft_executor_;
    std::unique_ptr<Executor>             target_executor_;
    std::unique_ptr<SchedulerBase>        scheduler_;
    bool                                  reuse_cache_ = false;
    std::shared_ptr<CacheManager>         draft_cache_manager_;
    std::shared_ptr<CacheManager>         target_cache_manager_;
    ResourceContext                       resource_context_;
};

}  // namespace rtp_llm

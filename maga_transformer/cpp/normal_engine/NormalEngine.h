#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "torch/all.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace rtp_llm {

class NormalEngine : public EngineBase {
public:
    NormalEngine(const MagaInitParams&                                                   params,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                 const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    ~NormalEngine();

    absl::Status enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::Status stop() override;

    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override;

    void removeLoRA(const int64_t lora_id) override;
    absl::Status step();
    absl::Status startLoop();

public:
    const ResourceContext& resourceContext() const override {
        return resource_context_;
    }
    const MagaInitParams magaInitParams() const {
        return params_;
    }

private:
    absl::Status    trySaveStepError() const;
    void            loop();
    void            initCacheManager();
    void            initSystemPrompt();

private:
    std::thread                           loop_thread_;
    std::atomic<bool>                     running_{false};
    std::unique_ptr<Executor>             executor_;
    std::unique_ptr<SchedulerBase>        scheduler_;
    std::shared_ptr<CacheManager>         cache_manager_;
    MagaInitParams                        params_;
    ResourceContext                       resource_context_;
    ft::NcclParam                         tensor_para_;
    ft::NcclParam                         pipeline_para_;
};

}  // namespace rtp_llm

#pragma once

#include <memory>

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/lora/LoraManager.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreBatchStreamProcessor.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSamplerOutput.h"

namespace rtp_llm {

class ScoreExecutor {
public:
    explicit ScoreExecutor(const EngineInitParams&                   params,
                           rtp_llm::DeviceBase*                           device,
                           const std::shared_ptr<CacheManager>&      cache_manager,
                           const std::shared_ptr<lora::LoraManager>& lora_manager,
                           bool                                      warm_up = false):
        device_(device),
        score_normal_executor_(params, cache_manager, device_, lora_manager, warm_up),
        normal_executor_(params, cache_manager, device_, lora_manager, warm_up) {
        const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
        score_normal_executor_.setBatchProcessor(
            std::move(std::make_unique<ScoreBatchStreamProcessor>(
            params.gpt_init_parameter, cache_config, warm_up)));
    }

    absl::Status normalProcess(const std::list<GenerateStreamPtr>& streams) {
        return normal_executor_.process(streams);
    }

    absl::StatusOr<ScoreOutput> score(const std::list<GenerateStreamPtr>& streams,
                                      const ProposeOutput&                proposer_output);

    absl::StatusOr<ScoreOutput> mtpScore(const std::list<GenerateStreamPtr>& streams,
                                         const ProposeOutput&                proposer_output,
                                         const std::list<GenerateStreamPtr>& prefill_streams);

private:
    rtp_llm::DeviceBase* device_;
    NormalExecutor  score_normal_executor_;
    NormalExecutor  normal_executor_;
};

}  // namespace rtp_llm
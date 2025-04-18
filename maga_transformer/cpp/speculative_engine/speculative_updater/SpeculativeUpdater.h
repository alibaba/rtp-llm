#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/lora/LoraManager.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSamplerOutput.h"
#include "maga_transformer/cpp/speculative_engine/speculative_updater/SpeculativeUpdaterConfig.h"

namespace rtp_llm {

class SpeculativeUpdater {
public:
    explicit SpeculativeUpdater(ResourceContext& resource_context, const SpeculativeUpdaterConfig& config):
        resource_context_(resource_context), config_(config) {}

    absl::Status update(std::list<GenerateStreamPtr>& streams, const SpeculativeSamplerOutput& sampler_output) {
        size_t stream_index = 0;
        for (GenerateStreamPtr& stream : streams) {
            if (stream->stoppedWithoutLock() || stream->finishedWithoutLock()) {
                continue;
            }
            RETURN_IF_STATUS_ERROR(propose_compact_kv_cache(stream, sampler_output.outputs[stream_index]));
            RETURN_IF_STATUS_ERROR(score_compact_kv_cache(stream, sampler_output.outputs[stream_index]));
            RETURN_IF_STATUS_ERROR(save_score_last_state(stream, sampler_output.outputs[stream_index]));
            RETURN_IF_STATUS_ERROR(dispatch(stream, sampler_output.outputs[stream_index]));
            stream_index++;
        }
        FT_LOG_DEBUG("speculative update done");
        return absl::OkStatus();
    };

private:
    absl::Status propose_compact_kv_cache(const GenerateStreamPtr&              stream,
                                          const SpeculativeSamplerStreamOutput& stream_output) const;
    absl::Status score_compact_kv_cache(const GenerateStreamPtr&              stream,
                                        const SpeculativeSamplerStreamOutput& stream_output) const;
    absl::Status save_score_last_state(const GenerateStreamPtr&              stream,
                                       const SpeculativeSamplerStreamOutput& stream_output) const;
    absl::Status dispatch(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const;

private:
    ResourceContext          resource_context_;
    SpeculativeUpdaterConfig config_;
};

}  // namespace rtp_llm
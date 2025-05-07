#pragma once

#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"

namespace rtp_llm {

class SpeculativeOnlineAdaptor {
public:
    void dynamicUpdateProposerConfig(std::unique_ptr<ProposeExecutor>&     proposer,
                                     const std::unique_ptr<SchedulerBase>& scheduler) const {
        RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
        // TODO(xyz): update proposer config based on the status of metrics and scheduler
    }
};

}  // namespace rtp_llm
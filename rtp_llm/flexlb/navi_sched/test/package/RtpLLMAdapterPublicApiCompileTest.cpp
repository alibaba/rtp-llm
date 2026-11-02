#include "aios/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"

namespace navi_sched {

void verifyRtpLLMAdapterPublicApiCompile()
{
    auto createAgent = &createRtpLLMAgentFromEnvironment;
    auto finishBatch = &finishRtpLLMBatch;
    auto retainBatch = &retainRtpLLMReadyBatch;
    bool (*startBatches)(const std::vector<NaviSchedLatencyQuery> &) =
        &startRetainedRtpLLMBatches;
    bool (*startBatchesAt)(const std::vector<NaviSchedLatencyQuery> &, int64_t) =
        &startRetainedRtpLLMBatches;
    (void)createAgent;
    (void)finishBatch;
    (void)retainBatch;
    (void)startBatches;
    (void)startBatchesAt;
}

}

#include <navi_sched/NaviSchedAgent.h>

#include "aios/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"

int main()
{
    navi_sched::NaviSchedConfig config;
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    navi_sched::NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.service = config;
    navi_sched::NaviSchedAgent agent(agentConfig);
    agent.stop();
    navi_sched::abortRetainedRtpLLMBatch(1);
    navi_sched::recordRtpLLMObservedPrefillBatch(nullptr, {}, 0, 0);
    const auto retained = navi_sched::retainRtpLLMForwardBatchWithResult(nullptr, {1});
    return retained.status == navi_sched::RtpLLMReadyBatchRetainStatus::INVALID_BATCH ? 0 : 1;
}

#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentControlToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentRequestEventToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedAgentEvent.h"
#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <memory>
#include <string>

namespace navi_sched {

static const int32_t NAVI_SCHED_AGENT_UPDATE_VERSION = 10;
static const int32_t NAVI_SCHED_AGENT_UPDATE_MIN_SUPPORTED_VERSION = 10;
static const std::string AGENT_UPDATE_TO_MASTER_TYPE_ID = "NaviSchedAgentUpdateType";

class AgentUpdateToMasterData : public navi::Data {
public:
    AgentUpdateToMasterData();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid() const;

public:
    int32_t version = NAVI_SCHED_AGENT_UPDATE_VERSION;
    NaviSchedAgentEvent::Type event = NaviSchedAgentEvent::UNKNOWN;
    int64_t eventTimeUs = 0;
    // Receiver-local monotonic timestamps; never serialized across machines.
    int64_t deserializeBeginMonoUs = 0;
    int64_t deserializeEndMonoUs = 0;
    AgentControlToMasterDataPtr control;
    AgentRequestEventToMasterDataPtr requestEvent;
    AgentCacheToMasterDataPtr cache;
    AgentPendingCacheToMasterDataPtr pendingCache;
    AgentLatencyToMasterDataPtr latency;
    NaviSchedWorkloadSnapshotPtr workload;
    std::string latencyModelFingerprint;
    bool latencyModelTrusted = false;
    bool latencyModelWarmStarted = false;
    uint64_t schedulerConfigRequestGeneration = 0;
    uint64_t schedulerConfigGeneration = 0;
    std::string schedulerConfigHash;
    std::string schedulerConfigJson;
    std::string schedulerConfigError;
};

NAVI_TYPEDEF_PTR(AgentUpdateToMasterData);
using AgentUpdateToMasterDataConstPtr = std::shared_ptr<const AgentUpdateToMasterData>;

} // namespace navi_sched

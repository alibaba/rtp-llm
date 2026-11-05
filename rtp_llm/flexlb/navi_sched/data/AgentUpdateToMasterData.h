#pragma once

#include "aios/navi_sched/data/AgentCacheToMasterData.h"
#include "aios/navi_sched/data/AgentControlToMasterData.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "aios/navi_sched/data/AgentRequestEventToMasterData.h"
#include "aios/navi_sched/data/NaviSchedAgentEvent.h"
#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

static const int32_t NAVI_SCHED_AGENT_UPDATE_VERSION = 24;
static const int32_t NAVI_SCHED_AGENT_UPDATE_MIN_SUPPORTED_VERSION = 24;
// The collector graph selects the ordered envelope format.
static const int32_t NAVI_SCHED_AGENT_UPDATE_BATCH_VERSION = 25;
static const int32_t NAVI_SCHED_AGENT_UPDATE_LEGACY_BATCH_VERSION = 13;
static const size_t NAVI_SCHED_AGENT_UPDATE_MAX_BATCH_SIZE = 8;
static const std::string AGENT_UPDATE_TO_MASTER_TYPE_ID = "NaviSchedAgentUpdateType";

class AgentUpdateToMasterData : public navi::Data {
public:
    AgentUpdateToMasterData();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid() const;
    bool validEnvelope() const;
    static std::shared_ptr<AgentUpdateToMasterData> makeBatch(
        const std::vector<std::shared_ptr<AgentUpdateToMasterData>> &updates);

private:
    void serializeCompactPayload(autil::DataBuffer &dataBuffer) const;
    void deserializeCompactPayload(autil::DataBuffer &dataBuffer);
    void deserializePayload(autil::DataBuffer &dataBuffer);

public:
    int32_t version = NAVI_SCHED_AGENT_UPDATE_VERSION;
    NaviSchedAgentEvent::Type event = NaviSchedAgentEvent::UNKNOWN;
    int64_t eventTimeUs = 0;
    // Receiver-local monotonic timestamps; never serialized across machines.
    int64_t deserializeBeginMonoUs = 0;
    int64_t deserializeEndMonoUs = 0;
    bool peerCacheReadAvailable = false;
    uint64_t peerCacheReadRevision = 0;
    int64_t peerCacheReadAgentStartTimeUs = 0;
    AgentControlToMasterDataPtr control;
    AgentRequestEventToMasterDataPtr requestEvent;
    AgentCacheToMasterDataPtr cache;
    AgentPendingCacheToMasterDataPtr pendingCache;
    AgentLatencyToMasterDataPtr latency;
    NaviSchedWorkloadSnapshotPtr workload;
    double executionUtilization = -1.0;
    double executionBusyUtilization = -1.0;
    std::string latencyModelFingerprint;
    bool latencyModelTrusted = false;
    bool latencyModelWarmStarted = false;
    uint64_t schedulerConfigRequestGeneration = 0;
    uint64_t schedulerConfigGeneration = 0;
    std::string schedulerConfigHash;
    std::string schedulerConfigJson;
    std::string schedulerConfigError;
    // Only events produced by one operation; never accumulate across operations.
    std::vector<std::shared_ptr<AgentUpdateToMasterData>> updates;
    bool mambaReplayStopShift = false;
    int64_t mambaRetentionInterval = -1;
    bool mambaAlign = false;
    int64_t fullAttentionQueryDropTokens = 0;
};

NAVI_TYPEDEF_PTR(AgentUpdateToMasterData);
using AgentUpdateToMasterDataConstPtr = std::shared_ptr<const AgentUpdateToMasterData>;

} // namespace navi_sched

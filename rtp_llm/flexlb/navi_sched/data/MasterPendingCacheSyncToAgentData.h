#pragma once

#include <cstdint>
#include <string>

#include "autil/DataBuffer.h"
#include "navi/engine/Data.h"

namespace navi_sched {

static const std::string MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID =
    "NaviSchedMasterPendingCacheSyncType";

class MasterPendingCacheSyncToAgentData : public navi::Data {
public:
    MasterPendingCacheSyncToAgentData();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    std::string collectorId;
    std::string reason;
    bool fullSnapshot = false;
    bool snapshotReadyBefore = false;
    bool candidateStateComputed = false;
    uint64_t masterRevisionBefore = 0;
    uint64_t expectedRevision = 0;
    uint64_t rejectedRevision = 0;
    uint64_t masterChecksumBefore = 0;
    uint64_t candidateChecksum = 0;
    uint64_t receivedChecksum = 0;
    int64_t masterBatchCountBefore = 0;
    int64_t candidateBatchCount = 0;
    int64_t receivedBatchCount = 0;
    int64_t masterBlockCountBefore = 0;
    int64_t candidateBlockCount = 0;
    int64_t receivedBlockCount = 0;
    int64_t masterTailBlockCountBefore = 0;
    int64_t candidateTailBlockCount = 0;
    int64_t receivedTailBlockCount = 0;
};

NAVI_TYPEDEF_PTR(MasterPendingCacheSyncToAgentData);

}

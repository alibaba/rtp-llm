#pragma once

#include "aios/navi_sched/collector/PPCacheCollectorState.h"
#include "autil/DataBuffer.h"
#include "navi/engine/Data.h"

namespace navi_sched {

extern const std::string PP_CACHE_SHARD_UPDATE_TYPE_ID;

class PPCacheShardUpdateData : public navi::Data {
public:
    PPCacheShardUpdateData();
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
    bool valid() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static constexpr uint32_t VERSION = 2;
    uint32_t version = VERSION;
    PPCacheShardUpdate update;
};

using PPCacheShardUpdateDataPtr = std::shared_ptr<PPCacheShardUpdateData>;

}

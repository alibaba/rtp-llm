#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "navi/engine/Data.h"
#include "autil/DataBuffer.h"

namespace navi_sched {

static const std::string V6D_CACHE_TOUCH_TYPE_ID = "NaviSchedV6dCacheTouch";
static const std::string V6D_CACHE_TOUCH_DATA_NAME = "navi_sched.v6d_cache_touch";

class V6dCacheTouchData : public navi::Data {
public:
    V6dCacheTouchData() : navi::Data(V6D_CACHE_TOUCH_TYPE_ID) {}
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    int32_t version = NAVI_SCHED_DATA_VERSION;
    uint64_t requestId = 0;
    NaviSchedBlockHashes blockHashes;
};

}

#pragma once

#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/Spec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

struct BizNodeInfo {
    VersionTy version = INVALID_VERSION_ID;
    Spec spec;
    MetaMapPtr metaMap;
};

using BizNodeInfoVec = std::vector<BizNodeInfo>;

} // namespace multi_call

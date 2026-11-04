#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class MiscUtil
{
public:
    MiscUtil();
    ~MiscUtil();

private:
    MiscUtil(const MiscUtil &);
    MiscUtil &operator=(const MiscUtil &);

public:
    static std::string createBizName(const std::string &cluster, const std::string &biz);
    static float scaleLinear(uint32_t begin, uint32_t end, float pos);

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(MiscUtil);

} // namespace multi_call


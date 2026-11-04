#pragma once

#include "navi/common.h"
#include "navi/proto/GraphVis.pb.h"
#include <autil/DataBuffer.h>

namespace navi {

class ScheduleInfoDef;

class ScheduleInfo {
public:
    ScheduleInfo()
        : enqueueTime(0)
        , dequeueTime(0)
        , runningThread(0)
        , threadCounter(0)
        , threadWaitCounter(0)
        , queueSize(0)
        , schedTid(0)
        , signalTid(0)
        , processTid(0)
        , inlineType(ICT_NO_INLINE)
    {
    }
public:
    static std::shared_ptr<ScheduleInfo> makeInlineSchedInfo();
public:
    void fillProto(ScheduleInfoDef *info) const;
public:
    int64_t enqueueTime;
    int64_t dequeueTime;
    int64_t runningThread;
    int64_t threadCounter;
    int64_t threadWaitCounter;
    size_t queueSize;
    int32_t schedTid;
    int32_t signalTid;
    int32_t processTid;
    InlineComputeType inlineType;
};

NAVI_TYPEDEF_PTR(ScheduleInfo);

}

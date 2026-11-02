#pragma once

#include "aios/navi_sched/scheduler/SchedulerBaseKernel.h"

namespace navi_sched {

class ScheduleReplayRunner;

class MaxCacheHitSchedulerKernel final : public SchedulerBaseKernel {
public:
    static const std::string KERNEL_ID;

public:
    void def(navi::KernelDefBuilder &builder) const override;

protected:
    bool schedule(NaviSchedScheduleContext &context) const override;

private:
    friend class ScheduleReplayRunner;
    static uint64_t getNodeRank(uint64_t requestId, uint64_t agentIdHash);

private:
    KERNEL_DEPEND_DECLARE_BASE(SchedulerBaseKernel);
};

}

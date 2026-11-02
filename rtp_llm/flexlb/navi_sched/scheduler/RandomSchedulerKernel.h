#pragma once

#include "aios/navi_sched/scheduler/SchedulerBaseKernel.h"

namespace navi_sched {

class ScheduleReplayRunner;

class RandomSchedulerKernel final : public SchedulerBaseKernel {
public:
    static const std::string KERNEL_ID;

public:
    void def(navi::KernelDefBuilder &builder) const override;

protected:
    bool schedule(NaviSchedScheduleContext &context) const override;

private:
    friend class ScheduleReplayRunner;
    uint64_t nextRandom() const;

private:
    KERNEL_DEPEND_DECLARE_BASE(SchedulerBaseKernel);
    mutable uint64_t _state = 0x9e3779b97f4a7c15ULL;
};

}

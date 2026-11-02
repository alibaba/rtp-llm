#pragma once

#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class CostSchedulerPrepareKernel;

class SchedulerBaseKernel : public navi::Kernel {
public:
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) final;

protected:
    void def(navi::KernelDefBuilder &builder, const std::string &name) const;
    virtual bool schedule(NaviSchedScheduleContext &context) const;

private:
    friend class CostSchedulerPrepareKernel;
    static bool validateScheduleInput(
        const NaviSchedScheduleContextPtr &context);
};

}

#pragma once

#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerConfig.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/SchedulerBaseKernel.h"
#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"

namespace navi_sched {

class ScheduleReplayRunner;

class CostSchedulerKernel final : public SchedulerBaseKernel {
public:
    static const std::string KERNEL_ID;

public:
    CostSchedulerKernel();
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;

protected:
    bool schedule(NaviSchedScheduleContext &context) const override;

private:
    friend class ScheduleReplayRunner;
    KERNEL_DEPEND_DECLARE_BASE(SchedulerBaseKernel);
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);
    CostSchedulerConfig _config;
    mutable multi_call::RandomGenerator _randomGenerator;
};

} // namespace navi_sched

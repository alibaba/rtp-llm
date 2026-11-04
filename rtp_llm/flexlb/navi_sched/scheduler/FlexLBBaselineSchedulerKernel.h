#pragma once

#include "aios/navi_sched/scheduler/SchedulerBaseKernel.h"

namespace navi_sched {

class ScheduleReplayRunner;
class FlexLBBaselineSchedulerKernelTestAccess;

class FlexLBBaselineSchedulerKernel final : public SchedulerBaseKernel {
public:
    static const std::string KERNEL_ID;

public:
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;

protected:
    bool schedule(NaviSchedScheduleContext &context) const override;

private:
    friend class ScheduleReplayRunner;
    friend class FlexLBBaselineSchedulerKernelTestAccess;
    std::string _predictorType = "flexlb_formula";
    double _computeTokenCost = 1.0;
    double _cacheHitTokenCost = 0.3;
    bool _useReportedLoad = false;
    bool _scoreTieRandomEnabled = true;
    double _scoreTieThresholdPct = 0.1;
    double _scoreTieThresholdMs = 20.0;
    double _hotspotMultiplier = 3.0;
    double _imbalanceMultiplier = 3.0;

    static uint64_t getNodeRank(uint64_t requestId, uint64_t agentIdHash);

private:
    KERNEL_DEPEND_DECLARE_BASE(SchedulerBaseKernel);
};

}

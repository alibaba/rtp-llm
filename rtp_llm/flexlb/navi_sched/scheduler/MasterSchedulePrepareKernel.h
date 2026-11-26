#pragma once

#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "aios/network/gig/multi_call/util/RandomGenerator.h"
#include "navi/engine/Kernel.h"
#include <functional>
#include <memory>
#include <string>

namespace navi_sched {

class MasterScheduleContextData;

// Admission and cache-derived cost inputs are prepared after the remote query.
class MasterSchedulePrepareKernel : public navi::Kernel {
    friend class ::NaviSchedResourceTest;
    friend class ScheduleReplayRunner;
    friend class MasterScheduleResultKernel;
public:
    static const std::string KERNEL_ID;
    MasterSchedulePrepareKernel();
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    bool prepareContext(const std::shared_ptr<MasterScheduleContextData> &data);
    static size_t scheduleRoundRequestCount(const NaviSchedScheduleContext &context);
    bool prepareAdmission(const std::shared_ptr<MasterScheduleContextData> &context,
                          const std::function<uint64_t()> &randomDraw = {});
    bool applyQueueOverloadLimit(
        const std::shared_ptr<MasterScheduleContextData> &context,
        const std::function<uint64_t()> &randomDraw = {},
        bool forceExecutionUtilization = false);
    void updateExecutionUtilization(const NaviSchedScheduleContext &context, bool enabled);
    static double calculateQueueOverloadDropProbability(double averagePredictedLatencyMs,
                                                        int64_t scheduleCycleMs,
                                                        int64_t startCycleMultiplier,
                                                        int64_t fullCycleMultiplier);
    bool shouldDropRequest(double dropProbability, uint64_t *sample = nullptr,
                           const std::function<uint64_t()> &randomDraw = {});
    void useConfig(const SchedulerConfigR::Snapshot &snapshot);
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON(MasterScheduleR, _masterScheduleR);

    ClusterExecutionUtilization _clusterExecutionUtilization;
    ExecutionUtilizationController _executionUtilizationController;
    bool _executionUtilizationLimitEnabled = false;
    bool _useMinimumBacklogForAdmission = false;
    bool _useSelectedQueueForAdmission = false;
    ExecutionUtilizationController::Config _executionUtilizationConfig;
    double _averageExecutionUtilization = 0.0;
    size_t _executionUtilizationReportingNodes = 0;
    int64_t _maxWaitScheduleTimeMs = 10;
    int64_t _queueOverloadStartCycleMultiplier = 10;
    int64_t _queueOverloadFullCycleMultiplier = 30;
    std::string _schedulerKernelId;
    multi_call::RandomGenerator _randomGenerator;
};

} // namespace navi_sched

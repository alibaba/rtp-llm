#pragma once

#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "aios/network/gig/multi_call/util/RandomGenerator.h"
#include "navi/engine/Kernel.h"
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

class NaviSchedResourceTest;

namespace navi_sched {

class MasterScheduleContextData;

class MasterScheduleContextKernel : public navi::Kernel {
    friend class ::NaviSchedResourceTest;
    friend class ScheduleReplayRunner;

public:
    static const std::string KERNEL_ID;

public:
    MasterScheduleContextKernel();
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    bool drainRequests();
    void retireCancelledRequests();
    bool retireRequest(const MasterScheduleRequestDataPtr &request,
                       int64_t currentTimeUs);
    bool shouldBuildContext(int64_t currentTimeUs) const;
    int64_t getNextScheduleDelayMs(int64_t currentTimeUs) const;
    bool stopScheduleTimer(navi::KernelComputeContext &ctx);
    bool updateScheduleTimer(navi::KernelComputeContext &ctx, int64_t currentTimeUs);
    std::shared_ptr<void> beginScheduleRound();
    void buildReadyContexts(int64_t activationTimeUs);
    bool prepareAdmission(const std::shared_ptr<MasterScheduleContextData> &context,
                          const std::function<uint64_t()> &randomDraw = {});
    bool applyQueueOverloadLimit(
        const std::shared_ptr<MasterScheduleContextData> &context,
        const std::function<uint64_t()> &randomDraw = {},
        bool forceExecutionUtilization = false);
    void updateExecutionUtilization(const NaviSchedScheduleContext &context,
                                    bool enabled);
    static double calculateQueueOverloadDropProbability(double averagePredictedLatencyMs,
                                                        int64_t scheduleCycleMs,
                                                        int64_t startCycleMultiplier,
                                                        int64_t fullCycleMultiplier);
    bool shouldDropRequest(double dropProbability, uint64_t *sample = nullptr,
                           const std::function<uint64_t()> &randomDraw = {});
    bool activate(navi::KernelComputeContext &ctx);

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterScheduleR, _masterScheduleR);
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);

private:
    ClusterExecutionUtilization _clusterExecutionUtilization;
    ExecutionUtilizationController _executionUtilizationController;
    bool _executionUtilizationLimitEnabled = false;
    ExecutionUtilizationController::Config _executionUtilizationConfig;
    double _averageExecutionUtilization = 0.0;
    size_t _executionUtilizationReportingNodes = 0;
    navi::AsyncPipePtr _requestPipe;
    navi::AsyncPipePtr _roundCompletionPipe;
    std::shared_ptr<std::atomic<bool>> _roundInFlight =
        std::make_shared<std::atomic<bool>>(false);
    std::deque<MasterScheduleRequestDataPtr> _pendingRequests;
    std::deque<std::shared_ptr<MasterScheduleContextData>> _readyContexts;
    size_t _maxWaitScheduleCount = 30;
    int64_t _maxWaitScheduleTimeMs = 10;
    int64_t _queueOverloadStartCycleMultiplier = 10;
    int64_t _queueOverloadFullCycleMultiplier = 30;
    double _timeCompression = 1.0;
    bool _peerCacheEnabled = false;
    uint64_t _peerCacheReadyConfigGeneration = 0;
    std::shared_ptr<const std::unordered_set<std::string>> _peerCacheReadyAgents;
    int64_t _requestPipeGeneration = 0;
    multi_call::RandomGenerator _randomGenerator;
    uint64_t _nextScheduleId = 1;
    std::string _schedulerKernelId;
    bool _requestEof = false;
};

class MasterScheduleResultKernel : public navi::Kernel {
    friend class ::NaviSchedResourceTest;

public:
    static const std::string KERNEL_ID;

public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
    static void recordReplayAdmission(NaviSchedScheduleContext &context);

protected:
    struct FrozenExecutionInputAudit {
        uint64_t requestId = 0;
        TPUPrefillRequestInput input;
        int64_t fullPrefixBlocksBeforeQueryDrop = -1;
    };

    struct AdmissionPredictionAudit {
        std::vector<FrozenExecutionInputAudit> frozenExecutionInputs;
        std::optional<TPUPrefillCacheObservation> cacheObservation = std::nullopt;
        double withWorkloadBaseMs = 0.0;
        double withWorkloadFinalMs = 0.0;
        double floorBaseMs = 0.0;
        double floorFinalMs = 0.0;
        double baseExecutionMs = 0.0;
        double finalExecutionMs = 0.0;
        double correctionMs = 0.0;
        bool valid = false;
    };

    static AdmissionPredictionAudit getAdmissionPredictionAudit(
        const NaviSchedScheduleContext &context, size_t requestIndex,
        bool includeFrozenExecutionInputs = false);
    void logAdmissionPredictions(const NaviSchedScheduleContext &context) const;
    bool finishContext(NaviSchedScheduleContext &context);
    bool prepareContext(NaviSchedScheduleContext &context, int64_t resultBeginUs);
    void reportResultTiming(NaviSchedScheduleContext &context,
                            int64_t resultBeginUs) const;
    void reportPredictionMetrics(const NaviSchedScheduleContext &context) const;
    void reportResultMetrics(const NaviSchedScheduleContext &context) const;

protected:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON(MasterLatencyModelR, _masterLatencyModelR);
    KERNEL_DEPEND_ON_FALSE(kmonitor::MetricsReporterR, _metricsReporterR);
    KERNEL_DEPEND_ON(MasterScheduleR, _masterScheduleR);
};

}

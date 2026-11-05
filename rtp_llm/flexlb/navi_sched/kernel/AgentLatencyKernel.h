#pragma once

#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Kernel.h"
#include <array>
#include <atomic>
#include <optional>
#include <unordered_map>
#include <unordered_set>

class NaviSchedResourceTest;

namespace navi_sched {

class AgentLatencyKernelTest;

class AgentLatencyKernel : public navi::Kernel {
public:
    static void buildGraph(navi::GraphBuilder &builder, uint64_t inputPipeGeneration);
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    friend class AgentLatencyKernelTest;
    friend class ::NaviSchedResourceTest;

    int64_t predictBatchLatencyUs(const NaviSchedBatchPtr &batch) const;
    struct UnifiedBatchState {
        NaviSchedLatencyFeatures features;
        NaviSchedLatencyFeatures precedingFeatures;
        int32_t remainingPrefillRequestCount = 0;
        int64_t readyTimeUs = 0;
        NaviSchedLocalBatchInfo localBatch;
    };

    struct QueueFeatures {
        double rawLatencyMs = 0.0;
        double precedingBatchCount = 0.0;
        double predictedLatencyMs = 0.0;
        NaviSchedQueueModel::Parameters weights{};
        double cacheHitMemberFraction = 0.0;
        double lookupMemberFraction = 0.0;
        uint64_t fitStep = 0;
        int64_t readyTimeUs = 0;
        int64_t queueEndCanRunTimeUs = 0;
        int32_t requestCount = 0;
        double masterPredictedQueueLatencyMs = 0.0;
        double masterPredictedExecutionLatencyMs = 0.0;
        bool contributesToUnifiedWorkload = false;
        UnifiedBatchState unifiedBatch;
    };

    struct RunningBatch {
        int64_t startTimeUs = 0;
        int64_t predictedLatencyUs = 0;
    };

    struct QueueLoadPrediction {
        double totalLatencyMs = 0.0;
        double decayHeadRemainingMs = 0.0;
        NaviSchedQueueModelState model;
    };

    struct FitInflightAudit {
        RequestLatencyTracker::Snapshot before;
        RequestLatencyTracker::Snapshot after;
        int64_t beforeTimeUs = 0;
        int64_t afterTimeUs = 0;
        bool beforeAvailable = false;
        bool afterAvailable = false;
    };

    double getRawQueueLatencyMs(int64_t currentTimeUs) const;
    QueueFeatures getQueueFeatures(int64_t currentTimeUs) const;
    double predictQueueLatencyMs(const QueueFeatures &features) const;
    double getQueueLatencyMs(int64_t currentTimeUs) const;
    QueueLoadPrediction getQueueLoadPrediction(int64_t currentTimeUs) const;
    static bool shouldFitQueueLatency(bool unifiedLatencyPrediction,
                                      AgentLatencyInputType inputType,
                                      bool hasQueueFeatures, bool latencySampleValid);
    static bool shouldRecordLatencySample(bool unifiedLatencyPrediction,
                                          AgentLatencyInputType inputType,
                                          bool unifiedLatencySample,
                                          bool latencySampleValid);
    static bool shouldRecordLatencyValidation(AgentLatencyInputType inputType,
                                              bool latencySampleValid,
                                              bool latencyValidationValid);
    void fitQueueLatency(const QueueFeatures &features, double actualQueueLatencyMs);
    void updateExecutionState(const AgentLatencyInput &input);
    void startRunningBatch(const AgentLatencyInput &input);
    double getRunningElapsedTimeMs(int64_t currentTimeUs) const;
    void reportExecutionLoad(const NaviSchedLocalBatchInfo &localBatch = {}) const;
    bool derivesUnifiedWorkloadFromBatches() const;
    bool addUnifiedBatch(const AgentLatencyInput &input, UnifiedBatchState &state,
                         NaviSchedLocalBatchInfo &admittedBatch);
    bool replaceUnifiedBatch(const AgentLatencyInput &input, UnifiedBatchState &state);
    bool retireUnifiedBatch(AgentLatencyInput &input, const UnifiedBatchState &state);
    void reportUnifiedWorkload(int64_t eventTimeUs,
                               const NaviSchedLocalBatchInfo &admittedBatch = {});
    NaviSchedWorkloadSnapshot makeUnifiedWorkloadSnapshot(
        int64_t eventTimeUs, const NaviSchedLocalBatchInfo &admittedBatch = {});
    NaviSchedLocalBatchInfo getLocalBatchInfo(int64_t batchId, bool erase);
    void recordLatencyValidation(const AgentLatencyInput &input,
                                 double agentPredictedExecutionLatencyMs,
                                 const QueueFeatures *queueFeatures,
                                 double agentPredictedQueueLatencyMs);
    void logQueueDebugSample(const AgentLatencyInput &input, int64_t scheduleBatchId,
                             const QueueFeatures &features,
                             double agentPredictedQueueLatencyMs,
                             double agentPredictedExecutionLatencyMs) const;
    static bool shouldLogQueueDebugSample(int64_t scheduleBatchId);
    static const char *getLatencyInputTypeName(AgentLatencyInputType type);
    void logLatencyTrainingSample(
        const AgentLatencyInput &input, uint64_t sampleSequence,
        const NaviSchedLatencySample &sample, const QueueFeatures *queueFeatures,
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
        const std::optional<NonLinearPrefillModelR::NonlinearOperatingPoint>
            &operatingPoint) const;
    void logLatencyTrainingUpdate(
        uint64_t sampleSequence, int64_t eventTimeUs,
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &before,
        const NonLinearPrefillModelR::OptimizerState &optimizerBefore,
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &after,
        const NonLinearPrefillModelR::OptimizerState &optimizerAfter,
        const FitInflightAudit &fitInflight) const;
    static void updateMean(double value, uint64_t count, double &mean);

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentLatencyInputR, _agentLatencyInputR);
    KERNEL_DEPEND_ON(AgentBatchR, _agentBatchR);
    KERNEL_DEPEND_ON(LatencyModelR, _latencyModelR);
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
    KERNEL_DEPEND_ON(AgentSchedulerConfigR, _agentSchedulerConfigR);
    KERNEL_DEPEND_ON_FALSE(kmonitor::MetricsReporterR, _metricsReporterR);
    navi::AsyncPipePtr _samplePipe;
    navi::AsyncPipePtr _executionUtilizationPipe;
    int64_t _executionUtilizationActivationCount = 0;
    std::atomic<int64_t> _queueLatencyUs{0};
    std::unordered_map<int64_t, int64_t> _batchLatencyUs;
    // Missing estimates remain explicit while lifecycle events continue.
    std::unordered_set<int64_t> _unknownQueueBatches;
    mutable uint64_t _queueProjectionFallbackCount = 0;
    mutable uint64_t _queueProjectionUnknownCount = 0;
    std::unordered_map<int64_t, QueueFeatures> _batchQueueFeatures;
    std::unordered_map<int64_t, NaviSchedLocalBatchInfo> _batchLocalInfos;
    std::unordered_map<int64_t, RunningBatch> _runningBatches;
    NaviSchedLatencyFeatures _unifiedWorkloadFeatures;
    int32_t _unifiedRemainingPrefillRequestCount = 0;
    NaviSchedQueueModel::Parameters _queueWeights{};
    NaviSchedQueueModel::Parameters _queueAdamMoment1{};
    NaviSchedQueueModel::Parameters _queueAdamMoment2{};
    uint64_t _queueFitStep = 1;
    double _queueLearningRate = 0.002;
    double _timeCompression = 1.0;
    int32_t _latestBatchWaitingRequests = 0;
    int32_t _latestQueuedBatchRequests = 0;
    int32_t _latestRunningRequests = 0;
    uint64_t _latencyTrainingSampleSequence = 0;
    NaviSchedLatencyValidationStats _latencyValidationStats;
    int64_t _inputPipeGeneration = 0;
};

}

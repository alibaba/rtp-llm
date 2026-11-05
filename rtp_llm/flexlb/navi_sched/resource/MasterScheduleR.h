#pragma once

#include "aios/navi_sched/data/MasterScheduleRequestData.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>

namespace navi_sched {

struct NaviSchedColdStartRoutingStats;

class MasterScheduleR : public navi::Resource {
public:
    struct BacklogStats {
        size_t pipePending = 0;
        size_t kernelPending = 0;
        uint64_t expired = 0;
        uint64_t cancelled = 0;
        double dequeueWaitMs = 0.0;
        double dequeueMaxWaitMs = 0.0;
    };

    struct ExecutionUtilizationStats {
        double rawAverage = 0.0;
        uint64_t running = 0;
        uint64_t waiting = 0;
        bool globalScaling = false;
        double average = 0.0;
        double integral = 0.0;
        size_t reportingNodes = 0;
        bool enabled = false;
    };
    struct QueueOverloadStats {
        uint64_t evaluatedRoundCount = 0;
        uint64_t evaluatedRequestCount = 0;
        uint64_t rejectedRequestCount = 0;
        double latestAveragePredictedLatencyMs = 0.0;
        double latestDropProbability = 0.0;
        bool active = false;
    };

    struct ColdStartRoutingStats {
        uint64_t evaluatedRoundCount = 0;
        uint64_t trustedRequestCount = 0;
        uint64_t untrustedRequestCount = 0;
        uint64_t rejectedRequestCount = 0;
        double latestTargetInflight = 0.0;
        uint64_t latestTrustedNodeCount = 0;
        uint64_t latestUntrustedNodeCount = 0;
        uint64_t latestUntrustedCapacity = 0;
        uint64_t latestRampingNodeCount = 0;
        uint64_t latestUnsafeWaitingMarginalNodeCount = 0;
        uint64_t unsafeWaitingMarginalNodeObservationCount = 0;
        double latestMinimumRampProgress = 1.0;
        double latestMinimumRampedTargetInflight = 0.0;
    };

public:
    ~MasterScheduleR() override;

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    uint64_t beginRequestPipeGeneration();
    bool setRequestPipe(uint64_t generation, const navi::AsyncPipePtr &requestPipe);
    bool waitRequestPipe(uint64_t generation, int64_t timeoutMs) const;
    void clearRequestPipe(uint64_t generation);
    MasterScheduleRequestDataPtr submit(const ClientRequestToMasterDataConstPtr &request,
                                        const navi::AsyncPipePtr &responsePipe);
    bool stopped() const;
    void stop();
    uint64_t getSubmitCount() const;
    uint64_t getRequestPipeGeneration() const;
    size_t getPendingRequestCount() const;
    size_t getSchedulingRequestCount() const;
    std::shared_ptr<void> trackSchedulingRequests(size_t count);
    BacklogStats getBacklogStats() const;
    void recordRetiredRequest(bool expired);
    void recordDequeueWait(int64_t totalUs, int64_t maxUs, size_t count);
    void recordKernelPendingRequests(uint64_t generation, size_t count);
    void recordQueueOverload(size_t requestCount, size_t rejectedRequestCount,
                             double averagePredictedLatencyMs, double dropProbability);
    QueueOverloadStats getQueueOverloadStats() const;
    void recordExecutionUtilization(double average, double integral,
                                    size_t reportingNodes, bool enabled,
                                    double rawAverage = 0.0, uint64_t running = 0,
                                    uint64_t waiting = 0, bool globalScaling = false);
    ExecutionUtilizationStats getExecutionUtilizationStats() const;
    void recordColdStartRouting(const NaviSchedColdStartRoutingStats &stats);
    ColdStartRoutingStats getColdStartRoutingStats() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    static uint64_t makeRequestIdSeed(const std::string &bindIp, int64_t startupTimeUs);
    uint64_t nextRequestId();
    static void rejectUnavailable(
        const ClientRequestToMasterDataConstPtr &request,
        const navi::AsyncPipePtr &responsePipe,
        const MasterScheduleRequestDataPtr &scheduleRequest = nullptr);

private:
    mutable autil::ReadWriteLock _lock;
    navi::AsyncPipePtr _requestPipe;
    uint64_t _requestPipeOwnerGeneration = 0;
    bool _stopped = false;
    std::atomic<uint64_t> _nextCanonicalRequestId{1};
    std::atomic<uint64_t> _submitCount{0};
    size_t _kernelPendingRequestCount = 0;
    std::shared_ptr<std::atomic<size_t>> _schedulingRequestCount =
        std::make_shared<std::atomic<size_t>>(0);
    std::atomic<uint64_t> _expiredRequestCount{0};
    std::atomic<uint64_t> _cancelledRequestCount{0};
    std::atomic<double> _dequeueWaitMs{0.0};
    std::atomic<double> _dequeueMaxWaitMs{0.0};
    std::atomic<uint64_t> _requestPipeGeneration{0};
    std::atomic<uint64_t> _queueOverloadEvaluatedRoundCount{0};
    std::atomic<uint64_t> _queueOverloadEvaluatedRequestCount{0};
    std::atomic<uint64_t> _queueOverloadRejectedRequestCount{0};
    std::atomic<double> _queueOverloadLatestAveragePredictedLatencyMs{0.0};
    std::atomic<double> _executionRawAverage{0.0};
    std::atomic<uint64_t> _executionRunning{0};
    std::atomic<uint64_t> _executionWaiting{0};
    std::atomic<bool> _executionGlobalScaling{false};
    std::atomic<double> _executionUtilizationAverage{0.0};
    std::atomic<double> _executionUtilizationIntegral{0.0};
    std::atomic<size_t> _executionUtilizationReportingNodes{0};
    std::atomic<bool> _executionUtilizationEnabled{false};
    std::atomic<double> _queueOverloadLatestDropProbability{0.0};
    std::atomic<uint64_t> _coldStartEvaluatedRoundCount{0};
    std::atomic<uint64_t> _coldStartTrustedRequestCount{0};
    std::atomic<uint64_t> _coldStartUntrustedRequestCount{0};
    std::atomic<uint64_t> _coldStartRejectedRequestCount{0};
    std::atomic<double> _coldStartLatestTargetInflight{0.0};
    std::atomic<uint64_t> _coldStartLatestTrustedNodeCount{0};
    std::atomic<uint64_t> _coldStartLatestUntrustedNodeCount{0};
    std::atomic<uint64_t> _coldStartLatestUntrustedCapacity{0};
    std::atomic<uint64_t> _coldStartLatestRampingNodeCount{0};
    std::atomic<uint64_t> _coldStartLatestUnsafeWaitingMarginalNodeCount{0};
    std::atomic<uint64_t> _coldStartUnsafeWaitingMarginalNodeObservationCount{0};
    std::atomic<double> _coldStartLatestMinimumRampProgress{1.0};
    std::atomic<double> _coldStartLatestMinimumRampedTargetInflight{0.0};
};

NAVI_TYPEDEF_PTR(MasterScheduleR);

}

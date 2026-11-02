#pragma once

#include <stdint.h>

namespace autil {
class DataBuffer;
}

namespace navi_sched {

class NonLinearPrefillModelR;
struct NaviSchedWorkloadSnapshot;

class NaviSchedLatencyFeatures {
public:
    void appendRequest(int64_t cacheHitTokenCount,
                       int64_t cacheMissTokenCount)
    {
        ++_batchSize;
        _cacheHitTokenCount += cacheHitTokenCount;
        _cacheMissTokenCount += cacheMissTokenCount;
        _cacheMissTokenSquareSum +=
            static_cast<double>(cacheMissTokenCount) *
            cacheMissTokenCount;
        _cacheHitMissTokenProductSum +=
            static_cast<double>(cacheHitTokenCount) *
            cacheMissTokenCount;
    }
    bool appendAggregate(int64_t batchCount, int64_t batchSize,
                         int64_t cacheHitTokenCount,
                         int64_t cacheMissTokenCount,
                         double cacheMissTokenSquareSum,
                         double cacheHitMissTokenProductSum);
    bool appendCapacitySplitAggregate(
        int64_t batchCount, int64_t runningBatchSize,
        int64_t runningCacheHitTokenCount,
        int64_t runningCacheMissTokenCount,
        double runningCacheMissTokenSquareSum,
        double runningCacheHitMissTokenProductSum,
        int64_t waitingBatchSize, int64_t waitingCacheHitTokenCount,
        int64_t waitingCacheMissTokenCount,
        double waitingCacheMissTokenSquareSum,
        double waitingCacheHitMissTokenProductSum);
    void appendWaitingRequest(int64_t cacheHitTokenCount,
                              int64_t cacheMissTokenCount);
    bool appendBatchCount(int64_t batchCount);
    bool appendPrecedingRequestCount(int64_t requestCount);
    bool appendRunningElapsedTimeMs(double elapsedTimeMs);
    bool appendRunningDecodeProgress(int64_t requestCount,
                                     int64_t generatedTokenCount,
                                     int64_t remainingTokenBudget = 0);
    bool appendPrefillWorkload(const NaviSchedWorkloadSnapshot &workload);
    bool append(const NaviSchedLatencyFeatures &other);
    bool subtract(const NaviSchedLatencyFeatures &other);
    int64_t getBatchCount() const;
    int64_t getPrecedingRequestCount() const;
    double getRunningElapsedTimeMs() const;
    int64_t getRunningDecodeRequestCount() const;
    int64_t getRunningDecodeGeneratedTokenCount() const;
    int64_t getRunningDecodeRemainingTokenBudget() const;
    int64_t getBatchSize() const;
    int64_t getCacheHitTokenCount() const;
    int64_t getCacheMissTokenCount() const;
    double getCacheMissTokenSquareSum() const;
    double getCacheHitMissTokenProductSum() const;
    bool usesCapacitySplit() const;
    int64_t getRunningBatchSize() const;
    int64_t getRunningCacheHitTokenCount() const;
    int64_t getRunningCacheMissTokenCount() const;
    double getRunningCacheMissTokenSquareSum() const;
    double getRunningCacheHitMissTokenProductSum() const;
    int64_t getWaitingBatchSize() const;
    int64_t getWaitingCacheHitTokenCount() const;
    int64_t getWaitingCacheMissTokenCount() const;
    double getWaitingCacheMissTokenSquareSum() const;
    double getWaitingCacheHitMissTokenProductSum() const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

private:
    friend class NonLinearPrefillModelR;
    int64_t _batchCount = 0;
    int64_t _precedingRequestCount = 0;
    double _runningElapsedTimeMs = 0.0;
    int64_t _runningDecodeRequestCount = 0;
    int64_t _runningDecodeGeneratedTokenCount = 0;
    int64_t _runningDecodeRemainingTokenBudget = 0;
    int64_t _batchSize = 0;
    int64_t _cacheHitTokenCount = 0;
    int64_t _cacheMissTokenCount = 0;
    double _cacheMissTokenSquareSum = 0.0;
    double _cacheHitMissTokenProductSum = 0.0;
    bool _capacitySplit = false;
    int64_t _waitingBatchSize = 0;
    int64_t _waitingCacheHitTokenCount = 0;
    int64_t _waitingCacheMissTokenCount = 0;
    double _waitingCacheMissTokenSquareSum = 0.0;
    double _waitingCacheHitMissTokenProductSum = 0.0;
};

struct NaviSchedLatencySample {
    NaviSchedLatencyFeatures features;
    double latencyMs = 0.0;
    int64_t eventTimeUs = 0;
    int64_t localBatchId = 0;
    int64_t latestAllocatedLocalBatchId = 0;
};

}

#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencySample.h"

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "autil/DataBuffer.h"
#include <cmath>
#include <limits>
#include <utility>

namespace navi_sched {

namespace {
bool validAggregate(int64_t batchCount, int64_t batchSize,
                    int64_t cacheHitTokenCount, int64_t cacheMissTokenCount,
                    double cacheMissTokenSquareSum,
                    double cacheHitMissTokenProductSum)
{
    return batchCount >= 0 && batchSize >= 0 && cacheHitTokenCount >= 0 &&
           cacheMissTokenCount >= 0 &&
           std::isfinite(cacheMissTokenSquareSum) &&
           cacheMissTokenSquareSum >= 0.0 &&
           std::isfinite(cacheHitMissTokenProductSum) &&
           cacheHitMissTokenProductSum >= 0.0;
}
} // namespace

bool NaviSchedLatencyFeatures::appendAggregate(
    int64_t batchCount, int64_t batchSize, int64_t cacheHitTokenCount,
    int64_t cacheMissTokenCount, double cacheMissTokenSquareSum,
    double cacheHitMissTokenProductSum)
{
    if (!validAggregate(batchCount, batchSize, cacheHitTokenCount,
                        cacheMissTokenCount, cacheMissTokenSquareSum,
                        cacheHitMissTokenProductSum)) {
        return false;
    }
    _batchCount += batchCount;
    _batchSize += batchSize;
    _cacheHitTokenCount += cacheHitTokenCount;
    _cacheMissTokenCount += cacheMissTokenCount;
    _cacheMissTokenSquareSum += cacheMissTokenSquareSum;
    _cacheHitMissTokenProductSum += cacheHitMissTokenProductSum;
    return true;
}

bool NaviSchedLatencyFeatures::appendCapacitySplitAggregate(
    int64_t batchCount, int64_t runningBatchSize,
    int64_t runningCacheHitTokenCount, int64_t runningCacheMissTokenCount,
    double runningCacheMissTokenSquareSum,
    double runningCacheHitMissTokenProductSum, int64_t waitingBatchSize,
    int64_t waitingCacheHitTokenCount, int64_t waitingCacheMissTokenCount,
    double waitingCacheMissTokenSquareSum,
    double waitingCacheHitMissTokenProductSum)
{
    if (!validAggregate(batchCount, runningBatchSize,
                        runningCacheHitTokenCount,
                        runningCacheMissTokenCount,
                        runningCacheMissTokenSquareSum,
                        runningCacheHitMissTokenProductSum) ||
        !validAggregate(0, waitingBatchSize, waitingCacheHitTokenCount,
                        waitingCacheMissTokenCount,
                        waitingCacheMissTokenSquareSum,
                        waitingCacheHitMissTokenProductSum)) {
        return false;
    }
    _capacitySplit = true;
    _batchCount += batchCount;
    _batchSize += runningBatchSize;
    _cacheHitTokenCount += runningCacheHitTokenCount;
    _cacheMissTokenCount += runningCacheMissTokenCount;
    _cacheMissTokenSquareSum += runningCacheMissTokenSquareSum;
    _cacheHitMissTokenProductSum += runningCacheHitMissTokenProductSum;
    _waitingBatchSize += waitingBatchSize;
    _waitingCacheHitTokenCount += waitingCacheHitTokenCount;
    _waitingCacheMissTokenCount += waitingCacheMissTokenCount;
    _waitingCacheMissTokenSquareSum += waitingCacheMissTokenSquareSum;
    _waitingCacheHitMissTokenProductSum +=
        waitingCacheHitMissTokenProductSum;
    return true;
}

void NaviSchedLatencyFeatures::appendWaitingRequest(
    int64_t cacheHitTokenCount, int64_t cacheMissTokenCount)
{
    _capacitySplit = true;
    ++_waitingBatchSize;
    _waitingCacheHitTokenCount += cacheHitTokenCount;
    _waitingCacheMissTokenCount += cacheMissTokenCount;
    _waitingCacheMissTokenSquareSum +=
        static_cast<double>(cacheMissTokenCount) * cacheMissTokenCount;
    _waitingCacheHitMissTokenProductSum +=
        static_cast<double>(cacheHitTokenCount) * cacheMissTokenCount;
}

bool NaviSchedLatencyFeatures::appendBatchCount(int64_t batchCount)
{
    if (batchCount < 0) {
        return false;
    }
    _batchCount += batchCount;
    return true;
}

bool NaviSchedLatencyFeatures::appendPrecedingRequestCount(
    int64_t requestCount)
{
    if (requestCount < 0 ||
        requestCount >
            std::numeric_limits<int64_t>::max() - _precedingRequestCount) {
        return false;
    }
    _precedingRequestCount += requestCount;
    return true;
}

bool NaviSchedLatencyFeatures::appendRunningElapsedTimeMs(double elapsedTimeMs)
{
    if (!std::isfinite(elapsedTimeMs) || elapsedTimeMs < 0.0 ||
        elapsedTimeMs > std::numeric_limits<double>::max() -
                            _runningElapsedTimeMs) {
        return false;
    }
    _runningElapsedTimeMs += elapsedTimeMs;
    return true;
}

bool NaviSchedLatencyFeatures::appendRunningDecodeProgress(
    int64_t requestCount, int64_t generatedTokenCount,
    int64_t remainingTokenBudget)
{
    if (requestCount < 0 || generatedTokenCount < requestCount ||
        remainingTokenBudget < 0 ||
        (requestCount == 0 && remainingTokenBudget != 0) ||
        requestCount > std::numeric_limits<int64_t>::max() -
                           _runningDecodeRequestCount ||
        generatedTokenCount > std::numeric_limits<int64_t>::max() -
                                  _runningDecodeGeneratedTokenCount ||
        remainingTokenBudget > std::numeric_limits<int64_t>::max() -
                                   _runningDecodeRemainingTokenBudget) {
        return false;
    }
    _runningDecodeRequestCount += requestCount;
    _runningDecodeGeneratedTokenCount += generatedTokenCount;
    _runningDecodeRemainingTokenBudget += remainingTokenBudget;
    return true;
}

bool NaviSchedLatencyFeatures::appendPrefillWorkload(
    const NaviSchedWorkloadSnapshot &workload)
{
    if (workload.prefillCapacitySplit) {
        auto updated = *this;
        if (!updated.appendCapacitySplitAggregate(
            workload.batchCount,
            workload.runningCapacityPrefillRequestCount,
            workload.runningCapacityPrefillContextTokenCount,
            workload.runningCapacityPrefillTokenCount,
            workload.runningCapacityPrefillTokenSquareSum,
            workload.runningCapacityContextRemainingPrefillTokenProductSum,
            workload.waitingCapacityPrefillRequestCount,
            workload.waitingCapacityPrefillContextTokenCount,
            workload.waitingCapacityPrefillTokenCount,
            workload.waitingCapacityPrefillTokenSquareSum,
            workload.waitingCapacityContextRemainingPrefillTokenProductSum) ||
            !updated.appendPrecedingRequestCount(workload.requestCount) ||
            !updated.appendRunningElapsedTimeMs(
                workload.runningElapsedTimeMs) ||
            !updated.appendRunningDecodeProgress(
                workload.runningDecodeRequestCount,
                workload.runningDecodeGeneratedTokenCount,
                workload.runningDecodeRemainingTokenBudget)) {
            return false;
        }
        *this = std::move(updated);
        return true;
    }
    auto updated = *this;
    if (!updated.appendAggregate(
            workload.batchCount, workload.remainingPrefillRequestCount,
            workload.remainingPrefillContextTokenCount,
            workload.remainingPrefillTokenCount,
            workload.remainingPrefillTokenSquareSum,
            workload.contextRemainingPrefillTokenProductSum) ||
        !updated.appendRunningElapsedTimeMs(workload.runningElapsedTimeMs)) {
        return false;
    }
    *this = std::move(updated);
    return true;
}

bool NaviSchedLatencyFeatures::append(const NaviSchedLatencyFeatures &other)
{
    const auto thisHasWorkload = getBatchSize() > 0;
    const auto otherHasWorkload = other.getBatchSize() > 0;
    if (thisHasWorkload && otherHasWorkload &&
        _capacitySplit != other._capacitySplit) {
        return false;
    }
    if (other._precedingRequestCount < 0 ||
        other._precedingRequestCount >
            std::numeric_limits<int64_t>::max() - _precedingRequestCount) {
        return false;
    }
    if (!std::isfinite(other._runningElapsedTimeMs) ||
        other._runningElapsedTimeMs < 0.0 ||
        other._runningElapsedTimeMs >
            std::numeric_limits<double>::max() - _runningElapsedTimeMs) {
        return false;
    }
    if (other._runningDecodeRequestCount < 0 ||
        other._runningDecodeGeneratedTokenCount <
            other._runningDecodeRequestCount ||
        other._runningDecodeRequestCount >
            std::numeric_limits<int64_t>::max() -
                _runningDecodeRequestCount ||
        other._runningDecodeGeneratedTokenCount >
            std::numeric_limits<int64_t>::max() -
                _runningDecodeGeneratedTokenCount ||
        other._runningDecodeRemainingTokenBudget < 0 ||
        (other._runningDecodeRequestCount == 0 &&
         other._runningDecodeRemainingTokenBudget != 0) ||
        other._runningDecodeRemainingTokenBudget >
            std::numeric_limits<int64_t>::max() -
                _runningDecodeRemainingTokenBudget) {
        return false;
    }
    if (other._capacitySplit) {
        _capacitySplit = true;
    }
    _batchCount += other._batchCount;
    _precedingRequestCount += other._precedingRequestCount;
    _runningElapsedTimeMs += other._runningElapsedTimeMs;
    _runningDecodeRequestCount += other._runningDecodeRequestCount;
    _runningDecodeGeneratedTokenCount +=
        other._runningDecodeGeneratedTokenCount;
    _runningDecodeRemainingTokenBudget +=
        other._runningDecodeRemainingTokenBudget;
    _batchSize += other._batchSize;
    _cacheHitTokenCount += other._cacheHitTokenCount;
    _cacheMissTokenCount += other._cacheMissTokenCount;
    _cacheMissTokenSquareSum += other._cacheMissTokenSquareSum;
    _cacheHitMissTokenProductSum += other._cacheHitMissTokenProductSum;
    _waitingBatchSize += other._waitingBatchSize;
    _waitingCacheHitTokenCount += other._waitingCacheHitTokenCount;
    _waitingCacheMissTokenCount += other._waitingCacheMissTokenCount;
    _waitingCacheMissTokenSquareSum += other._waitingCacheMissTokenSquareSum;
    _waitingCacheHitMissTokenProductSum +=
        other._waitingCacheHitMissTokenProductSum;
    return true;
}

bool NaviSchedLatencyFeatures::subtract(const NaviSchedLatencyFeatures &other)
{
    if (_capacitySplit != other._capacitySplit ||
        _batchCount < other._batchCount ||
        _precedingRequestCount < other._precedingRequestCount ||
        _runningElapsedTimeMs < other._runningElapsedTimeMs ||
        _runningDecodeRequestCount < other._runningDecodeRequestCount ||
        _runningDecodeGeneratedTokenCount <
            other._runningDecodeGeneratedTokenCount ||
        _runningDecodeRemainingTokenBudget <
            other._runningDecodeRemainingTokenBudget ||
        _batchSize < other._batchSize ||
        _cacheHitTokenCount < other._cacheHitTokenCount ||
        _cacheMissTokenCount < other._cacheMissTokenCount ||
        _cacheMissTokenSquareSum < other._cacheMissTokenSquareSum ||
        _cacheHitMissTokenProductSum < other._cacheHitMissTokenProductSum ||
        _waitingBatchSize < other._waitingBatchSize ||
        _waitingCacheHitTokenCount < other._waitingCacheHitTokenCount ||
        _waitingCacheMissTokenCount < other._waitingCacheMissTokenCount ||
        _waitingCacheMissTokenSquareSum <
            other._waitingCacheMissTokenSquareSum ||
        _waitingCacheHitMissTokenProductSum <
            other._waitingCacheHitMissTokenProductSum) {
        return false;
    }
    const auto remainingDecodeRequestCount =
        _runningDecodeRequestCount - other._runningDecodeRequestCount;
    const auto remainingDecodeGeneratedTokenCount =
        _runningDecodeGeneratedTokenCount -
        other._runningDecodeGeneratedTokenCount;
    const auto remainingDecodeTokenBudget =
        _runningDecodeRemainingTokenBudget -
        other._runningDecodeRemainingTokenBudget;
    if (remainingDecodeGeneratedTokenCount < remainingDecodeRequestCount) {
        return false;
    }
    if (remainingDecodeRequestCount == 0 && remainingDecodeTokenBudget != 0) {
        return false;
    }
    _batchCount -= other._batchCount;
    _precedingRequestCount -= other._precedingRequestCount;
    _runningElapsedTimeMs -= other._runningElapsedTimeMs;
    _runningDecodeRequestCount = remainingDecodeRequestCount;
    _runningDecodeGeneratedTokenCount = remainingDecodeGeneratedTokenCount;
    _runningDecodeRemainingTokenBudget = remainingDecodeTokenBudget;
    _batchSize -= other._batchSize;
    _cacheHitTokenCount -= other._cacheHitTokenCount;
    _cacheMissTokenCount -= other._cacheMissTokenCount;
    _cacheMissTokenSquareSum -= other._cacheMissTokenSquareSum;
    _cacheHitMissTokenProductSum -= other._cacheHitMissTokenProductSum;
    _waitingBatchSize -= other._waitingBatchSize;
    _waitingCacheHitTokenCount -= other._waitingCacheHitTokenCount;
    _waitingCacheMissTokenCount -= other._waitingCacheMissTokenCount;
    _waitingCacheMissTokenSquareSum -= other._waitingCacheMissTokenSquareSum;
    _waitingCacheHitMissTokenProductSum -=
        other._waitingCacheHitMissTokenProductSum;
    return true;
}

int64_t NaviSchedLatencyFeatures::getBatchCount() const { return _batchCount; }

int64_t NaviSchedLatencyFeatures::getPrecedingRequestCount() const
{
    return _precedingRequestCount;
}

double NaviSchedLatencyFeatures::getRunningElapsedTimeMs() const
{
    return _runningElapsedTimeMs;
}

int64_t NaviSchedLatencyFeatures::getRunningDecodeRequestCount() const
{
    return _runningDecodeRequestCount;
}

int64_t NaviSchedLatencyFeatures::getRunningDecodeGeneratedTokenCount() const
{
    return _runningDecodeGeneratedTokenCount;
}

int64_t NaviSchedLatencyFeatures::getRunningDecodeRemainingTokenBudget() const
{
    return _runningDecodeRemainingTokenBudget;
}

int64_t NaviSchedLatencyFeatures::getBatchSize() const
{
    return _batchSize + _waitingBatchSize;
}

int64_t NaviSchedLatencyFeatures::getCacheHitTokenCount() const
{
    return _cacheHitTokenCount + _waitingCacheHitTokenCount;
}

int64_t NaviSchedLatencyFeatures::getCacheMissTokenCount() const
{
    return _cacheMissTokenCount + _waitingCacheMissTokenCount;
}

double NaviSchedLatencyFeatures::getCacheMissTokenSquareSum() const
{
    return _cacheMissTokenSquareSum + _waitingCacheMissTokenSquareSum;
}

double NaviSchedLatencyFeatures::getCacheHitMissTokenProductSum() const
{
    return _cacheHitMissTokenProductSum +
           _waitingCacheHitMissTokenProductSum;
}

bool NaviSchedLatencyFeatures::usesCapacitySplit() const
{
    return _capacitySplit;
}

int64_t NaviSchedLatencyFeatures::getRunningBatchSize() const
{
    return _batchSize;
}

int64_t NaviSchedLatencyFeatures::getRunningCacheHitTokenCount() const
{
    return _cacheHitTokenCount;
}

int64_t NaviSchedLatencyFeatures::getRunningCacheMissTokenCount() const
{
    return _cacheMissTokenCount;
}

double NaviSchedLatencyFeatures::getRunningCacheMissTokenSquareSum() const
{
    return _cacheMissTokenSquareSum;
}

double NaviSchedLatencyFeatures::getRunningCacheHitMissTokenProductSum() const
{
    return _cacheHitMissTokenProductSum;
}

int64_t NaviSchedLatencyFeatures::getWaitingBatchSize() const
{
    return _waitingBatchSize;
}

int64_t NaviSchedLatencyFeatures::getWaitingCacheHitTokenCount() const
{
    return _waitingCacheHitTokenCount;
}

int64_t NaviSchedLatencyFeatures::getWaitingCacheMissTokenCount() const
{
    return _waitingCacheMissTokenCount;
}

double NaviSchedLatencyFeatures::getWaitingCacheMissTokenSquareSum() const
{
    return _waitingCacheMissTokenSquareSum;
}

double NaviSchedLatencyFeatures::getWaitingCacheHitMissTokenProductSum() const
{
    return _waitingCacheHitMissTokenProductSum;
}

void NaviSchedLatencyFeatures::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(_batchCount);
    dataBuffer.write(_batchSize);
    dataBuffer.write(_cacheHitTokenCount);
    dataBuffer.write(_cacheMissTokenCount);
    dataBuffer.write(_cacheMissTokenSquareSum);
    dataBuffer.write(_cacheHitMissTokenProductSum);
    dataBuffer.write(_capacitySplit);
    dataBuffer.write(_waitingBatchSize);
    dataBuffer.write(_waitingCacheHitTokenCount);
    dataBuffer.write(_waitingCacheMissTokenCount);
    dataBuffer.write(_waitingCacheMissTokenSquareSum);
    dataBuffer.write(_waitingCacheHitMissTokenProductSum);
    dataBuffer.write(_precedingRequestCount);
    dataBuffer.write(_runningElapsedTimeMs);
    dataBuffer.write(_runningDecodeRequestCount);
    dataBuffer.write(_runningDecodeGeneratedTokenCount);
    dataBuffer.write(_runningDecodeRemainingTokenBudget);
}

void NaviSchedLatencyFeatures::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(_batchCount);
    dataBuffer.read(_batchSize);
    dataBuffer.read(_cacheHitTokenCount);
    dataBuffer.read(_cacheMissTokenCount);
    dataBuffer.read(_cacheMissTokenSquareSum);
    dataBuffer.read(_cacheHitMissTokenProductSum);
    dataBuffer.read(_capacitySplit);
    dataBuffer.read(_waitingBatchSize);
    dataBuffer.read(_waitingCacheHitTokenCount);
    dataBuffer.read(_waitingCacheMissTokenCount);
    dataBuffer.read(_waitingCacheMissTokenSquareSum);
    dataBuffer.read(_waitingCacheHitMissTokenProductSum);
    dataBuffer.read(_precedingRequestCount);
    dataBuffer.read(_runningElapsedTimeMs);
    dataBuffer.read(_runningDecodeRequestCount);
    dataBuffer.read(_runningDecodeGeneratedTokenCount);
    dataBuffer.read(_runningDecodeRemainingTokenBudget);
}

} // namespace navi_sched

#include "rtp_llm/flexlb/navi_sched/kernel/AgentLatencyKernel.h"
#include "autil/TimeUtility.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace navi_sched {

namespace {
constexpr double QUEUE_BETA_1 = 0.9;
constexpr double QUEUE_BETA_2 = 0.95;
constexpr double QUEUE_EPSILON = 1e-20;
constexpr double QUEUE_RAW_LATENCY_SCALE_MS = 1000.0;
constexpr int64_t QUEUE_DEBUG_SAMPLE_INTERVAL = 128;
constexpr size_t MAX_EXECUTION_LATENCY_BUCKET_COUNT = 64;

template <typename Values>
std::string formatValues(const Values &values, size_t count)
{
    std::ostringstream stream;
    stream << std::setprecision(17);
    for (size_t index = 0; index < std::min(count, values.size()); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

std::string formatRequestIds(const AgentLatencyInput &input)
{
    std::ostringstream stream;
    for (size_t index = 0; index < input.queries.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << input.queries[index].requestId;
    }
    if (input.queries.empty()) {
        for (size_t index = 0; index < input.localBatch.requestIds.size(); ++index) {
            if (index > 0) {
                stream << ',';
            }
            stream << input.localBatch.requestIds[index];
        }
    }
    return stream.str();
}
} // namespace

int64_t AgentLatencyKernel::predictBatchLatencyUs(const NaviSchedBatchPtr &batch) const
{
    if (!batch) {
        return 0;
    }
    const auto &requests = AgentBatchR::getRequestStats(*batch);
    NaviSchedLatencyFeatures features;
    features.appendBatchCount(1);
    for (const auto &request : requests) {
        AgentLatencyInput::appendFeatures(features, request);
    }
    return std::llround(_latencyModelR->predict(features) * 1000.0);
}

double AgentLatencyKernel::getRawQueueLatencyMs(int64_t currentTimeUs) const
{
    auto latencyUs = _queueLatencyUs.load(std::memory_order_relaxed);
    int64_t earliestStartTimeUs = std::numeric_limits<int64_t>::max();
    int64_t earliestBatchId = std::numeric_limits<int64_t>::max();
    int64_t headElapsedUs = 0;
    int64_t headPredictedLatencyUs = 0;
    for (const auto &[batchId, running] : _runningBatches) {
        if (running.startTimeUs > earliestStartTimeUs ||
            (running.startTimeUs == earliestStartTimeUs && batchId >= earliestBatchId)) {
            continue;
        }
        earliestStartTimeUs = running.startTimeUs;
        earliestBatchId = batchId;
        headElapsedUs = std::max<int64_t>(
            0, static_cast<int64_t>((currentTimeUs - running.startTimeUs) *
                                    _timeCompression));
        headPredictedLatencyUs = running.predictedLatencyUs;
    }
    latencyUs -= std::min(headElapsedUs, headPredictedLatencyUs);
    return std::max<int64_t>(0, latencyUs) / 1000.0;
}

AgentLatencyKernel::QueueFeatures AgentLatencyKernel::getQueueFeatures(
    int64_t currentTimeUs) const
{
    QueueFeatures features{getRawQueueLatencyMs(currentTimeUs),
                           static_cast<double>(_batchLatencyUs.size())};
    features.weights = _queueWeights;
    features.fitStep = _queueFitStep - 1;
    features.predictedLatencyMs = predictQueueLatencyMs(features);
    return features;
}

double AgentLatencyKernel::predictQueueLatencyMs(const QueueFeatures &features) const
{
    const auto prediction =
        features.rawLatencyMs + _queueWeights[0] +
        _queueWeights[1] * features.rawLatencyMs / QUEUE_RAW_LATENCY_SCALE_MS +
        _queueWeights[2] * features.precedingBatchCount;
    return std::max(0.0, prediction);
}

double AgentLatencyKernel::getQueueLatencyMs(int64_t currentTimeUs) const
{
    return getQueueFeatures(currentTimeUs).predictedLatencyMs;
}

AgentLatencyKernel::QueueLoadPrediction AgentLatencyKernel::getQueueLoadPrediction(
    int64_t currentTimeUs) const
{
    QueueLoadPrediction prediction;
    if (_batchLatencyUs.empty()) {
        return prediction;
    }
    prediction.totalLatencyMs = getQueueLatencyMs(currentTimeUs);
    int64_t headRemainingUs = 0;
    int64_t earliestStartTimeUs = std::numeric_limits<int64_t>::max();
    int64_t earliestBatchId = std::numeric_limits<int64_t>::max();
    for (const auto &[batchId, running] : _runningBatches) {
        if (running.startTimeUs > earliestStartTimeUs ||
            (running.startTimeUs == earliestStartTimeUs && batchId >= earliestBatchId)) {
            continue;
        }
        const auto elapsedUs = std::max<int64_t>(
            0, static_cast<int64_t>((currentTimeUs - running.startTimeUs) *
                                    _timeCompression));
        earliestStartTimeUs = running.startTimeUs;
        earliestBatchId = batchId;
        headRemainingUs = std::max<int64_t>(0, running.predictedLatencyUs - elapsedUs);
    }
    prediction.decayHeadRemainingMs = std::min(
        prediction.totalLatencyMs, std::max<int64_t>(0, headRemainingUs) / 1000.0);
    return prediction;
}

bool AgentLatencyKernel::shouldFitQueueLatency(bool unifiedLatencyPrediction,
                                               AgentLatencyInputType inputType,
                                               bool hasQueueFeatures,
                                               bool latencySampleValid)
{
    return latencySampleValid && !unifiedLatencyPrediction &&
           AgentLatencyInputType::BATCH_FINISHED == inputType && hasQueueFeatures;
}

bool AgentLatencyKernel::shouldRecordLatencySample(bool unifiedLatencyPrediction,
                                                   AgentLatencyInputType inputType,
                                                   bool unifiedLatencySample,
                                                   bool latencySampleValid)
{
    return latencySampleValid && AgentLatencyInputType::BATCH_ABORTED != inputType &&
           (!unifiedLatencyPrediction || unifiedLatencySample);
}

bool AgentLatencyKernel::shouldRecordLatencyValidation(AgentLatencyInputType inputType,
                                                       bool latencySampleValid,
                                                       bool latencyValidationValid)
{
    return AgentLatencyInputType::BATCH_FINISHED == inputType && latencySampleValid &&
           latencyValidationValid;
}

void AgentLatencyKernel::fitQueueLatency(const QueueFeatures &features,
                                         double actualQueueLatencyMs)
{
    if (!std::isfinite(actualQueueLatencyMs) || actualQueueLatencyMs < 0.0) {
        return;
    }
    const std::array<double, 3> input = {
        1.0,
        features.rawLatencyMs / QUEUE_RAW_LATENCY_SCALE_MS,
        features.precedingBatchCount,
    };
    const auto difference = features.predictedLatencyMs - actualQueueLatencyMs;
    const auto correction = _queueLearningRate *
                            std::sqrt(1.0 - std::pow(QUEUE_BETA_2, _queueFitStep)) /
                            (1.0 - std::pow(QUEUE_BETA_1, _queueFitStep));
    for (size_t index = 0; index < _queueWeights.size(); ++index) {
        const auto gradient = difference * input[index];
        _queueAdamMoment1[index] =
            QUEUE_BETA_1 * _queueAdamMoment1[index] + (1.0 - QUEUE_BETA_1) * gradient;
        _queueAdamMoment2[index] = QUEUE_BETA_2 * _queueAdamMoment2[index] +
                                   (1.0 - QUEUE_BETA_2) * gradient * gradient;
        _queueWeights[index] -= correction * _queueAdamMoment1[index] /
                                std::sqrt(_queueAdamMoment2[index] + QUEUE_EPSILON);
    }
    ++_queueFitStep;
}

void AgentLatencyKernel::updateExecutionState(const AgentLatencyInput &input)
{
    _latestBatchWaitingRequests = input.batchWaitingRequests;
    _latestQueuedBatchRequests = input.queuedBatchRequests;
    _latestRunningRequests = input.runningRequests;
}

void AgentLatencyKernel::startRunningBatch(const AgentLatencyInput &input)
{
    const auto latencyIt = _batchLatencyUs.find(input.batchId);
    _runningBatches[input.batchId] = RunningBatch{
        input.eventTimeUs,
        _batchLatencyUs.end() == latencyIt ? 0 : latencyIt->second,
    };
}

double AgentLatencyKernel::getRunningElapsedTimeMs(int64_t currentTimeUs) const
{
    if (_latestRunningRequests <= 0) {
        return 0.0;
    }
    long double elapsedTimeMs = 0.0;
    for (const auto &[batchId, running] : _runningBatches) {
        const auto queueFeaturesIt = _batchQueueFeatures.find(batchId);
        if (queueFeaturesIt == _batchQueueFeatures.end()) {
            continue;
        }
        elapsedTimeMs += std::max<int64_t>(0, currentTimeUs - running.startTimeUs) /
                         1000.0L * _timeCompression *
                         std::max(0, queueFeaturesIt->second.requestCount);
    }
    return static_cast<double>(
        std::min<long double>(elapsedTimeMs, std::numeric_limits<double>::max()));
}

void AgentLatencyKernel::reportExecutionLoad(
    const NaviSchedLocalBatchInfo &localBatch) const
{
    const auto prediction = getQueueLoadPrediction(autil::TimeUtility::currentTime());
    _agentReportR->reportExecutionLoad(
        _latestBatchWaitingRequests, _latestQueuedBatchRequests, _latestRunningRequests,
        prediction.totalLatencyMs, prediction.decayHeadRemainingMs,
        _agentBatchR->getTimeoutBatchCount(), _agentBatchR->getAbnormalBatchFinishCount(),
        0, localBatch);
}

bool AgentLatencyKernel::derivesUnifiedWorkloadFromBatches() const
{
    return _agentReportR->pdSeparated() && _agentReportR->usesUnifiedLatencyPrediction();
}

bool AgentLatencyKernel::addUnifiedBatch(const AgentLatencyInput &input,
                                         UnifiedBatchState &state,
                                         NaviSchedLocalBatchInfo &admittedBatch)
{
    admittedBatch = NaviSchedLocalBatchInfo();
    if (!input.batch || input.batchId <= 0 || input.batch->batchId != input.batchId) {
        return false;
    }
    const auto &requests = AgentBatchR::getRequestStats(*input.batch);
    if (requests.empty()) {
        return false;
    }
    state.precedingFeatures = _unifiedWorkloadFeatures;
    if (!state.precedingFeatures.appendRunningElapsedTimeMs(
            getRunningElapsedTimeMs(input.eventTimeUs))) {
        return false;
    }
    state.readyTimeUs = input.eventTimeUs;
    state.localBatch = input.batch->makeLocalBatchInfo();
    if (!state.localBatch.valid()) {
        return false;
    }
    if (!state.features.appendBatchCount(1)) {
        return false;
    }
    for (const auto &request : requests) {
        if (!request.valid()) {
            return false;
        }
        AgentLatencyInput::appendFeatures(state.features, request);
        if (request.cacheHitTokenCount < request.totalTokenCount) {
            ++state.remainingPrefillRequestCount;
        }
    }
    auto updatedFeatures = _unifiedWorkloadFeatures;
    if (!updatedFeatures.append(state.features) ||
        updatedFeatures.getBatchCount() > std::numeric_limits<int32_t>::max() ||
        updatedFeatures.getBatchSize() > std::numeric_limits<int32_t>::max() ||
        state.remainingPrefillRequestCount >
            std::numeric_limits<int32_t>::max() - _unifiedRemainingPrefillRequestCount) {
        return false;
    }
    _unifiedWorkloadFeatures = std::move(updatedFeatures);
    _unifiedRemainingPrefillRequestCount += state.remainingPrefillRequestCount;
    admittedBatch = state.localBatch;
    return true;
}

bool AgentLatencyKernel::replaceUnifiedBatch(const AgentLatencyInput &input,
                                             UnifiedBatchState &state)
{
    if (!input.batch || input.batchId <= 0 || input.batch->batchId != input.batchId) {
        return false;
    }
    UnifiedBatchState replacement;
    replacement.precedingFeatures = state.precedingFeatures;
    replacement.readyTimeUs = state.readyTimeUs;
    replacement.localBatch = input.batch->makeLocalBatchInfo();
    if (!replacement.localBatch.valid() || !replacement.features.appendBatchCount(1)) {
        return false;
    }
    const auto &requests = AgentBatchR::getRequestStats(*input.batch);
    if (requests.empty()) {
        return false;
    }
    for (const auto &request : requests) {
        if (!request.valid()) {
            return false;
        }
        AgentLatencyInput::appendFeatures(replacement.features, request);
        if (request.cacheHitTokenCount < request.totalTokenCount) {
            ++replacement.remainingPrefillRequestCount;
        }
    }
    if (_unifiedRemainingPrefillRequestCount < state.remainingPrefillRequestCount) {
        return false;
    }
    auto updatedFeatures = _unifiedWorkloadFeatures;
    if (!updatedFeatures.subtract(state.features) ||
        !updatedFeatures.append(replacement.features) ||
        updatedFeatures.getBatchCount() > std::numeric_limits<int32_t>::max() ||
        updatedFeatures.getBatchSize() > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    const auto remainingPrefillRequestCount = _unifiedRemainingPrefillRequestCount -
                                              state.remainingPrefillRequestCount +
                                              replacement.remainingPrefillRequestCount;
    if (remainingPrefillRequestCount < 0) {
        return false;
    }
    _unifiedWorkloadFeatures = std::move(updatedFeatures);
    _unifiedRemainingPrefillRequestCount = remainingPrefillRequestCount;
    state = std::move(replacement);
    return true;
}

bool AgentLatencyKernel::retireUnifiedBatch(AgentLatencyInput &input,
                                            const UnifiedBatchState &state)
{
    if (_unifiedRemainingPrefillRequestCount < state.remainingPrefillRequestCount) {
        return false;
    }
    auto updatedFeatures = _unifiedWorkloadFeatures;
    if (!updatedFeatures.subtract(state.features)) {
        return false;
    }
    if (AgentLatencyInputType::BATCH_FINISHED == input.type) {
        input.precedingFeatures = state.precedingFeatures;
        input.unifiedLatencySample = true;
        input.modelLatencyMs =
            std::max<int64_t>(0, input.eventTimeUs - state.readyTimeUs) / 1000.0 *
            _timeCompression;
    }
    _unifiedWorkloadFeatures = std::move(updatedFeatures);
    _unifiedRemainingPrefillRequestCount -= state.remainingPrefillRequestCount;
    return true;
}

NaviSchedWorkloadSnapshot AgentLatencyKernel::makeUnifiedWorkloadSnapshot(
    int64_t eventTimeUs, const NaviSchedLocalBatchInfo &admittedBatch)
{
    NaviSchedWorkloadSnapshot snapshot;
    snapshot.eventTimeUs = eventTimeUs;
    snapshot.batchCount = static_cast<int32_t>(_unifiedWorkloadFeatures.getBatchCount());
    snapshot.requestCount = static_cast<int32_t>(_unifiedWorkloadFeatures.getBatchSize());
    snapshot.remainingPrefillRequestCount = _unifiedRemainingPrefillRequestCount;
    snapshot.runningRequestCount = _latestRunningRequests;
    snapshot.runningElapsedTimeMs = getRunningElapsedTimeMs(eventTimeUs);
    snapshot.waitingRequestCount = _latestQueuedBatchRequests;
    snapshot.remainingPrefillContextTokenCount =
        _unifiedWorkloadFeatures.getCacheHitTokenCount();
    snapshot.remainingPrefillTokenCount =
        _unifiedWorkloadFeatures.getCacheMissTokenCount();
    snapshot.remainingPrefillTokenSquareSum =
        _unifiedWorkloadFeatures.getCacheMissTokenSquareSum();
    snapshot.contextRemainingPrefillTokenProductSum =
        _unifiedWorkloadFeatures.getCacheHitMissTokenProductSum();
    snapshot.remainingTotalContextTokenCount = snapshot.remainingPrefillContextTokenCount;
    snapshot.remainingTotalTokenCount = snapshot.remainingPrefillTokenCount;
    snapshot.remainingTotalTokenSquareSum = snapshot.remainingPrefillTokenSquareSum;
    snapshot.contextRemainingTotalTokenProductSum =
        snapshot.contextRemainingPrefillTokenProductSum;
    snapshot.totalPromptTokenCount =
        snapshot.remainingTotalContextTokenCount + snapshot.remainingTotalTokenCount;
    snapshot.admittedBatch = admittedBatch;
    return snapshot;
}

void AgentLatencyKernel::reportUnifiedWorkload(
    int64_t eventTimeUs, const NaviSchedLocalBatchInfo &admittedBatch)
{
    auto snapshot = makeUnifiedWorkloadSnapshot(
        eventTimeUs > 0 ? eventTimeUs : autil::TimeUtility::currentTime(), admittedBatch);
    const auto revision = _agentReportR->reportWorkloadUpdated(snapshot);
    if (0 == revision && _agentReportR->getSubscriberCount() > 0) {
        NAVI_KERNEL_LOG(ERROR,
                        "failed to report unified workload, scheduleBatchId [%ld], "
                        "localBatchId [%ld]",
                        admittedBatch.scheduleBatchId, admittedBatch.localBatchId);
    }
}

NaviSchedLocalBatchInfo AgentLatencyKernel::getLocalBatchInfo(int64_t batchId, bool erase)
{
    const auto it = _batchLocalInfos.find(batchId);
    if (_batchLocalInfos.end() == it) {
        return {};
    }
    const auto localBatch = it->second;
    if (erase) {
        _batchLocalInfos.erase(it);
    }
    return localBatch;
}

void AgentLatencyKernel::updateMean(double value, uint64_t count, double &mean)
{
    mean += (value - mean) / static_cast<double>(count);
}

void AgentLatencyKernel::recordLatencyValidation(const AgentLatencyInput &input,
                                                 double agentPredictedExecutionLatencyMs,
                                                 const QueueFeatures *queueFeatures,
                                                 double agentPredictedQueueLatencyMs)
{
    if (!std::isfinite(agentPredictedExecutionLatencyMs) ||
        agentPredictedExecutionLatencyMs < 0.0 ||
        !std::isfinite(input.backendQueueLatencyMs) ||
        input.backendQueueLatencyMs < 0.0) {
        return;
    }
    auto &stats = _latencyValidationStats;
    const auto count = ++stats.sampleCount;
    const auto cacheHitError = static_cast<double>(input.actualCacheHitTokenCount -
                                                   input.predictedCacheHitTokenCount);
    const auto actualQueueMs = input.backendQueueLatencyMs;
    const auto actualExecutionMs =
        input.unifiedLatencySample ? input.modelLatencyMs : input.latencyMs;
    const auto predictedTotalMs =
        input.predictedQueueLatencyMs + input.predictedExecutionLatencyMs;
    const auto actualTotalMs = input.unifiedLatencySample
                                   ? input.modelLatencyMs
                                   : actualQueueMs + actualExecutionMs;
    const auto queueErrorMs = actualQueueMs - input.predictedQueueLatencyMs;
    const auto masterExecutionErrorMs =
        actualExecutionMs - input.predictedExecutionLatencyMs;
    const auto agentExecutionErrorMs =
        actualExecutionMs - agentPredictedExecutionLatencyMs;
    const auto totalErrorMs = actualTotalMs - predictedTotalMs;
    const auto executionDivisor = std::max(actualExecutionMs, 1e-9);
    const auto totalDivisor = std::max(actualTotalMs, 1e-9);
    const auto bucketId = static_cast<size_t>(
        std::clamp(std::floor(std::log2(std::max(actualExecutionMs, 1.0))), 0.0,
                   static_cast<double>(MAX_EXECUTION_LATENCY_BUCKET_COUNT - 1)));
    if (_metricsReporterR) {
        const auto &reporter = _metricsReporterR->getReporter();
        if (reporter) {
            const auto report = [&reporter](const char *name, double value,
                                            const kmonitor::MetricsTags *tags = nullptr) {
                reporter->report(value, name, kmonitor::MetricType::GAUGE, tags,
                                 kmonitor::MetricLevel::FATAL);
            };
            report("agent.validation_cache_hit_tokens_error", cacheHitError);
            report("agent.validation_cache_hit_tokens_absolute_error",
                   std::abs(cacheHitError));
            report("agent.validation_queue_error_ms", queueErrorMs);
            report("agent.validation_queue_absolute_error_ms", std::abs(queueErrorMs));
            if (queueFeatures) {
                const auto rawQueueErrorMs = actualQueueMs - queueFeatures->rawLatencyMs;
                const auto agentQueueErrorMs =
                    actualQueueMs - agentPredictedQueueLatencyMs;
                report("agent.queue_model_raw_error_ms", rawQueueErrorMs);
                report("agent.queue_model_raw_absolute_error_ms",
                       std::abs(rawQueueErrorMs));
                report("agent.queue_model_error_ms", agentQueueErrorMs);
                report("agent.queue_model_absolute_error_ms",
                       std::abs(agentQueueErrorMs));
            }
            if (input.predictedQueueLatencyMs <= 1e-9) {
                report("agent.validation_queue_zero_absolute_error_ms",
                       std::abs(queueErrorMs));
            } else {
                report("agent.validation_queue_positive_error_ms", queueErrorMs);
                report("agent.validation_queue_positive_absolute_error_ms",
                       std::abs(queueErrorMs));
            }
            report("agent.validation_execution_master_error_ms", masterExecutionErrorMs);
            report("agent.validation_execution_master_absolute_error_ms",
                   std::abs(masterExecutionErrorMs));
            report("agent.validation_execution_master_absolute_percentage_error_pct",
                   std::abs(masterExecutionErrorMs) / executionDivisor * 100.0);
            report("agent.validation_execution_agent_error_ms", agentExecutionErrorMs);
            report("agent.validation_execution_agent_absolute_error_ms",
                   std::abs(agentExecutionErrorMs));
            report("agent.validation_execution_agent_absolute_percentage_error_pct",
                   std::abs(agentExecutionErrorMs) / executionDivisor * 100.0);
            report("agent.validation_total_error_ms", totalErrorMs);
            report("agent.validation_total_absolute_error_ms", std::abs(totalErrorMs));
            report("agent.validation_total_absolute_percentage_error_pct",
                   std::abs(totalErrorMs) / totalDivisor * 100.0);
            const kmonitor::MetricsTags tags("bucket_id", std::to_string(bucketId));
            report("agent.validation_execution_bucket_master_absolute_error_ms",
                   std::abs(masterExecutionErrorMs), &tags);
            report(
                "agent.validation_execution_bucket_master_absolute_percentage_error_pct",
                std::abs(masterExecutionErrorMs) / executionDivisor * 100.0, &tags);
            report("agent.validation_execution_bucket_agent_absolute_error_ms",
                   std::abs(agentExecutionErrorMs), &tags);
            report(
                "agent.validation_execution_bucket_agent_absolute_percentage_error_pct",
                std::abs(agentExecutionErrorMs) / executionDivisor * 100.0, &tags);
        }
    }
    updateMean(input.completedRequestCount, count, stats.batchRequestCountMean);
    updateMean(input.predictedCacheHitTokenCount, count,
               stats.predictedCacheHitTokensMean);
    updateMean(input.actualCacheHitTokenCount, count, stats.actualCacheHitTokensMean);
    updateMean(cacheHitError, count, stats.cacheHitTokensBiasMean);
    updateMean(std::abs(cacheHitError), count, stats.cacheHitTokensMae);
    updateMean(input.predictedQueueLatencyMs, count, stats.masterPredictedQueueMeanMs);
    updateMean(actualQueueMs, count, stats.actualQueueMeanMs);
    updateMean(input.backendQueueLatencyMs, count, stats.backendQueueMeanMs);
    updateMean(queueErrorMs, count, stats.queueBiasMeanMs);
    updateMean(std::abs(queueErrorMs), count, stats.queueMaeMs);
    if (queueFeatures) {
        const auto rawQueueErrorMs = actualQueueMs - queueFeatures->rawLatencyMs;
        const auto agentQueueErrorMs = actualQueueMs - agentPredictedQueueLatencyMs;
        updateMean(queueFeatures->rawLatencyMs, count, stats.queueRawPredictedMeanMs);
        updateMean(rawQueueErrorMs, count, stats.queueRawBiasMeanMs);
        updateMean(std::abs(rawQueueErrorMs), count, stats.queueRawMaeMs);
        updateMean(agentPredictedQueueLatencyMs, count, stats.agentPredictedQueueMeanMs);
        updateMean(agentQueueErrorMs, count, stats.agentQueueBiasMeanMs);
        updateMean(std::abs(agentQueueErrorMs), count, stats.agentQueueMaeMs);
        updateMean(queueFeatures->precedingBatchCount, count,
                   stats.queueFeatureBatchCountMean);
    }
    stats.queueModelWeights.assign(_queueWeights.begin(), _queueWeights.end());
    stats.queueModelFitStep = _queueFitStep - 1;
    if (input.predictedQueueLatencyMs <= 1e-9) {
        const auto zeroCount = ++stats.zeroPredictedQueueSampleCount;
        updateMean(actualQueueMs, zeroCount, stats.zeroPredictedQueueActualMeanMs);
        updateMean(std::abs(queueErrorMs), zeroCount, stats.zeroPredictedQueueMaeMs);
    } else {
        const auto positiveCount = ++stats.positivePredictedQueueSampleCount;
        updateMean(input.predictedQueueLatencyMs, positiveCount,
                   stats.positivePredictedQueueMeanMs);
        updateMean(actualQueueMs, positiveCount,
                   stats.positivePredictedQueueActualMeanMs);
        updateMean(queueErrorMs, positiveCount, stats.positivePredictedQueueBiasMeanMs);
        updateMean(std::abs(queueErrorMs), positiveCount,
                   stats.positivePredictedQueueMaeMs);
    }
    updateMean(input.predictedExecutionLatencyMs, count,
               stats.masterPredictedExecutionMeanMs);
    updateMean(agentPredictedExecutionLatencyMs, count,
               stats.agentPredictedExecutionMeanMs);
    updateMean(actualExecutionMs, count, stats.actualExecutionMeanMs);
    updateMean(masterExecutionErrorMs, count, stats.masterExecutionBiasMeanMs);
    updateMean(std::abs(masterExecutionErrorMs), count, stats.masterExecutionMaeMs);
    updateMean(std::abs(masterExecutionErrorMs) / executionDivisor * 100.0, count,
               stats.masterExecutionMapePct);
    updateMean(agentExecutionErrorMs, count, stats.agentExecutionBiasMeanMs);
    updateMean(std::abs(agentExecutionErrorMs), count, stats.agentExecutionMaeMs);
    updateMean(std::abs(agentExecutionErrorMs) / executionDivisor * 100.0, count,
               stats.agentExecutionMapePct);
    if (stats.executionLatencyBuckets.size() <= bucketId) {
        stats.executionLatencyBuckets.resize(bucketId + 1);
    }
    auto &bucket = stats.executionLatencyBuckets[bucketId];
    const auto bucketCount = ++bucket.sampleCount;
    updateMean(std::abs(masterExecutionErrorMs), bucketCount,
               bucket.masterExecutionMaeMs);
    updateMean(std::abs(masterExecutionErrorMs) / executionDivisor * 100.0, bucketCount,
               bucket.masterExecutionMapePct);
    updateMean(std::abs(agentExecutionErrorMs), bucketCount, bucket.agentExecutionMaeMs);
    updateMean(std::abs(agentExecutionErrorMs) / executionDivisor * 100.0, bucketCount,
               bucket.agentExecutionMapePct);
    updateMean(predictedTotalMs, count, stats.masterPredictedTotalMeanMs);
    updateMean(actualTotalMs, count, stats.actualTotalMeanMs);
    updateMean(totalErrorMs, count, stats.totalBiasMeanMs);
    updateMean(std::abs(totalErrorMs), count, stats.totalMaeMs);
    updateMean(std::abs(totalErrorMs) / totalDivisor * 100.0, count, stats.totalMapePct);
}

bool AgentLatencyKernel::shouldLogQueueDebugSample(int64_t scheduleBatchId)
{
    return scheduleBatchId > 0 && scheduleBatchId % QUEUE_DEBUG_SAMPLE_INTERVAL == 0;
}

const char *AgentLatencyKernel::getLatencyInputTypeName(AgentLatencyInputType type)
{
    switch (type) {
        case AgentLatencyInputType::BATCH_FINISHED:
            return "batch_finished";
        case AgentLatencyInputType::OBSERVED_BATCH_FINISHED:
            return "observed_batch_finished";
        default:
            return "unexpected";
    }
}

void AgentLatencyKernel::logLatencyTrainingSample(
    const AgentLatencyInput &input, uint64_t sampleSequence,
    const NaviSchedLatencySample &sample, const QueueFeatures *queueFeatures,
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
    const std::optional<NonLinearPrefillModelR::NonlinearOperatingPoint>
        &operatingPoint) const
{
    const auto basePrediction =
        snapshot && snapshot->valid()
            ? _latencyModelR->predict(sample.features, snapshot->parameters)
            : 0.0;
    const auto predictionTimeUs =
        sample.eventTimeUs > 0 ? sample.eventTimeUs : autil::TimeUtility::currentTime();
    const auto residualConfig =
        _agentSchedulerConfigR->get().config.residualRuntimeConfig();
    const auto residualEvaluation =
        snapshot && snapshot->valid()
            ? LatencyResidualModel::evaluate(basePrediction, snapshot->residualParameters,
                                             snapshot->residualLastUpdateTimeUs,
                                             predictionTimeUs, residualConfig)
            : LatencyResidualModel::Evaluation{basePrediction};
    const auto residualDecay =
        snapshot && snapshot->valid()
            ? LatencyResidualModel::decayFactor(snapshot->residualLastUpdateTimeUs,
                                                predictionTimeUs, residualConfig)
            : 0.0;
    const char *region = "invalid";
    if (operatingPoint) {
        region = operatingPoint->region < 0 ? "negative_far"
                 : operatingPoint->region > 0 ? "positive_far" : "transition";
    }
    const auto requestIds = formatRequestIds(input);
    NAVI_KERNEL_LOG(
        INFO,
        "[navi-latency-training-sample] sequence=%lu type=%s event_time_us=%ld "
        "batch_id=%ld schedule_batch_id=%ld request_ids=[%s] "
        "reported_inflight_requests=%d "
        "batch_count=%ld batch_size=%ld "
        "preceding_request_count=%ld "
        "predicted_hit_tokens=%ld "
        "actual_hit_tokens=%ld hit_tokens=%ld miss_tokens=%ld "
        "hit_mismatch_requests=%d hit_error_abs_tokens=%ld "
        "hit_error_max_tokens=%ld "
        "miss_square_sum=%.0f hit_miss_sum=%.0f target_ms=%.9f "
        "capacity_split=%d running_batch_size=%ld running_hit_tokens=%ld "
        "running_miss_tokens=%ld running_miss_square_sum=%.0f "
        "running_hit_miss_sum=%.0f running_elapsed_time_ms=%.9f "
        "running_decode_request_count=%ld "
        "running_decode_generated_tokens=%ld "
        "running_decode_remaining_token_budget=%ld "
        "waiting_batch_size=%ld "
        "waiting_hit_tokens=%ld waiting_miss_tokens=%ld "
        "waiting_miss_square_sum=%.0f waiting_hit_miss_sum=%.0f "
        "unified_sample=%d preceding_batch_count=%ld preceding_batch_size=%ld "
        "preceding_hit_tokens=%ld preceding_miss_tokens=%ld "
        "preceding_miss_square_sum=%.0f preceding_hit_miss_sum=%.0f "
        "preceding_running_decode_request_count=%ld "
        "preceding_running_decode_generated_tokens=%ld "
        "preceding_running_decode_remaining_token_budget=%ld "
        "master_prediction_ms=%.9f base_prediction_before_ms=%.9f "
        "residual_correction_before_ms=%.9f prediction_before_ms=%.9f "
        "residual_decay_factor_before=%.9f "
        "nonlinear_u_before=%.9f nonlinear_curvature_before=%.9f "
        "nonlinear_r_before=%.9f nonlinear_region_before=%s "
        "master_queue_prediction_ms=%.9f backend_queue_ms=%.9f "
        "queue_features_valid=%d queue_raw_at_ready_ms=%.9f "
        "agent_queue_at_ready_ms=%.9f queue_preceding_batch_count=%.0f "
        "queue_fit_step_at_ready=%lu "
        "queue_weights_at_ready=[%.9f,%.9f,%.9f] "
        "latest_local_batch_id=%ld revision_before=%ld",
        sampleSequence, getLatencyInputTypeName(input.type), sample.eventTimeUs,
        input.batchId, input.localBatch.scheduleBatchId, requestIds.c_str(),
        input.inflightRequests,
        sample.features.getBatchCount(), sample.features.getBatchSize(),
        sample.features.getPrecedingRequestCount(), input.predictedCacheHitTokenCount,
        input.actualCacheHitTokenCount, sample.features.getCacheHitTokenCount(),
        sample.features.getCacheMissTokenCount(), input.cacheHitMismatchRequestCount,
        input.cacheHitAbsoluteErrorTokenCount, input.cacheHitMaxAbsoluteErrorTokenCount,
        sample.features.getCacheMissTokenSquareSum(),
        sample.features.getCacheHitMissTokenProductSum(), sample.latencyMs,
        sample.features.usesCapacitySplit() ? 1 : 0,
        sample.features.getRunningBatchSize(),
        sample.features.getRunningCacheHitTokenCount(),
        sample.features.getRunningCacheMissTokenCount(),
        sample.features.getRunningCacheMissTokenSquareSum(),
        sample.features.getRunningCacheHitMissTokenProductSum(),
        sample.features.getRunningElapsedTimeMs(),
        sample.features.getRunningDecodeRequestCount(),
        sample.features.getRunningDecodeGeneratedTokenCount(),
        sample.features.getRunningDecodeRemainingTokenBudget(),
        sample.features.getWaitingBatchSize(),
        sample.features.getWaitingCacheHitTokenCount(),
        sample.features.getWaitingCacheMissTokenCount(),
        sample.features.getWaitingCacheMissTokenSquareSum(),
        sample.features.getWaitingCacheHitMissTokenProductSum(),
        input.unifiedLatencySample ? 1 : 0, input.precedingFeatures.getBatchCount(),
        input.precedingFeatures.getBatchSize(),
        input.precedingFeatures.getCacheHitTokenCount(),
        input.precedingFeatures.getCacheMissTokenCount(),
        input.precedingFeatures.getCacheMissTokenSquareSum(),
        input.precedingFeatures.getCacheHitMissTokenProductSum(),
        input.precedingFeatures.getRunningDecodeRequestCount(),
        input.precedingFeatures.getRunningDecodeGeneratedTokenCount(),
        input.precedingFeatures.getRunningDecodeRemainingTokenBudget(),
        input.predictedExecutionLatencyMs, basePrediction,
        residualEvaluation.correctionMs, residualEvaluation.predictionMs, residualDecay,
        operatingPoint ? operatingPoint->u : 0.0,
        operatingPoint ? operatingPoint->curvature : 0.0,
        operatingPoint ? operatingPoint->r : 0.0, region,
        input.predictedQueueLatencyMs, input.backendQueueLatencyMs, queueFeatures ? 1 : 0,
        queueFeatures ? queueFeatures->rawLatencyMs : 0.0,
        queueFeatures ? queueFeatures->predictedLatencyMs : 0.0,
        queueFeatures ? queueFeatures->precedingBatchCount : 0.0,
        queueFeatures ? queueFeatures->fitStep : 0,
        queueFeatures ? queueFeatures->weights[0] : 0.0,
        queueFeatures ? queueFeatures->weights[1] : 0.0,
        queueFeatures ? queueFeatures->weights[2] : 0.0,
        sample.latestAllocatedLocalBatchId, snapshot ? snapshot->revision : 0);
}

void AgentLatencyKernel::logLatencyTrainingUpdate(
    uint64_t sampleSequence, int64_t eventTimeUs,
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &before,
    const NonLinearPrefillModelR::OptimizerState &optimizerBefore,
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &after,
    const NonLinearPrefillModelR::OptimizerState &optimizerAfter,
    const FitInflightAudit &fitInflight) const
{
    if (!before || !before->valid() || !after || !after->valid()) {
        NAVI_KERNEL_LOG(WARN,
                        "skip latency training update audit with invalid snapshot, "
                        "sequence=%lu",
                        sampleSequence);
        return;
    }
    const auto parameterCount = before->parameters.size();
    const auto parametersBefore = formatValues(before->parameters, parameterCount);
    const auto parametersAfter = formatValues(after->parameters, parameterCount);
    const auto moment1Before = formatValues(optimizerBefore.adamMoment1, parameterCount);
    const auto moment2Before = formatValues(optimizerBefore.adamMoment2, parameterCount);
    const auto moment1After = formatValues(optimizerAfter.adamMoment1, parameterCount);
    const auto moment2After = formatValues(optimizerAfter.adamMoment2, parameterCount);
    const auto residualParametersBefore =
        formatValues(before->residualParameters, before->residualParameters.size());
    const auto residualParametersAfter =
        formatValues(after->residualParameters, after->residualParameters.size());
    NAVI_KERNEL_LOG(INFO,
                    "[navi-latency-training-update] sequence=%lu event_time_us=%ld "
                    "fit_inflight_before_valid=%d fit_inflight_before_requests=%d "
                    "fit_inflight_before_revision=%lu fit_inflight_before_time_us=%ld "
                    "fit_inflight_after_valid=%d fit_inflight_after_requests=%d "
                    "fit_inflight_after_revision=%lu fit_inflight_after_time_us=%ld "
                    "revision_before=%ld revision_after=%ld "
                    "optimizer_step_before=%ld optimizer_step_after=%ld "
                    "parameter_count=%lu parameters_before=[%s] parameters_after=[%s] "
                    "adam_m1_before=[%s] adam_m2_before=[%s] "
                    "adam_m1_after=[%s] adam_m2_after=[%s] "
                    "residual_parameters_before=[%s] residual_parameters_after=[%s] "
                    "residual_last_update_time_us_before=%ld "
                    "residual_last_update_time_us_after=%ld",
                    sampleSequence, eventTimeUs,
                    fitInflight.beforeAvailable ? 1 : 0,
                    fitInflight.before.requestCount, fitInflight.before.revision,
                    fitInflight.beforeTimeUs, fitInflight.afterAvailable ? 1 : 0,
                    fitInflight.after.requestCount, fitInflight.after.revision,
                    fitInflight.afterTimeUs, before->revision, after->revision,
                    optimizerBefore.step, optimizerAfter.step, parameterCount,
                    parametersBefore.c_str(), parametersAfter.c_str(),
                    moment1Before.c_str(), moment2Before.c_str(), moment1After.c_str(),
                    moment2After.c_str(), residualParametersBefore.c_str(),
                    residualParametersAfter.c_str(), before->residualLastUpdateTimeUs,
                    after->residualLastUpdateTimeUs);
}

void AgentLatencyKernel::logQueueDebugSample(
    const AgentLatencyInput &input, int64_t scheduleBatchId,
    const QueueFeatures &features, double agentPredictedQueueLatencyMs,
    double agentPredictedExecutionLatencyMs) const
{
    if (!shouldLogQueueDebugSample(scheduleBatchId)) {
        return;
    }
    NAVI_KERNEL_LOG(INFO,
                    "[navi-queue-debug] schedule_batch_id=%ld local_batch_id=%ld "
                    "request_count=%d ready_time_us=%ld queue_end_canrun_time_us=%ld "
                    "finish_time_us=%ld "
                    "master_queue_ms=%.6f raw_queue_ms=%.6f agent_queue_at_ready_ms=%.6f "
                    "actual_queue_ms=%.6f preceding_batch_count=%.0f "
                    "master_execution_ms=%.6f agent_execution_ms=%.6f "
                    "actual_execution_ms=%.6f queue_fit_step=%lu "
                    "queue_weights=[%.9f,%.9f,%.9f]",
                    scheduleBatchId, input.batchId, features.requestCount,
                    features.readyTimeUs, features.queueEndCanRunTimeUs,
                    input.eventTimeUs, features.masterPredictedQueueLatencyMs,
                    features.rawLatencyMs, agentPredictedQueueLatencyMs,
                    input.backendQueueLatencyMs, features.precedingBatchCount,
                    features.masterPredictedExecutionLatencyMs,
                    agentPredictedExecutionLatencyMs, input.latencyMs, features.fitStep,
                    features.weights[0], features.weights[1], features.weights[2]);
}

void AgentLatencyKernel::buildGraph(navi::GraphBuilder &builder,
                                    uint64_t inputPipeGeneration)
{
    auto latency = builder.node("navi_sched_agent_latency")
                       .kernel("navi_sched.agent_latency_k")
                       .integerAttr("input_pipe_generation",
                                    static_cast<int64_t>(inputPipeGeneration));
    auto historySave = builder.node("navi_sched_agent_history_save")
                           .kernel("navi_sched.agent_history_save_k");
    latency.out("save").to(historySave.in("save"));
    latency.out("finished").asGraphOutput("latency_finished");
    historySave.out("finished").asGraphOutput("history_save_finished");
}

void AgentLatencyKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_latency_k").output("save", "").output("finished", "");
}

bool AgentLatencyKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "queue_learning_rate", _queueLearningRate, _queueLearningRate);
    NAVI_JSONIZE(ctx, "time_compression", _timeCompression, _timeCompression);
    NAVI_JSONIZE(ctx, "input_pipe_generation", _inputPipeGeneration,
                 _inputPipeGeneration);
    return std::isfinite(_queueLearningRate) && _queueLearningRate > 0.0 &&
           std::isfinite(_timeCompression) && _timeCompression > 0.0 &&
           _inputPipeGeneration > 0;
}

navi::ErrorCode AgentLatencyKernel::init(navi::KernelInitContext &ctx)
{
    _samplePipe = ctx.createAsyncPipe();
    if (!_samplePipe) {
        NAVI_KERNEL_LOG(ERROR, "create Agent latency input pipe failed, generation [%ld]",
                        _inputPipeGeneration);
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!_agentLatencyInputR->setInputPipe(static_cast<uint64_t>(_inputPipeGeneration),
                                           _samplePipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "install Agent latency input pipe failed, generation [%ld], "
                        "resourceGeneration [%lu]",
                        _inputPipeGeneration,
                        _agentLatencyInputR->getInputPipeGeneration());
        return navi::EC_INIT_SUB_GRAPH;
    }
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updatedSnapshot;
    if (!_latencyModelR->fitHistoryOnStart(updatedSnapshot)) {
        NAVI_KERNEL_LOG(ERROR,
                        "fit Agent latency history on graph start failed, "
                        "generation [%ld]",
                        _inputPipeGeneration);
        return navi::EC_INIT_RESOURCE;
    }
    _agentReportR->setLatencyModelSnapshot(_latencyModelR->getSnapshot());
    if (updatedSnapshot) {
        _agentReportR->reportLatencyModelUpdated(updatedSnapshot);
    }
    if (derivesUnifiedWorkloadFromBatches()) {
        reportUnifiedWorkload(autil::TimeUtility::currentTime());
    }
    return navi::EC_NONE;
}

navi::ErrorCode AgentLatencyKernel::compute(navi::KernelComputeContext &ctx)
{
    const auto runtimeConfig = _agentSchedulerConfigR->get().config;
    _queueLearningRate = runtimeConfig.queueLearningRate;
    _timeCompression = runtimeConfig.timeCompression;
    navi::DataPtr data;
    bool eof = false;
    _samplePipe->getData(data, eof);
    std::vector<AgentLatencyInput> inputs;
    _agentLatencyInputR->take(inputs);
    bool saveNeeded = false;
    for (auto input : inputs) {
        std::optional<QueueFeatures> completedQueueFeatures;
        const bool observedBatch =
            AgentLatencyInputType::OBSERVED_BATCH_FINISHED == input.type;
        if (!observedBatch) {
            updateExecutionState(input);
        }
        const bool batchDerivedUnified = derivesUnifiedWorkloadFromBatches();
        if (AgentLatencyInputType::BATCH_READY == input.type) {
            if (_batchLatencyUs.count(input.batchId) > 0 ||
                _batchQueueFeatures.count(input.batchId) > 0 ||
                _batchLocalInfos.count(input.batchId) > 0) {
                NAVI_KERNEL_LOG(ERROR, "duplicate batch ready, batchId [%ld]",
                                input.batchId);
                continue;
            }
            auto queueFeatures = getQueueFeatures(input.eventTimeUs);
            queueFeatures.readyTimeUs = input.eventTimeUs;
            if (input.batch) {
                const auto &requests = AgentBatchR::getRequestStats(*input.batch);
                queueFeatures.requestCount = static_cast<int32_t>(requests.size());
                if (!requests.empty()) {
                    queueFeatures.masterPredictedQueueLatencyMs =
                        requests.front().predictedQueueLatencyMs;
                    queueFeatures.masterPredictedExecutionLatencyMs =
                        requests.front().predictedExecutionLatencyMs;
                }
            }
            auto localBatch = input.batch ? input.batch->makeLocalBatchInfo()
                                          : NaviSchedLocalBatchInfo();
            if (input.batch && input.batch->scheduleBatchId > 0 && !localBatch.valid()) {
                NAVI_KERNEL_LOG(ERROR, "invalid local batch metadata, batchId [%ld]",
                                input.batchId);
                continue;
            }
            NaviSchedLocalBatchInfo admittedBatch;
            if (batchDerivedUnified) {
                if (!addUnifiedBatch(input, queueFeatures.unifiedBatch, admittedBatch)) {
                    NAVI_KERNEL_LOG(ERROR, "failed to add unified batch, batchId [%ld]",
                                    input.batchId);
                    continue;
                }
                queueFeatures.contributesToUnifiedWorkload = true;
            }
            auto latencyUs = _agentReportR->usesUnifiedLatencyPrediction()
                                 ? 0
                                 : predictBatchLatencyUs(input.batch);
            _batchLatencyUs.emplace(input.batchId, latencyUs);
            _batchQueueFeatures.emplace(input.batchId, std::move(queueFeatures));
            _queueLatencyUs.fetch_add(latencyUs, std::memory_order_relaxed);
            _batchLocalInfos[input.batchId] = localBatch;
            if (batchDerivedUnified) {
                reportUnifiedWorkload(input.eventTimeUs, admittedBatch);
            }
            reportExecutionLoad(batchDerivedUnified ? NaviSchedLocalBatchInfo()
                                                    : localBatch);
            continue;
        }
        if (AgentLatencyInputType::BATCH_STARTED == input.type) {
            auto queueFeaturesIt = _batchQueueFeatures.find(input.batchId);
            if (_batchQueueFeatures.end() != queueFeaturesIt) {
                queueFeaturesIt->second.queueEndCanRunTimeUs = input.eventTimeUs;
            }
            startRunningBatch(input);
            if (batchDerivedUnified) {
                reportUnifiedWorkload(input.eventTimeUs);
            }
            reportExecutionLoad();
            continue;
        }
        if (AgentLatencyInputType::EXECUTION_LOAD == input.type) {
            if (batchDerivedUnified) {
                reportUnifiedWorkload(input.eventTimeUs);
            }
            reportExecutionLoad();
            continue;
        }
        if (AgentLatencyInputType::BATCH_REQUESTS_ABORTED == input.type) {
            auto latencyIt = _batchLatencyUs.find(input.batchId);
            auto queueFeaturesIt = _batchQueueFeatures.find(input.batchId);
            auto localBatchIt = _batchLocalInfos.find(input.batchId);
            const auto remainingLocalBatch = input.batch
                                                 ? input.batch->makeLocalBatchInfo()
                                                 : NaviSchedLocalBatchInfo();
            const bool tracked = latencyIt != _batchLatencyUs.end() &&
                                 queueFeaturesIt != _batchQueueFeatures.end() &&
                                 localBatchIt != _batchLocalInfos.end() &&
                                 remainingLocalBatch.valid();
            if (!tracked) {
                NAVI_KERNEL_LOG(
                    ERROR,
                    "partial batch abort has no complete latency state, batchId [%ld]",
                    input.batchId);
            } else {
                const bool replaceUnified =
                    batchDerivedUnified &&
                    queueFeaturesIt->second.contributesToUnifiedWorkload;
                const bool unifiedReplaced =
                    !replaceUnified ||
                    replaceUnifiedBatch(input, queueFeaturesIt->second.unifiedBatch);
                if (!unifiedReplaced) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "failed to replace partially aborted unified batch, "
                                    "batchId [%ld]",
                                    input.batchId);
                } else {
                    const auto oldLatencyUs = latencyIt->second;
                    const auto newLatencyUs =
                        _agentReportR->usesUnifiedLatencyPrediction()
                            ? 0
                            : predictBatchLatencyUs(input.batch);
                    latencyIt->second = newLatencyUs;
                    _queueLatencyUs.fetch_add(newLatencyUs - oldLatencyUs,
                                              std::memory_order_relaxed);
                    queueFeaturesIt->second.requestCount = static_cast<int32_t>(
                        AgentBatchR::getRequestStats(*input.batch).size());
                    localBatchIt->second = remainingLocalBatch;
                    auto runningIt = _runningBatches.find(input.batchId);
                    if (runningIt != _runningBatches.end()) {
                        runningIt->second.predictedLatencyUs = newLatencyUs;
                    }
                    if (replaceUnified) {
                        reportUnifiedWorkload(input.eventTimeUs);
                    }
                }
            }
            const auto queuePrediction =
                getQueueLoadPrediction(autil::TimeUtility::currentTime());
            if (0 == _agentReportR->reportBatchRequestsAborted(
                         input.inflightRequests, _latestBatchWaitingRequests,
                         _latestQueuedBatchRequests, _latestRunningRequests,
                         queuePrediction.totalLatencyMs,
                         queuePrediction.decayHeadRemainingMs,
                         _agentBatchR->getTimeoutBatchCount(),
                         _agentBatchR->getAbnormalBatchFinishCount(),
                         input.completedRequestCount, input.eventTimeUs,
                         input.localBatch) &&
                _agentReportR->getSubscriberCount() > 0) {
                NAVI_KERNEL_LOG(ERROR,
                                "failed to report partially aborted batch, "
                                "batchId [%ld]",
                                input.batchId);
            }
            continue;
        }
        if (AgentLatencyInputType::BATCH_FINISHED == input.type || observedBatch ||
            AgentLatencyInputType::BATCH_ABORTED == input.type) {
            auto queueFeaturesIt = _batchQueueFeatures.find(input.batchId);
            const auto hasQueueFeatures = _batchQueueFeatures.end() != queueFeaturesIt;
            if (!observedBatch) {
                _runningBatches.erase(input.batchId);
            }
            if (batchDerivedUnified && !observedBatch) {
                if (!hasQueueFeatures ||
                    !queueFeaturesIt->second.contributesToUnifiedWorkload ||
                    !retireUnifiedBatch(input, queueFeaturesIt->second.unifiedBatch)) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "failed to retire unified batch, batchId [%ld]",
                                    input.batchId);
                } else {
                    reportUnifiedWorkload(input.eventTimeUs);
                }
            }
            const auto sample = input.makeLatencySample();
            const auto agentPredictedExecutionLatencyMs =
                _agentReportR->latencyValidationEnabled() &&
                        AgentLatencyInputType::BATCH_FINISHED == input.type
                    ? _latencyModelR->predict(sample.features)
                    : 0.0;
            if (hasQueueFeatures) {
                completedQueueFeatures = queueFeaturesIt->second;
            }
            const auto agentPredictedQueueLatencyMs =
                hasQueueFeatures ? queueFeaturesIt->second.predictedLatencyMs : 0.0;
            const auto recordedLocalBatch = getLocalBatchInfo(input.batchId, false);
            const auto &localBatch =
                input.localBatch.valid() ? input.localBatch : recordedLocalBatch;
            const auto scheduleBatchId = localBatch.scheduleBatchId;
            if (_agentReportR->latencyValidationEnabled() &&
                shouldRecordLatencyValidation(input.type, input.latencySampleValid,
                                              input.latencyValidationValid)) {
                recordLatencyValidation(
                    input, agentPredictedExecutionLatencyMs,
                    hasQueueFeatures ? &queueFeaturesIt->second : nullptr,
                    agentPredictedQueueLatencyMs);
            }
            if (shouldFitQueueLatency(_agentReportR->usesUnifiedLatencyPrediction(),
                                      input.type, hasQueueFeatures,
                                      input.latencySampleValid)) {
                logQueueDebugSample(input, scheduleBatchId, queueFeaturesIt->second,
                                    agentPredictedQueueLatencyMs,
                                    agentPredictedExecutionLatencyMs);
            }
            if (!_agentReportR->usesUnifiedLatencyPrediction() &&
                AgentLatencyInputType::BATCH_FINISHED == input.type && hasQueueFeatures &&
                input.latencySampleValid) {
                fitQueueLatency(queueFeaturesIt->second, input.backendQueueLatencyMs);
                _latencyValidationStats.queueModelWeights.assign(_queueWeights.begin(),
                                                                 _queueWeights.end());
                _latencyValidationStats.queueModelFitStep = _queueFitStep - 1;
            }
            auto latencyIt = _batchLatencyUs.find(input.batchId);
            if (_batchLatencyUs.end() != latencyIt) {
                _queueLatencyUs.fetch_sub(latencyIt->second, std::memory_order_relaxed);
                _batchLatencyUs.erase(latencyIt);
            }
            if (hasQueueFeatures) {
                _batchQueueFeatures.erase(queueFeaturesIt);
            }
            getLocalBatchInfo(input.batchId, true);
            int64_t cacheHitTokens = 0;
            int64_t cacheMissTokens = 0;
            for (const auto &query : input.queries) {
                cacheHitTokens += query.cacheHitTokenCount;
                cacheMissTokens += query.totalTokenCount - query.cacheHitTokenCount;
            }
            const auto queuePrediction =
                getQueueLoadPrediction(autil::TimeUtility::currentTime());
            const bool reportSubscriberReady = _agentReportR->getSubscriberCount() > 0;
            const auto reportCount = _agentReportR->reportBatchFinished(
                input.inflightRequests, _latestBatchWaitingRequests,
                _latestQueuedBatchRequests, _latestRunningRequests, input.latencyMs,
                queuePrediction.totalLatencyMs, queuePrediction.decayHeadRemainingMs,
                NaviSchedCacheUpdate(), input.completedRequestCount, cacheHitTokens,
                cacheMissTokens, _agentBatchR->getTimeoutBatchCount(),
                _agentBatchR->getAbnormalBatchFinishCount(),
                AgentLatencyInputType::BATCH_ABORTED == input.type, input.eventTimeUs,
                localBatch, agentPredictedExecutionLatencyMs, input.backendQueueLatencyMs,
                _latencyValidationStats);
            if (reportSubscriberReady && reportCount == 0) {
                NAVI_KERNEL_LOG(ERROR,
                                "failed to report batch finished, batchId [%ld], "
                                "scheduleBatchId [%ld], localBatchId [%ld], "
                                "completedRequestCount [%d], latencyMs [%f], "
                                "predictedExecutionLatencyMs [%f], "
                                "backendQueueLatencyMs [%f]",
                                input.batchId, localBatch.scheduleBatchId,
                                localBatch.localBatchId, input.completedRequestCount,
                                input.latencyMs, agentPredictedExecutionLatencyMs,
                                input.backendQueueLatencyMs);
            }
            if (!shouldRecordLatencySample(_agentReportR->usesUnifiedLatencyPrediction(),
                                           input.type, input.unifiedLatencySample,
                                           input.latencySampleValid)) {
                continue;
            }
        }
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> updatedSnapshot;
        bool currentSaveNeeded = false;
        auto sample = input.makeLatencySample();
        sample.localBatchId = input.batchId;
        sample.latestAllocatedLocalBatchId = _agentBatchR->getLatestLocalBatchId();
        const auto sampleSequence = ++_latencyTrainingSampleSequence;
        const auto auditTraining = _agentReportR->latencyValidationEnabled();
        const auto snapshotBefore = _latencyModelR->getSnapshot();
        const auto operatingPoint =
            snapshotBefore && snapshotBefore->valid()
                ? _latencyModelR->getNonlinearOperatingPoint(
                      sample.features, snapshotBefore->parameters)
                : std::nullopt;
        if (operatingPoint && _metricsReporterR) {
            const auto &reporter = _metricsReporterR->getReporter();
            if (reporter) {
                const char *region = operatingPoint->region < 0 ? "negative_far"
                                     : operatingPoint->region > 0 ? "positive_far"
                                                                  : "transition";
                const kmonitor::MetricsTags tags("region", region);
                const auto report = [&reporter, &tags](const char *name,
                                                        double value) {
                    reporter->report(value, name, kmonitor::MetricType::GAUGE,
                                     &tags, kmonitor::MetricLevel::FATAL);
                };
                report("agent.latency_nonlinear_u", operatingPoint->u);
                report("agent.latency_nonlinear_curvature",
                       operatingPoint->curvature);
                report("agent.latency_nonlinear_r", operatingPoint->r);
            }
        }
        NonLinearPrefillModelR::OptimizerState optimizerBefore;
        if (auditTraining) {
            optimizerBefore = _latencyModelR->getOptimizerState();
            logLatencyTrainingSample(
                input, sampleSequence, sample,
                completedQueueFeatures ? &completedQueueFeatures.value() : nullptr,
                snapshotBefore, operatingPoint);
        }
        FitInflightAudit fitInflight;
        if (auditTraining) {
            fitInflight.beforeTimeUs = autil::TimeUtility::currentTime();
            fitInflight.beforeAvailable =
                _agentReportR->getLiveInflightStats(fitInflight.before);
        }
        if (!_latencyModelR->recordSample(sample, updatedSnapshot, currentSaveNeeded,
                                          input.getLatencyTrainingPhase())) {
            // Reporting a bad sample must not stop the resident Agent graph.
            NAVI_KERNEL_LOG(WARN, "discard invalid latency sample");
            continue;
        }
        if (auditTraining) {
            fitInflight.afterAvailable =
                _agentReportR->getLiveInflightStats(fitInflight.after);
            fitInflight.afterTimeUs = autil::TimeUtility::currentTime();
        }
        saveNeeded = saveNeeded || currentSaveNeeded;
        if (auditTraining) {
            const auto optimizerAfter = _latencyModelR->getOptimizerState();
            if (optimizerAfter.step != optimizerBefore.step) {
                logLatencyTrainingUpdate(
                    sampleSequence, sample.eventTimeUs, snapshotBefore, optimizerBefore,
                    updatedSnapshot ? updatedSnapshot : _latencyModelR->getSnapshot(),
                    optimizerAfter, fitInflight);
            }
        }
        if (updatedSnapshot) {
            _agentReportR->reportLatencyModelUpdated(updatedSnapshot);
        }
    }
    if (saveNeeded || eof) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, eof);
    }
    if (eof) {
        ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr, true);
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentLatencyKernel);

} // namespace navi_sched

#include "rtp_llm/flexlb/navi_sched/resource/LatencyHistorySampler.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"

#include "rapidjson/document.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace navi_sched {
namespace {

constexpr uint32_t MERGE_COUNT = 1U << 0;
constexpr uint32_t MERGE_HIT = 1U << 1;
constexpr uint32_t MERGE_MISS = 1U << 2;
constexpr uint32_t MERGE_MISS_SQUARE = 1U << 3;
constexpr uint32_t MERGE_HIT_MISS = 1U << 4;
constexpr uint32_t MERGE_ALL = (1U << 5) - 1;
constexpr const char *TRAINING_LOSS = "log1p_squared";

struct InputData {
    std::vector<NaviSchedLatencySample> samples;
    size_t rampSampleCount = 0;
};

struct RapidRefitConfig {
    bool enabled = false;
    size_t distributionWindow = 64;
    size_t stableWindows = 2;
    size_t maxHistory = 2000;
    size_t maxTriggers = 8;
    size_t currentBucketSampleCount = 2;
    double currentSampleWeight = 4.0;
    double minDistributionShift = 0.5;
    double maxStableWindowShift = 0.25;
};

enum class TrainingScope {
    ALL,
    LOW_PRESSURE,
    HIGH_PRESSURE,
};

const char *trainingScopeName(TrainingScope scope)
{
    switch (scope) {
    case TrainingScope::ALL:
        return "all";
    case TrainingScope::LOW_PRESSURE:
        return "low_pressure";
    case TrainingScope::HIGH_PRESSURE:
        return "high_pressure";
    }
    return "unknown";
}

bool isInTrainingScope(const NaviSchedLatencySample &sample,
                       TrainingScope scope, double lowPressureMaxMs)
{
    if (scope == TrainingScope::ALL) {
        return true;
    }
    const auto isLowPressure = sample.latencyMs <= lowPressureMaxMs;
    return scope == TrainingScope::LOW_PRESSURE ? isLowPressure
                                                 : !isLowPressure;
}

struct Metrics {
    std::vector<double> absoluteErrors;
    std::vector<double> relativeErrors;
    double targetSum = 0.0;
    double absoluteErrorSum = 0.0;
    double signedErrorSum = 0.0;

    void add(double prediction, double target)
    {
        const auto error = prediction - target;
        const auto absoluteError = std::abs(error);
        absoluteErrors.emplace_back(absoluteError);
        relativeErrors.emplace_back(absoluteError / std::max(1.0, target));
        targetSum += target;
        absoluteErrorSum += absoluteError;
        signedErrorSum += error;
    }
};

struct ReplayResult {
    std::map<int32_t, Metrics> baseBucketMetrics;
    std::map<int32_t, Metrics> bucketMetrics;
    std::map<std::string, Metrics> baseStageMetrics;
    std::map<std::string, Metrics> stageMetrics;
    std::map<std::string, Metrics> basePressureStageMetrics;
    std::map<std::string, Metrics> pressureStageMetrics;
    Metrics baseAllMetrics;
    Metrics allMetrics;
    Metrics baseLowPressureMetrics;
    Metrics lowPressureMetrics;
    Metrics baseHighPressureMetrics;
    Metrics highPressureMetrics;
    std::vector<double> baseZeroPoints;
    std::vector<double> finalZeroPoints;
    std::vector<double> baseClassificationGaps;
    std::vector<double> classificationGaps;
    std::vector<double> finalParameters;
    std::map<std::string, Metrics> rapidAdaptationMetrics;
    size_t rapidTriggerCount = 0;
    size_t rapidReplayedSamples = 0;
};

double distributionShift(const std::vector<int32_t> &left,
                         const std::vector<int32_t> &right)
{
    std::map<int32_t, size_t> leftCounts;
    std::map<int32_t, size_t> rightCounts;
    for (const auto bucket : left) {
        ++leftCounts[bucket];
    }
    for (const auto bucket : right) {
        ++rightCounts[bucket];
    }
    const auto leftSize = static_cast<double>(std::max<size_t>(1, left.size()));
    const auto rightSize = static_cast<double>(std::max<size_t>(1, right.size()));
    std::map<int32_t, bool> buckets;
    for (const auto &[bucket, unused] : leftCounts) {
        buckets[bucket] = true;
    }
    for (const auto &[bucket, unused] : rightCounts) {
        buckets[bucket] = true;
    }
    double totalVariation = 0.0;
    for (const auto &[bucket, unused] : buckets) {
        totalVariation += std::abs(leftCounts[bucket] / leftSize -
                                   rightCounts[bucket] / rightSize);
    }
    return 0.5 * totalVariation;
}

class StableDistributionDetector {
public:
    explicit StableDistributionDetector(const RapidRefitConfig &config)
        : _config(config)
    {
    }

    bool observe(int32_t bucket)
    {
        _current.emplace_back(bucket);
        if (_current.size() < _config.distributionWindow) {
            return false;
        }
        auto window = std::move(_current);
        _current.clear();
        if (_reference.empty()) {
            _reference = window;
            _previous = std::move(window);
            return false;
        }
        const auto referenceShift = distributionShift(_reference, window);
        const auto previousShift = distributionShift(_previous, window);
        if (referenceShift >= _config.minDistributionShift &&
            (_previous.empty() ||
             previousShift <= _config.maxStableWindowShift)) {
            ++_stableWindowCount;
        } else if (referenceShift >= _config.minDistributionShift) {
            _stableWindowCount = 1;
        } else {
            _stableWindowCount = 0;
        }
        _previous = window;
        if (_stableWindowCount < _config.stableWindows) {
            return false;
        }
        _reference = std::move(window);
        _previous.clear();
        _stableWindowCount = 0;
        return true;
    }

private:
    const RapidRefitConfig &_config;
    std::vector<int32_t> _current;
    std::vector<int32_t> _reference;
    std::vector<int32_t> _previous;
    size_t _stableWindowCount = 0;
};

std::string variantName(uint32_t mergeMask)
{
    if (mergeMask == 0) {
        return "split";
    }
    if (mergeMask == MERGE_ALL) {
        return "shared_all";
    }
    std::string name = "merge";
    for (const auto &[bit, suffix] :
         {std::pair{MERGE_COUNT, "_count"}, std::pair{MERGE_HIT, "_hit"},
          std::pair{MERGE_MISS, "_miss"},
          std::pair{MERGE_MISS_SQUARE, "_miss_square"},
          std::pair{MERGE_HIT_MISS, "_hit_x_miss"}}) {
        if (mergeMask & bit) {
            name += suffix;
        }
    }
    return name;
}

NaviSchedLatencySample transformSample(const NaviSchedLatencySample &source,
                                       uint32_t mergeMask)
{
    if (mergeMask == 0) {
        return source;
    }
    const auto &features = source.features;
    auto runningCount = features.getRunningBatchSize();
    auto runningHit = features.getRunningCacheHitTokenCount();
    auto runningMiss = features.getRunningCacheMissTokenCount();
    auto runningSquare = features.getRunningCacheMissTokenSquareSum();
    auto runningProduct = features.getRunningCacheHitMissTokenProductSum();
    auto waitingCount = features.getWaitingBatchSize();
    auto waitingHit = features.getWaitingCacheHitTokenCount();
    auto waitingMiss = features.getWaitingCacheMissTokenCount();
    auto waitingSquare = features.getWaitingCacheMissTokenSquareSum();
    auto waitingProduct = features.getWaitingCacheHitMissTokenProductSum();

    if (mergeMask & MERGE_COUNT) {
        runningCount += waitingCount;
        waitingCount = 0;
    }
    if (mergeMask & MERGE_HIT) {
        runningHit += waitingHit;
        waitingHit = 0;
    }
    if (mergeMask & MERGE_MISS) {
        runningMiss += waitingMiss;
        waitingMiss = 0;
    }
    if (mergeMask & MERGE_MISS_SQUARE) {
        runningSquare += waitingSquare;
        waitingSquare = 0.0;
    }
    if (mergeMask & MERGE_HIT_MISS) {
        runningProduct += waitingProduct;
        waitingProduct = 0.0;
    }

    NaviSchedLatencySample transformed;
    if (!transformed.features.appendCapacitySplitAggregate(
            features.getBatchCount(), runningCount, runningHit, runningMiss,
            runningSquare, runningProduct, waitingCount, waitingHit, waitingMiss,
            waitingSquare, waitingProduct) ||
        !transformed.features.appendPrecedingRequestCount(
            features.getPrecedingRequestCount()) ||
        !transformed.features.appendRunningElapsedTimeMs(
            features.getRunningElapsedTimeMs())) {
        throw std::runtime_error("failed to transform latency sample");
    }
    transformed.latencyMs = source.latencyMs;
    transformed.eventTimeUs = source.eventTimeUs;
    return transformed;
}

NaviSchedLatencySample moveOneWaitingRequestToRunning(
    const NaviSchedLatencySample &sample)
{
    const auto sourceWaitingCount = sample.features.getWaitingBatchSize();
    if (sourceWaitingCount <= 0) {
        return sample;
    }
    const auto &features = sample.features;
    const auto movedHit = features.getWaitingCacheHitTokenCount() /
                          sourceWaitingCount;
    const auto movedMiss = features.getWaitingCacheMissTokenCount() /
                           sourceWaitingCount;
    const auto movedSquare = features.getWaitingCacheMissTokenSquareSum() /
                             static_cast<double>(sourceWaitingCount);
    const auto movedProduct = features.getWaitingCacheHitMissTokenProductSum() /
                              static_cast<double>(sourceWaitingCount);
    NaviSchedLatencySample moved;
    if (!moved.features.appendCapacitySplitAggregate(
            features.getBatchCount(),
            features.getRunningBatchSize() + 1,
            features.getRunningCacheHitTokenCount() + movedHit,
            features.getRunningCacheMissTokenCount() + movedMiss,
            features.getRunningCacheMissTokenSquareSum() + movedSquare,
            features.getRunningCacheHitMissTokenProductSum() + movedProduct,
            features.getWaitingBatchSize() - 1,
            features.getWaitingCacheHitTokenCount() - movedHit,
            features.getWaitingCacheMissTokenCount() - movedMiss,
            features.getWaitingCacheMissTokenSquareSum() - movedSquare,
            features.getWaitingCacheHitMissTokenProductSum() - movedProduct) ||
        !moved.features.appendPrecedingRequestCount(
            features.getPrecedingRequestCount()) ||
        !moved.features.appendRunningElapsedTimeMs(
            features.getRunningElapsedTimeMs()) ||
        !moved.features.appendRunningDecodeProgress(
            features.getRunningDecodeRequestCount(),
            features.getRunningDecodeGeneratedTokenCount())) {
        throw std::runtime_error("failed to move waiting request");
    }
    moved.latencyMs = sample.latencyMs;
    moved.eventTimeUs = sample.eventTimeUs;
    return moved;
}

NaviSchedLatencySample readSample(const rapidjson::Value &event)
{
    const auto &features = event["features"];
    if (!features.IsArray() ||
        (features.Size() != 12 && features.Size() != 13 &&
         features.Size() != 15)) {
        throw std::runtime_error("invalid sample features");
    }
    NaviSchedLatencySample sample;
    if (!sample.features.appendCapacitySplitAggregate(
            features[0].GetInt64(), features[1].GetInt64(),
            features[2].GetInt64(), features[3].GetInt64(),
            features[4].GetDouble(), features[5].GetDouble(),
            features[6].GetInt64(), features[7].GetInt64(),
            features[8].GetInt64(), features[9].GetDouble(),
            features[10].GetDouble()) ||
        !sample.features.appendPrecedingRequestCount(features[11].GetInt64()) ||
        (features.Size() >= 13 &&
         !sample.features.appendRunningElapsedTimeMs(features[12].GetDouble())) ||
        (features.Size() == 15 &&
         !sample.features.appendRunningDecodeProgress(
             features[13].GetInt64(), features[14].GetInt64()))) {
        throw std::runtime_error("invalid sample aggregate");
    }
    sample.latencyMs = event["target_ms"].GetDouble();
    sample.eventTimeUs = event["event_time_us"].GetInt64();
    return sample;
}

InputData readInput(const std::filesystem::path &path)
{
    InputData data;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        rapidjson::Document event;
        event.Parse(line.c_str());
        if (event.HasParseError() || !event.IsObject() ||
            !event.HasMember("type")) {
            throw std::runtime_error("invalid replay event in " + path.string());
        }
        const std::string type = event["type"].GetString();
        if (type == "sample") {
            data.samples.emplace_back(readSample(event));
        } else if (type == "update" && data.rampSampleCount == 0) {
            data.rampSampleCount = data.samples.size();
        }
    }
    if (data.rampSampleCount == 0 || data.rampSampleCount >= data.samples.size()) {
        throw std::runtime_error("missing usable ramp boundary in " + path.string());
    }
    return data;
}

double quantile(std::vector<double> values, double probability)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto position = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const auto upper = static_cast<size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

bool applyBatches(NonLinearPrefillModelR &model,
                  const std::vector<std::vector<NaviSchedLatencySample>> &batches,
                  std::vector<double> &parameters,
                  std::vector<double> &residualParameters,
                  int64_t &residualTimeUs, double learningRateScale)
{
    for (const auto &batch : batches) {
        std::vector<double> updated;
        if (!model.fit(batch, parameters, updated, learningRateScale, 1.0)) {
            std::cerr << "model fit failed, batch_size=" << batch.size()
                      << '\n';
            return false;
        }
        parameters = std::move(updated);
        const auto &current = batch.back();
        const auto base = model.predictConfigured(current.features, parameters);
        if (!LatencyResidualModel::update(base, current.latencyMs,
                                          residualParameters, residualTimeUs,
                                          current.eventTimeUs)) {
            std::cerr << "residual update failed, base=" << base
                      << " target=" << current.latencyMs << '\n';
            return false;
        }
    }
    return true;
}

bool applyRapidRefit(NonLinearPrefillModelR &model,
                     const LatencyHistorySampler::History &history,
                     std::vector<double> &parameters,
                     const RapidRefitConfig &config,
                     size_t &replayedSamples)
{
    if (history.empty()) {
        return false;
    }
    std::vector<NaviSchedLatencySample> retained(history.begin(), history.end());
    if (retained.size() > config.maxHistory) {
        retained.erase(retained.begin(),
                       retained.end() - config.maxHistory);
    }
    const auto currentSampleCount =
        std::min(retained.size(), config.distributionWindow * config.stableWindows);
    const auto currentBegin = retained.size() - currentSampleCount;

    // Rebuild the online sampler in timestamp order. Each replay step has one
    // current sample and the same per-bucket context used by online Adam.
    LatencyHistorySampler replaySampler(1);
    replaySampler.configure(config.maxHistory, 64);
    replayedSamples = 0;
    for (size_t index = 0; index < retained.size(); ++index) {
        const auto &sample = retained[index];
        replaySampler.append(sample);
        const auto batches = replaySampler.getOnlineFitBatches(
            1, config.currentBucketSampleCount);
        if (batches.size() != 1 || batches.front().empty()) {
            return false;
        }
        std::vector<double> updatedParameters;
        const auto currentSampleWeight =
            index >= currentBegin ? config.currentSampleWeight : 1.0;
        if (!model.fit(batches.front(), parameters, updatedParameters, 1.0,
                       currentSampleWeight)) {
            return false;
        }
        parameters = std::move(updatedParameters);
        ++replayedSamples;
    }
    return replayedSamples > 0;
}

void addRapidAdaptationMetrics(ReplayResult &result, size_t triggerIndex,
                               size_t samplesSinceTrigger, double prediction,
                               double basePrediction, double target,
                               int32_t bucket, double lowPressureMaxMs)
{
    constexpr std::array<size_t, 5> WINDOW_ENDS = {64, 128, 256, 512, 1024};
    if (samplesSinceTrigger == 0 || samplesSinceTrigger > WINDOW_ENDS.back()) {
        return;
    }
    const auto end = *std::find_if(
        WINDOW_ENDS.begin(), WINDOW_ENDS.end(),
        [samplesSinceTrigger](size_t value) { return samplesSinceTrigger <= value; });
    const auto begin = end == 64 ? 1 : end / 2 + 1;
    const auto prefix = "trigger_" + std::to_string(triggerIndex) + "_post_" +
                        (begin < 10 ? "000" : begin < 100 ? "00" : begin < 1000 ? "0" : "") +
                        std::to_string(begin) + "_" +
                        (end < 10 ? "000" : end < 100 ? "00" : end < 1000 ? "0" : "") +
                        std::to_string(end) + "_";
    result.rapidAdaptationMetrics[prefix + "all"].add(prediction, target);
    result.rapidAdaptationMetrics[prefix + "base_all"].add(basePrediction, target);
    const auto pressure = target <= lowPressureMaxMs ? "low" : "high";
    result.rapidAdaptationMetrics[prefix + pressure].add(prediction, target);
    result.rapidAdaptationMetrics[prefix + "base_" + pressure].add(
        basePrediction, target);
    result.rapidAdaptationMetrics[prefix + "bucket_" + std::to_string(bucket)].add(
        prediction, target);
    result.rapidAdaptationMetrics[prefix + "base_bucket_" +
                                   std::to_string(bucket)].add(basePrediction,
                                                               target);
}

ReplayResult replay(const InputData &input, uint32_t mergeMask, uint64_t seed,
                    double learningRateScale, TrainingScope trainingScope,
                    double lowPressureMaxMs, size_t currentBucketSampleCount,
                    const RapidRefitConfig &rapidConfig)
{
    NonLinearPrefillModelR model;
    model._learningRate = 0.011;
    model._useBatchCountFeature = false;
    model._useCapacitySplitFeature = true;
    model._zeroAnchorMaxMs = lowPressureMaxMs;
    auto parameters = model.getInitialParameters();
    auto residualParameters = LatencyResidualModel::makeInitialParameters();
    int64_t residualTimeUs = 0;
    LatencyHistorySampler sampler(seed);
    sampler.configure(2000, 64);
    size_t pendingSampleCount = 0;
    ReplayResult result;
    std::optional<StableDistributionDetector> rapidDetector;
    if (trainingScope == TrainingScope::ALL) {
        rapidDetector.emplace(rapidConfig);
    }
    size_t rapidTriggerId = 0;
    size_t samplesSinceRapidTrigger = 0;
    bool rapidTriggerPending = false;

    for (size_t index = 0; index < input.samples.size(); ++index) {
        auto sample = transformSample(input.samples[index], mergeMask);
        if (index >= input.rampSampleCount) {
            const auto scoredIndex = index - input.rampSampleCount;
            const auto scoredCount = input.samples.size() - input.rampSampleCount;
            const auto segment = std::min<size_t>(
                9, scoredIndex * 10 / std::max<size_t>(1, scoredCount));
            const auto stage = scoredIndex < 256
                ? "early_0001_0256"
                : (scoredIndex < 1024 ? "early_0257_1024" : "steady_after_1024");
            const auto segmentName = "segment_" + std::to_string(segment);
            const auto base = model.predictConfigured(sample.features, parameters);
            const auto prediction = LatencyResidualModel::evaluate(
                base, residualParameters, residualTimeUs,
                sample.eventTimeUs).predictionMs;
            result.baseAllMetrics.add(base, sample.latencyMs);
            result.allMetrics.add(prediction, sample.latencyMs);
            result.baseStageMetrics[stage].add(base, sample.latencyMs);
            result.stageMetrics[stage].add(prediction, sample.latencyMs);
            result.baseStageMetrics[segmentName].add(base, sample.latencyMs);
            result.stageMetrics[segmentName].add(prediction, sample.latencyMs);
            const auto lowPressure = sample.latencyMs <= lowPressureMaxMs;
            const std::string pressurePrefix =
                lowPressure ? "low_pressure_" : "high_pressure_";
            result.basePressureStageMetrics[pressurePrefix + stage].add(
                base, sample.latencyMs);
            result.pressureStageMetrics[pressurePrefix + stage].add(
                prediction, sample.latencyMs);
            result.basePressureStageMetrics[pressurePrefix + segmentName].add(
                base, sample.latencyMs);
            result.pressureStageMetrics[pressurePrefix + segmentName].add(
                prediction, sample.latencyMs);
            if (lowPressure) {
                result.baseLowPressureMetrics.add(base, sample.latencyMs);
                result.lowPressureMetrics.add(prediction, sample.latencyMs);
            } else {
                result.baseHighPressureMetrics.add(base, sample.latencyMs);
                result.highPressureMetrics.add(prediction, sample.latencyMs);
            }
            result.baseBucketMetrics[LatencyHistorySampler::getBucketIndex(
                sample.latencyMs)].add(base, sample.latencyMs);
            result.bucketMetrics[LatencyHistorySampler::getBucketIndex(
                sample.latencyMs)].add(prediction, sample.latencyMs);
            const auto baseZeroPoint =
                NonLinearPrefillModelR::calculateZeroPoint(parameters);
            result.baseZeroPoints.emplace_back(baseZeroPoint);
            result.finalZeroPoints.emplace_back(
                LatencyResidualModel::evaluate(
                    baseZeroPoint, residualParameters, residualTimeUs,
                    sample.eventTimeUs).predictionMs);
            if (rapidTriggerId > 0) {
                ++samplesSinceRapidTrigger;
                addRapidAdaptationMetrics(
                    result, rapidTriggerId, samplesSinceRapidTrigger, prediction,
                    base, sample.latencyMs,
                    LatencyHistorySampler::getBucketIndex(sample.latencyMs),
                    lowPressureMaxMs);
            }
            if (input.samples[index].features.getWaitingBatchSize() > 0) {
                const auto movedSource =
                    moveOneWaitingRequestToRunning(input.samples[index]);
                const auto oneMoved = transformSample(movedSource, mergeMask);
                const auto oneMovedBase = model.predictConfigured(
                    oneMoved.features, parameters);
                const auto oneMovedPrediction = LatencyResidualModel::evaluate(
                    oneMovedBase, residualParameters, residualTimeUs,
                    sample.eventTimeUs).predictionMs;
                result.baseClassificationGaps.emplace_back(
                    std::abs(base - oneMovedBase));
                result.classificationGaps.emplace_back(
                    std::abs(prediction - oneMovedPrediction));
            }
        }

        if (!isInTrainingScope(sample, trainingScope, lowPressureMaxMs)) {
            continue;
        }
        sampler.append(sample);
        ++pendingSampleCount;
        rapidTriggerPending = rapidTriggerPending ||
            (rapidDetector && index >= input.rampSampleCount &&
             rapidDetector->observe(
                 LatencyHistorySampler::getBucketIndex(sample.latencyMs)));
        // Ramp samples are collected but must not be fitted until the first
        // normal sample, which triggers the same replay as the agent path.
        const auto rampFinished = index >= input.rampSampleCount;
        const auto rampReplay = index == input.rampSampleCount;
        if (!rampFinished || (!rampReplay && pendingSampleCount < 4)) {
            continue;
        }
        auto batches = sampler.getOnlineFitBatches(pendingSampleCount,
                                                   currentBucketSampleCount);
        if (!applyBatches(model, batches, parameters, residualParameters,
                          residualTimeUs, learningRateScale)) {
            throw std::runtime_error("online fit failed");
        }
        pendingSampleCount = 0;

        if (rapidTriggerPending) {
            if (rapidTriggerId < rapidConfig.maxTriggers) {
                if (rapidConfig.enabled) {
                    size_t replayedSamples = 0;
                    if (!applyRapidRefit(model, *sampler.getHistory(),
                                         parameters, rapidConfig,
                                         replayedSamples)) {
                        throw std::runtime_error("rapid refit failed");
                    }
                    result.rapidReplayedSamples += replayedSamples;
                }
                ++result.rapidTriggerCount;
            }
            ++rapidTriggerId;
            samplesSinceRapidTrigger = 0;
            rapidTriggerPending = false;
        }
    }
    result.finalParameters = parameters;
    return result;
}

void writeMetrics(std::ostream &output, const std::string &host,
                  uint32_t mergeMask, uint64_t seed,
                  TrainingScope trainingScope, const std::string &predictionKind,
                  const std::string &scope,
                  const Metrics &metrics, const std::vector<double> &zeroPoints,
                  const std::vector<double> &classificationGaps)
{
    const auto count = metrics.absoluteErrors.size();
    if (count == 0) {
        return;
    }
    const auto denominator = static_cast<double>(count);
    const auto zeroMin = zeroPoints.empty()
        ? 0.0
        : *std::min_element(zeroPoints.begin(), zeroPoints.end());
    const auto zeroMax = zeroPoints.empty()
        ? 0.0
        : *std::max_element(zeroPoints.begin(), zeroPoints.end());
    const auto classificationGapMean = classificationGaps.empty()
        ? 0.0
        : std::accumulate(classificationGaps.begin(), classificationGaps.end(), 0.0) /
              static_cast<double>(classificationGaps.size());
    const auto classificationGapMax = classificationGaps.empty()
        ? 0.0
        : *std::max_element(classificationGaps.begin(), classificationGaps.end());
    output << host << '\t' << variantName(mergeMask) << '\t' << mergeMask
           << '\t' << seed << '\t' << TRAINING_LOSS << '\t'
           << trainingScopeName(trainingScope) << '\t'
           << predictionKind << '\t'
           << scope << '\t' << count << '\t'
           << 100.0 * std::accumulate(metrics.relativeErrors.begin(),
                                      metrics.relativeErrors.end(), 0.0) /
                  denominator
           << '\t' << 100.0 * metrics.absoluteErrorSum /
                          std::max(1.0, metrics.targetSum)
           << '\t' << metrics.absoluteErrorSum / denominator << '\t'
           << quantile(metrics.absoluteErrors, 0.50) << '\t'
           << quantile(metrics.absoluteErrors, 0.95) << '\t'
           << quantile(metrics.relativeErrors, 0.95) * 100.0 << '\t'
           << metrics.signedErrorSum / denominator << '\t' << zeroMin << '\t'
           << zeroMax << '\t' << zeroMax - zeroMin << '\t'
           << (zeroPoints.empty() ? 0.0 : zeroPoints.back()) << '\t'
           << classificationGaps.size() << '\t'
           << classificationGapMean << '\t'
           << quantile(classificationGaps, 0.95) << '\t'
           << classificationGapMax << '\t' << -1 << '\t' << 0.0 << '\n';
}

void writeParameters(std::ostream &output, const std::string &host,
                     uint32_t mergeMask, uint64_t seed,
                     TrainingScope trainingScope,
                     const std::vector<double> &parameters)
{
    for (size_t index = 0; index < parameters.size(); ++index) {
        output << host << '\t' << variantName(mergeMask) << '\t' << mergeMask
               << '\t' << seed << '\t' << TRAINING_LOSS << '\t'
               << trainingScopeName(trainingScope) << "\tparameter\tp" << index;
        for (size_t column = 0; column < 16; ++column) {
            output << '\t' << 0.0;
        }
        output << '\t' << index << '\t' << parameters[index] << '\n';
    }
}

void writeRapidSummary(std::ostream &output, const std::string &host,
                       uint32_t mergeMask, uint64_t seed,
                       const ReplayResult &result)
{
    output << host << '\t' << variantName(mergeMask) << '\t' << mergeMask
           << '\t' << seed << '\t' << TRAINING_LOSS << "\tall\trapid_trigger\t"
           << "rapid_trigger_summary\t" << result.rapidTriggerCount
           << "\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t-2\t"
           << result.rapidReplayedSamples << '\n';
}

void runFile(const std::filesystem::path &path, uint64_t seed,
             int32_t requestedMask, double learningRateScale,
             double lowPressureMaxMs, size_t currentBucketSampleCount,
             const RapidRefitConfig &rapidConfig, std::ostream &output)
{
    const auto input = readInput(path);
    const auto host = path.stem().string();
    for (const auto trainingScope : {TrainingScope::ALL,
                                     TrainingScope::LOW_PRESSURE,
                                     TrainingScope::HIGH_PRESSURE}) {
        for (uint32_t mergeMask = 0; mergeMask <= MERGE_ALL; ++mergeMask) {
            if (requestedMask >= 0 &&
                mergeMask != static_cast<uint32_t>(requestedMask)) {
                continue;
            }
            const auto result = replay(input, mergeMask, seed, learningRateScale,
                                       trainingScope, lowPressureMaxMs,
                                       currentBucketSampleCount, rapidConfig);
            writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                         "eval_all", result.baseAllMetrics,
                         result.baseZeroPoints, result.baseClassificationGaps);
            writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                         "eval_all", result.allMetrics,
                         result.finalZeroPoints, result.classificationGaps);
            writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                         "eval_low_pressure", result.baseLowPressureMetrics, {},
                         {});
            writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                         "eval_low_pressure", result.lowPressureMetrics, {}, {});
            writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                         "eval_high_pressure", result.baseHighPressureMetrics, {},
                         {});
            writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                         "eval_high_pressure", result.highPressureMetrics, {}, {});
            for (const auto &[bucket, metrics] : result.baseBucketMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                             "bucket_" + std::to_string(bucket), metrics, {}, {});
            }
            for (const auto &[bucket, metrics] : result.bucketMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                             "bucket_" + std::to_string(bucket), metrics, {}, {});
            }
            for (const auto &[stage, metrics] : result.baseStageMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                             stage, metrics, {}, {});
            }
            for (const auto &[stage, metrics] : result.stageMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                             stage, metrics, {}, {});
            }
            for (const auto &[stage, metrics] :
                 result.basePressureStageMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "base",
                             stage, metrics, {}, {});
            }
            for (const auto &[stage, metrics] : result.pressureStageMetrics) {
                writeMetrics(output, host, mergeMask, seed, trainingScope, "final",
                             stage, metrics, {}, {});
            }
            writeParameters(output, host, mergeMask, seed, trainingScope,
                            result.finalParameters);
            if (trainingScope == TrainingScope::ALL) {
                for (const auto &[scope, metrics] :
                     result.rapidAdaptationMetrics) {
                    writeMetrics(output, host, mergeMask, seed, trainingScope,
                                 "rapid", scope, metrics, {}, {});
                }
                if (result.rapidTriggerCount > 0 || rapidConfig.enabled) {
                    writeRapidSummary(output, host, mergeMask, seed, result);
                }
            }
        }
    }
}

} // namespace
} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 14) {
        std::cerr << "usage: latency_formula_experiment INPUT_DIR OUTPUT SEED "
                     "[LEARNING_RATE_SCALE] [MERGE_MASK] "
                     "[LOW_PRESSURE_MAX_MS] [CURRENT_BUCKET_SAMPLE_COUNT] "
                     "[RAPID_REFIT_ENABLED] "
                     "[RAPID_WINDOW] [RAPID_STABLE_WINDOWS] "
                     "[RAPID_MIN_SHIFT] [RAPID_WEIGHT] [RAPID_MAX_TRIGGERS]\n";
        return 2;
    }
    try {
        std::vector<std::filesystem::path> paths;
        for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() == ".jsonl") {
                paths.emplace_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        std::ofstream output(argv[2]);
        output << std::setprecision(17)
               << "host\tvariant\tmerge_mask\tseed\tloss\ttraining_scope"
                  "\tprediction_kind\tscope"
                  "\tcount\tmape_pct\twmape_pct"
                  "\tmae_ms\tp50_ae_ms\tp95_ae_ms\tp95_ape_pct\tbias_ms"
                  "\tzero_min_ms\tzero_max_ms\tzero_range_ms\tzero_final_ms"
                  "\tclassification_gap_count"
                  "\tclassification_gap_mean_ms\tclassification_gap_p95_ms"
                  "\tclassification_gap_max_ms\tparameter_index"
                  "\tparameter_value\n";
        const auto seed = std::stoull(argv[3]);
        const auto learningRateScale = argc >= 5 ? std::stod(argv[4]) : 6.0;
        const auto requestedMask = argc >= 6 ? std::stoi(argv[5]) : -1;
        const auto lowPressureMaxMs = argc >= 7 ? std::stod(argv[6]) : 256.0;
        const auto currentBucketSampleCount =
            argc >= 8 ? std::stoull(argv[7]) : 0;
        navi_sched::RapidRefitConfig rapidConfig;
        rapidConfig.enabled = argc >= 9 ? std::stoi(argv[8]) != 0 : false;
        rapidConfig.distributionWindow = argc >= 10 ? std::stoull(argv[9]) : 64;
        rapidConfig.stableWindows = argc >= 11 ? std::stoull(argv[10]) : 2;
        rapidConfig.minDistributionShift =
            argc >= 12 ? std::stod(argv[11]) : 0.5;
        rapidConfig.currentSampleWeight =
            argc >= 13 ? std::stod(argv[12]) : 4.0;
        rapidConfig.maxTriggers = argc >= 14 ? std::stoull(argv[13]) : 8;
        rapidConfig.currentBucketSampleCount = currentBucketSampleCount;
        if (!std::isfinite(learningRateScale) || learningRateScale <= 0.0) {
            throw std::runtime_error("invalid learning rate scale");
        }
        if (!std::isfinite(lowPressureMaxMs) || lowPressureMaxMs < 0.0) {
            throw std::runtime_error("invalid low pressure cutoff");
        }
        if (rapidConfig.enabled &&
            (rapidConfig.distributionWindow == 0 ||
             rapidConfig.stableWindows == 0 || rapidConfig.maxTriggers == 0 ||
             !std::isfinite(rapidConfig.minDistributionShift) ||
             rapidConfig.minDistributionShift < 0.0 ||
             !std::isfinite(rapidConfig.currentSampleWeight) ||
             rapidConfig.currentSampleWeight <= 0.0)) {
            throw std::runtime_error("invalid rapid refit configuration");
        }
        if (requestedMask < -1 ||
            requestedMask > static_cast<int32_t>(navi_sched::MERGE_ALL)) {
            throw std::runtime_error("invalid merge mask");
        }
        for (const auto &path : paths) {
            navi_sched::runFile(path, seed, requestedMask, learningRateScale,
                                lowPressureMaxMs, currentBucketSampleCount,
                                rapidConfig, output);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

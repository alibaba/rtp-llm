#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace navi_sched {
namespace {
constexpr double LN_2 = 0.69314718055994530942;
struct IsotonicBlock {
    size_t begin = 0;
    size_t end = 0;
    double sum = 0.0;
    size_t count = 0;

    double average() const { return sum / static_cast<double>(count); }
};

double smootherStep(double value)
{
    return value * value * value *
           (value * (value * 6.0 - 15.0) + 10.0);
}

double smootherStepDerivative(double value)
{
    const auto complement = value - 1.0;
    return 30.0 * value * value * complement * complement;
}

double minimumPredictionMs()
{
    return std::ldexp(1.0, LatencyResidualModel::MIN_BUCKET);
}

double maximumPredictionMs()
{
    return std::ldexp(1.0, LatencyResidualModel::MAX_BUCKET);
}

double maximumNegativeDifference(int32_t lowerBucket)
{
    // Maximize smootherStep'(f) / 2^f on [0, 1]. This is the exact
    // adjacent-parameter limit that keeps d(base + residual) / d(base) >= 0.
    static const double scale = []() {
        const auto fraction =
            (LN_2 + 4.0 - std::sqrt(LN_2 * LN_2 + 16.0)) / (2.0 * LN_2);
        const auto maximum =
            smootherStepDerivative(fraction) / std::exp2(fraction);
        return LN_2 / maximum;
    }();
    return scale * std::ldexp(1.0, lowerBucket);
}

bool projectParameters(std::vector<double> &parameters)
{
    const auto parameterCount = parameters.size();
    if (parameterCount != LatencyResidualModel::PARAMETER_COUNT ||
        parameters.front() != 0.0 || parameters.back() != 0.0 ||
        !std::all_of(parameters.begin(), parameters.end(), [](double value) {
            return std::isfinite(value);
        })) {
        return false;
    }

    std::vector<double> offsets(parameterCount, 0.0);
    for (size_t index = 1; index < parameterCount; ++index) {
        const auto bucket = LatencyResidualModel::MIN_BUCKET +
                            static_cast<int32_t>(index) - 1;
        offsets[index] = offsets[index - 1] + maximumNegativeDifference(bucket);
    }

    std::vector<IsotonicBlock> blocks;
    blocks.reserve(parameterCount - 2);
    for (size_t index = 1; index + 1 < parameterCount; ++index) {
        const auto transformed =
            std::clamp(parameters[index] + offsets[index], 0.0, offsets.back());
        blocks.push_back({index, index + 1, transformed, 1});
        while (blocks.size() >= 2 &&
               blocks[blocks.size() - 2].average() > blocks.back().average()) {
            auto right = blocks.back();
            blocks.pop_back();
            auto &left = blocks.back();
            left.end = right.end;
            left.sum += right.sum;
            left.count += right.count;
        }
    }
    for (const auto &block : blocks) {
        const auto transformed = block.average();
        for (size_t index = block.begin; index < block.end; ++index) {
            parameters[index] = transformed - offsets[index];
        }
    }
    return true;
}
} // namespace

bool LatencyResidualModel::RuntimeConfig::valid() const
{
    return std::isfinite(learningRate) && learningRate >= 0.0 &&
           learningRate <= 1.0 && std::isfinite(updateErrorFloorMs) &&
           updateErrorFloorMs >= 0.0 && std::isfinite(updateErrorRatio) &&
           updateErrorRatio >= 0.0 && idleDecayStartUs >= 0 &&
           idleDecayEndUs > idleDecayStartUs;
}

std::vector<double> LatencyResidualModel::makeInitialParameters()
{
    return std::vector<double>(PARAMETER_COUNT, 0.0);
}

bool LatencyResidualModel::validateParameters(
    const std::vector<double> &parameters)
{
    if (parameters.size() != PARAMETER_COUNT || parameters.front() != 0.0 ||
        parameters.back() != 0.0) {
        return false;
    }
    for (size_t index = 0; index + 1 < parameters.size(); ++index) {
        const auto limit = maximumNegativeDifference(
            MIN_BUCKET + static_cast<int32_t>(index));
        const auto tolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(parameters[index]),
                      std::abs(parameters[index + 1]), limit});
        if (!std::isfinite(parameters[index]) ||
            parameters[index + 1] - parameters[index] < -limit - tolerance) {
            return false;
        }
    }
    return std::isfinite(parameters.back());
}

bool LatencyResidualModel::expired(int64_t lastUpdateTimeUs,
                                   int64_t currentTimeUs)
{
    return expired(lastUpdateTimeUs, currentTimeUs, RuntimeConfig());
}

bool LatencyResidualModel::expired(int64_t lastUpdateTimeUs,
                                   int64_t currentTimeUs,
                                   const RuntimeConfig &config)
{
    return decayFactor(lastUpdateTimeUs, currentTimeUs, config) == 0.0;
}

double LatencyResidualModel::decayFactor(int64_t lastUpdateTimeUs,
                                         int64_t currentTimeUs)
{
    return decayFactor(lastUpdateTimeUs, currentTimeUs, RuntimeConfig());
}

double LatencyResidualModel::decayFactor(int64_t lastUpdateTimeUs,
                                         int64_t currentTimeUs,
                                         const RuntimeConfig &config)
{
    if (config.learningRate == 0.0) {
        return 0.0;
    }
    if (lastUpdateTimeUs <= 0 || currentTimeUs <= lastUpdateTimeUs) {
        return 1.0;
    }
    const auto idleTimeUs = currentTimeUs - lastUpdateTimeUs;
    if (idleTimeUs <= config.idleDecayStartUs) {
        return 1.0;
    }
    if (idleTimeUs >= config.idleDecayEndUs) {
        return 0.0;
    }
    return static_cast<double>(config.idleDecayEndUs - idleTimeUs) /
           static_cast<double>(config.idleDecayEndUs -
                               config.idleDecayStartUs);
}

bool LatencyResidualModel::applyIdleDecay(std::vector<double> &parameters,
                                          int64_t lastUpdateTimeUs,
                                          int64_t currentTimeUs)
{
    return applyIdleDecay(parameters, lastUpdateTimeUs, currentTimeUs,
                          RuntimeConfig());
}

bool LatencyResidualModel::applyIdleDecay(std::vector<double> &parameters,
                                          int64_t lastUpdateTimeUs,
                                          int64_t currentTimeUs,
                                          const RuntimeConfig &config)
{
    if (!validateParameters(parameters) || lastUpdateTimeUs < 0 ||
        currentTimeUs < 0) {
        return false;
    }
    if (config.learningRate == 0.0) {
        return true;
    }
    const auto factor = decayFactor(lastUpdateTimeUs, currentTimeUs, config);
    if (factor < 1.0) {
        for (auto &parameter : parameters) {
            parameter *= factor;
        }
    }
    return validateParameters(parameters);
}

LatencyResidualModel::Evaluation LatencyResidualModel::evaluate(
    double basePredictionMs, const std::vector<double> &parameters,
    int64_t lastUpdateTimeUs, int64_t currentTimeUs)
{
    return evaluate(basePredictionMs, parameters, lastUpdateTimeUs,
                    currentTimeUs, RuntimeConfig());
}

LatencyResidualModel::Evaluation LatencyResidualModel::evaluate(
    double basePredictionMs, const std::vector<double> &parameters,
    int64_t lastUpdateTimeUs, int64_t currentTimeUs,
    const RuntimeConfig &config)
{
    Evaluation result;
    result.predictionMs = basePredictionMs;
    if (config.learningRate == 0.0) {
        return result;
    }
    if (!std::isfinite(basePredictionMs) || basePredictionMs < minimumPredictionMs() ||
        basePredictionMs > maximumPredictionMs() ||
        !validateParameters(parameters)) {
        return result;
    }

    const auto position = std::log2(basePredictionMs);
    if (!std::isfinite(position)) {
        return result;
    }
    auto lowerBucket = static_cast<int32_t>(std::floor(position));
    if (lowerBucket >= MAX_BUCKET) {
        result.applied = true;
        return result;
    }
    lowerBucket = std::max(MIN_BUCKET, lowerBucket);
    const auto lowerIndex = static_cast<size_t>(lowerBucket - MIN_BUCKET);
    const auto fraction = std::clamp(position - lowerBucket, 0.0, 1.0);
    const auto upperWeight = smootherStep(fraction);
    const auto difference = parameters[lowerIndex + 1] - parameters[lowerIndex];
    const auto idleDecay = decayFactor(lastUpdateTimeUs, currentTimeUs, config);
    const auto correction =
        idleDecay * (parameters[lowerIndex] + upperWeight * difference);
    const auto derivativeFactor =
        1.0 + idleDecay * difference * smootherStepDerivative(fraction) /
                  (basePredictionMs * LN_2);
    const auto unboundedPrediction = basePredictionMs + correction;
    if (!std::isfinite(correction) || !std::isfinite(derivativeFactor) ||
        !std::isfinite(unboundedPrediction)) {
        return result;
    }
    result.correctionMs = correction;
    result.predictionMs = std::max(0.0, unboundedPrediction);
    result.derivativeFactor =
        unboundedPrediction > 0.0 ? std::max(0.0, derivativeFactor) : 0.0;
    result.applied = true;
    return result;
}

bool LatencyResidualModel::update(double basePredictionMs, double actualLatencyMs,
                                  std::vector<double> &parameters,
                                  int64_t &lastUpdateTimeUs,
                                  int64_t currentTimeUs)
{
    return update(basePredictionMs, actualLatencyMs, parameters,
                  lastUpdateTimeUs, currentTimeUs, RuntimeConfig());
}

bool LatencyResidualModel::update(double basePredictionMs, double actualLatencyMs,
                                  std::vector<double> &parameters,
                                  int64_t &lastUpdateTimeUs,
                                  int64_t currentTimeUs,
                                  const RuntimeConfig &config)
{
    if (!std::isfinite(basePredictionMs) || !std::isfinite(actualLatencyMs) ||
        actualLatencyMs < 0.0 || basePredictionMs < 0.0 ||
        basePredictionMs > maximumPredictionMs() ||
        !validateParameters(parameters) || lastUpdateTimeUs < 0 ||
        currentTimeUs < 0) {
        return false;
    }
    if (config.learningRate == 0.0) {
        return true;
    }
    // The residual model has no bucket below 2^MIN_BUCKET. A valid low
    // prediction is still a valid training sample for the base model; leave
    // residual state untouched instead of rejecting the whole fit batch.
    if (basePredictionMs < minimumPredictionMs()) {
        return true;
    }
    const auto position = std::log2(basePredictionMs);
    auto lowerBucket = static_cast<int32_t>(std::floor(position));
    if (lowerBucket >= MAX_BUCKET) {
        return true;
    }
    lowerBucket = std::max(MIN_BUCKET, lowerBucket);
    const auto lowerIndex = static_cast<size_t>(lowerBucket - MIN_BUCKET);
    const auto fraction = std::clamp(position - lowerBucket, 0.0, 1.0);
    const auto upperWeight = smootherStep(fraction);
    const auto lowerWeight = 1.0 - upperWeight;
    auto updated = parameters;
    const auto updateTimeUs = std::max(lastUpdateTimeUs, currentTimeUs);
    if (!applyIdleDecay(updated, lastUpdateTimeUs, updateTimeUs, config)) {
        return false;
    }
    const auto correction = lowerWeight * updated[lowerIndex] +
                            upperWeight * updated[lowerIndex + 1];
    const auto targetCorrection = actualLatencyMs - basePredictionMs;
    const auto errorLimit = std::max(config.updateErrorFloorMs,
                                     config.updateErrorRatio * basePredictionMs);
    const auto error = std::clamp(targetCorrection - correction,
                                  -errorLimit, errorLimit);
    const auto denominator = lowerWeight * lowerWeight + upperWeight * upperWeight;
    if (!std::isfinite(error) || !std::isfinite(denominator) || denominator <= 0.0) {
        return false;
    }
    if (lowerIndex != 0) {
        updated[lowerIndex] +=
            config.learningRate * error * lowerWeight / denominator;
    }
    if (lowerIndex + 1 != PARAMETER_COUNT - 1) {
        updated[lowerIndex + 1] +=
            config.learningRate * error * upperWeight / denominator;
    }
    if (!projectParameters(updated) || !validateParameters(updated)) {
        return false;
    }
    parameters = std::move(updated);
    lastUpdateTimeUs = updateTimeUs;
    return true;
}

} // namespace navi_sched

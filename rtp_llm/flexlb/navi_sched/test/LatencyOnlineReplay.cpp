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
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace navi_sched {
namespace {

struct FeatureScales {
    double runningCount = 1.0;
    double waitingCount = 1.0;
    double runningTokens = 1.0;
    double waitingTokens = 1.0;
    double precedingCount = 1.0;
    double runningElapsed = 1.0;
    double runningDecodeCount = 1.0;
    double runningDecodeTokens = 1.0;
    // Test-only mapping into P18, used to compare completed and remaining
    // decode work without changing the production parameter layout.
    double runningDecodeRemainingBudget = 0.0;
    bool enableBatchCount = false;
};

struct ResponseNormalizedRateConfig {
    enum class Signal {
        ABSOLUTE_PREDICTION_MS,
        RELATIVE_PREDICTION,
        RELATIVE_PARAMETER,
        BATCH_MEAN_LOSS_IMPACT,
    };

    std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> targets{};
    std::array<bool, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> enabled{};
    double alpha = 0.05;
    double minimum = 0.25;
    double maximum = 2.0;
    size_t calibrationUpdates = 40;
    double denominatorFloor = 100.0;
    Signal signal = Signal::ABSOLUTE_PREDICTION_MS;
};

constexpr std::array<size_t, 3> RESPONSE_RATE_PARAMETER_INDICES = {2, 3, 5};

struct FeatureConditionedRateConfig {
    int64_t runningThreshold = 9;
    double activeP2Factor = 0.25;
    double otherP8Factor = 0.5;
    double minimumP8Value = 0.0;
    double maximumP8Value = std::numeric_limits<double>::max();
    bool resetP15StaleMomentum = false;
};

enum class BaseOptimizerMode {
    ADAM,
    SGD,
};

struct ReparameterizedSgdConfig {
    double gradientEmaAlpha = 0.01;
    double learningRateMultiplier = 1.0;
    double scalePower = 1.0;
    bool useAdamMoment1 = false;
};

struct BaseUpdateProbe {
    std::vector<double> parametersBefore;
    std::vector<double> parametersAfter;
    double learningRateScale = 0.0;
    uint64_t sourceSequence = 0;
    size_t sourceFitIndex = 0;
    size_t horizon = 0;
    bool traceParameters = false;
};

struct AdamFitTrace {
    double totalSampleWeight = 0.0;
    double correction = 0.0;
    std::vector<double> gradient;
    std::vector<double> moment1Before;
    std::vector<double> moment1Candidate;
    std::vector<double> moment2Before;
    std::vector<double> moment2Candidate;
    std::vector<double> unprojectedParameters;
};

struct ReplayState {
    NonLinearPrefillModelR model;
    LatencyHistorySampler sampler;
    std::vector<double> parameters;
    std::vector<double> residualParameters;
    int64_t residualTimeUs = 0;
    std::vector<NaviSchedLatencySample> pending;
    std::unordered_map<uint64_t, NaviSchedLatencySample> historyBySequence;
    std::vector<double> previousLoggedParameters;
    std::deque<BaseUpdateProbe> baseUpdateProbes;
    double runningElapsedFeatureScale = 1.0;
    BaseOptimizerMode optimizerMode = BaseOptimizerMode::ADAM;
    std::optional<ReparameterizedSgdConfig> reparameterizedSgdConfig;
    std::vector<double> gradientMagnitudeEma;
    std::optional<ResponseNormalizedRateConfig> responseRateConfig;
    std::optional<FeatureConditionedRateConfig> featureRateConfig;
    std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> responseEma{};
    std::array<size_t, NonLinearPrefillModelR::MAX_PARAMETER_COUNT>
        responseSampleCount{};
    std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> responseSum{};
    std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> responseWeight{};
    size_t seenUpdateEvents = 0;
    bool anchored = false;
    bool residualPredictionEnabled = true;
    std::optional<double> p14Floor;

    ReplayState(
        uint64_t seed,
        double runningElapsedFeatureScale,
        bool enableBatchCount,
        BaseOptimizerMode optimizerMode,
        const std::optional<ResponseNormalizedRateConfig> &responseConfig,
        const std::optional<FeatureConditionedRateConfig> &featureConfig)
        : sampler(seed)
        , runningElapsedFeatureScale(runningElapsedFeatureScale)
        , optimizerMode(optimizerMode)
        , responseRateConfig(responseConfig)
        , featureRateConfig(featureConfig)
    {
        sampler.configure(2000, 64);
        model._learningRate = 0.05;
        model._useBatchCountFeature = enableBatchCount;
        model._useCapacitySplitFeature = true;
    }
};

std::vector<double> readVector(const rapidjson::Value &object, const char *name)
{
    const auto &value = object[name];
    if (!value.IsArray()) {
        throw std::runtime_error(std::string("invalid array: ") + name);
    }
    std::vector<double> result;
    result.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
        result.emplace_back(entry.GetDouble());
    }
    return result;
}

AdamFitTrace calculateAdamFitTrace(
    const NonLinearPrefillModelR &model,
    const std::vector<NaviSchedLatencySample> &samples,
    const std::vector<double> &currentParameters, double learningRateScale,
    double currentSampleWeight,
    const std::vector<double> *parameterLearningRateScales)
{
    constexpr double BETA1 = 0.9;
    constexpr double BETA2 = 0.95;
    constexpr double EPSILON = 1e-20;
    const auto parameterCount = currentParameters.size();
    const auto optimizer = model.getOptimizerState();
    auto weights = NonLinearPrefillModelR::toValues(currentParameters);
    auto moment1 = optimizer.adamMoment1;
    auto moment2 = optimizer.adamMoment2;
    const auto nonlinearScale = model.getNonlinearScale();
    NonLinearPrefillModelR::reparameterize(
        weights, moment1, moment2, parameterCount,
        NonLinearPrefillModelR::NONLINEAR_SCALE, nonlinearScale);
    if (!model.usesBatchCountFeature()) {
        weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        moment1[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        moment2[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    const auto disableParameter = [&](size_t index, bool enabled) {
        if (!enabled && index < parameterCount) {
            weights[index] = 0.0;
            moment1[index] = 0.0;
            moment2[index] = 0.0;
        }
    };
    disableParameter(
        NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX,
        model._useRunningDecodeRequestCountFeature);
    disableParameter(
        NonLinearPrefillModelR::
            RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX,
        model._useRunningDecodeGeneratedTokenCountFeature);

    NonLinearPrefillModelR::Values gradient{};
    double totalSampleWeight = 0.0;
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const auto &sample = samples[sampleIndex];
        auto input = NonLinearPrefillModelR::makeInput(sample.features);
        if (!model.usesBatchCountFeature()) {
            input[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        }
        if (!model._useRunningDecodeRequestCountFeature) {
            input[14] = 0.0;
        }
        if (!model._useRunningDecodeGeneratedTokenCountFeature) {
            input[15] = 0.0;
        }
        NonLinearPrefillModelR::Values predictionGradient{};
        const auto prediction = NonLinearPrefillModelR::predict(
            input, weights, parameterCount, &predictionGradient, nonlinearScale);
        const auto difference =
            (std::log1p(prediction) - std::log1p(sample.latencyMs)) /
            (1.0 + prediction);
        const auto sampleWeight = sampleIndex + 1 == samples.size()
                                      ? currentSampleWeight
                                      : 1.0;
        totalSampleWeight += sampleWeight;
        for (size_t index = 0; index < parameterCount; ++index) {
            gradient[index] +=
                sampleWeight * difference * predictionGradient[index];
        }
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        gradient[index] /= totalSampleWeight;
    }

    const auto moment1BeforeInternal = moment1;
    const auto moment2BeforeInternal = moment2;
    const auto correction =
        model._learningRate * learningRateScale *
        std::sqrt(1.0 - std::pow(BETA2, optimizer.step)) /
        (1.0 - std::pow(BETA1, optimizer.step));
    for (size_t index = 0; index < parameterCount; ++index) {
        moment1[index] =
            BETA1 * moment1[index] + (1.0 - BETA1) * gradient[index];
        moment2[index] = BETA2 * moment2[index] +
                         (1.0 - BETA2) * gradient[index] * gradient[index];
        const auto parameterScale = parameterLearningRateScales
                                        ? (*parameterLearningRateScales)[index]
                                        : 1.0;
        weights[index] -= correction * parameterScale * moment1[index] /
                          std::sqrt(moment2[index] + EPSILON);
    }

    auto gradientMoment2 = gradient;
    for (size_t index = 0; index < parameterCount; ++index) {
        gradientMoment2[index] *= gradientMoment2[index];
    }
    auto gradientOutput = gradient;
    auto gradientWeights = weights;
    NonLinearPrefillModelR::reparameterize(
        gradientWeights, gradientOutput, gradientMoment2, parameterCount,
        nonlinearScale, NonLinearPrefillModelR::NONLINEAR_SCALE);

    auto moment1Before = moment1BeforeInternal;
    auto moment2Before = moment2BeforeInternal;
    auto beforeWeights = NonLinearPrefillModelR::toValues(currentParameters);
    NonLinearPrefillModelR::reparameterize(
        beforeWeights, moment1Before, moment2Before, parameterCount,
        nonlinearScale, NonLinearPrefillModelR::NONLINEAR_SCALE);
    NonLinearPrefillModelR::reparameterize(
        weights, moment1, moment2, parameterCount, nonlinearScale,
        NonLinearPrefillModelR::NONLINEAR_SCALE);

    AdamFitTrace trace;
    trace.totalSampleWeight = totalSampleWeight;
    trace.correction = correction;
    trace.gradient.assign(gradientOutput.begin(),
                          gradientOutput.begin() + parameterCount);
    trace.moment1Before.assign(moment1Before.begin(),
                               moment1Before.begin() + parameterCount);
    trace.moment1Candidate.assign(moment1.begin(),
                                  moment1.begin() + parameterCount);
    trace.moment2Before.assign(moment2Before.begin(),
                               moment2Before.begin() + parameterCount);
    trace.moment2Candidate.assign(moment2.begin(),
                                  moment2.begin() + parameterCount);
    trace.unprojectedParameters.assign(weights.begin(),
                                       weights.begin() + parameterCount);
    return trace;
}

std::optional<std::vector<double>> parseParameterLearningRateScales(
    const std::string &value)
{
    constexpr const char *PREFIX = "parameter_rates:";
    if (value.rfind(PREFIX, 0) != 0) {
        return std::nullopt;
    }
    std::vector<double> scales(NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT,
                               1.0);
    std::array<bool, NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT> seen{};
    std::stringstream entries(value.substr(std::char_traits<char>::length(PREFIX)));
    std::string entry;
    size_t count = 0;
    while (std::getline(entries, entry, ',')) {
        const auto separator = entry.find('=');
        if (separator == std::string::npos || separator == 0 ||
            separator + 1 == entry.size()) {
            throw std::runtime_error("parameter rate must be INDEX=MULTIPLIER");
        }
        size_t parsed = 0;
        const auto index = std::stoull(entry.substr(0, separator), &parsed);
        if (parsed != separator || index >= scales.size() || seen[index]) {
            throw std::runtime_error("invalid or duplicate parameter rate index");
        }
        const auto rate = entry.substr(separator + 1);
        scales[index] = std::stod(rate, &parsed);
        const auto mayDisableDecodeFeature =
            index == NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX ||
            index == NonLinearPrefillModelR::
                         RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX;
        if (parsed != rate.size() || !std::isfinite(scales[index]) ||
            scales[index] < 0.0 ||
            (scales[index] == 0.0 && !mayDisableDecodeFeature)) {
            throw std::runtime_error(
                "parameter rate must be positive except disabled decode features");
        }
        seen[index] = true;
        ++count;
    }
    if (count == 0) {
        throw std::runtime_error("parameter_rates requires at least one index");
    }
    return scales;
}

std::optional<double> parseP14Floor(const std::string &value)
{
    constexpr const char *PREFIX = "p14_floor:";
    if (value.rfind(PREFIX, 0) != 0) {
        return std::nullopt;
    }
    const auto text = value.substr(std::char_traits<char>::length(PREFIX));
    size_t parsed = 0;
    const auto floor = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(floor) || floor < 0.0) {
        throw std::runtime_error("p14_floor must be a finite nonnegative value");
    }
    return floor;
}

std::optional<ResponseNormalizedRateConfig> parseResponseNormalizedRates(
    const std::string &value)
{
    constexpr const char *LEGACY_PREFIX = "response_normalize:";
    constexpr const char *PREDICTION_PREFIX = "prediction_feedback:";
    constexpr const char *PARAMETER_PREFIX = "parameter_feedback:";
    constexpr const char *LOSS_PREFIX = "loss_feedback:";
    const char *prefix = nullptr;
    auto signal = ResponseNormalizedRateConfig::Signal::ABSOLUTE_PREDICTION_MS;
    if (value.rfind(LEGACY_PREFIX, 0) == 0) {
        prefix = LEGACY_PREFIX;
    } else if (value.rfind(PREDICTION_PREFIX, 0) == 0) {
        prefix = PREDICTION_PREFIX;
        signal = ResponseNormalizedRateConfig::Signal::RELATIVE_PREDICTION;
    } else if (value.rfind(PARAMETER_PREFIX, 0) == 0) {
        prefix = PARAMETER_PREFIX;
        signal = ResponseNormalizedRateConfig::Signal::RELATIVE_PARAMETER;
    } else if (value.rfind(LOSS_PREFIX, 0) == 0) {
        prefix = LOSS_PREFIX;
        signal = ResponseNormalizedRateConfig::Signal::BATCH_MEAN_LOSS_IMPACT;
    } else {
        return std::nullopt;
    }
    std::stringstream entries(value.substr(std::char_traits<char>::length(prefix)));
    std::vector<std::string> values;
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        values.emplace_back(entry);
    }
    ResponseNormalizedRateConfig config;
    config.signal = signal;
    std::vector<double *> numbers;
    size_t calibrationIndex = 0;
    if (signal == ResponseNormalizedRateConfig::Signal::ABSOLUTE_PREDICTION_MS) {
        if (values.size() != 7) {
            throw std::runtime_error("response_normalize requires 7 values");
        }
        for (size_t index = 0; index < RESPONSE_RATE_PARAMETER_INDICES.size(); ++index) {
            config.enabled[RESPONSE_RATE_PARAMETER_INDICES[index]] = true;
            numbers.emplace_back(
                &config.targets[RESPONSE_RATE_PARAMETER_INDICES[index]]);
        }
        numbers.insert(numbers.end(), {&config.alpha, &config.minimum,
                                       &config.maximum});
        calibrationIndex = 6;
    } else {
        const auto expectedValueCount =
            signal == ResponseNormalizedRateConfig::Signal::BATCH_MEAN_LOSS_IMPACT
                ? 5
                : 6;
        if (values.size() != expectedValueCount) {
            throw std::runtime_error(
                signal == ResponseNormalizedRateConfig::Signal::BATCH_MEAN_LOSS_IMPACT
                    ? "loss feedback requires TARGET,ALPHA,MIN,MAX,CALIBRATION"
                    : "feedback mode requires TARGET,ALPHA,MIN,MAX,CALIBRATION,FLOOR");
        }
        config.enabled.fill(true);
        config.targets.fill(0.0);
        numbers = {&config.targets[0], &config.alpha, &config.minimum,
                   &config.maximum};
        calibrationIndex = 4;
    }
    for (size_t index = 0; index < numbers.size(); ++index) {
        size_t parsed = 0;
        *numbers[index] = std::stod(values[index], &parsed);
        if (parsed != values[index].size() || !std::isfinite(*numbers[index]) ||
            *numbers[index] <= 0.0) {
            throw std::runtime_error(
                "feedback values must be positive finite numbers");
        }
    }
    if (signal != ResponseNormalizedRateConfig::Signal::ABSOLUTE_PREDICTION_MS) {
        config.targets.fill(config.targets[0]);
    }
    if (signal == ResponseNormalizedRateConfig::Signal::RELATIVE_PREDICTION ||
        signal == ResponseNormalizedRateConfig::Signal::RELATIVE_PARAMETER) {
        size_t parsed = 0;
        config.denominatorFloor = std::stod(values[5], &parsed);
        if (parsed != values[5].size() ||
            !std::isfinite(config.denominatorFloor) ||
            config.denominatorFloor <= 0.0) {
            throw std::runtime_error("feedback denominator floor must be positive");
        }
    }
    size_t parsed = 0;
    config.calibrationUpdates = std::stoull(values[calibrationIndex], &parsed);
    if (parsed != values[calibrationIndex].size() ||
        config.calibrationUpdates == 0 ||
        config.alpha > 1.0 || config.maximum < config.minimum) {
        throw std::runtime_error("invalid response_normalize bounds or calibration count");
    }
    return config;
}

std::optional<FeatureConditionedRateConfig> parseFeatureConditionedRates(
    const std::string &value)
{
    constexpr const char *PREFIX = "feature_rates:";
    if (value.rfind(PREFIX, 0) != 0) {
        return std::nullopt;
    }
    std::stringstream entries(value.substr(std::char_traits<char>::length(PREFIX)));
    std::array<std::string, 6> values;
    std::string entry;
    size_t count = 0;
    while (std::getline(entries, entry, ',')) {
        if (count >= values.size()) {
            throw std::runtime_error("feature_rates requires 3 to 6 values");
        }
        values[count++] = entry;
    }
    if (count < 3 || count > 6) {
        throw std::runtime_error("feature_rates requires 3 to 6 values");
    }
    FeatureConditionedRateConfig config;
    size_t parsed = 0;
    config.runningThreshold = std::stoll(values[0], &parsed);
    if (parsed != values[0].size() || config.runningThreshold <= 0) {
        throw std::runtime_error("feature_rates threshold must be positive");
    }
    std::array<double *, 2> factors = {&config.activeP2Factor,
                                      &config.otherP8Factor};
    for (size_t index = 0; index < factors.size(); ++index) {
        parsed = 0;
        *factors[index] = std::stod(values[index + 1], &parsed);
        if (parsed != values[index + 1].size() ||
            !std::isfinite(*factors[index]) || *factors[index] <= 0.0) {
            throw std::runtime_error("feature_rates factors must be finite and positive");
        }
    }
    if (count >= 4) {
        parsed = 0;
        config.minimumP8Value = std::stod(values[3], &parsed);
        if (parsed != values[3].size() ||
            !std::isfinite(config.minimumP8Value) ||
            config.minimumP8Value < 0.0) {
            throw std::runtime_error("feature_rates p8 minimum must be finite and nonnegative");
        }
    }
    if (count >= 5) {
        parsed = 0;
        config.maximumP8Value = std::stod(values[4], &parsed);
        if (parsed != values[4].size() ||
            !std::isfinite(config.maximumP8Value) ||
            config.maximumP8Value < config.minimumP8Value) {
            throw std::runtime_error("feature_rates p8 maximum must be finite and above minimum");
        }
    }
    if (count >= 6) {
        if (values[5] != "0" && values[5] != "1") {
            throw std::runtime_error("feature_rates p15 momentum reset must be 0 or 1");
        }
        config.resetP15StaleMomentum = values[5] == "1";
    }
    return config;
}

std::optional<ReparameterizedSgdConfig>
parseReparameterizedSgdConfig(const std::string &value)
{
    constexpr const char *SGD_PREFIX = "sgd_reparameterized:";
    constexpr const char *MOMENTUM_PREFIX = "momentum_reparameterized:";
    const bool useAdamMoment1 = value.rfind(MOMENTUM_PREFIX, 0) == 0;
    const auto prefix = useAdamMoment1 ? MOMENTUM_PREFIX : SGD_PREFIX;
    if (value.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }
    std::stringstream stream(value.substr(std::char_traits<char>::length(prefix)));
    std::string token;
    std::vector<double> values;
    while (std::getline(stream, token, ',')) {
        size_t parsed = 0;
        const auto number = std::stod(token, &parsed);
        if (parsed != token.size() || !std::isfinite(number)) {
            throw std::runtime_error("invalid reparameterized optimizer value");
        }
        values.push_back(number);
    }
    if (values.size() != 3 || values[0] <= 0.0 || values[0] > 1.0 ||
        values[1] <= 0.0 || values[2] <= 0.0 || values[2] > 1.0) {
        throw std::runtime_error(
            "reparameterized optimizer requires EMA_ALPHA,LR_MULTIPLIER,SCALE_POWER");
    }
    return ReparameterizedSgdConfig{values[0], values[1], values[2],
                                    useAdamMoment1};
}

std::optional<std::pair<double, double>>
parseResidualErrorLimit(const std::string &value)
{
    constexpr const char *PREFIX = "residual_error_limit:";
    if (value.rfind(PREFIX, 0) != 0) {
        return std::nullopt;
    }
    const auto settings = value.substr(std::char_traits<char>::length(PREFIX));
    const auto separator = settings.find(',');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == settings.size()) {
        throw std::runtime_error("residual_error_limit requires FLOOR_MS,RATIO");
    }
    size_t parsed = 0;
    const auto floorMs = std::stod(settings.substr(0, separator), &parsed);
    if (parsed != separator || !std::isfinite(floorMs) || floorMs < 0.0) {
        throw std::runtime_error("invalid residual error floor");
    }
    const auto ratio = std::stod(settings.substr(separator + 1), &parsed);
    if (parsed != settings.size() - separator - 1 || !std::isfinite(ratio) ||
        ratio < 0.0) {
        throw std::runtime_error("invalid residual error ratio");
    }
    return std::make_pair(floorMs, ratio);
}

bool fitSgd(NonLinearPrefillModelR &model,
            const std::vector<NaviSchedLatencySample> &samples,
            const std::vector<double> &currentParameters,
            std::vector<double> &newParameters, double learningRateScale,
            double currentSampleWeight,
            const std::optional<ReparameterizedSgdConfig> &reparameterizedConfig,
            std::vector<double> &gradientMagnitudeEma,
            std::vector<double> *appliedScales,
            std::vector<double> *appliedMoment1)
{
    constexpr double MOMENTUM_BETA1 = 0.9;
    if (samples.empty() || !NonLinearPrefillModelR::validateParameters(currentParameters) ||
        !std::isfinite(learningRateScale) || learningRateScale <= 0.0 ||
        !std::isfinite(currentSampleWeight) || currentSampleWeight <= 0.0) {
        return false;
    }
    const auto parameterCount = currentParameters.size();
    auto optimizer = model.getOptimizerState();
    if (appliedMoment1) {
        appliedMoment1->assign(optimizer.adamMoment1.begin(),
                               optimizer.adamMoment1.begin() + parameterCount);
    }
    auto weights = NonLinearPrefillModelR::toValues(currentParameters);
    NonLinearPrefillModelR::Values unusedMoment1{};
    NonLinearPrefillModelR::Values unusedMoment2{};
    const auto nonlinearScale = model.getNonlinearScale();
    NonLinearPrefillModelR::reparameterize(
        weights, unusedMoment1, unusedMoment2, parameterCount,
        NonLinearPrefillModelR::NONLINEAR_SCALE, nonlinearScale);
    NonLinearPrefillModelR::Values internalFactors{};
    internalFactors.fill(1.0);
    if (reparameterizedConfig) {
        NonLinearPrefillModelR::reparameterize(
            internalFactors, unusedMoment1, unusedMoment2, parameterCount,
            NonLinearPrefillModelR::NONLINEAR_SCALE, nonlinearScale);
    }
    if (!model.usesBatchCountFeature()) {
        weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (reparameterizedConfig && !gradientMagnitudeEma.empty() &&
        gradientMagnitudeEma.size() != parameterCount) {
        return false;
    }
    std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT> scales{};
    scales.fill(1.0);
    if (reparameterizedConfig && !gradientMagnitudeEma.empty()) {
        std::vector<double> activeMagnitudes;
        for (size_t index = 0; index < parameterCount; ++index) {
            if (index != NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX ||
                model.usesBatchCountFeature()) {
                if (gradientMagnitudeEma[index] > 0.0) {
                    activeMagnitudes.push_back(gradientMagnitudeEma[index]);
                }
            }
        }
        if (!activeMagnitudes.empty()) {
            std::sort(activeMagnitudes.begin(), activeMagnitudes.end());
            const auto reference = activeMagnitudes[activeMagnitudes.size() / 2];
            for (size_t index = 0; index < parameterCount; ++index) {
                if (gradientMagnitudeEma[index] > 0.0) {
                    scales[index] = std::pow(
                        gradientMagnitudeEma[index] / reference,
                        reparameterizedConfig->scalePower);
                    if (!std::isfinite(scales[index]) || scales[index] <= 0.0) {
                        return false;
                    }
                }
            }
        }
    }
    if (appliedScales) {
        appliedScales->assign(scales.begin(), scales.begin() + parameterCount);
    }
    NonLinearPrefillModelR::Values gradient{};
    double totalSampleWeight = 0.0;
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const auto &sample = samples[sampleIndex];
        auto input = NonLinearPrefillModelR::makeInput(sample.features);
        if (!model.usesBatchCountFeature()) {
            input[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        }
        NonLinearPrefillModelR::Values predictionGradient{};
        double prediction = 0.0;
        if (reparameterizedConfig) {
            std::array<AutoGrad<NonLinearPrefillModelR::MAX_PARAMETER_COUNT>,
                       NonLinearPrefillModelR::MAX_PARAMETER_COUNT> formulaWeights;
            for (size_t index = 0; index < parameterCount; ++index) {
                const auto optimizedParameter =
                    AutoGrad<NonLinearPrefillModelR::MAX_PARAMETER_COUNT>::parameter(
                        currentParameters[index] * scales[index], index);
                formulaWeights[index] =
                    optimizedParameter * (internalFactors[index] / scales[index]);
            }
            const auto result = NonLinearPrefillModelR::evaluate(
                input, formulaWeights, nonlinearScale);
            prediction = result.value();
            for (size_t index = 0; index < parameterCount; ++index) {
                predictionGradient[index] = result.gradient(index);
            }
        } else {
            prediction = NonLinearPrefillModelR::predict(
                input, weights, parameterCount, &predictionGradient,
                nonlinearScale);
        }
        if (!std::isfinite(sample.latencyMs) || sample.latencyMs < 0.0 ||
            !std::isfinite(prediction) || prediction < 0.0) {
            newParameters = currentParameters;
            return true;
        }
        const auto difference =
            (std::log1p(prediction) - std::log1p(sample.latencyMs)) /
            (1.0 + prediction);
        const auto sampleWeight = sampleIndex + 1 == samples.size()
                                      ? currentSampleWeight
                                      : 1.0;
        totalSampleWeight += sampleWeight;
        for (size_t index = 0; index < parameterCount; ++index) {
            gradient[index] +=
                sampleWeight * difference * predictionGradient[index];
        }
    }
    const auto previousWeights = weights;
    const bool useAdamMoment1 =
        reparameterizedConfig && reparameterizedConfig->useAdamMoment1;
    if (useAdamMoment1 && optimizer.step < 1) {
        return false;
    }
    const auto biasCorrection =
        useAdamMoment1 ? 1.0 - std::pow(MOMENTUM_BETA1, optimizer.step) : 1.0;
    const auto stepSize = model._learningRate * learningRateScale *
                          (reparameterizedConfig
                               ? reparameterizedConfig->learningRateMultiplier
                               : 1.0);
    for (size_t index = 0; index < parameterCount; ++index) {
        auto optimizedParameter = reparameterizedConfig
                                      ? currentParameters[index] * scales[index]
                                      : weights[index];
        auto update = stepSize * gradient[index] / totalSampleWeight;
        if (useAdamMoment1) {
            const auto realGradient = gradient[index] * scales[index] /
                                      totalSampleWeight;
            optimizer.adamMoment1[index] =
                MOMENTUM_BETA1 * optimizer.adamMoment1[index] +
                (1.0 - MOMENTUM_BETA1) * realGradient;
            update = stepSize * optimizer.adamMoment1[index] /
                     (scales[index] * biasCorrection);
        }
        optimizedParameter -= update;
        weights[index] = reparameterizedConfig
                             ? optimizedParameter * internalFactors[index] /
                                   scales[index]
                             : optimizedParameter;
    }
    const auto unprojectedWeights = weights;
    const auto scaleRatio = NonLinearPrefillModelR::NONLINEAR_SCALE / nonlinearScale;
    auto projectedMoment1 = useAdamMoment1 ? optimizer.adamMoment1 : unusedMoment1;
    auto projectedMoment2 = unusedMoment2;
    if (useAdamMoment1) {
        auto conversionWeights = NonLinearPrefillModelR::toValues(currentParameters);
        NonLinearPrefillModelR::reparameterize(
            conversionWeights, projectedMoment1, projectedMoment2, parameterCount,
            NonLinearPrefillModelR::NONLINEAR_SCALE, nonlinearScale);
    }
    const auto previousProjectedMoment1 = projectedMoment1;
    model.projectOutputParameters(weights, projectedMoment1, projectedMoment2,
                                  NonLinearPrefillModelR::MIN_NONLINEAR_CURVATURE *
                                      scaleRatio * scaleRatio, scaleRatio);
    for (const auto [index, minimum] :
         {std::pair{NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX, 0.0},
          std::pair{NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX,
                    NonLinearPrefillModelR::MIN_NONLINEAR_CURVATURE * scaleRatio *
                        scaleRatio}}) {
        if (previousWeights[index] > minimum && unprojectedWeights[index] <= minimum) {
            weights[index] = previousWeights[index];
            if (useAdamMoment1) {
                projectedMoment1[index] = previousProjectedMoment1[index];
            }
        }
    }
    NonLinearPrefillModelR::reparameterize(
        weights, projectedMoment1, projectedMoment2, parameterCount, nonlinearScale,
        NonLinearPrefillModelR::NONLINEAR_SCALE);
    if (useAdamMoment1) {
        optimizer.adamMoment1 = projectedMoment1;
    }
    newParameters.assign(weights.begin(), weights.begin() + parameterCount);
    if (!NonLinearPrefillModelR::validateParameters(newParameters)) {
        return false;
    }
    for (const auto &sample : samples) {
        const auto prediction = model.predictConfigured(sample.features, newParameters);
        if (!std::isfinite(prediction) || prediction < 0.0) {
            newParameters = currentParameters;
            return true;
        }
    }
    if (reparameterizedConfig) {
        auto nextMagnitudeEma = gradientMagnitudeEma.empty()
                                    ? std::vector<double>(parameterCount, 0.0)
                                    : gradientMagnitudeEma;
        for (size_t index = 0; index < parameterCount; ++index) {
            const auto rawGradient =
                std::abs(gradient[index] * scales[index] / totalSampleWeight);
            nextMagnitudeEma[index] +=
                reparameterizedConfig->gradientEmaAlpha *
                (rawGradient - nextMagnitudeEma[index]);
            if (!std::isfinite(nextMagnitudeEma[index])) {
                return false;
            }
        }
        gradientMagnitudeEma = std::move(nextMagnitudeEma);
    }
    if (useAdamMoment1) {
        ++optimizer.step;
        if (!model.restoreOptimizerState(optimizer)) {
            return false;
        }
    }
    if (appliedMoment1) {
        appliedMoment1->assign(optimizer.adamMoment1.begin(),
                               optimizer.adamMoment1.begin() + parameterCount);
    }
    return true;
}

bool equalParameters(const std::vector<double> &left, const std::vector<double> &right)
{
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 0; index < left.size(); ++index) {
        const auto tolerance = 1e-6 + 1e-10 * std::abs(right[index]);
        if (std::abs(left[index] - right[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

double maxDifference(const std::vector<double> &left, const std::vector<double> &right)
{
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

FeatureScales parseFeatureScales(const std::string &value)
{
    FeatureScales result;
    if (value == "all") {
        return result;
    }
    if (value == "all+batch") {
        result.enableBatchCount = true;
        return result;
    }
    std::array<double *, 9> fields = {
        &result.runningCount,  &result.waitingCount,   &result.runningTokens,
        &result.waitingTokens, &result.precedingCount, &result.runningElapsed,
        &result.runningDecodeCount, &result.runningDecodeTokens,
        &result.runningDecodeRemainingBudget,
    };
    std::stringstream stream(value);
    std::string token;
    size_t index = 0;
    while (std::getline(stream, token, ',')) {
        if (index >= fields.size()) {
            throw std::runtime_error("too many feature scales");
        }
        *fields[index++] = std::stod(token);
    }
    if (index != 6 && index != 8 && index != fields.size()) {
        throw std::runtime_error(
            "feature scales must contain six, eight, or nine values");
    }
    for (const auto field : fields) {
        if (!std::isfinite(*field) || *field < 0.0) {
            throw std::runtime_error("feature scales must be finite and non-negative");
        }
    }
    return result;
}

int64_t scaleInteger(int64_t value, double scale)
{
    return static_cast<int64_t>(std::llround(static_cast<double>(value) * scale));
}

NaviSchedLatencySample readSample(const rapidjson::Value &event,
                                  const FeatureScales &scales)
{
    const auto &features = event["features"];
    if (!features.IsArray() ||
        (features.Size() != 12 && features.Size() != 13 &&
         features.Size() != 15 && features.Size() != 16)) {
        throw std::runtime_error("invalid sample features");
    }
    NaviSchedLatencySample sample;
    const auto hasDecodeProgress = features.Size() >= 15;
    const auto runningDecodeCount =
        hasDecodeProgress ? features[13].GetInt64() : 0;
    const auto runningDecodeTokens =
        hasDecodeProgress ? features[14].GetInt64() : 0;
    const auto runningDecodeRemainingTokenBudget =
        features.Size() == 16 ? features[15].GetInt64() : 0;
    if (!sample.features.appendCapacitySplitAggregate(
            features[0].GetInt64(),
            scaleInteger(features[1].GetInt64(), scales.runningCount),
            scaleInteger(features[2].GetInt64(), scales.runningTokens),
            scaleInteger(features[3].GetInt64(), scales.runningTokens),
            features[4].GetDouble() * scales.runningTokens,
            features[5].GetDouble() * scales.runningTokens,
            scaleInteger(features[6].GetInt64(), scales.waitingCount),
            scaleInteger(features[7].GetInt64(), scales.waitingTokens),
            scaleInteger(features[8].GetInt64(), scales.waitingTokens),
            features[9].GetDouble() * scales.waitingTokens,
            features[10].GetDouble() * scales.waitingTokens) ||
        !sample.features.appendPrecedingRequestCount(
            scaleInteger(features[11].GetInt64(), scales.precedingCount)) ||
        (features.Size() >= 13 &&
         !sample.features.appendRunningElapsedTimeMs(features[12].GetDouble() *
                                                     scales.runningElapsed)) ||
        (hasDecodeProgress &&
         !sample.features.appendRunningDecodeProgress(
             runningDecodeCount, runningDecodeTokens,
             runningDecodeRemainingTokenBudget))) {
        throw std::runtime_error("invalid sample aggregate");
    }
    // Replay feature ablations must mask P17/P18 independently and can map the
    // remaining decode budget into the P18 parameter slot. Keep production
    // validation on the raw observation, then apply this test-only transform.
    if (hasDecodeProgress) {
        sample.features._runningDecodeRequestCount =
            scaleInteger(runningDecodeCount, scales.runningDecodeCount);
        sample.features._runningDecodeGeneratedTokenCount =
            scaleInteger(runningDecodeTokens, scales.runningDecodeTokens) +
            scaleInteger(runningDecodeRemainingTokenBudget,
                         scales.runningDecodeRemainingBudget);
    }
    sample.latencyMs = event["target_ms"].GetDouble();
    sample.eventTimeUs = event["event_time_us"].GetInt64();
    return sample;
}

void restoreAnchor(ReplayState &state, const rapidjson::Value &event,
                   bool beforeUpdate = false, bool resetControllers = true)
{
    const auto suffix = beforeUpdate ? "_before" : "_after";
    const auto readStateVector = [&](const char *name) {
        const auto fieldName = std::string(name) + suffix;
        return readVector(event, fieldName.c_str());
    };
    const auto loggedParameters = readStateVector("parameters");
    state.parameters = loggedParameters;
    state.residualParameters = readStateVector("residual_parameters");
    state.residualTimeUs = event[beforeUpdate ? "residual_time_before"
                                              : "residual_time_after"].GetInt64();
    auto moment1 = readStateVector("adam_m1");
    auto moment2 = readStateVector("adam_m2");
    NonLinearPrefillModelR::OptimizerState optimizer;
    if (moment1.size() != NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT ||
        moment2.size() != NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT) {
        throw std::runtime_error("invalid optimizer state size");
    }
    std::copy(moment1.begin(), moment1.end(), optimizer.adamMoment1.begin());
    std::copy(moment2.begin(), moment2.end(), optimizer.adamMoment2.begin());
    if (state.runningElapsedFeatureScale > 0.0 &&
        state.runningElapsedFeatureScale != 1.0) {
        const auto index =
            NonLinearPrefillModelR::RUNNING_ELAPSED_TIME_PARAMETER_INDEX;
        state.parameters[index] /= state.runningElapsedFeatureScale;
        optimizer.adamMoment1[index] *= state.runningElapsedFeatureScale;
        optimizer.adamMoment2[index] *= state.runningElapsedFeatureScale *
                                        state.runningElapsedFeatureScale;
    }
    optimizer.step = event[beforeUpdate ? "step_before" : "step_after"].GetInt64();
    if (!state.model.restoreOptimizerState(optimizer)) {
        throw std::runtime_error("restore optimizer failed");
    }
    state.previousLoggedParameters = loggedParameters;
    if (!beforeUpdate) {
        state.pending.clear();
    }
    if (!resetControllers) {
        state.anchored = true;
        return;
    }
    state.responseEma = {};
    state.responseSampleCount = {};
    state.responseSum = {};
    state.responseWeight = {};
    state.gradientMagnitudeEma.clear();
    state.baseUpdateProbes.clear();
    state.anchored = true;
}

void evaluateBaseUpdateProbes(std::ostream &output, const std::string &host,
                              double weight, uint64_t seed, uint64_t currentSequence,
                              size_t currentFitIndex,
                              const NaviSchedLatencySample &sample, ReplayState &state)
{
    for (auto &probe : state.baseUpdateProbes) {
        ++probe.horizon;
        const auto predictionBefore =
            state.model.predictConfigured(sample.features, probe.parametersBefore);
        const auto predictionAfter =
            state.model.predictConfigured(sample.features, probe.parametersAfter);
        const auto errorBefore = sample.latencyMs - predictionBefore;
        const auto errorAfter = sample.latencyMs - predictionAfter;
        const auto predictionShift = predictionAfter - predictionBefore;
        const auto direction = errorBefore < 0.0 ? -1.0 : 1.0;
        const auto alignedShift = direction * predictionShift;
        const auto alignedResponseMsPerScale = alignedShift / probe.learningRateScale;
        const auto absoluteError = std::max(100.0, std::abs(errorBefore));
        output << "R\t" << host << '\t' << weight << '\t' << seed << '\t'
               << probe.sourceSequence << '\t' << probe.sourceFitIndex << '\t'
               << currentSequence << '\t' << currentFitIndex << '\t' << probe.horizon
               << '\t' << sample.eventTimeUs << '\t' << probe.learningRateScale << '\t'
               << sample.latencyMs << '\t' << predictionBefore << '\t' << predictionAfter
               << '\t' << std::abs(errorBefore) - std::abs(errorAfter) << '\t'
               << alignedShift << '\t'
               << (std::abs(errorBefore) - std::abs(errorAfter)) / absoluteError << '\t'
               << alignedShift / absoluteError << '\t' << alignedResponseMsPerScale
               << '\n';
        if (probe.traceParameters) {
            output << "Q\t" << host << '\t' << weight << '\t' << seed << '\t'
                   << probe.sourceSequence << '\t' << probe.sourceFitIndex << '\t'
                   << currentSequence << '\t' << currentFitIndex << '\t'
                   << probe.horizon << '\t' << sample.eventTimeUs << '\t'
                   << probe.parametersBefore.size();
            for (size_t index = 0; index < probe.parametersBefore.size(); ++index) {
                auto oneParameter = probe.parametersBefore;
                oneParameter[index] = probe.parametersAfter[index];
                output << '\t'
                       << state.model.predictConfigured(sample.features, oneParameter) -
                              predictionBefore;
            }
            output << '\n';
        }
    }
    while (!state.baseUpdateProbes.empty() &&
           state.baseUpdateProbes.front().horizon >= 4) {
        state.baseUpdateProbes.pop_front();
    }
}

void writeAdamFitTrace(std::ostream &output, const std::string &host, double weight,
                       uint64_t seed, uint64_t sequence, size_t fitIndex,
                       const NaviSchedLatencySample &sample, double learningRateScale,
                       const AdamFitTrace &trace,
                       const std::vector<double> &parametersBefore,
                       const ReplayState &state)
{
    constexpr double EPSILON = 1e-20;
    const auto optimizerAfter = state.model.getOptimizerState();
    const auto predictionBefore =
        state.model.predictConfigured(sample.features, parametersBefore);
    const auto predictionAfter =
        state.model.predictConfigured(sample.features, state.parameters);
    output << "D\t" << host << '\t' << weight << '\t' << seed << '\t' << sequence
           << '\t' << fitIndex << '\t' << sample.eventTimeUs << '\t'
           << sample.latencyMs << '\t' << trace.totalSampleWeight << '\t'
           << learningRateScale << '\t' << trace.correction << '\t'
           << predictionBefore << '\t' << predictionAfter << '\t'
           << parametersBefore.size();
    for (size_t index = 0; index < parametersBefore.size(); ++index) {
        const auto candidateDelta =
            trace.unprojectedParameters[index] - parametersBefore[index];
        const auto actualDelta = state.parameters[index] - parametersBefore[index];
        auto oneParameter = parametersBefore;
        oneParameter[index] = state.parameters[index];
        const auto predictionShift =
            state.model.predictConfigured(sample.features, oneParameter) -
            predictionBefore;
        output << '\t' << trace.gradient[index] << '\t'
               << trace.moment1Before[index] << '\t'
               << trace.moment1Candidate[index] << '\t'
               << optimizerAfter.adamMoment1[index] << '\t'
               << trace.moment2Before[index] << '\t'
               << trace.moment2Candidate[index] << '\t'
               << optimizerAfter.adamMoment2[index] << '\t'
               << trace.moment1Candidate[index] /
                      std::sqrt(trace.moment2Candidate[index] + EPSILON)
               << '\t' << parametersBefore[index] << '\t'
               << trace.unprojectedParameters[index] << '\t'
               << state.parameters[index] << '\t' << candidateDelta << '\t'
               << actualDelta << '\t' << actualDelta - candidateDelta << '\t'
               << predictionShift;
    }
    output << '\n';
}

void writePrediction(std::ostream &output, const std::string &host, double weight,
                     uint64_t seed, uint64_t sequence,
                     const NaviSchedLatencySample &sample, const rapidjson::Value &event,
                     const ReplayState &state)
{
    const auto base = state.model.predictConfigured(sample.features, state.parameters);
    const auto evaluation = state.residualPredictionEnabled
                                ? LatencyResidualModel::evaluate(
                                      base, state.residualParameters, state.residualTimeUs,
                                      sample.eventTimeUs)
                                : LatencyResidualModel::Evaluation{base};
    output << "P\t" << host << '\t' << weight << '\t' << seed << '\t' << sequence << '\t'
           << sample.eventTimeUs << '\t' << sample.latencyMs << '\t'
           << sample.features.getPrecedingRequestCount() << '\t' << base << '\t'
           << evaluation.predictionMs << '\t' << evaluation.correctionMs << '\t'
           << event["base_prediction_before_ms"].GetDouble() << '\t'
           << event["prediction_before_ms"].GetDouble() << '\n';
}

void writeZeroPoint(std::ostream &output, const std::string &host, double weight,
                    uint64_t seed, uint64_t sequence, int64_t eventTimeUs,
                    const ReplayState &state)
{
    const auto base = NonLinearPrefillModelR::calculateZeroPoint(state.parameters);
    const auto evaluation = state.residualPredictionEnabled
                                ? LatencyResidualModel::evaluate(
                                      base, state.residualParameters, state.residualTimeUs,
                                      eventTimeUs)
                                : LatencyResidualModel::Evaluation{base};
    output << "Z\t" << host << '\t' << weight << '\t' << seed << '\t' << sequence << '\t'
           << eventTimeUs << '\t' << base << '\t' << evaluation.predictionMs << '\n';
}

void writeModelState(std::ostream &output, const std::string &host, double weight,
                     uint64_t seed, uint64_t sequence, size_t fitIndex,
                     const NaviSchedLatencySample &sample, const ReplayState &state)
{
    const auto base = state.model.predictConfigured(sample.features, state.parameters);
    const auto evaluation = state.residualPredictionEnabled
                                ? LatencyResidualModel::evaluate(
                                      base, state.residualParameters, state.residualTimeUs,
                                      sample.eventTimeUs)
                                : LatencyResidualModel::Evaluation{base};
    output << "M\t" << host << '\t' << weight << '\t' << seed << '\t' << sequence << '\t'
           << fitIndex << '\t' << sample.eventTimeUs << '\t' << sample.latencyMs << '\t'
           << sample.features.getBatchCount() << '\t'
           << sample.features.getRunningBatchSize() << '\t'
           << sample.features.getRunningCacheHitTokenCount() << '\t'
           << sample.features.getRunningCacheMissTokenCount() << '\t'
           << sample.features.getRunningCacheMissTokenSquareSum() << '\t'
           << sample.features.getRunningCacheHitMissTokenProductSum() << '\t'
           << sample.features.getWaitingBatchSize() << '\t'
           << sample.features.getWaitingCacheHitTokenCount() << '\t'
           << sample.features.getWaitingCacheMissTokenCount() << '\t'
           << sample.features.getWaitingCacheMissTokenSquareSum() << '\t'
           << sample.features.getWaitingCacheHitMissTokenProductSum() << '\t'
           << sample.features.getPrecedingRequestCount() << '\t'
           << sample.features.getRunningElapsedTimeMs() << '\t' << base << '\t'
           << evaluation.predictionMs << '\t' << evaluation.correctionMs << '\t'
           << state.parameters.size();
    for (const auto parameter : state.parameters) {
        output << '\t' << parameter;
    }
    output << '\t' << state.residualParameters.size();
    for (const auto parameter : state.residualParameters) {
        output << '\t' << parameter;
    }
    output << '\n';
}

std::pair<double, double> calculateFitBatchLossImpact(
    const NonLinearPrefillModelR &model,
    const std::vector<NaviSchedLatencySample> &batch,
    const std::vector<double> &parametersBefore,
    const std::vector<double> &parametersAfter,
    double currentSampleWeight)
{
    double weightedImpact = 0.0;
    double totalWeight = 0.0;
    for (size_t index = 0; index < batch.size(); ++index) {
        const auto predictionBefore =
            model.predictConfigured(batch[index].features, parametersBefore);
        const auto predictionAfter =
            model.predictConfigured(batch[index].features, parametersAfter);
        const auto target = std::log1p(batch[index].latencyMs);
        const auto residualBefore = std::log1p(predictionBefore) - target;
        const auto residualAfter = std::log1p(predictionAfter) - target;
        const auto lossBefore = 0.5 * residualBefore * residualBefore;
        const auto lossAfter = 0.5 * residualAfter * residualAfter;
        const auto sampleWeight = index + 1 == batch.size()
                                      ? currentSampleWeight
                                      : 1.0;
        weightedImpact += sampleWeight * std::abs(lossAfter - lossBefore);
        totalWeight += sampleWeight;
    }
    return {weightedImpact / totalWeight, totalWeight};
}

void applyUpdate(std::ostream &output, const std::string &host, double weight,
                 uint64_t seed, uint64_t sequence, int64_t eventTimeUs,
                 const rapidjson::Value &event, double learningRateScale,
                 const LatencyResidualModel::RuntimeConfig &residualConfig,
                 bool overrideLoggedRates, bool overrideResidualErrorLimit,
                 bool traceModelState,
                 bool freezeBase, bool freezeResidual,
                 int64_t diagnosticReanchorUntilUs,
                 const std::vector<double> *parameterLearningRateScales,
                 ReplayState &state)
{
    state.residualPredictionEnabled =
        overrideLoggedRates || !event.HasMember("residual_learning_rate")
            ? residualConfig.learningRate > 0.0
            : event["residual_learning_rate"].GetDouble() > 0.0;
    if (event.HasMember("exact_replayable") &&
        !event["exact_replayable"].GetBool()) {
        restoreAnchor(state, event, false, true);
        output << "K\t" << host << '\t' << weight << '\t' << seed << '\t'
               << sequence << '\t' << eventTimeUs
               << "\tcold_start_state_transition\n";
        ++state.seenUpdateEvents;
        return;
    }
    const auto loggedBefore = readVector(event, "parameters_before");
    if (!state.anchored) {
        const bool hasBeforeState = event.HasMember("adam_m1_before");
        if (hasBeforeState &&
            (!event.HasMember("adam_m2_before") ||
             !event.HasMember("residual_parameters_before") ||
             !event.HasMember("residual_time_before"))) {
            throw std::runtime_error("incomplete pre-update replay anchor");
        }
        // Preserve the first fit when the log contains its complete input state.
        restoreAnchor(state, event, hasBeforeState, true);
        if (!hasBeforeState) {
            return;
        }
    }
    if (freezeBase && freezeResidual) {
        state.previousLoggedParameters = readVector(event, "parameters_after");
        state.pending.clear();
        return;
    }
    if (!equalParameters(loggedBefore, state.previousLoggedParameters)) {
        restoreAnchor(state, event, false, true);
        return;
    }
    if (diagnosticReanchorUntilUs > 0 && eventTimeUs < diagnosticReanchorUntilUs) {
        restoreAnchor(state, event, true, false);
    }

    auto effectiveLearningRateScale = learningRateScale;
    auto effectiveCurrentSampleWeight = weight;
    auto effectiveResidualConfig = residualConfig;

    if (event.HasMember("base_learning_rate")) {
        state.model._learningRate = event["base_learning_rate"].GetDouble();
    }
    if (!overrideLoggedRates && event.HasMember("base_learning_rate_scale")) {
        effectiveLearningRateScale =
            event["base_learning_rate_scale"].GetDouble();
    }
    if (!overrideLoggedRates && event.HasMember("current_sample_weight")) {
        effectiveCurrentSampleWeight =
            event["current_sample_weight"].GetDouble();
    }
    if (event.HasMember("nonlinear_scale")) {
        state.model._nonlinearScale = event["nonlinear_scale"].GetDouble();
    }
    if (event.HasMember("zero_anchor_max_ms")) {
        state.model._zeroAnchorMaxMs = event["zero_anchor_max_ms"].GetDouble();
    }
    if (!overrideLoggedRates && event.HasMember("residual_learning_rate")) {
        effectiveResidualConfig.learningRate =
            event["residual_learning_rate"].GetDouble();
    }
    if (!overrideResidualErrorLimit &&
        event.HasMember("residual_update_error_floor_ms")) {
        effectiveResidualConfig.updateErrorFloorMs =
            event["residual_update_error_floor_ms"].GetDouble();
    }
    if (!overrideResidualErrorLimit &&
        event.HasMember("residual_update_error_ratio")) {
        effectiveResidualConfig.updateErrorRatio =
            event["residual_update_error_ratio"].GetDouble();
    }
    if (event.HasMember("residual_idle_decay_start_us")) {
        effectiveResidualConfig.idleDecayStartUs =
            event["residual_idle_decay_start_us"].GetInt64();
    }
    if (event.HasMember("residual_idle_decay_end_us")) {
        effectiveResidualConfig.idleDecayEndUs =
            event["residual_idle_decay_end_us"].GetInt64();
    }
    if (!std::isfinite(state.model._learningRate) ||
        state.model._learningRate <= 0.0 ||
        !std::isfinite(effectiveLearningRateScale) ||
        effectiveLearningRateScale <= 0.0 ||
        !std::isfinite(effectiveCurrentSampleWeight) ||
        effectiveCurrentSampleWeight <= 0.0 ||
        !std::isfinite(state.model._nonlinearScale) ||
        state.model._nonlinearScale <= 0.0 ||
        !std::isfinite(state.model._zeroAnchorMaxMs) ||
        state.model._zeroAnchorMaxMs <= 0.0 ||
        !effectiveResidualConfig.valid()) {
        throw std::runtime_error("invalid per-update runtime config");
    }

    std::optional<std::vector<double>> effectiveParameterLearningRateScales;
    if (event.HasMember("parameter_learning_rate_scales")) {
        effectiveParameterLearningRateScales =
            readVector(event, "parameter_learning_rate_scales");
        if (!effectiveParameterLearningRateScales->empty() &&
            (effectiveParameterLearningRateScales->size() != state.parameters.size() ||
             std::any_of(effectiveParameterLearningRateScales->begin(),
                         effectiveParameterLearningRateScales->end(),
                         [index = size_t{0}](double scale) mutable {
                             const auto current = index++;
                             return !std::isfinite(scale) || scale < 0.0 ||
                                    (scale == 0.0 &&
                                     current < NonLinearPrefillModelR::
                                                   RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX);
                         }))) {
            throw std::runtime_error("invalid update parameter learning rate scales");
        }
    }
    if (parameterLearningRateScales) {
        // Compare a counterfactual rate against the exact rate logged for this update.
        if (parameterLearningRateScales->size() != state.parameters.size()) {
            throw std::runtime_error("invalid candidate parameter learning rate scales");
        }
        if (!effectiveParameterLearningRateScales ||
            effectiveParameterLearningRateScales->empty()) {
            effectiveParameterLearningRateScales =
                std::vector<double>(state.parameters.size(), 1.0);
        }
        for (size_t index = 0; index < state.parameters.size(); ++index) {
            (*effectiveParameterLearningRateScales)[index] *=
                (*parameterLearningRateScales)[index];
        }
    }
    const auto activeParameterLearningRateScales =
        effectiveParameterLearningRateScales &&
                !effectiveParameterLearningRateScales->empty()
            ? &*effectiveParameterLearningRateScales
            : nullptr;
    if (activeParameterLearningRateScales &&
        activeParameterLearningRateScales->size() >
            NonLinearPrefillModelR::
                RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX) {
        state.model._useRunningDecodeRequestCountFeature =
            (*activeParameterLearningRateScales)
                [NonLinearPrefillModelR::
                     RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] > 0.0;
        state.model._useRunningDecodeGeneratedTokenCountFeature =
            (*activeParameterLearningRateScales)
                [NonLinearPrefillModelR::
                     RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] > 0.0;
    }

    const auto expected = static_cast<size_t>(event["step_after"].GetInt64() -
                                              event["step_before"].GetInt64());
    std::vector<std::vector<NaviSchedLatencySample>> batches;
    if (event.HasMember("fit_batches")) {
        const auto &fitBatches = event["fit_batches"];
        if (!fitBatches.IsArray()) {
            throw std::runtime_error("invalid exact fit batches");
        }
        batches.reserve(fitBatches.Size());
        for (const auto &fitBatch : fitBatches.GetArray()) {
            if (!fitBatch.IsArray() || fitBatch.Empty()) {
                throw std::runtime_error("invalid exact fit batch");
            }
            auto &batch = batches.emplace_back();
            batch.reserve(fitBatch.Size());
            for (const auto &sequenceValue : fitBatch.GetArray()) {
                const auto historySequence = sequenceValue.GetUint64();
                const auto iterator = state.historyBySequence.find(historySequence);
                if (iterator == state.historyBySequence.end()) {
                    throw std::runtime_error(
                        "exact fit batch references unknown history sequence");
                }
                batch.emplace_back(iterator->second);
            }
        }
    } else {
        batches = state.sampler.getOnlineFitBatches(state.pending.size());
    }
    if (batches.size() != expected) {
        throw std::runtime_error("online update count mismatch");
    }
    size_t fitIndex = 0;
    for (auto &batch : batches) {
        const auto current = batch.back();
        evaluateBaseUpdateProbes(output, host, weight, seed, sequence, fitIndex, current,
                                 state);
        auto currentLearningRateScale = effectiveLearningRateScale;
        auto currentResidualConfig = effectiveResidualConfig;
        const auto parametersBefore = state.parameters;
        std::vector<double> fitParameterScales;
        const auto *fitParameterLearningRateScales =
            activeParameterLearningRateScales;
        if (state.responseRateConfig) {
            fitParameterScales = activeParameterLearningRateScales
                                     ? *activeParameterLearningRateScales
                                     : std::vector<double>(state.parameters.size(), 1.0);
            output << "N\t" << host << '\t' << weight << '\t' << seed << '\t'
                   << sequence << '\t' << fitIndex;
            for (size_t index = 0; index < state.parameters.size(); ++index) {
                if (!state.responseRateConfig->enabled[index]) {
                    continue;
                }
                double factor = 1.0;
                if (state.responseSampleCount[index] >=
                        state.responseRateConfig->calibrationUpdates &&
                    state.responseEma[index] > 0.0) {
                    factor = std::clamp(
                        state.responseRateConfig->targets[index] /
                            state.responseEma[index],
                        state.responseRateConfig->minimum,
                        state.responseRateConfig->maximum);
                }
                fitParameterScales[index] *= factor;
                const auto meanResponse = state.responseWeight[index] > 0.0
                                              ? state.responseSum[index] /
                                                    state.responseWeight[index]
                                              : 0.0;
                output << '\t' << index << '\t' << factor << '\t'
                       << state.responseEma[index] << '\t'
                       << state.responseSampleCount[index] << '\t'
                       << meanResponse;
            }
            output << '\n';
            fitParameterLearningRateScales = &fitParameterScales;
        }
        if (state.featureRateConfig) {
            if (fitParameterScales.empty()) {
                fitParameterScales = activeParameterLearningRateScales
                                         ? *activeParameterLearningRateScales
                                         : std::vector<double>(state.parameters.size(),
                                                               1.0);
            }
            const auto running = current.features.getRunningBatchSize();
            if (running >= state.featureRateConfig->runningThreshold) {
                fitParameterScales[2] *= state.featureRateConfig->activeP2Factor;
            } else if (state.parameters[8] >=
                           state.featureRateConfig->minimumP8Value &&
                       state.parameters[8] <=
                           state.featureRateConfig->maximumP8Value) {
                fitParameterScales[8] *= state.featureRateConfig->otherP8Factor;
            }
            fitParameterLearningRateScales = &fitParameterScales;
            output << "F\t" << host << '\t' << weight << '\t' << seed << '\t'
                   << sequence << '\t' << fitIndex << '\t' << running << '\t'
                   << fitParameterScales[2] << '\t' << fitParameterScales[8] << '\n';
        }
        if (state.featureRateConfig &&
            state.featureRateConfig->resetP15StaleMomentum &&
            state.optimizerMode == BaseOptimizerMode::ADAM &&
            state.parameters[8] > state.featureRateConfig->maximumP8Value &&
            state.parameters[15] >
                NonLinearPrefillModelR::MIN_PRECEDING_REQUEST_COEFFICIENT_MS) {
            const auto probe = calculateAdamFitTrace(
                state.model, batch, state.parameters,
                currentLearningRateScale, effectiveCurrentSampleWeight,
                fitParameterLearningRateScales);
            const auto prediction =
                state.model.predictConfigured(current.features, state.parameters);
            if (probe.gradient[15] > 0.0 &&
                probe.moment1Before[15] < 0.0 &&
                prediction > current.latencyMs + 500.0) {
                auto optimizer = state.model.getOptimizerState();
                optimizer.adamMoment1[15] = 0.0;
                if (!state.model.restoreOptimizerState(optimizer)) {
                    throw std::runtime_error("failed to reset p15 Adam momentum");
                }
                output << "J\t" << host << '\t' << weight << '\t' << seed << '\t'
                       << sequence << '\t' << fitIndex << '\t'
                       << probe.gradient[15] << '\t'
                       << probe.moment1Before[15] << '\t'
                       << prediction << '\t' << current.latencyMs << '\t'
                       << state.parameters[8] << '\n';
            }
        }
        const auto adamFitTrace =
            traceModelState && state.optimizerMode == BaseOptimizerMode::ADAM
                ? std::make_optional(calculateAdamFitTrace(
                      state.model, batch, state.parameters,
                      currentLearningRateScale, effectiveCurrentSampleWeight,
                      fitParameterLearningRateScales))
                : std::optional<AdamFitTrace>();
        std::vector<double> updatedParameters;
        std::vector<double> sgdScales;
        std::vector<double> sgdMoment1;
        const auto fitOk = freezeBase
                               ? true
                               : state.optimizerMode == BaseOptimizerMode::SGD
                               ? fitSgd(state.model, batch, state.parameters,
                                        updatedParameters, currentLearningRateScale,
                                        effectiveCurrentSampleWeight,
                                        state.reparameterizedSgdConfig,
                                        state.gradientMagnitudeEma,
                                        traceModelState ? &sgdScales : nullptr,
                                        traceModelState ? &sgdMoment1 : nullptr)
                               : state.model.fit(batch, state.parameters,
                                                 updatedParameters,
                                                 currentLearningRateScale,
                                                 effectiveCurrentSampleWeight,
                                                 fitParameterLearningRateScales,
                                                 event.HasMember("reject_nonpositive_waiting_candidate") &&
                                                     event["reject_nonpositive_waiting_candidate"].GetBool());
        if (!fitOk) {
            throw std::runtime_error("base model fit failed");
        }
        if (!freezeBase) {
            state.parameters = std::move(updatedParameters);
            constexpr auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + 4;
            if (state.p14Floor && state.parameters[index] < *state.p14Floor) {
                state.parameters[index] = *state.p14Floor;
                auto optimizer = state.model.getOptimizerState();
                optimizer.adamMoment1[index] = std::min(0.0, optimizer.adamMoment1[index]);
                if (!state.model.restoreOptimizerState(optimizer)) {
                    throw std::runtime_error("failed to project P14 optimizer state");
                }
            }
        }
        state.baseUpdateProbes.push_back({parametersBefore, state.parameters,
                                          currentLearningRateScale, sequence, fitIndex,
                                          0, traceModelState});
        if (adamFitTrace) {
            writeAdamFitTrace(output, host, weight, seed, sequence, fitIndex,
                              current, currentLearningRateScale, *adamFitTrace,
                              parametersBefore, state);
        }
        if (traceModelState && state.optimizerMode == BaseOptimizerMode::SGD) {
            const auto traceMoment1 = state.reparameterizedSgdConfig &&
                                      state.reparameterizedSgdConfig->useAdamMoment1;
            output << (traceMoment1 ? "M\t" : "S\t") << host << '\t'
                   << weight << '\t' << seed << '\t'
                   << sequence << '\t' << fitIndex << '\t' << current.eventTimeUs
                   << '\t' << currentLearningRateScale << '\t'
                   << state.parameters.size();
            for (size_t index = 0; index < state.parameters.size(); ++index) {
                output << '\t' << parametersBefore[index] << '\t'
                       << state.parameters[index] << '\t'
                       << state.parameters[index] - parametersBefore[index] << '\t'
                       << sgdScales[index] << '\t'
                       << (state.reparameterizedSgdConfig
                               ? state.gradientMagnitudeEma[index]
                               : 0.0);
                if (traceMoment1) {
                    output << '\t' << sgdMoment1[index];
                }
            }
            output << '\n';
        }
        if (state.responseRateConfig) {
            const auto beforePrediction =
                state.model.predictConfigured(current.features, parametersBefore);
            for (size_t index = 0; index < state.parameters.size(); ++index) {
                if (!state.responseRateConfig->enabled[index]) {
                    continue;
                }
                auto oneParameter = parametersBefore;
                oneParameter[index] = state.parameters[index];
                const auto predictionShift = std::abs(
                    state.model.predictConfigured(current.features, oneParameter) -
                    beforePrediction);
                double response = predictionShift;
                switch (state.responseRateConfig->signal) {
                case ResponseNormalizedRateConfig::Signal::ABSOLUTE_PREDICTION_MS:
                    break;
                case ResponseNormalizedRateConfig::Signal::RELATIVE_PREDICTION:
                    response /= std::max(
                        {std::abs(beforePrediction), std::abs(current.latencyMs),
                         state.responseRateConfig->denominatorFloor});
                    break;
                case ResponseNormalizedRateConfig::Signal::RELATIVE_PARAMETER:
                    response = std::abs(state.parameters[index] -
                                        parametersBefore[index]) /
                               std::max(std::abs(parametersBefore[index]),
                                        state.responseRateConfig->denominatorFloor);
                    break;
                case ResponseNormalizedRateConfig::Signal::BATCH_MEAN_LOSS_IMPACT: {
                    const auto [lossImpact, batchWeight] =
                        calculateFitBatchLossImpact(
                            state.model, batch, parametersBefore, oneParameter,
                            effectiveCurrentSampleWeight);
                    response = lossImpact;
                    state.responseSum[index] += response * batchWeight;
                    state.responseWeight[index] += batchWeight;
                    break;
                }
                }
                if (!std::isfinite(response) || response <= 1e-12) {
                    continue;
                }
                if (state.responseSampleCount[index] == 0) {
                    state.responseEma[index] = response;
                } else {
                    const auto alpha = state.responseRateConfig->alpha;
                    state.responseEma[index] =
                        (1.0 - alpha) * state.responseEma[index] + alpha * response;
                }
                ++state.responseSampleCount[index];
            }
        }
        const auto base =
            state.model.predictConfigured(current.features, state.parameters);
        if (!freezeResidual &&
            (!std::isfinite(base) || base < 0.0 ||
             base > NonLinearPrefillModelR::MAX_FIT_PREDICTION_MS)) {
            throw std::runtime_error("base prediction out of range");
        }
        if (!freezeResidual && currentResidualConfig.learningRate > 0.0) {
            if (!LatencyResidualModel::update(
                    base, current.latencyMs, state.residualParameters,
                    state.residualTimeUs, current.eventTimeUs,
                    currentResidualConfig)) {
                throw std::runtime_error("residual update failed");
            }
        }
        if (traceModelState) {
            writeModelState(output, host, weight, seed, sequence, fitIndex, current,
                            state);
        }
        ++fitIndex;
    }
    const auto expectedParameters = readVector(event, "parameters_after");
    const auto expectedResidual = readVector(event, "residual_parameters_after");
    const auto expectedMoment1 = readVector(event, "adam_m1_after");
    const auto expectedMoment2 = readVector(event, "adam_m2_after");
    const auto optimizer = state.model.getOptimizerState();
    const std::vector<double> actualMoment1(optimizer.adamMoment1.begin(),
                                            optimizer.adamMoment1.end());
    const std::vector<double> actualMoment2(optimizer.adamMoment2.begin(),
                                            optimizer.adamMoment2.end());
    output << "U\t" << host << '\t' << weight << '\t' << seed << '\t' << sequence << '\t'
           << eventTimeUs << '\t' << maxDifference(state.parameters, expectedParameters)
           << '\t' << maxDifference(state.residualParameters, expectedResidual) << '\t'
           << maxDifference(actualMoment1, expectedMoment1) << '\t'
           << maxDifference(actualMoment2, expectedMoment2) << '\t'
           << std::abs(state.residualTimeUs - event["residual_time_after"].GetInt64())
           << '\n';
    state.previousLoggedParameters = readVector(event, "parameters_after");
    state.pending.clear();
    writeZeroPoint(output, host, weight, seed, sequence, eventTimeUs, state);
    ++state.seenUpdateEvents;
}

void replayFile(const std::filesystem::path &path, double weight, uint64_t seed,
                double learningRateScale, double residualLearningRate,
                const FeatureScales &featureScales, bool overrideLoggedRates,
                bool traceModelState,
                bool freezeBase, bool freezeResidual,
                int64_t diagnosticReanchorUntilUs,
                BaseOptimizerMode optimizerMode,
                const std::vector<double> *parameterLearningRateScales,
                const std::optional<ResponseNormalizedRateConfig> &responseRateConfig,
                const std::optional<FeatureConditionedRateConfig> &featureRateConfig,
                const std::optional<ReparameterizedSgdConfig> &reparameterizedSgdConfig,
                const std::optional<std::pair<double, double>> &residualErrorLimit,
                const std::optional<std::pair<int64_t, double>> &weightSwitch,
                const std::optional<double> &p14Floor,
                std::ostream &output)
{
    const auto host = path.stem().string();
    ReplayState state(seed,
                      featureScales.runningElapsed, featureScales.enableBatchCount,
                      optimizerMode, responseRateConfig, featureRateConfig);
    state.residualPredictionEnabled = residualLearningRate > 0.0;
    state.p14Floor = p14Floor;
    if (parameterLearningRateScales) {
        state.model._useRunningDecodeRequestCountFeature =
            (*parameterLearningRateScales)
                [NonLinearPrefillModelR::
                     RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] > 0.0;
        state.model._useRunningDecodeGeneratedTokenCountFeature =
            (*parameterLearningRateScales)
                [NonLinearPrefillModelR::
                     RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] > 0.0;
    } else if (responseRateConfig) {
        state.model._useRunningDecodeRequestCountFeature = true;
        state.model._useRunningDecodeGeneratedTokenCountFeature = true;
    }
    state.reparameterizedSgdConfig = reparameterizedSgdConfig;
    LatencyResidualModel::RuntimeConfig residualConfig;
    residualConfig.learningRate = residualLearningRate;
    if (residualErrorLimit) {
        residualConfig.updateErrorFloorMs = residualErrorLimit->first;
        residualConfig.updateErrorRatio = residualErrorLimit->second;
    }
    std::ifstream input(path);
    std::string line;
    uint64_t sequence = 0;
    int64_t eventTimeUs = 0;
    while (std::getline(input, line)) {
        rapidjson::Document event;
        event.Parse(line.c_str());
        if (event.HasParseError() || !event.IsObject() || !event.HasMember("type")) {
            throw std::runtime_error("invalid replay event in " + path.string());
        }
        const std::string type = event["type"].GetString();
        if (type == "sample") {
            sequence = event["sequence"].GetUint64();
            eventTimeUs = event["event_time_us"].GetInt64();
            auto sample = readSample(event, featureScales);
            if (state.anchored) {
                writePrediction(output, host, weight, seed, sequence, sample, event,
                                state);
            }
            if (event.HasMember("history_sequence")) {
                state.sampler.append(sample);
                state.historyBySequence[event["history_sequence"].GetUint64()] = sample;
            }
            state.pending.emplace_back(std::move(sample));
        } else if (type == "update") {
            const auto updateWeight = weightSwitch && eventTimeUs >= weightSwitch->first
                                          ? weightSwitch->second : weight;
            applyUpdate(output, host, updateWeight, seed, sequence, eventTimeUs, event,
                        learningRateScale, residualConfig, overrideLoggedRates,
                        residualErrorLimit.has_value(), traceModelState,
                        freezeBase, freezeResidual, diagnosticReanchorUntilUs,
                        parameterLearningRateScales, state);
        } else {
            throw std::runtime_error("unknown replay event type");
        }
    }
    if (state.responseRateConfig) {
        output << "C\t" << host;
        for (size_t index = 0; index < state.parameters.size(); ++index) {
            if (!state.responseRateConfig->enabled[index]) {
                continue;
            }
            const auto meanResponse = state.responseWeight[index] > 0.0
                                          ? state.responseSum[index] /
                                                state.responseWeight[index]
                                          : 0.0;
            output << '\t' << index << '\t' << meanResponse << '\t'
                   << state.responseWeight[index] << '\t'
                   << state.responseSampleCount[index] << '\t'
                   << state.responseEma[index];
        }
        output << '\n';
    }
}

} // namespace
} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc < 5 || argc > 12) {
        std::cerr << "usage: latency_online_replay INPUT_DIR OUTPUT "
                     "CURRENT_SAMPLE_WEIGHT SEED [LEARNING_RATE_SCALE] "
                     "[RESIDUAL_LEARNING_RATE] "
                     "[RUN_COUNT,WAIT_COUNT,RUN_TOKENS,WAIT_TOKENS,"
                     "PRECEDING,RUN_ELAPSED[,RUN_DECODE_COUNT,"
                     "RUN_DECODE_TOKENS[,RUN_DECODE_REMAINING_BUDGET]]|"
                     "all|all+batch] [TRACE_MODEL_STATE] "
                     "[fixed|counterfactual_fixed|freeze|freeze_base|freeze_residual|sgd|"
                     "parameter_rates:INDEX=MULTIPLIER[,INDEX=MULTIPLIER...]|"
                     "fixed_reanchor_diagnostic|logged_reanchor_diagnostic|"
                     "counterfactual_anchor_until:TIME_US] "
                     "[parameter_rates:INDEX=MULTIPLIER[,INDEX=MULTIPLIER...]|"
                     "response_normalize:TARGET_P2,TARGET_P3,TARGET_P5,"
                     "ALPHA,MIN,MAX,CALIBRATION_UPDATES|"
                     "prediction_feedback:TARGET_RATIO,ALPHA,MIN,MAX,"
                     "CALIBRATION_UPDATES,DENOMINATOR_FLOOR|"
                     "parameter_feedback:TARGET_RATIO,ALPHA,MIN,MAX,"
                     "CALIBRATION_UPDATES,DENOMINATOR_FLOOR|"
                     "loss_feedback:TARGET,ALPHA,MIN,MAX,CALIBRATION_UPDATES|"
                     "feature_rates:RUNNING_THRESHOLD,P2_FACTOR,P8_FACTOR"
                     "[,MIN_P8[,MAX_P8[,RESET_STALE_P15_MOMENT]]]|"
                     "p14_floor:NONNEGATIVE_VALUE|"
                     "sgd_reparameterized:EMA_ALPHA,LR_MULTIPLIER,SCALE_POWER|"
                     "momentum_reparameterized:EMA_ALPHA,LR_MULTIPLIER,SCALE_POWER "
                     "[WEIGHT_SWITCH_TIME_US:NEW_WEIGHT|"
                     "residual_error_limit:FLOOR_MS,RATIO] "
                     "(candidate multipliers apply to the active per-update rates)\n";
        return 2;
    }
    try {
        const auto weight = std::stod(argv[3]);
        const auto seed = std::stoull(argv[4]);
        const auto learningRateScale = argc >= 6 ? std::stod(argv[5]) : 6.0;
        const auto residualLearningRate = argc >= 7 ? std::stod(argv[6]) : 0.03;
        const auto featureScales = argc >= 8 ? navi_sched::parseFeatureScales(argv[7])
                                             : navi_sched::FeatureScales();
        const auto traceModelState = argc >= 9 && std::string(argv[8]) != "0";
        const auto mode = argc >= 10 ? std::string(argv[9]) : std::string();
        const bool supportedMode = mode.empty() || mode == "fixed" ||
            mode == "counterfactual_fixed" || mode == "freeze" ||
            mode == "freeze_base" || mode == "freeze_residual" || mode == "sgd" ||
            mode == "fixed_reanchor_diagnostic" || mode == "logged_reanchor_diagnostic" ||
            mode.rfind("counterfactual_anchor_until:", 0) == 0 ||
            mode.rfind("parameter_rates:", 0) == 0;
        if (!supportedMode) throw std::runtime_error("unsupported replay mode: " + mode);
        // Diagnostic reanchoring can use either logged or candidate rates.
        const bool overrideLoggedRates =
            mode != "fixed" && mode != "logged_reanchor_diagnostic";
        const bool freezeBase = mode == "freeze" || mode == "freeze_base";
        const bool freezeResidual = mode == "freeze" || mode == "freeze_residual";
        const bool fixedReanchorDiagnostic =
            mode == "fixed_reanchor_diagnostic";
        int64_t diagnosticReanchorUntilUs = 0;
        if (fixedReanchorDiagnostic || mode == "logged_reanchor_diagnostic") {
            diagnosticReanchorUntilUs = std::numeric_limits<int64_t>::max();
        } else if (mode.rfind("counterfactual_anchor_until:", 0) == 0) {
            const auto value = mode.substr(mode.find(':') + 1);
            size_t parsed = 0;
            diagnosticReanchorUntilUs = std::stoll(value, &parsed);
            if (parsed != value.size() || diagnosticReanchorUntilUs <= 0) {
                throw std::runtime_error("invalid replay anchor time");
            }
        }
        const auto reparameterizedSgdConfig =
            argc >= 11
                ? navi_sched::parseReparameterizedSgdConfig(argv[10])
                : std::optional<navi_sched::ReparameterizedSgdConfig>();
        const auto optimizerMode =
            ((argc >= 10 && std::string(argv[9]) == "sgd") ||
             reparameterizedSgdConfig)
                ? navi_sched::BaseOptimizerMode::SGD
                : navi_sched::BaseOptimizerMode::ADAM;
        const auto standaloneParameterLearningRateScales =
            argc >= 10
                ? navi_sched::parseParameterLearningRateScales(argv[9])
                : std::optional<std::vector<double>>();
        const auto secondaryParameterLearningRateScales =
            argc >= 11
                ? navi_sched::parseParameterLearningRateScales(argv[10])
                : std::optional<std::vector<double>>();
        const auto parameterLearningRateScales =
            secondaryParameterLearningRateScales
                ? secondaryParameterLearningRateScales
                : standaloneParameterLearningRateScales;
        const auto responseRateConfig =
            argc >= 11
                ? navi_sched::parseResponseNormalizedRates(argv[10])
                : std::optional<navi_sched::ResponseNormalizedRateConfig>();
        const auto featureRateConfig =
            argc >= 11
                ? navi_sched::parseFeatureConditionedRates(argv[10])
                : std::optional<navi_sched::FeatureConditionedRateConfig>();
        const auto p14Floor =
            argc >= 11 ? navi_sched::parseP14Floor(argv[10])
                       : std::optional<double>();
        if (argc >= 11 && !reparameterizedSgdConfig &&
            (optimizerMode != navi_sched::BaseOptimizerMode::ADAM ||
             (!parameterLearningRateScales && !responseRateConfig &&
              !featureRateConfig && !p14Floor))) {
            throw std::runtime_error(
                "a second mode must be parameter_rates, response_normalize, feature_rates, p14_floor, sgd_reparameterized, or momentum_reparameterized");
        }
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw std::runtime_error("weight must be positive");
        }
        if (!std::isfinite(learningRateScale) || learningRateScale <= 0.0) {
            throw std::runtime_error("learning rate scale must be positive");
        }
        if (!std::isfinite(residualLearningRate) || residualLearningRate < 0.0 ||
            residualLearningRate > 1.0) {
            throw std::runtime_error("residual learning rate must be in [0, 1]");
        }
        std::optional<std::pair<int64_t, double>> weightSwitch;
        std::optional<std::pair<double, double>> residualErrorLimit;
        if (argc >= 12) {
            const std::string argument = argv[11];
            residualErrorLimit = navi_sched::parseResidualErrorLimit(argument);
            if (!residualErrorLimit) {
                const auto separator = argument.find(':');
                if (separator == std::string::npos ||
                    separator == 0 || separator + 1 == argument.size()) {
                    throw std::runtime_error("weight switch must be TIME_US:NEW_WEIGHT");
                }
                size_t parsed = 0;
                const auto timeUs = std::stoll(argument.substr(0, separator), &parsed);
                if (parsed != separator || timeUs <= 0) {
                    throw std::runtime_error("invalid weight switch time");
                }
                const auto newWeight = std::stod(argument.substr(separator + 1), &parsed);
                if (parsed != argument.size() - separator - 1 ||
                    !std::isfinite(newWeight) || newWeight <= 0.0) {
                    throw std::runtime_error("invalid switched weight");
                }
                weightSwitch = std::make_pair(timeUs, newWeight);
            }
        }
        std::ofstream output(argv[2]);
        output << std::setprecision(17);
        std::vector<std::filesystem::path> paths;
        for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() == ".jsonl") {
                paths.emplace_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        for (const auto &path : paths) {
            navi_sched::replayFile(path, weight, seed, learningRateScale,
                                   residualLearningRate, featureScales,
                                   overrideLoggedRates, traceModelState,
                                   freezeBase, freezeResidual,
                                   diagnosticReanchorUntilUs,
                                   optimizerMode,
                                   parameterLearningRateScales
                                       ? &*parameterLearningRateScales
                                       : nullptr,
                                   responseRateConfig, featureRateConfig,
                                   reparameterizedSgdConfig,
                                   residualErrorLimit,
                                   weightSwitch,
                                   p14Floor,
                                   output);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

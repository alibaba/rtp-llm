#include "aios/navi_sched/resource/testdata/legacy_v76/LegacyV76NonLinearPrefillModelR.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "kmonitor/client/MetricsReporter.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace navi_sched {
constexpr size_t LINEAR_PARAMETER_COUNT = 7;
constexpr double TOKEN_SCALE = 1024.0;
constexpr double COEFFICIENT_3 = 320.0;
constexpr double BETA_1 = 0.9;
constexpr double BETA_2 = 0.95;
constexpr double EPSILON = 1e-20;

void reportOptimizerStep(
    const kmonitor::MetricsReporterPtr &reporter,
    const LegacyV76NonLinearPrefillModelR::OptimizerStepDiagnostics &diagnostics)
{
    if (!reporter || diagnostics.parameterCount == 0) {
        return;
    }
    reporter->report(diagnostics.step, "agent.latency_model_optimizer_step",
                     kmonitor::MetricType::GAUGE, nullptr, kmonitor::MetricLevel::FATAL);
    for (size_t index = 0; index < diagnostics.parameterCount; ++index) {
        const kmonitor::MetricsTags tags("parameter_index", std::to_string(index));
        const auto report = [&](const char *name, double value) {
            reporter->report(value, name, kmonitor::MetricType::GAUGE, &tags,
                             kmonitor::MetricLevel::FATAL);
        };
        report("agent.latency_model_raw_gradient", diagnostics.rawGradient[index]);
        report("agent.latency_model_adam_moment_1", diagnostics.adamMoment1[index]);
        report("agent.latency_model_adam_moment_2", diagnostics.adamMoment2[index]);
        report("agent.latency_model_adam_normalized_gradient",
               diagnostics.normalizedGradient[index]);
        report("agent.latency_model_adam_candidate_update",
               diagnostics.candidateUpdate[index]);
        report("agent.latency_model_actual_update", diagnostics.actualUpdate[index]);
        report("agent.latency_model_parameter_prediction_shift_ms",
               diagnostics.predictionShiftMs[index]);
    }
}

double valueOf(double value) { return value; }

template <typename Value>
double valueOf(const Value &value)
{
    return value.value();
}

const std::array<double, LegacyV76NonLinearPrefillModelR::PARAMETER_COUNT>
    INITIAL_PARAMETERS = {
        -36.56241493030776,
        2.7998347700525095,
        0.5296454912518096,
        0.27178684866553887,
        0.003629661106608585,
        0.00015333761680912973,
        0.0,
        250.19523675146382,
        1.9329205277529811,
        1e-6 / LegacyV76NonLinearPrefillModelR::PARAMETER_SCALES[9],
};
const std::array<double, LegacyV76NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT>
    CAPACITY_SPLIT_INITIAL_PARAMETERS = {
        -36.56241493030776,
        2.7998347700525095,
        0.5296454912518096,
        0.27178684866553887,
        0.003629661106608585,
        0.00015333761680912973,
        0.0,
        250.19523675146382,
        1.9329205277529811,
        1e-6 / LegacyV76NonLinearPrefillModelR::PARAMETER_SCALES[9],
        0.30370055600496004,
        0.053103146391784761,
        0.0034376753181673777,
        1.4994382069729369e-05,
        1.4458939266851801e-05,
        205.87344937782723,
        0.0,
        0.0,
        0.0,
};

const std::string LegacyV76NonLinearPrefillModelR::RESOURCE_ID =
    "navi_sched.non_linear_prefill_model_r";

void LegacyV76NonLinearPrefillModelR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool LegacyV76NonLinearPrefillModelR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "pd_separated", _pdSeparated, _pdSeparated);
    NAVI_JSONIZE(ctx, "execution_geometry_enabled", _executionGeometryEnabled,
                 _executionGeometryEnabled);
    NAVI_JSONIZE(ctx, "projected_execution_geometry_enabled",
                 _projectedExecutionGeometryEnabled, _projectedExecutionGeometryEnabled);
    NAVI_JSONIZE(ctx, "learning_rate", _learningRate, _learningRate);
    NAVI_JSONIZE(ctx, "nonlinear_scale", _nonlinearScale, _nonlinearScale);
    NAVI_JSONIZE(ctx, "zero_anchor_max_ms", _zeroAnchorMaxMs, _zeroAnchorMaxMs);
    NAVI_JSONIZE(ctx, "check_gradient", _checkGradient, _checkGradient);
    NAVI_JSONIZE(ctx, "use_batch_count_feature", _useBatchCountFeature,
                 _useBatchCountFeature);
    NAVI_JSONIZE(ctx, "use_capacity_split_feature", _useCapacitySplitFeature,
                 _useCapacitySplitFeature);
    return std::isfinite(_learningRate) && _learningRate > 0.0 &&
           std::isfinite(_nonlinearScale) && _nonlinearScale > 0.0 &&
           std::isfinite(_zeroAnchorMaxMs) && _zeroAnchorMaxMs > 0.0;
}

navi::ErrorCode LegacyV76NonLinearPrefillModelR::init(navi::ResourceInitContext &ctx)
{
    return navi::EC_NONE;
}

int32_t LegacyV76NonLinearPrefillModelR::getFormula() const
{
    return NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
}

bool LegacyV76NonLinearPrefillModelR::usesPdCoordinates() const
{
    return _pdSeparated && !_useCapacitySplitFeature;
}

int64_t LegacyV76NonLinearPrefillModelR::getBinaryVersion() const
{
    if (usesProjectedExecutionGeometry()) {
        return PROJECTED_GEOMETRY_BINARY_VERSION;
    }
    return usesExecutionGeometry()
               ? GEOMETRY_BINARY_VERSION
               : (usesPdCoordinates() ? PD_BINARY_VERSION : BINARY_VERSION);
}

const LegacyV76NonLinearPrefillModelR::ParameterScales &
LegacyV76NonLinearPrefillModelR::getConfiguredParameterScales() const
{
    return usesPdCoordinates() ? PD_PARAMETER_SCALES : PARAMETER_SCALES;
}

bool LegacyV76NonLinearPrefillModelR::supportsBinaryVersion(int64_t version)
{
    return version == BINARY_VERSION || version == PD_BINARY_VERSION ||
           usesGeometryForVersion(version);
}

std::vector<double> LegacyV76NonLinearPrefillModelR::parametersForLegacyPrediction(
    const std::vector<double> &parameters, int64_t version)
{
    if (!supportsBinaryVersion(version) ||
        ((version == PD_BINARY_VERSION || usesGeometryForVersion(version)) &&
         parameters.size() != PARAMETER_COUNT)) {
        return {};
    }
    auto result = parameters;
    if (version == PD_BINARY_VERSION || usesGeometryForVersion(version)) {
        for (size_t index = 0; index < result.size(); ++index) {
            result[index] *= PD_PARAMETER_SCALES[index] / PARAMETER_SCALES[index];
        }
    }
    return result;
}

bool LegacyV76NonLinearPrefillModelR::validateParametersForVersion(
    const std::vector<double> &parameters, int64_t version)
{
    if (!supportsBinaryVersion(version)) {
        return false;
    }
    if (version == BINARY_VERSION) {
        return validateParameters(parameters);
    }
    if (parameters.size() != PARAMETER_COUNT) {
        return false;
    }
    std::array<double, PARAMETER_COUNT> scaled;
    for (size_t index = 0; index < scaled.size(); ++index) {
        // Preserve the conversion's operation order at validation boundaries.
        scaled[index] = parameters[index];
        scaled[index] *= PD_PARAMETER_SCALES[index] / PARAMETER_SCALES[index];
    }
    return validateParameterValues(scaled.data(), scaled.size());
}

bool LegacyV76NonLinearPrefillModelR::validateConfiguredParameters(
    const std::vector<double> &parameters) const
{
    return validateParametersForVersion(parameters, getBinaryVersion());
}

std::vector<double> LegacyV76NonLinearPrefillModelR::getInitialParameters() const
{
    std::vector<double> parameters =
        _useCapacitySplitFeature
            ? std::vector<double>(CAPACITY_SPLIT_INITIAL_PARAMETERS.begin(),
                                  CAPACITY_SPLIT_INITIAL_PARAMETERS.end())
            : std::vector<double>(INITIAL_PARAMETERS.begin(), INITIAL_PARAMETERS.end());
    if (_useBatchCountFeature) {
        parameters[BATCH_COUNT_PARAMETER_INDEX] =
            positiveParameterFloor(BATCH_COUNT_PARAMETER_INDEX, 1.0);
    }
    if (_useCapacitySplitFeature && _useRunningDecodeRequestCountFeature) {
        parameters[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] =
            positiveParameterFloor(RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX, 1.0);
    }
    if (usesPdCoordinates()) {
        for (size_t index = 0; index < parameters.size(); ++index) {
            parameters[index] *= PARAMETER_SCALES[index] / PD_PARAMETER_SCALES[index];
        }
    }
    return parameters;
}

double LegacyV76NonLinearPrefillModelR::getParameterScale(size_t index)
{
    return index < PARAMETER_SCALES.size() ? PARAMETER_SCALES[index] : 0.0;
}

bool LegacyV76NonLinearPrefillModelR::validateParameters(
    const std::vector<double> &parameters)
{
    return validateParameterValues(parameters.data(), parameters.size());
}

bool LegacyV76NonLinearPrefillModelR::validateParameterValues(const double *parameters,
                                                              size_t parameterCount)
{
    if (PARAMETER_COUNT != parameterCount &&
        CAPACITY_SPLIT_PARAMETER_COUNT != parameterCount) {
        return false;
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        if (!std::isfinite(parameters[index])) {
            return false;
        }
    }
    for (const auto index : POSITIVE_PARAMETER_INDEXES) {
        if (index < parameterCount && parameters[index] < 0.0) {
            return false;
        }
    }
    if (parameters[NONLINEAR_CURVATURE_PARAMETER_INDEX] *
                PARAMETER_SCALES[NONLINEAR_CURVATURE_PARAMETER_INDEX] <
            MIN_NONLINEAR_CURVATURE ||
        (parameterCount == CAPACITY_SPLIT_PARAMETER_COUNT &&
         parameters[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                 PARAMETER_SCALES[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] <
             MIN_PRECEDING_REQUEST_COEFFICIENT_MS)) {
        return false;
    }
    return true;
}

bool LegacyV76NonLinearPrefillModelR::usesCapacitySplitParameters(
    const std::vector<double> &parameters)
{
    return CAPACITY_SPLIT_PARAMETER_COUNT == parameters.size();
}

bool LegacyV76NonLinearPrefillModelR::fit(
    const std::vector<NaviSchedLatencySample> &samples,
    const std::vector<double> &currentParameters, std::vector<double> &newParameters,
    double learningRateScale, double currentSampleWeight,
    const std::vector<double> *parameterLearningRateScales) const
{
    const auto parameterCount =
        _useCapacitySplitFeature ? CAPACITY_SPLIT_PARAMETER_COUNT : PARAMETER_COUNT;
    if (samples.empty() || !validateConfiguredParameters(currentParameters) ||
        currentParameters.size() != parameterCount || !std::isfinite(learningRateScale) ||
        learningRateScale <= 0.0 || !std::isfinite(currentSampleWeight) ||
        currentSampleWeight <= 0.0 ||
        (parameterLearningRateScales &&
         ((parameterLearningRateScales->size() != parameterCount &&
           !(parameterCount == PARAMETER_COUNT &&
             parameterLearningRateScales->size() == CAPACITY_SPLIT_PARAMETER_COUNT)) ||
          std::any_of(parameterLearningRateScales->begin(),
                      parameterLearningRateScales->end(),
                      [index = size_t{0}](double scale) mutable {
                          const auto current = index++;
                          return !std::isfinite(scale) || scale < 0.0 ||
                                 (scale == 0.0 &&
                                  current < RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX);
                      })))) {
        return false;
    }
    if (usesExecutionGeometry() &&
        std::any_of(samples.begin(), samples.end(), [](const auto &sample) {
            return !sample.features.hasExecutionGeometry();
        })) {
        return false;
    }
    const auto &scales = getConfiguredParameterScales();
    autil::ScopedLock lock(_lock);
    auto weights = toValues(currentParameters);
    auto adamMoment1 = _adamMoment1;
    auto adamMoment2 = _adamMoment2;
    const auto nonlinearScale = _nonlinearScale;
    reparameterize(weights, adamMoment1, adamMoment2, parameterCount, NONLINEAR_SCALE,
                   nonlinearScale);
    if (!_useBatchCountFeature) {
        weights[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment1[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment2[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    const auto disableParameter = [&](size_t index, bool enabled) {
        if (!enabled && index < parameterCount) {
            weights[index] = 0.0;
            adamMoment1[index] = 0.0;
            adamMoment2[index] = 0.0;
        }
    };
    disableParameter(RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX,
                     _useRunningDecodeRequestCountFeature);
    disableParameter(RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX,
                     _useRunningDecodeGeneratedTokenCountFeature);
    const auto previousOutputParameters = weights;
    const auto previousAdamMoment1 = adamMoment1;
    const auto previousAdamMoment2 = adamMoment2;
    Values gradient{};
    double totalSampleWeight = 0.0;
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const auto &sample = samples[sampleIndex];
        if (sample.features.usesCapacitySplit() != _useCapacitySplitFeature) {
            return false;
        }
        auto input = makeInput(sample.features, usesExecutionGeometry());
        if (!_useBatchCountFeature) {
            input[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        }
        if (!_useRunningDecodeRequestCountFeature) {
            input[14] = 0.0;
        }
        if (!_useRunningDecodeGeneratedTokenCountFeature) {
            input[15] = 0.0;
        }
        if (_checkGradient) {
            size_t failedIndex = 0;
            if (!checkGradient(input, weights, parameterCount, &failedIndex,
                               nonlinearScale, scales)) {
                NAVI_KERNEL_LOG(
                    ERROR,
                    "gradient check failed, sampleIndex [%lu], parameterIndex [%lu]",
                    sampleIndex, failedIndex);
                return false;
            }
        }
        Values predictionGradient{};
        auto prediction = predict(input, weights, parameterCount, &predictionGradient,
                                  nonlinearScale, scales);
        if (!std::isfinite(sample.latencyMs) || sample.latencyMs < 0.0 ||
            !std::isfinite(prediction) || prediction < 0.0) {
            ++_invalidCandidateRecoveryCount;
            NAVI_KERNEL_LOG(WARN,
                            "skip latency model update with invalid current prediction, "
                            "prediction [%f], sample_latency_ms [%f]",
                            prediction, sample.latencyMs);
            newParameters = currentParameters;
            return true;
        }
        const auto logResidual = std::log1p(prediction) - std::log1p(sample.latencyMs);
        const auto difference = logResidual / (1.0 + prediction);
        const auto sampleWeight =
            sampleIndex + 1 == samples.size() ? currentSampleWeight : 1.0;
        totalSampleWeight += sampleWeight;
        for (size_t index = 0; index < parameterCount; ++index) {
            gradient[index] += sampleWeight * difference * predictionGradient[index];
        }
    }
    const auto scaleRatio = NONLINEAR_SCALE / nonlinearScale;
    auto correction = _learningRate * learningRateScale *
                      std::sqrt(1.0 - std::pow(BETA_2, _step)) /
                      (1.0 - std::pow(BETA_1, _step));
    for (size_t index = 0; index < parameterCount; ++index) {
        gradient[index] /= totalSampleWeight;
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        adamMoment1[index] =
            BETA_1 * adamMoment1[index] + (1.0 - BETA_1) * gradient[index];
        adamMoment2[index] = BETA_2 * adamMoment2[index] +
                             (1.0 - BETA_2) * gradient[index] * gradient[index];
        const auto parameterScale =
            parameterLearningRateScales && index < parameterLearningRateScales->size()
                ? (*parameterLearningRateScales)[index]
                : 1.0;
        const auto update = correction * parameterScale * adamMoment1[index] /
                            std::sqrt(adamMoment2[index] + EPSILON);
        weights[index] -= update;
    }
    const auto updatedOutputParameters = weights;
    const auto minimumCurvature = MIN_NONLINEAR_CURVATURE * scaleRatio * scaleRatio /
                                  scales[NONLINEAR_CURVATURE_PARAMETER_INDEX];
    projectOutputParameters(weights, adamMoment1, adamMoment2, minimumCurvature,
                            scaleRatio);
    bool restoredOutputParameter = false;
    const auto restoreProjectedOutputParameter = [&](size_t index, double minimumValue) {
        if (previousOutputParameters[index] > minimumValue &&
            updatedOutputParameters[index] <= minimumValue) {
            restoredOutputParameter = true;
            weights[index] = previousOutputParameters[index];
            adamMoment1[index] = previousAdamMoment1[index];
            adamMoment2[index] = previousAdamMoment2[index];
            NAVI_KERNEL_LOG(
                WARN,
                "restore latency output parameter after projection, index [%lu], "
                "updated [%f], minimum [%f], restored [%f]",
                index, updatedOutputParameters[index], minimumValue, weights[index]);
        }
    };
    restoreProjectedOutputParameter(
        BASE_LATENCY_PARAMETER_INDEX,
        positiveParameterFloor(BASE_LATENCY_PARAMETER_INDEX, 1.0));
    restoreProjectedOutputParameter(NONLINEAR_CURVATURE_PARAMETER_INDEX,
                                    minimumCurvature);
    if (restoredOutputParameter) {
        _outputParameterProjectionRecoveryCount.fetch_add(1, std::memory_order_relaxed);
    }
    const auto &latestSample = samples.back();
    auto latestInput = makeInput(latestSample.features, usesExecutionGeometry());
    if (!_useBatchCountFeature) {
        latestInput[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeRequestCountFeature) {
        latestInput[14] = 0.0;
    }
    if (!_useRunningDecodeGeneratedTokenCountFeature) {
        latestInput[15] = 0.0;
    }
    reparameterize(weights, adamMoment1, adamMoment2, parameterCount, nonlinearScale,
                   NONLINEAR_SCALE);
    const auto candidateOutputParameters = weights;
    std::vector<double> outputParameters(weights.begin(),
                                         weights.begin() + parameterCount);
    if (!validateConfiguredParameters(outputParameters)) {
        return false;
    }
    bool resetUnsafeWaitingState = false;
    for (const auto &sample : samples) {
        double waitingCost = 0.0;
        if (_useCapacitySplitFeature &&
            !hasNonNegativeWaitingWorkloadCost(outputParameters, sample.features,
                                               &waitingCost)) {
            _unsafeWaitingCandidateRecoveryCount.fetch_add(1, std::memory_order_relaxed);
            NAVI_KERNEL_LOG(WARN,
                            "skip latency model update with negative waiting workload "
                            "cost, waiting_cost [%f], waiting_batch_size [%ld], "
                            "sample_latency_ms [%f]",
                            waitingCost, sample.features.getWaitingBatchSize(),
                            sample.latencyMs);
            if (_metricsReporterR) {
                const auto &reporter = _metricsReporterR->getReporter();
                if (reporter) {
                    reporter->report(
                        1, "agent.latency_model_unsafe_waiting_candidate_recovery_qps",
                        kmonitor::MetricType::QPS, nullptr, kmonitor::MetricLevel::FATAL);
                }
            }
            double currentWaitingCost = 0.0;
            const auto currentSafe = hasNonNegativeWaitingWorkloadCost(
                currentParameters, sample.features, &currentWaitingCost);
            const auto replacementParameters =
                currentSafe ? currentParameters : getInitialParameters();
            for (size_t index = 0; index < WAITING_PARAMETER_COUNT; ++index) {
                const auto parameterIndex = WAITING_PARAMETER_OFFSET + index;
                weights[parameterIndex] = replacementParameters[parameterIndex];
                outputParameters[parameterIndex] = replacementParameters[parameterIndex];
                adamMoment1[parameterIndex] =
                    currentSafe ? _adamMoment1[parameterIndex] : 0.0;
                adamMoment2[parameterIndex] =
                    currentSafe ? _adamMoment2[parameterIndex] : 0.0;
            }
            if (!currentSafe && !resetUnsafeWaitingState) {
                resetUnsafeWaitingState = true;
                _unsafeWaitingStateResetCount.fetch_add(1, std::memory_order_relaxed);
                NAVI_KERNEL_LOG(ERROR,
                                "reset unsafe waiting latency parameters, "
                                "current_waiting_cost [%f], candidate_waiting_cost [%f], "
                                "waiting_batch_size [%ld]",
                                currentWaitingCost, waitingCost,
                                sample.features.getWaitingBatchSize());
                if (_metricsReporterR) {
                    const auto &reporter = _metricsReporterR->getReporter();
                    if (reporter) {
                        reporter->report(
                            1, "agent.latency_model_unsafe_waiting_state_reset_qps",
                            kmonitor::MetricType::QPS, nullptr,
                            kmonitor::MetricLevel::FATAL);
                    }
                }
            }
            if (_useCapacitySplitFeature &&
                !hasNonNegativeWaitingWorkloadCost(outputParameters, sample.features,
                                                   &waitingCost)) {
                return false;
            }
        }
        const auto candidatePrediction =
            predictConfigured(sample.features, outputParameters);
        if (!std::isfinite(candidatePrediction) || candidatePrediction < 0.0) {
            // Keep a bad update from poisoning future batches. The copied Adam
            // state is intentionally discarded together with this candidate.
            ++_invalidCandidateRecoveryCount;
            NAVI_KERNEL_LOG(
                WARN,
                "skip latency model update with invalid candidate prediction, "
                "prediction [%f], sample_latency_ms [%f]",
                candidatePrediction, sample.latencyMs);
            newParameters = currentParameters;
            return true;
        }
    }
    const OptimizerState outputOptimizer{adamMoment1, adamMoment2, _step + 1};
    if (!validateOptimizerState(outputOptimizer, parameterCount)) {
        return false;
    }
    OptimizerStepDiagnostics diagnostics;
    diagnostics.parameterCount = parameterCount;
    diagnostics.step = _step;
    const auto previousParameters = toValues(currentParameters);
    const auto predictionBefore = predict(latestInput, previousParameters, parameterCount,
                                          nullptr, NONLINEAR_SCALE, scales);
    for (size_t index = 0; index < parameterCount; ++index) {
        diagnostics.rawGradient[index] = gradient[index];
        diagnostics.adamMoment1[index] = adamMoment1[index];
        diagnostics.adamMoment2[index] = adamMoment2[index];
        diagnostics.normalizedGradient[index] =
            adamMoment1[index] / std::sqrt(adamMoment2[index] + EPSILON);
        diagnostics.candidateUpdate[index] =
            candidateOutputParameters[index] - previousParameters[index];
        diagnostics.actualUpdate[index] = weights[index] - previousParameters[index];
        auto oneParameterUpdated = previousParameters;
        oneParameterUpdated[index] = weights[index];
        diagnostics.predictionShiftMs[index] =
            predict(latestInput, oneParameterUpdated, parameterCount, nullptr,
                    NONLINEAR_SCALE, scales) -
            predictionBefore;
    }
    _adamMoment1 = adamMoment1;
    _adamMoment2 = adamMoment2;
    _optimizerStepDiagnostics = diagnostics;
    ++_step;
    newParameters.assign(weights.begin(), weights.begin() + parameterCount);
    if (_metricsReporterR) {
        reportOptimizerStep(_metricsReporterR->getReporter(), diagnostics);
    }
    return true;
}

void LegacyV76NonLinearPrefillModelR::applyRuntimeConfig(
    const AgentSchedulerConfig &config)
{
    autil::ScopedLock lock(_lock);
    _learningRate = config.latencyLearningRate;
    _checkGradient = config.latencyCheckGradient;
    _nonlinearScale = config.latencyNonlinearScale;
    _zeroAnchorMaxMs = config.latencyZeroAnchorMaxMs;
    const auto &scales = config.latencyParameterLearningRateScales;
    _useRunningDecodeRequestCountFeature =
        scales.size() == CAPACITY_SPLIT_PARAMETER_COUNT &&
        scales[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] > 0.0;
    _useRunningDecodeGeneratedTokenCountFeature =
        scales.size() == CAPACITY_SPLIT_PARAMETER_COUNT &&
        scales[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] > 0.0;
    if (!_useRunningDecodeRequestCountFeature) {
        _adamMoment1[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = 0.0;
        _adamMoment2[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeGeneratedTokenCountFeature) {
        _adamMoment1[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] = 0.0;
        _adamMoment2[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] = 0.0;
    }
}

double LegacyV76NonLinearPrefillModelR::getNonlinearScale() const
{
    autil::ScopedLock lock(_lock);
    return _nonlinearScale;
}

bool LegacyV76NonLinearPrefillModelR::fitHistory(
    const std::deque<NaviSchedLatencySample> &history, size_t fitBatchSize,
    size_t epochCount, const std::vector<double> &currentParameters,
    std::vector<double> &newParameters) const
{
    if (history.empty() || fitBatchSize == 0 || epochCount == 0 ||
        !validateConfiguredParameters(currentParameters)) {
        return false;
    }
    auto parameters = currentParameters;
    std::vector<NaviSchedLatencySample> batch;
    batch.reserve(fitBatchSize);
    for (size_t epoch = 0; epoch < epochCount; ++epoch) {
        for (size_t offset = 0; offset < history.size(); ++offset) {
            batch.emplace_back(history[(epoch + offset) % history.size()]);
            if (batch.size() != fitBatchSize) {
                continue;
            }
            if (!fit(batch, parameters, newParameters, 1.0, 1.0)) {
                return false;
            }
            parameters = std::move(newParameters);
            batch.clear();
        }
    }
    if (!batch.empty() && !fit(batch, parameters, newParameters, 1.0, 1.0)) {
        return false;
    }
    if (batch.empty()) {
        newParameters = std::move(parameters);
    }
    return validateConfiguredParameters(newParameters);
}

double LegacyV76NonLinearPrefillModelR::predict(const NaviSchedLatencyFeatures &features,
                                                const std::vector<double> &parameters)
{
    if (!validateParameters(parameters) ||
        features.usesCapacitySplit() != usesCapacitySplitParameters(parameters)) {
        return 0.0;
    }
    return predict(makeInput(features), toValues(parameters), parameters.size(), nullptr);
}

double LegacyV76NonLinearPrefillModelR::predictForVersion(
    const NaviSchedLatencyFeatures &features, const std::vector<double> &parameters,
    int64_t binaryVersion)
{
    const bool geometry = usesGeometryForVersion(binaryVersion);
    if (!validateParametersForVersion(parameters, binaryVersion) ||
        features.usesCapacitySplit() != usesCapacitySplitParameters(parameters) ||
        (geometry && !features.hasExecutionGeometry())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto &scales =
        binaryVersion == BINARY_VERSION ? PARAMETER_SCALES : PD_PARAMETER_SCALES;
    return predict(makeInput(features, geometry), toValues(parameters), parameters.size(),
                   nullptr, NONLINEAR_SCALE, scales);
}

double LegacyV76NonLinearPrefillModelR::calculateZeroPoint(
    const std::vector<double> &parameters, std::vector<double> *gradient,
    int64_t binaryVersion)
{
    if (!validateParametersForVersion(parameters, binaryVersion)) {
        if (gradient) {
            gradient->clear();
        }
        return 0.0;
    }
    const auto &scales =
        (binaryVersion == PD_BINARY_VERSION || usesGeometryForVersion(binaryVersion))
            ? PD_PARAMETER_SCALES
            : PARAMETER_SCALES;
    if (gradient) {
        gradient->assign(parameters.size(), 0.0);
        (*gradient)[ZERO_POINT_PARAMETER_INDEX] = scales[ZERO_POINT_PARAMETER_INDEX];
    }
    return parameters[ZERO_POINT_PARAMETER_INDEX] * scales[ZERO_POINT_PARAMETER_INDEX];
}

bool LegacyV76NonLinearPrefillModelR::checkGradientConfigured(
    const NaviSchedLatencyFeatures &features, const std::vector<double> &parameters) const
{
    if (!validateConfiguredParameters(parameters) ||
        features.usesCapacitySplit() != _useCapacitySplitFeature ||
        parameters.size() != getInitialParameters().size() ||
        (usesExecutionGeometry() && !features.hasExecutionGeometry())) {
        return false;
    }
    auto input = makeInput(features, usesExecutionGeometry());
    if (!_useBatchCountFeature) {
        input[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeRequestCountFeature) {
        input[14] = 0.0;
    }
    if (!_useRunningDecodeGeneratedTokenCountFeature) {
        input[15] = 0.0;
    }
    return checkGradient(input, toValues(parameters), parameters.size(), nullptr,
                         NONLINEAR_SCALE, getConfiguredParameterScales());
}

double LegacyV76NonLinearPrefillModelR::predictConfigured(
    const NaviSchedLatencyFeatures &features, const std::vector<double> &parameters) const
{
    if (usesExecutionGeometry() && !features.hasExecutionGeometry()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!validateConfiguredParameters(parameters) ||
        features.usesCapacitySplit() != _useCapacitySplitFeature ||
        parameters.size() != getInitialParameters().size()) {
        return 0.0;
    }
    auto input = makeInput(features, usesExecutionGeometry());
    if (!_useBatchCountFeature) {
        input[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeRequestCountFeature) {
        input[14] = 0.0;
    }
    if (!_useRunningDecodeGeneratedTokenCountFeature) {
        input[15] = 0.0;
    }
    return predict(input, toValues(parameters), parameters.size(), nullptr,
                   NONLINEAR_SCALE, getConfiguredParameterScales());
}

std::optional<LegacyV76NonLinearPrefillModelR::NonlinearOperatingPoint>
LegacyV76NonLinearPrefillModelR::getNonlinearOperatingPointConfigured(
    const NaviSchedLatencyFeatures &features, const std::vector<double> &parameters) const
{
    if (usesExecutionGeometry() && !features.hasExecutionGeometry()) {
        return std::nullopt;
    }
    if (!validateConfiguredParameters(parameters) ||
        features.usesCapacitySplit() != _useCapacitySplitFeature ||
        parameters.size() != getInitialParameters().size()) {
        return std::nullopt;
    }
    auto input = makeInput(features, usesExecutionGeometry());
    if (!_useBatchCountFeature) {
        input[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeRequestCountFeature) {
        input[14] = 0.0;
    }
    if (!_useRunningDecodeGeneratedTokenCountFeature) {
        input[15] = 0.0;
    }
    const auto weights = toValues(parameters);
    const auto u = evaluateLinear(input, weights, getConfiguredParameterScales());
    const auto curvature =
        weights[NONLINEAR_CURVATURE_PARAMETER_INDEX] *
        getConfiguredParameterScales()[NONLINEAR_CURVATURE_PARAMETER_INDEX];
    const auto r = u / std::sqrt(curvature);
    if (!std::isfinite(r)) {
        return std::nullopt;
    }
    return NonlinearOperatingPoint{u, curvature, r, r < -3.0 ? -1 : (r > 3.0 ? 1 : 0)};
}

double LegacyV76NonLinearPrefillModelR::calculateRequestLinearCost(
    const std::vector<double> &weights, int64_t totalTokenCount,
    int64_t cacheHitTokenCount)
{
    if (weights.size() < LINEAR_PARAMETER_COUNT) {
        return 0.0;
    }
    NaviSchedLatencyFeatures features;
    features.appendRequest(cacheHitTokenCount, totalTokenCount - cacheHitTokenCount);
    return calculateLinearCost(weights, features);
}

double LegacyV76NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
    const std::vector<double> &weights, int64_t totalTokenCount,
    int64_t cacheHitTokenCount)
{
    if (!usesCapacitySplitParameters(weights)) {
        return calculateRequestLinearCost(weights, totalTokenCount, cacheHitTokenCount);
    }
    NaviSchedLatencyFeatures features;
    features.appendWaitingRequest(cacheHitTokenCount,
                                  totalTokenCount - cacheHitTokenCount);
    return calculateLinearCost(weights, features);
}

double LegacyV76NonLinearPrefillModelR::calculateLinearCost(
    const std::vector<double> &weights, const NaviSchedLatencyFeatures &features)
{
    if (weights.size() < LINEAR_PARAMETER_COUNT) {
        return 0.0;
    }
    auto input = makeInput(features);
    double cost = 0.0;
    for (size_t index = 1; index < LINEAR_PARAMETER_COUNT; ++index) {
        cost += input[index] * weights[index] * PARAMETER_SCALES[index];
    }
    if (usesCapacitySplitParameters(weights) && features.usesCapacitySplit()) {
        for (size_t index = 0; index < WAITING_PARAMETER_COUNT; ++index) {
            const auto parameterIndex = WAITING_PARAMETER_OFFSET + index;
            cost += input[LINEAR_PARAMETER_COUNT + index] * weights[parameterIndex] *
                    PARAMETER_SCALES[parameterIndex];
        }
        cost += input[13] * weights[RUNNING_ELAPSED_TIME_PARAMETER_INDEX] *
                PARAMETER_SCALES[RUNNING_ELAPSED_TIME_PARAMETER_INDEX];
        cost += input[14] * weights[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] *
                PARAMETER_SCALES[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX];
        cost += input[15] *
                weights[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] *
                PARAMETER_SCALES[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX];
    }
    return cost;
}

bool LegacyV76NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(
    const std::vector<double> &weights, const NaviSchedLatencyFeatures &features,
    double *waitingCost)
{
    if (waitingCost) {
        *waitingCost = 0.0;
    }
    if (!validateParameters(weights) ||
        features.usesCapacitySplit() != usesCapacitySplitParameters(weights)) {
        return false;
    }
    if (!features.usesCapacitySplit()) {
        return true;
    }
    const auto input = makeInput(features);
    double cost = 0.0;
    double absoluteTermSum = 0.0;
    for (size_t index = 0; index < WAITING_PARAMETER_COUNT; ++index) {
        const auto parameterIndex = WAITING_PARAMETER_OFFSET + index;
        const auto term = input[LINEAR_PARAMETER_COUNT + index] *
                          weights[parameterIndex] * PARAMETER_SCALES[parameterIndex];
        cost += term;
        absoluteTermSum += std::abs(term);
    }
    if (waitingCost) {
        *waitingCost = cost;
    }
    const auto tolerance = 1e-12 * std::max(1.0, absoluteTermSum);
    return std::isfinite(cost) && cost >= -tolerance;
}

AutoGradTape::Value LegacyV76NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(
    const std::vector<double> &weights, const AutoGradTape::Value &requestCostSum,
    const AutoGradTape::Value &totalPromptTokenCount, double precedingRequestCount)
{
    auto linear = (requestCostSum + weights[0] * PARAMETER_SCALES[0] +
                   weights[BATCH_COUNT_PARAMETER_INDEX] *
                       PARAMETER_SCALES[BATCH_COUNT_PARAMETER_INDEX]) /
                  COEFFICIENT_3;
    const auto precedingRequestLatency =
        usesCapacitySplitParameters(weights)
            ? weights[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                  PARAMETER_SCALES[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                  precedingRequestCount
            : 0.0;
    auto latency = evaluateNonLinear(linear, weights) +
                   weights[TOTAL_PROMPT_PARAMETER_INDEX] *
                       PARAMETER_SCALES[TOTAL_PROMPT_PARAMETER_INDEX] *
                       totalPromptTokenCount / TOKEN_SCALE +
                   precedingRequestLatency;
    return latency.value() < 0.0 ? AutoGradTape::Value(0.0) : latency;
}

double LegacyV76NonLinearPrefillModelR::calculateLatencyAndDerivativesFromRequestCostSum(
    const std::vector<double> &weights, double requestCostSum,
    double totalPromptTokenCount, double precedingRequestCount,
    double &requestCostDerivative, double &totalPromptTokenDerivative)
{
    auto requestCost = AutoGrad<1>::parameter(requestCostSum, 0);
    auto linear = (requestCost + weights[0] * PARAMETER_SCALES[0] +
                   weights[BATCH_COUNT_PARAMETER_INDEX] *
                       PARAMETER_SCALES[BATCH_COUNT_PARAMETER_INDEX]) /
                  COEFFICIENT_3;
    auto nonlinearLatency = evaluateNonLinear(linear, weights);
    requestCostDerivative = nonlinearLatency.gradient(0);
    totalPromptTokenDerivative = weights[TOTAL_PROMPT_PARAMETER_INDEX] *
                                 PARAMETER_SCALES[TOTAL_PROMPT_PARAMETER_INDEX] /
                                 TOKEN_SCALE;
    const auto precedingRequestLatency =
        usesCapacitySplitParameters(weights)
            ? weights[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                  PARAMETER_SCALES[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                  precedingRequestCount
            : 0.0;
    auto latency = nonlinearLatency.value() +
                   weights[TOTAL_PROMPT_PARAMETER_INDEX] *
                       PARAMETER_SCALES[TOTAL_PROMPT_PARAMETER_INDEX] *
                       totalPromptTokenCount / TOKEN_SCALE +
                   precedingRequestLatency;
    if (latency < 0.0) {
        requestCostDerivative = 0.0;
        totalPromptTokenDerivative = 0.0;
        return 0.0;
    }
    return latency;
}

bool LegacyV76NonLinearPrefillModelR::checkGradient(
    const NaviSchedLatencyFeatures &features, const std::vector<double> &parameters)
{
    if (!validateParameters(parameters) ||
        features.usesCapacitySplit() != usesCapacitySplitParameters(parameters)) {
        return false;
    }
    return checkGradient(makeInput(features), toValues(parameters), parameters.size(),
                         nullptr);
}

LegacyV76NonLinearPrefillModelR::OptimizerState
LegacyV76NonLinearPrefillModelR::getOptimizerState() const
{
    autil::ScopedLock lock(_lock);
    return {_adamMoment1, _adamMoment2, _step};
}

LegacyV76NonLinearPrefillModelR::OptimizerStepDiagnostics
LegacyV76NonLinearPrefillModelR::getOptimizerStepDiagnostics() const
{
    autil::ScopedLock lock(_lock);
    return _optimizerStepDiagnostics;
}

bool LegacyV76NonLinearPrefillModelR::validateOptimizerState(const OptimizerState &state,
                                                             size_t parameterCount)
{
    if ((parameterCount != PARAMETER_COUNT &&
         parameterCount != CAPACITY_SPLIT_PARAMETER_COUNT) ||
        state.step <= 0) {
        return false;
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        if (!std::isfinite(state.adamMoment1[index]) ||
            !std::isfinite(state.adamMoment2[index]) || state.adamMoment2[index] < 0.0) {
            return false;
        }
    }
    return true;
}

NaviSchedLatencyOptimizerState LegacyV76NonLinearPrefillModelR::makeOptimizerStateData(
    const OptimizerState &state, size_t parameterCount)
{
    NaviSchedLatencyOptimizerState data;
    if (!validateOptimizerState(state, parameterCount)) {
        return data;
    }
    data.adamMoment1.assign(state.adamMoment1.begin(),
                            state.adamMoment1.begin() + parameterCount);
    data.adamMoment2.assign(state.adamMoment2.begin(),
                            state.adamMoment2.begin() + parameterCount);
    data.step = state.step;
    return data;
}

bool LegacyV76NonLinearPrefillModelR::makeOptimizerState(
    const NaviSchedLatencyOptimizerState &data, OptimizerState &state)
{
    if (!data.valid(data.adamMoment1.size()) ||
        (data.adamMoment1.size() != PARAMETER_COUNT &&
         data.adamMoment1.size() != CAPACITY_SPLIT_PARAMETER_COUNT)) {
        return false;
    }
    state = OptimizerState();
    std::copy(data.adamMoment1.begin(), data.adamMoment1.end(),
              state.adamMoment1.begin());
    std::copy(data.adamMoment2.begin(), data.adamMoment2.end(),
              state.adamMoment2.begin());
    state.step = data.step;
    return true;
}

bool LegacyV76NonLinearPrefillModelR::sanitizeInactiveParameters(
    std::vector<double> &parameters, OptimizerState &state) const
{
    if (!validateConfiguredParameters(parameters) ||
        !validateOptimizerState(state, parameters.size())) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    const auto clear = [&](size_t index, bool enabled) {
        if (!enabled && index < parameters.size()) {
            parameters[index] = 0.0;
            state.adamMoment1[index] = 0.0;
            state.adamMoment2[index] = 0.0;
        }
    };
    clear(RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX,
          _useRunningDecodeRequestCountFeature);
    clear(RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX,
          _useRunningDecodeGeneratedTokenCountFeature);
    return true;
}

bool LegacyV76NonLinearPrefillModelR::restoreOptimizerState(const OptimizerState &state)
{
    const auto parameterCount = getInitialParameters().size();
    if (!validateOptimizerState(state, parameterCount)) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    _adamMoment1 = state.adamMoment1;
    _adamMoment2 = state.adamMoment2;
    _step = state.step;
    _optimizerStepDiagnostics = {};
    return true;
}

void LegacyV76NonLinearPrefillModelR::resetOptimizerState()
{
    autil::ScopedLock lock(_lock);
    _adamMoment1 = {};
    _adamMoment2 = {};
    _step = 1;
    _optimizerStepDiagnostics = {};
}

bool LegacyV76NonLinearPrefillModelR::usesBatchCountFeature() const
{
    return _useBatchCountFeature;
}

bool LegacyV76NonLinearPrefillModelR::checkGradient(
    const Input &input, const Values &weights, size_t parameterCount, size_t *failedIndex,
    double nonlinearScale, const ParameterScales &scales)
{
    Values gradient{};
    auto prediction =
        predict(input, weights, parameterCount, &gradient, nonlinearScale, scales);
    for (size_t index = 0; index < parameterCount; ++index) {
        auto testWeights = weights;
        const auto parameterScale = scales[index];
        const auto formulaCoefficient = testWeights[index] * parameterScale;
        const auto formulaDelta =
            0.0 == formulaCoefficient
                ? 1e-6
                : std::max(-10.0, std::min(10.0, formulaCoefficient * 0.0001));
        const auto delta = formulaDelta / parameterScale;
        testWeights[index] += delta;
        auto difference =
            predict(input, testWeights, parameterCount, nullptr, nonlinearScale, scales) -
            prediction;
        const auto expectedDifference = delta * gradient[index];
        const auto error = std::abs(expectedDifference - difference);
        const auto scale =
            std::max({std::abs(expectedDifference), std::abs(difference), 1e-9});
        if (error > std::max(1e-8, scale * 0.01)) {
            if (failedIndex) {
                *failedIndex = index;
            }
            return false;
        }
    }
    return true;
}

navi::SummaryErrorCode LegacyV76NonLinearPrefillModelR::getSummary(
    const std::string &query, navi::SummaryBuilder &builder) const
{
    Values adamMoment1;
    Values adamMoment2;
    int64_t step = 0;
    double zeroAnchorMaxMs = 0.0;
    {
        autil::ScopedLock lock(_lock);
        adamMoment1 = _adamMoment1;
        adamMoment2 = _adamMoment2;
        step = _step;
        zeroAnchorMaxMs = _zeroAnchorMaxMs;
    }
    auto overview = builder.newTable("NonLinearPrefillModelOverview");
    TABLE_ADD_COLUMN_VALUE(overview, "formula_id", getFormula());
    TABLE_ADD_COLUMN_VALUE(overview, "formula_name",
                           getNaviSchedLatencyFormulaName(getFormula()));
    TABLE_ADD_COLUMN_VALUE(overview, "binary_version", getBinaryVersion());
    const auto parameterCount = getInitialParameters().size();
    TABLE_ADD_COLUMN_VALUE(overview, "parameter_count", parameterCount);
    TABLE_ADD_COLUMN_VALUE(overview, "use_batch_count_feature", _useBatchCountFeature);
    TABLE_ADD_COLUMN_VALUE(overview, "use_capacity_split_feature",
                           _useCapacitySplitFeature);
    TABLE_ADD_COLUMN_VALUE(overview, "fit_step", step - 1);
    TABLE_ADD_COLUMN_VALUE(overview, "nonlinear_scale", getNonlinearScale());
    TABLE_ADD_COLUMN_VALUE(overview, "zero_anchor_max_ms", zeroAnchorMaxMs);
    TABLE_ADD_COLUMN_VALUE(
        overview, "output_parameter_projection_recovery_count",
        _outputParameterProjectionRecoveryCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        overview, "invalid_candidate_recovery_count",
        _invalidCandidateRecoveryCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        overview, "unsafe_waiting_candidate_recovery_count",
        _unsafeWaitingCandidateRecoveryCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "unsafe_waiting_state_reset_count",
                           _unsafeWaitingStateResetCount.load(std::memory_order_relaxed));
    if (SummaryUtil::getSummaryLevel(query) < 2) {
        return navi::SEC_OK;
    }
    auto optimizer = builder.newTable("NonLinearPrefillOptimizer");
    for (size_t index = 0; index < parameterCount; ++index) {
        TABLE_ADD_COLUMN_VALUE(optimizer, "index", index);
        TABLE_ADD_COLUMN_VALUE(optimizer, "adam_moment_1", adamMoment1[index]);
        TABLE_ADD_COLUMN_VALUE(optimizer, "adam_moment_2", adamMoment2[index]);
    }
    return navi::SEC_OK;
}

LegacyV76NonLinearPrefillModelR::Input LegacyV76NonLinearPrefillModelR::makeInput(
    const NaviSchedLatencyFeatures &features, bool geometry)
{
    auto batchSize = features.getRunningBatchSize();
    auto reuse = features.getRunningCacheHitTokenCount() / TOKEN_SCALE;
    auto compute = features.getRunningCacheMissTokenCount() / TOKEN_SCALE;
    auto computeSquare =
        features.getRunningCacheMissTokenSquareSum() / (TOKEN_SCALE * TOKEN_SCALE);
    auto reuseCompute =
        features.getRunningCacheHitMissTokenProductSum() / (TOKEN_SCALE * TOKEN_SCALE);
    auto waitingBatchSize = features.getWaitingBatchSize();
    auto waitingReuse = features.getWaitingCacheHitTokenCount() / TOKEN_SCALE;
    auto waitingCompute = features.getWaitingCacheMissTokenCount() / TOKEN_SCALE;
    auto waitingComputeSquare =
        features.getWaitingCacheMissTokenSquareSum() / (TOKEN_SCALE * TOKEN_SCALE);
    auto waitingReuseCompute =
        features.getWaitingCacheHitMissTokenProductSum() / (TOKEN_SCALE * TOKEN_SCALE);
    return {1.0,
            static_cast<double>(batchSize),
            reuse,
            geometry ? features.getGlobalPaddedTokens() / TOKEN_SCALE : compute,
            computeSquare,
            reuseCompute,
            static_cast<double>(geometry ? features.getExecutionStepCount()
                                         : features._batchCount),
            static_cast<double>(waitingBatchSize),
            waitingReuse,
            waitingCompute,
            waitingComputeSquare,
            waitingReuseCompute,
            static_cast<double>(features._precedingRequestCount),
            features._runningElapsedTimeMs / RUNNING_ELAPSED_TIME_SCALE_MS,
            static_cast<double>(features._runningDecodeRequestCount),
            features._runningDecodeGeneratedTokenCount / TOKEN_SCALE,
            reuse + compute + waitingReuse + waitingCompute};
}

bool LegacyV76NonLinearPrefillModelR::isZeroWorkload(
    const NaviSchedLatencyFeatures &features)
{
    if (features.getBatchCount() != 0 || features.getPrecedingRequestCount() != 0 ||
        features.getRunningElapsedTimeMs() != 0.0 ||
        features.getRunningDecodeRequestCount() != 0 ||
        features.getRunningDecodeGeneratedTokenCount() != 0 ||
        features.getRunningDecodeRemainingTokenBudget() != 0) {
        return false;
    }
    if (!features.usesCapacitySplit()) {
        return features.getBatchSize() == 0 && features.getCacheHitTokenCount() == 0 &&
               features.getCacheMissTokenCount() == 0 &&
               features.getCacheMissTokenSquareSum() == 0.0 &&
               features.getCacheHitMissTokenProductSum() == 0.0;
    }
    return features.getRunningBatchSize() == 0 &&
           features.getRunningCacheHitTokenCount() == 0 &&
           features.getRunningCacheMissTokenCount() == 0 &&
           features.getRunningCacheMissTokenSquareSum() == 0.0 &&
           features.getRunningCacheHitMissTokenProductSum() == 0.0 &&
           features.getWaitingBatchSize() == 0 &&
           features.getWaitingCacheHitTokenCount() == 0 &&
           features.getWaitingCacheMissTokenCount() == 0 &&
           features.getWaitingCacheMissTokenSquareSum() == 0.0 &&
           features.getWaitingCacheHitMissTokenProductSum() == 0.0;
}

double LegacyV76NonLinearPrefillModelR::predict(const Input &input, const Values &weights,
                                                size_t parameterCount, Values *gradient,
                                                double nonlinearScale,
                                                const ParameterScales &scales)
{
    if (!gradient) {
        return evaluate(input, weights, nonlinearScale, scales);
    }
    std::array<AutoGrad<MAX_PARAMETER_COUNT>, MAX_PARAMETER_COUNT> autoGradWeights;
    for (size_t index = 0; index < weights.size(); ++index) {
        autoGradWeights[index] =
            AutoGrad<MAX_PARAMETER_COUNT>::parameter(weights[index], index);
    }
    auto prediction = evaluate(input, autoGradWeights, nonlinearScale, scales);
    for (size_t index = 0; index < parameterCount; ++index) {
        (*gradient)[index] = prediction.gradient(index);
    }
    // Keep the training residual on the exact same forward path as inference.
    return evaluate(input, weights, nonlinearScale, scales);
}

template <typename Value>
Value LegacyV76NonLinearPrefillModelR::evaluateLinear(
    const Input &input, const std::array<Value, MAX_PARAMETER_COUNT> &weights,
    const ParameterScales &scales)
{
    Value linear = 0.0;
    for (size_t index = 0; index < LINEAR_PARAMETER_COUNT; ++index) {
        linear = linear + input[index] * weights[index] * scales[index];
    }
    for (size_t index = 0; index < WAITING_PARAMETER_COUNT; ++index) {
        const auto parameterIndex = WAITING_PARAMETER_OFFSET + index;
        linear = linear + input[LINEAR_PARAMETER_COUNT + index] *
                              weights[parameterIndex] * scales[parameterIndex];
    }
    linear = linear + input[13] * weights[RUNNING_ELAPSED_TIME_PARAMETER_INDEX] *
                          scales[RUNNING_ELAPSED_TIME_PARAMETER_INDEX];
    linear = linear + input[14] * weights[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] *
                          scales[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX];
    linear = linear + input[15] *
                          weights[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX] *
                          scales[RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX];
    return linear / COEFFICIENT_3;
}

template <typename Value>
Value LegacyV76NonLinearPrefillModelR::evaluate(
    const Input &input, const std::array<Value, MAX_PARAMETER_COUNT> &weights,
    double nonlinearScale, const ParameterScales &scales)
{
    const auto linear = evaluateLinear(input, weights, scales);
    auto latency = evaluateNonLinear(linear, weights, nonlinearScale, scales) +
                   weights[TOTAL_PROMPT_PARAMETER_INDEX] *
                       scales[TOTAL_PROMPT_PARAMETER_INDEX] * input[16] +
                   weights[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] *
                       scales[PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] * input[12];
    return valueOf(latency) < 0.0 ? Value(0.0) : latency;
}

template <typename Value, typename Weights>
Value LegacyV76NonLinearPrefillModelR::evaluateNonLinear(const Value &linear,
                                                         const Weights &weights,
                                                         double nonlinearScale,
                                                         const ParameterScales &scales)
{
    using std::sqrt;
    const auto curvature = weights[NONLINEAR_CURVATURE_PARAMETER_INDEX] *
                           scales[NONLINEAR_CURVATURE_PARAMETER_INDEX];
    const auto shape = [&](const auto &value) {
        auto squareRoot = sqrt(value * value + curvature);
        return valueOf(value) >= 0.0 ? value + squareRoot
                                     : curvature / (squareRoot - value);
    };
    const auto zeroLinear = weights[0] * scales[0] / COEFFICIENT_3;
    return weights[ZERO_POINT_PARAMETER_INDEX] * scales[ZERO_POINT_PARAMETER_INDEX] +
           nonlinearScale * (shape(linear) - shape(zeroLinear));
}

LegacyV76NonLinearPrefillModelR::Values LegacyV76NonLinearPrefillModelR::toValues(
    const std::vector<double> &parameters)
{
    Values values{};
    std::copy(parameters.begin(), parameters.end(), values.begin());
    return values;
}

double LegacyV76NonLinearPrefillModelR::positiveParameterFloor(size_t, double)
{
    return 0.0;
}

void LegacyV76NonLinearPrefillModelR::reparameterize(Values &parameters,
                                                     Values &adamMoment1,
                                                     Values &adamMoment2,
                                                     size_t parameterCount,
                                                     double fromScale, double toScale)
{
    const auto ratio = fromScale / toScale;
    const auto ratioSquared = ratio * ratio;
    const auto transformLinear = [&](size_t index) {
        parameters[index] *= ratio;
        adamMoment1[index] /= ratio;
        adamMoment2[index] /= ratioSquared;
    };
    for (size_t index = 0; index < LINEAR_PARAMETER_COUNT; ++index) {
        transformLinear(index);
    }
    if (parameterCount == CAPACITY_SPLIT_PARAMETER_COUNT) {
        for (size_t index = 0; index < WAITING_PARAMETER_COUNT; ++index) {
            transformLinear(WAITING_PARAMETER_OFFSET + index);
        }
        transformLinear(PRECEDING_REQUEST_COUNT_PARAMETER_INDEX);
        transformLinear(RUNNING_ELAPSED_TIME_PARAMETER_INDEX);
        transformLinear(RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX);
        transformLinear(RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX);
    }
    constexpr auto curvatureIndex = NONLINEAR_CURVATURE_PARAMETER_INDEX;
    parameters[curvatureIndex] *= ratioSquared;
    adamMoment1[curvatureIndex] /= ratioSquared;
    adamMoment2[curvatureIndex] /= ratioSquared * ratioSquared;
}

void LegacyV76NonLinearPrefillModelR::projectOutputParameters(
    Values &parameters, Values &adamMoment1, Values &adamMoment2, double minimumCurvature,
    double linearScaleRatio) const
{
    if (!_useBatchCountFeature) {
        parameters[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment1[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment2[BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    }
    if (!_useRunningDecodeRequestCountFeature) {
        parameters[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment1[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = 0.0;
        adamMoment2[RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = 0.0;
    }
    for (const auto index : POSITIVE_PARAMETER_INDEXES) {
        if ((index == BATCH_COUNT_PARAMETER_INDEX && !_useBatchCountFeature) ||
            (index == RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX &&
             !_useRunningDecodeRequestCountFeature) ||
            index >= (_useCapacitySplitFeature ? CAPACITY_SPLIT_PARAMETER_COUNT
                                               : PARAMETER_COUNT)) {
            continue;
        }
        const auto minimumValue = positiveParameterFloor(index, linearScaleRatio);
        if (parameters[index] >= minimumValue) {
            continue;
        }
        parameters[index] = minimumValue;
        if (adamMoment1[index] > 0.0) {
            adamMoment1[index] = 0.0;
        }
    }
    constexpr auto curvatureIndex = NONLINEAR_CURVATURE_PARAMETER_INDEX;
    if (parameters[curvatureIndex] < minimumCurvature) {
        parameters[curvatureIndex] = minimumCurvature;
        if (adamMoment1[curvatureIndex] > 0.0) {
            adamMoment1[curvatureIndex] = 0.0;
        }
    }
    if (_useCapacitySplitFeature) {
        constexpr auto index = PRECEDING_REQUEST_COUNT_PARAMETER_INDEX;
        const auto minimumValue = MIN_PRECEDING_REQUEST_COEFFICIENT_MS *
                                  linearScaleRatio / PARAMETER_SCALES[index];
        if (parameters[index] < minimumValue) {
            parameters[index] = minimumValue;
            if (adamMoment1[index] > 0.0) {
                adamMoment1[index] = 0.0;
            }
        }
    }
}

bool LegacyV76NonLinearPrefillModelR::usesExecutionGeometry() const
{
    return _executionGeometryEnabled && usesPdCoordinates();
}

bool LegacyV76NonLinearPrefillModelR::usesProjectedExecutionGeometry() const
{
    return _projectedExecutionGeometryEnabled && usesExecutionGeometry();
}

void LegacyV76NonLinearPrefillModelR::setProjectedExecutionGeometryEnabled(bool enabled)
{
    autil::ScopedLock lock(_lock);
    if (_projectedExecutionGeometryEnabled == enabled) {
        return;
    }
    _projectedExecutionGeometryEnabled = enabled;
    _adamMoment1.fill(0.0);
    _adamMoment2.fill(0.0);
    _step = 1;
    _optimizerStepDiagnostics = {};
}

void LegacyV76NonLinearPrefillModelR::setExecutionGeometryEnabled(bool enabled)
{
    autil::ScopedLock lock(_lock);
    if (_executionGeometryEnabled == enabled) {
        return;
    }
    _executionGeometryEnabled = enabled;
    _adamMoment1.fill(0.0);
    _adamMoment2.fill(0.0);
    _step = 1;
    _optimizerStepDiagnostics = {};
}

// Offline oracle only: do not register the archived resource.

} // namespace navi_sched

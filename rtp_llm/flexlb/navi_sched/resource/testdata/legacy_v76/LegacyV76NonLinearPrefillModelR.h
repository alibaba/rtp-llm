#pragma once

#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/scheduler/AutoGrad.h"
#include "autil/Lock.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "navi/engine/Resource.h"
#include <array>
#include <atomic>
#include <deque>
#include <optional>

namespace navi_sched {

class LegacyV76NonLinearPrefillModelR : public navi::Resource {
public:
    static constexpr size_t PARAMETER_COUNT = 10;
    static constexpr size_t CAPACITY_SPLIT_PARAMETER_COUNT = 19;
    static constexpr size_t MAX_PARAMETER_COUNT = CAPACITY_SPLIT_PARAMETER_COUNT;
    static constexpr size_t WAITING_PARAMETER_OFFSET = PARAMETER_COUNT;
    static constexpr size_t WAITING_PARAMETER_COUNT = 5;
    static constexpr int64_t BINARY_VERSION = 24;
    static constexpr int64_t PD_BINARY_VERSION = 25;
    static constexpr int64_t GEOMETRY_BINARY_VERSION = 26;
    static constexpr int64_t PROJECTED_GEOMETRY_BINARY_VERSION = 29;
    static bool usesGeometryForVersion(int64_t version)
    {
        return version == GEOMETRY_BINARY_VERSION ||
               version == PROJECTED_GEOMETRY_BINARY_VERSION;
    }
    static constexpr size_t BATCH_COUNT_PARAMETER_INDEX = 6;
    static constexpr size_t BASE_LATENCY_PARAMETER_INDEX = 7;
    static constexpr size_t ZERO_POINT_PARAMETER_INDEX = BASE_LATENCY_PARAMETER_INDEX;
    static constexpr size_t NONLINEAR_CURVATURE_PARAMETER_INDEX = 8;
    static constexpr size_t TOTAL_PROMPT_PARAMETER_INDEX = 9;
    static constexpr size_t PRECEDING_REQUEST_COUNT_PARAMETER_INDEX = 15;
    static constexpr size_t RUNNING_ELAPSED_TIME_PARAMETER_INDEX = 16;
    static constexpr size_t RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX = 17;
    static constexpr size_t RUNNING_DECODE_GENERATED_TOKEN_COUNT_PARAMETER_INDEX = 18;
    static constexpr double NONLINEAR_SCALE = 1500.0;
    static constexpr double MIN_NONLINEAR_CURVATURE = 1e-8;
    static constexpr double DEFAULT_ZERO_ANCHOR_MAX_MS = 256.0;
    // The feature is the sum across all running requests, so normalize at the
    // aggregate scale rather than at one request-second.
    static constexpr double RUNNING_ELAPSED_TIME_SCALE_MS = 1'000'000.0;
    static constexpr double INITIAL_PRECEDING_REQUEST_COEFFICIENT_MS = 5.0;
    static constexpr double MIN_PRECEDING_REQUEST_COEFFICIENT_MS = 1.0;
    inline static constexpr std::array<double, MAX_PARAMETER_COUNT> PARAMETER_SCALES = {
        0.878862443703,
        0.119054644545,
        0.0881092493705,
        2.45290259606,
        0.918359382716,
        19.5646708383,
        1.0,
        1.23972355201,
        0.00229934153041,
        0.211011741932,
        1.09757235126,
        0.878792874576,
        193.929503215,
        222.305482002,
        207.484099949,
        0.0242867645882,
        1892.09311928,
        0.0299401936353,
        0.403880546944,
    };

    using ParameterScales = std::array<double, MAX_PARAMETER_COUNT>;
    inline static constexpr ParameterScales PD_PARAMETER_SCALES = {
        4.9909829158298527,   0.46968907810397831,    0.1666947481746503,
        0.03433754612169089,  0.00013582507257929477, 0.0063937915256001903,
        0.7230158819771636,   2.0095874101382516,     0.15246041271834707,
        0.12173033959514495,  1.09757235126,          0.87879287457599997,
        193.92950321500001,   222.30548200199999,     207.48409994900001,
        0.024286764588200001, 1892.0931192800001,     0.0299401936353,
        0.403880546944,
    };

    struct OptimizerState {
        std::array<double, MAX_PARAMETER_COUNT> adamMoment1{};
        std::array<double, MAX_PARAMETER_COUNT> adamMoment2{};
        int64_t step = 1;
    };

    struct OptimizerStepDiagnostics {
        size_t parameterCount = 0;
        int64_t step = 0;
        std::array<double, MAX_PARAMETER_COUNT> rawGradient{};
        std::array<double, MAX_PARAMETER_COUNT> adamMoment1{};
        std::array<double, MAX_PARAMETER_COUNT> adamMoment2{};
        std::array<double, MAX_PARAMETER_COUNT> normalizedGradient{};
        std::array<double, MAX_PARAMETER_COUNT> candidateUpdate{};
        std::array<double, MAX_PARAMETER_COUNT> actualUpdate{};
        std::array<double, MAX_PARAMETER_COUNT> predictionShiftMs{};
    };

    struct NonlinearOperatingPoint {
        double u = 0.0;
        double curvature = 0.0;
        double r = 0.0;
        // -1: negative far field, 0: transition, 1: positive far field.
        int region = 0;
    };

    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    int32_t getFormula() const;
    int64_t getBinaryVersion() const;
    std::vector<double> getInitialParameters() const;
    static double getParameterScale(size_t index);
    const ParameterScales &getConfiguredParameterScales() const;
    bool usesPdCoordinates() const;
    bool validateConfiguredParameters(const std::vector<double> &parameters) const;
    static bool supportsBinaryVersion(int64_t binaryVersion);
    // Coordinate conversion only. For version 26, callers must additionally
    // plan batch geometry and use the explicit geometry model; conversion is
    // not an equivalent legacy-feature prediction.
    static std::vector<double> parametersForLegacyPrediction(
        const std::vector<double> &parameters, int64_t binaryVersion);
    static bool validateParametersForVersion(const std::vector<double> &parameters,
                                             int64_t binaryVersion);
    static bool validateParameters(const std::vector<double> &parameters);
    static bool usesCapacitySplitParameters(const std::vector<double> &parameters);
    bool fit(const std::vector<NaviSchedLatencySample> &samples,
             const std::vector<double> &currentParameters,
             std::vector<double> &newParameters, double learningRateScale = 1.0,
             double currentSampleWeight = 1.0,
             const std::vector<double> *parameterLearningRateScales = nullptr) const;
    // Enable only before training; changing feature semantics clears Adam state.
    void setExecutionGeometryEnabled(bool enabled);
    bool usesExecutionGeometry() const;
    void setProjectedExecutionGeometryEnabled(bool enabled);
    bool usesProjectedExecutionGeometry() const;
    void applyRuntimeConfig(const AgentSchedulerConfig &config);
    double getNonlinearScale() const;
    bool fitHistory(const std::deque<NaviSchedLatencySample> &history,
                    size_t fitBatchSize, size_t epochCount,
                    const std::vector<double> &currentParameters,
                    std::vector<double> &newParameters) const;
    static double predict(const NaviSchedLatencyFeatures &features,
                          const std::vector<double> &parameters);
    static double predictForVersion(const NaviSchedLatencyFeatures &features,
                                    const std::vector<double> &parameters,
                                    int64_t binaryVersion);
    static double calculateZeroPoint(const std::vector<double> &parameters,
                                     std::vector<double> *gradient = nullptr,
                                     int64_t binaryVersion = BINARY_VERSION);
    bool checkGradientConfigured(const NaviSchedLatencyFeatures &features,
                                 const std::vector<double> &parameters) const;
    double predictConfigured(const NaviSchedLatencyFeatures &features,
                             const std::vector<double> &parameters) const;
    std::optional<NonlinearOperatingPoint> getNonlinearOperatingPointConfigured(
        const NaviSchedLatencyFeatures &features,
        const std::vector<double> &parameters) const;
    static double calculateRequestLinearCost(const std::vector<double> &weights,
                                             int64_t totalTokenCount,
                                             int64_t cacheHitTokenCount);
    static double calculateWaitingRequestLinearCost(const std::vector<double> &weights,
                                                    int64_t totalTokenCount,
                                                    int64_t cacheHitTokenCount);
    static double calculateLinearCost(const std::vector<double> &weights,
                                      const NaviSchedLatencyFeatures &features);
    static bool hasNonNegativeWaitingWorkloadCost(
        const std::vector<double> &weights, const NaviSchedLatencyFeatures &features,
        double *waitingCost = nullptr);
    static AutoGradTape::Value calculateLatencyFromRequestCostSum(
        const std::vector<double> &weights, const AutoGradTape::Value &requestCostSum,
        const AutoGradTape::Value &totalPromptTokenCount,
        double precedingRequestCount = 0.0);
    static double calculateLatencyAndDerivativesFromRequestCostSum(
        const std::vector<double> &weights, double requestCostSum,
        double totalPromptTokenCount, double precedingRequestCount,
        double &requestCostDerivative, double &totalPromptTokenDerivative);
    static bool checkGradient(const NaviSchedLatencyFeatures &features,
                              const std::vector<double> &parameters);
    OptimizerState getOptimizerState() const;
    OptimizerStepDiagnostics getOptimizerStepDiagnostics() const;
    static bool validateOptimizerState(const OptimizerState &state,
                                       size_t parameterCount);
    static NaviSchedLatencyOptimizerState makeOptimizerStateData(
        const OptimizerState &state, size_t parameterCount);
    static bool makeOptimizerState(const NaviSchedLatencyOptimizerState &data,
                                   OptimizerState &state);
    bool sanitizeInactiveParameters(std::vector<double> &parameters,
                                    OptimizerState &state) const;
    bool restoreOptimizerState(const OptimizerState &state);
    void resetOptimizerState();
    bool usesBatchCountFeature() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    RESOURCE_DEPEND_DECLARE();

private:
    RESOURCE_DEPEND_ON_FALSE(kmonitor::MetricsReporterR, _metricsReporterR);
    using Input = std::array<double, 17>;
    using Values = std::array<double, MAX_PARAMETER_COUNT>;
    inline static constexpr std::array<size_t, 14> POSITIVE_PARAMETER_INDEXES = {
        1,
        2,
        3,
        4,
        5,
        BATCH_COUNT_PARAMETER_INDEX,
        BASE_LATENCY_PARAMETER_INDEX,
        TOTAL_PROMPT_PARAMETER_INDEX,
        WAITING_PARAMETER_OFFSET,
        WAITING_PARAMETER_OFFSET + 1,
        WAITING_PARAMETER_OFFSET + 2,
        WAITING_PARAMETER_OFFSET + 3,
        WAITING_PARAMETER_OFFSET + 4,
        RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX,
    };

private:
    static bool validateParameterValues(const double *parameters, size_t parameterCount);
    static Input makeInput(const NaviSchedLatencyFeatures &features,
                           bool geometry = false);
    static bool isZeroWorkload(const NaviSchedLatencyFeatures &features);
    template <typename Value>
    static Value evaluate(const Input &input,
                          const std::array<Value, MAX_PARAMETER_COUNT> &weights,
                          double nonlinearScale = NONLINEAR_SCALE,
                          const ParameterScales &scales = PARAMETER_SCALES);
    template <typename Value>
    static Value evaluateLinear(const Input &input,
                                const std::array<Value, MAX_PARAMETER_COUNT> &weights,
                                const ParameterScales &scales = PARAMETER_SCALES);
    template <typename Value, typename Weights>
    static Value evaluateNonLinear(const Value &linear, const Weights &weights,
                                   double nonlinearScale = NONLINEAR_SCALE,
                                   const ParameterScales &scales = PARAMETER_SCALES);
    static double predict(const Input &input, const Values &weights,
                          size_t parameterCount, Values *gradient,
                          double nonlinearScale = NONLINEAR_SCALE,
                          const ParameterScales &scales = PARAMETER_SCALES);
    static bool checkGradient(const Input &input, const Values &weights,
                              size_t parameterCount, size_t *failedIndex,
                              double nonlinearScale = NONLINEAR_SCALE,
                              const ParameterScales &scales = PARAMETER_SCALES);
    static Values toValues(const std::vector<double> &parameters);
    static double positiveParameterFloor(size_t index, double linearScaleRatio);
    static void reparameterize(Values &parameters, Values &adamMoment1,
                               Values &adamMoment2, size_t parameterCount,
                               double fromScale, double toScale);
    void projectOutputParameters(Values &parameters, Values &adamMoment1,
                                 Values &adamMoment2, double minimumCurvature,
                                 double linearScaleRatio = 1.0) const;

private:
    mutable autil::ThreadMutex _lock;
    mutable Values _adamMoment1{};
    mutable Values _adamMoment2{};
    mutable int64_t _step = 1;
    mutable OptimizerStepDiagnostics _optimizerStepDiagnostics;
    double _learningRate = 0.05;
    double _nonlinearScale = NONLINEAR_SCALE;
    double _zeroAnchorMaxMs = DEFAULT_ZERO_ANCHOR_MAX_MS;
    bool _checkGradient = false;
    bool _pdSeparated = false;
    bool _executionGeometryEnabled = false;
    bool _projectedExecutionGeometryEnabled = false;
    bool _useBatchCountFeature = true;
    bool _useCapacitySplitFeature = false;
    bool _useRunningDecodeRequestCountFeature = false;
    bool _useRunningDecodeGeneratedTokenCountFeature = false;
    mutable std::atomic<uint64_t> _outputParameterProjectionRecoveryCount{0};
    mutable std::atomic<uint64_t> _invalidCandidateRecoveryCount{0};
    mutable std::atomic<uint64_t> _unsafeWaitingCandidateRecoveryCount{0};
    mutable std::atomic<uint64_t> _unsafeWaitingStateResetCount{0};
};

} // namespace navi_sched

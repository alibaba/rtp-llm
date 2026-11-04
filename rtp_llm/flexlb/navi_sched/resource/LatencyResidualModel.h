#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace navi_sched {

class LatencyResidualModel {
public:
    struct RuntimeConfig {
        double learningRate = 0.8;
        double updateErrorFloorMs = 100.0;
        double updateErrorRatio = 0.1;
        int64_t idleDecayStartUs = 2 * 60 * 1000 * 1000;
        int64_t idleDecayEndUs = 5 * 60 * 1000 * 1000;

        bool valid() const;
    };

    struct Evaluation {
        double predictionMs = 0.0;
        double correctionMs = 0.0;
        double derivativeFactor = 1.0;
        bool applied = false;
    };

    static constexpr int32_t VERSION = NAVI_SCHED_LATENCY_RESIDUAL_VERSION;
    static constexpr int32_t MIN_BUCKET = 4;
    static constexpr int32_t MAX_BUCKET = 22;
    static constexpr size_t PARAMETER_COUNT =
        NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT;
    static_assert(PARAMETER_COUNT ==
                  static_cast<size_t>(MAX_BUCKET - MIN_BUCKET + 1));
    static constexpr int64_t IDLE_DECAY_START_US = 2 * 60 * 1000 * 1000;
    static constexpr int64_t IDLE_DECAY_END_US = 5 * 60 * 1000 * 1000;
    static constexpr double LEARNING_RATE = 0.03;
    static constexpr double UPDATE_ERROR_FLOOR_MS = 100.0;
    static constexpr double UPDATE_ERROR_RATIO = 0.1;

    static std::vector<double> makeInitialParameters();
    static bool validateParameters(const std::vector<double> &parameters);
    static double decayFactor(int64_t lastUpdateTimeUs, int64_t currentTimeUs);
    static double decayFactor(int64_t lastUpdateTimeUs, int64_t currentTimeUs,
                              const RuntimeConfig &config);
    static bool expired(int64_t lastUpdateTimeUs, int64_t currentTimeUs);
    static bool expired(int64_t lastUpdateTimeUs, int64_t currentTimeUs,
                        const RuntimeConfig &config);
    static bool applyIdleDecay(std::vector<double> &parameters,
                               int64_t lastUpdateTimeUs,
                               int64_t currentTimeUs);
    static bool applyIdleDecay(std::vector<double> &parameters,
                               int64_t lastUpdateTimeUs,
                               int64_t currentTimeUs,
                               const RuntimeConfig &config);
    static Evaluation evaluate(double basePredictionMs,
                               const std::vector<double> &parameters,
                               int64_t lastUpdateTimeUs = 0,
                               int64_t currentTimeUs = 0);
    static Evaluation evaluate(double basePredictionMs,
                               const std::vector<double> &parameters,
                               int64_t lastUpdateTimeUs,
                               int64_t currentTimeUs,
                               const RuntimeConfig &config);
    static bool update(double basePredictionMs, double actualLatencyMs,
                       std::vector<double> &parameters,
                       int64_t &lastUpdateTimeUs, int64_t currentTimeUs);
    static bool update(double basePredictionMs, double actualLatencyMs,
                       std::vector<double> &parameters,
                       int64_t &lastUpdateTimeUs, int64_t currentTimeUs,
                       const RuntimeConfig &config);
};

}

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <autil/DataBuffer.h>

namespace navi_sched {

struct NaviSchedQueueModelState {
    double baseClipAdjustmentMs = 0.0;
    double cacheMemberWeightMs = 0.0;
    double lookupMemberWeightMs = 0.0;

    bool valid(double totalMs) const
    {
        return std::isfinite(totalMs) && totalMs >= 0.0 &&
               std::isfinite(baseClipAdjustmentMs) && baseClipAdjustmentMs <= 0.0 &&
               (baseClipAdjustmentMs == 0.0 || totalMs == 0.0) &&
               std::isfinite(cacheMemberWeightMs) &&
               std::isfinite(lookupMemberWeightMs);
    }

    double baseMs(double totalMs) const { return totalMs + baseClipAdjustmentMs; }

    double decayedBaseMs(double totalMs, double rawElapsedMs) const
    {
        return baseMs(totalMs) - rawElapsedMs;
    }

    void setBaseMs(double base, double &totalMs)
    {
        totalMs = std::max(0.0, base);
        baseClipAdjustmentMs = std::min(0.0, base);
    }

    void serialize(autil::DataBuffer &buffer) const
    {
        buffer.write(baseClipAdjustmentMs);
        buffer.write(cacheMemberWeightMs);
        buffer.write(lookupMemberWeightMs);
    }

    void deserialize(autil::DataBuffer &buffer)
    {
        buffer.read(baseClipAdjustmentMs);
        buffer.read(cacheMemberWeightMs);
        buffer.read(lookupMemberWeightMs);
    }
};

struct NaviSchedQueueModel {
    using Parameters = std::array<double, 5>;
    static constexpr double TARGET_SCALE_MS = 54.674371877821976;
    static constexpr Parameters PARAMETER_SCALES = {
        54.674371877821976, 328.1617891459539, 54.636513104526756, 54.674371877821976,
        54.674371877821976};

    static double basePredictionMs(double rawQueueMs, double precedingBatchCount,
                                   const Parameters &parameters)
    {
        return rawQueueMs + parameters[0] + parameters[1] * rawQueueMs / 1000.0 +
               parameters[2] * precedingBatchCount;
    }

    static double predictMs(double baseMs, double cacheMemberWeightMs,
                            double lookupMemberWeightMs, double cacheHitMemberWeight,
                            double lookupMemberWeight, double batchWeight,
                            double *hitDerivative = nullptr,
                            double *lookupDerivative = nullptr,
                            double *batchDerivative = nullptr)
    {
        // Integer batches use their hit-member fraction. Fractional activation
        // below one keeps continuous optimization finite at an empty node.
        const auto denominator = std::max(1.0, batchWeight);
        const auto conditional = cacheMemberWeightMs * cacheHitMemberWeight +
                                 lookupMemberWeightMs * lookupMemberWeight;
        const auto prediction = baseMs + conditional / denominator;
        if (hitDerivative) {
            *hitDerivative = prediction > 0.0 ? cacheMemberWeightMs / denominator : 0.0;
        }
        if (lookupDerivative) {
            *lookupDerivative =
                prediction > 0.0 ? lookupMemberWeightMs / denominator : 0.0;
        }
        if (batchDerivative) {
            *batchDerivative = prediction > 0.0 && batchWeight > 1.0
                                   ? -conditional / (denominator * denominator)
                                   : 0.0;
        }
        return std::max(0.0, prediction);
    }
};

} // namespace navi_sched

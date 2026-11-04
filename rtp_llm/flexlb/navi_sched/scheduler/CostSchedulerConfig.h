#pragma once

#include <autil/legacy/jsonizable.h>

#include <cstddef>
#include <cstdint>

namespace navi_sched {

struct CostSchedulerConfig : public autil::legacy::Jsonizable {
    static constexpr int32_t MAX_LOOP_COUNT = 1000;
    static constexpr int32_t MAX_PARALLELISM = 1024;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("lambda", lambda, lambda);
        json.Jsonize("alpha", alpha, alpha);
        json.Jsonize("alpha_decay", alphaDecay, alphaDecay);
        json.Jsonize("min_alpha", minAlpha, minAlpha);
        json.Jsonize("max_loop_count", maxLoopCount, maxLoopCount);
        json.Jsonize("min_loop_count", minLoopCount, minLoopCount);
        json.Jsonize("optimization_time_budget_us",
                     optimizationTimeBudgetUs,
                     optimizationTimeBudgetUs);
        json.Jsonize("parallelism", parallelism, parallelism);
        json.Jsonize("latency_weight_concavity", latencyWeightConcavity,
                     latencyWeightConcavity);
        json.Jsonize("latency_token_normalization_power",
                     latencyTokenNormalizationPower,
                     latencyTokenNormalizationPower);
        json.Jsonize("initial_weight_seed", initialWeightSeed,
                     initialWeightSeed);
        json.Jsonize("discrete_check_interval", discreteCheckInterval,
                     discreteCheckInterval);
        json.Jsonize("discrete_gap_threshold", discreteGapThreshold,
                     discreteGapThreshold);
        json.Jsonize("discrete_repair_alpha_decay",
                     discreteRepairAlphaDecay,
                     discreteRepairAlphaDecay);
    }

    bool valid() const;

    double lambda = 0.5;
    double alpha = 512.0;
    double alphaDecay = 1.0;
    double minAlpha = 0.0;
    int32_t maxLoopCount = 10;
    int32_t minLoopCount = 2;
    int64_t optimizationTimeBudgetUs = 0;
    int32_t parallelism = 0;
    double latencyWeightConcavity = 128.0;
    double latencyTokenNormalizationPower = 1.0;
    uint64_t initialWeightSeed = 0;
    int32_t discreteCheckInterval = 5;
    double discreteGapThreshold = 0.25;
    double discreteRepairAlphaDecay = 0.7;
};

}

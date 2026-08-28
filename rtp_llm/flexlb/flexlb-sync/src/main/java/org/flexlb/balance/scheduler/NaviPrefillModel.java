package org.flexlb.balance.scheduler;

/**
 * Stateless numeric core of navi_sched's non-linear prefill latency model,
 * translated from {@code NonLinearPrefillModelR}. It maps a per-endpoint
 * 9-parameter vector plus request token features onto a predicted prefill
 * latency (and its derivative), which the {@link NaviPgdOptimizer} uses to
 * build its scheduling objective.
 *
 * <p>The parameter layout matches {@link org.flexlb.balance.prediction.LearningPredictor}
 * exactly: indexes {@code [0..5]} are the linear weights (bias, batchSize,
 * reuse, compute, compute^2, reuse*compute) and {@code [6..8]} are the
 * non-linear weights. This lets the optimizer consume the learned
 * per-endpoint snapshot without any reshaping.
 */
final class NaviPrefillModel {

    /** Token normalization scale; mirrors navi {@code TOKEN_SCALE}. */
    static final double TOKEN_SCALE = 1024.0;

    /** Non-linear parameter[6] divisor; mirrors navi {@code COEFFICIENT_1}. */
    static final double COEFFICIENT_1 = 0.005;

    /** Non-linear parameter[7] divisor; mirrors navi {@code COEFFICIENT_2}. */
    static final double COEFFICIENT_2 = 0.02;

    /** Linear-output divisor; mirrors navi {@code COEFFICIENT_3}. */
    static final double COEFFICIENT_3 = 320.0;

    /** Count of linear parameters; mirrors navi {@code LINEAR_PARAMETER_COUNT}. */
    static final int LINEAR_PARAMETER_COUNT = 6;

    private NaviPrefillModel() {
    }

    /**
     * Linear cost contribution of one request on one endpoint. Mirrors navi
     * {@code NonLinearPrefillModelR::calculateRequestLinearCost}: builds the
     * single-request input vector via {@code makeInput} semantics and sums
     * {@code input[i] * params[i]} for {@code i} in {@code [1, 5]}, skipping the
     * bias term {@code params[0]} (it is added once per node in the latency
     * stage, not per request).
     *
     * @param params        per-endpoint 9-parameter vector
     * @param totalTokens    total prompt tokens of the request
     * @param cacheHitTokens tokens served from cache (reuse); the remainder is compute
     */
    static double calculateRequestLinearCost(
            double[] params, long totalTokens, long cacheHitTokens) {
        if (params == null || params.length < LINEAR_PARAMETER_COUNT) {
            return 0.0;
        }
        long cacheMissTokens = totalTokens - cacheHitTokens;
        // makeInput for a single appended request (batchSize == 1):
        //   input[1] = batchSize, input[2] = reuse, input[3] = compute,
        //   input[4] = compute^2, input[5] = reuse * compute.
        double reuse = cacheHitTokens / TOKEN_SCALE;
        double compute = cacheMissTokens / TOKEN_SCALE;
        double computeSquare = compute * compute;
        double reuseCompute = reuse * compute;
        double cost = 0.0;
        cost += 1.0 * params[1];            // batchSize == 1
        cost += reuse * params[2];
        cost += compute * params[3];
        cost += computeSquare * params[4];
        cost += reuseCompute * params[5];
        return cost;
    }

    /**
     * Node latency and its derivative w.r.t. {@code requestCostSum}. Mirrors
     * navi {@code calculateLatencyAndDerivativeFromRequestCostSum} composed with
     * {@code evaluateNonLinear}.
     *
     * <p>{@code linear = (requestCostSum + params[0]) / 320};
     * {@code z = linear + params[8] + 1};
     * {@code latency = params[6]/0.005 + params[7]/0.02 * (z + sqrt(z^2 + 4))};
     * {@code derivative = params[7]/0.02 * (1 + z / sqrt(z^2 + 4)) / 320}.
     *
     * @return a two-element array {@code {latency, derivative}}
     */
    static double[] calculateLatencyAndDerivative(
            double requestCostSum, double[] params) {
        double parameter6 = params[LINEAR_PARAMETER_COUNT] / COEFFICIENT_1;
        double parameter7 = params[LINEAR_PARAMETER_COUNT + 1] / COEFFICIENT_2;
        double linear = (requestCostSum + params[0]) / COEFFICIENT_3;
        double linearOffset = linear + params[LINEAR_PARAMETER_COUNT + 2] + 1.0;
        double squareRoot = Math.sqrt(linearOffset * linearOffset + 4.0);
        double nonLinear = linearOffset + squareRoot;
        double latency = parameter6 + parameter7 * nonLinear;
        // d(latency)/d(requestCostSum)
        //   = parameter7 * d(nonLinear)/d(linearOffset) * d(linear)/d(requestCostSum)
        //   = parameter7 * (1 + linearOffset / squareRoot) / COEFFICIENT_3.
        double derivative =
                parameter7 * (1.0 + linearOffset / squareRoot) / COEFFICIENT_3;
        return new double[] {latency, derivative};
    }
}

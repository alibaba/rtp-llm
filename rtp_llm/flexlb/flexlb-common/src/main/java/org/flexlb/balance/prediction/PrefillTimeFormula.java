package org.flexlb.balance.prediction;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Shared configurable prefill-time formula engine.
 *
 * <h3>Syntax</h3>
 * <ul>
 *   <li><b>Operators</b>: {@code + - * / ^} (power, right-associative)</li>
 *   <li><b>Functions</b>: {@code sqrt(x) log(x) exp(x) abs(x) max(a,b) min(a,b) pow(a,b)}</li>
 *   <li><b>Batch aggregate</b>: {@code sum(expr)} evaluates {@code expr} per request and sums it</li>
 *   <li><b>Numbers</b>: decimal ({@code 3.14}) or scientific ({@code 1.2e-8})</li>
 *   <li><b>Parentheses</b>: {@code ( expr )}</li>
 * </ul>
 *
 * <h3>Variables</h3>
 * <table>
 *   <tr><th>Symbol</th><th>Meaning</th></tr>
 *   <tr><td>{@code inputTokens}</td><td>request input tokens</td></tr>
 *   <tr><td>{@code hitCacheTokens}</td><td>observed reusable KV-cache tokens</td></tr>
 *   <tr><td>{@code computeTokens}</td><td>{@code inputTokens - hitCacheTokens}</td></tr>
 *   <tr><td>{@code hasHitCache}</td><td>1 if {@code hitCacheTokens > 0}, otherwise 0</td></tr>
 *   <tr><td>{@code batchSize}</td><td>number of requests in the batch</td></tr>
 *   <tr><td>{@code totalInputTokens}</td><td>sum of input tokens across the batch</td></tr>
 *   <tr><td>{@code totalHitCacheTokens}</td><td>sum of cache-hit tokens across the batch</td></tr>
 *   <tr><td>{@code totalComputeTokens}</td><td>{@code totalInputTokens - totalHitCacheTokens}</td></tr>
 *   <tr><td>{@code maxInputTokens}</td><td>maximum request input length in the batch</td></tr>
 *   <tr><td>{@code maxComputeTokens}</td><td>maximum request compute length in the batch</td></tr>
 * </table>
 * <p>Batch-scoped variables ({@code batchSize}, {@code total*}, and {@code max*}) must be used
 * outside {@code sum(expr)}. Use the explicit {@code total*} variables for nonlinear batch-total terms. For example,
 * {@code totalComputeTokens^2} squares the batch total, while
 * {@code sum(computeTokens^2)} sums per-request squares. These expressions are intentionally
 * different and must not be substituted for each other. Use {@code sum(expr)} only when the
 * per-request distribution is part of the model.
 *
 * <h3>Example</h3>
 * <pre>{@code
 *   "param(base, 100) + param(batch, 2)*batchSize
 *       + param(compute, 0.01)*totalComputeTokens
 *       + param(compute2, 1e-8)*totalComputeTokens^2
 *       + param(maxCompute, 0.001)*maxComputeTokens
 *       + param(distribution, 1e-8)*sum(computeTokens^2)"
 * }</pre>
 */
public final class PrefillTimeFormula {

    private static final Set<String> BATCH_SCOPED_VARIABLES = Set.of(
            "batchSize", "totalInputTokens", "totalHitCacheTokens",
            "totalComputeTokens", "maxInputTokens", "maxComputeTokens");

    static final int IDX_BATCH_SIZE = 0;
    static final int IDX_INPUT_TOKENS = 1;
    static final int IDX_HIT_CACHE_TOKENS = 2;
    static final int IDX_COMPUTE_TOKENS = 3;
    static final int IDX_HAS_HIT_CACHE = 4;
    static final int IDX_TOTAL_INPUT_TOKENS = 5;
    static final int IDX_TOTAL_HIT_CACHE_TOKENS = 6;
    static final int IDX_TOTAL_COMPUTE_TOKENS = 7;
    static final int IDX_MAX_INPUT_TOKENS = 8;
    static final int IDX_MAX_COMPUTE_TOKENS = 9;
    static final int VAR_COUNT = 10;

    private static final Map<String, Integer> VAR_INDEX_MAP = Map.of(
            "batchSize", IDX_BATCH_SIZE,
            "inputTokens", IDX_INPUT_TOKENS,
            "hitCacheTokens", IDX_HIT_CACHE_TOKENS,
            "computeTokens", IDX_COMPUTE_TOKENS,
            "hasHitCache", IDX_HAS_HIT_CACHE,
            "totalInputTokens", IDX_TOTAL_INPUT_TOKENS,
            "totalHitCacheTokens", IDX_TOTAL_HIT_CACHE_TOKENS,
            "totalComputeTokens", IDX_TOTAL_COMPUTE_TOKENS,
            "maxInputTokens", IDX_MAX_INPUT_TOKENS,
            "maxComputeTokens", IDX_MAX_COMPUTE_TOKENS
    );

    private final ArithmeticFormula formula;

    private PrefillTimeFormula(ArithmeticFormula formula) {
        this.formula = formula;
    }

    /**
     * Parse a formula string.
     *
     * @throws IllegalArgumentException if the formula is malformed or references unknown variables.
     */
    public static PrefillTimeFormula parse(String formula) {
        return new PrefillTimeFormula(ArithmeticFormula.parse(
                formula, VAR_INDEX_MAP, BATCH_SCOPED_VARIABLES, true));
    }

    /**
     * Evaluate the formula with aggregate-aware per-request bindings.
     * {@code sum(expr)} evaluates {@code expr} for each array in {@code itemVars}.
     */
    public long evaluate(double[] vars, List<double[]> itemVars) {
        return (long) evaluateAsDouble(vars, itemVars);
    }

    /** Retain fractional and non-finite results until the prediction boundary validates them. */
    public double evaluateAsDouble(double[] vars, List<double[]> itemVars) {
        return formula.evaluateAsDouble(vars, itemVars);
    }
}

package org.flexlb.util.formula;

import java.util.Map;

/**
 * Counts uncached input tokens and evaluates a configured scalar work score.
 * Formulas are compiled once; this predictor has no learned state or batch inputs.
 */
public final class PrefillTimePredictor {
    private final ArithmeticFormula formula;

    public PrefillTimePredictor() {
        this("computeTokens");
    }

    public PrefillTimePredictor(String expression) {
        formula =
                "computeTokens".equals(expression)
                        ? null
                        : ArithmeticFormula.parse(expression, Map.of("computeTokens", 0));
    }

    /**
     * Default singleton time model from the upstream Prefill formula, in milliseconds.
     * Its coefficients are time estimates; the configurable ranking score has no time unit.
     */
    public static long estimateDurationMs(long inputTokens, long hitTokens) {
        long input = Math.max(0L, inputTokens);
        long hit = Math.clamp(hitTokens, 0L, input);
        double duration = Math.ceil((input - hit) + 0.02 * hit);
        return duration >= Long.MAX_VALUE ? Long.MAX_VALUE : (long) duration;
    }

    public long computeTokens(long inputTokens, long cacheHitTokens) {
        long input = Math.max(0L, inputTokens);
        return input - Math.clamp(cacheHitTokens, 0L, input);
    }

    public boolean usesDefaultFormula() {
        return formula == null;
    }

    public double score(long computeTokens) {
        double score =
                formula == null
                        ? computeTokens
                        : formula.evaluateAsDouble(new double[] {computeTokens});
        if (!Double.isFinite(score) || score < 0.0)
            throw new IllegalArgumentException(
                    "Prefill cost must be finite and non-negative: " + score);
        return score;
    }
}

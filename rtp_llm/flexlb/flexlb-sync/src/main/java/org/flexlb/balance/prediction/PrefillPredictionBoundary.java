package org.flexlb.balance.prediction;

import org.flexlb.balance.projection.WorkSnapshot;

import static com.google.common.math.LongMath.saturatedAdd;

/** Sole validation boundary for values returned by Prefill predictors. */
public final class PrefillPredictionBoundary {

    private PrefillPredictionBoundary() {
    }

    /** Absolute visibility estimate. Unknown preceding work deliberately produces no deadline. */
    public static long predictedVisibilityDeadlineMs(WorkSnapshot work, long unstartedMs,
                                                     double lifetime, long nowMs) {
        if (unstartedMs < 0L || !Double.isFinite(lifetime) || lifetime < 1.0) {
            throw new IllegalArgumentException("invalid delivery prediction inputs");
        }
        var preceding = java.util.Objects.requireNonNull(work).totalRemainingWorkMsAt(nowMs);
        if (preceding.isEmpty()) { return 0L; }
        double scaled = Math.ceil(saturatedAdd(preceding.getAsLong(), unstartedMs) * lifetime);
        long duration = scaled >= Long.MAX_VALUE ? Long.MAX_VALUE : (long) scaled;
        return saturatedAdd(nowMs, saturatedAdd(duration, 10_000L));
    }

    /** Evaluate one request and require a non-negative duration. */
    public static long predictSingleRequestMs(
            PrefillTimePredictor.Evaluator evaluator,
            long totalTokens,
            long hitTokens) {
        long predictedMs = evaluator.estimateMs(totalTokens, hitTokens);
        if (predictedMs < 0L) {
            throw new InvalidPrefillPredictionException(
                    "SINGLE_REQUEST",
                    predictedMs);
        }
        return predictedMs;
    }

    /** Evaluate committed work and convert it to lifecycle milliseconds. */
    public static long predictCommittedBatchMs(
            PrefillTimePredictor.Evaluator evaluator,
            PrefillBatchFeatures features) {
        return committedDecisionGroupMs(
                evaluator.predictBatchMs(features));
    }

    /** Convert an already planned group duration to lifecycle milliseconds. */
    public static long committedDecisionGroupMs(double predictedMs) {
        requireValidDecisionGroupMs(predictedMs);
        return predictedMs >= Long.MAX_VALUE
                ? Long.MAX_VALUE
                : (long) Math.ceil(predictedMs);
    }

    /** Require a finite, non-negative group duration. */
    public static double requireValidDecisionGroupMs(double predictedMs) {
        if (!Double.isFinite(predictedMs) || predictedMs < 0.0) {
            throw new InvalidPrefillPredictionException(
                    "DECISION_GROUP",
                    predictedMs);
        }
        return predictedMs;
    }
}

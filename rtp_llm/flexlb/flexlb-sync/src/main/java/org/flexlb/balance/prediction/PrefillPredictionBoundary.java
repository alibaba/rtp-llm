package org.flexlb.balance.prediction;

/** Sole validation boundary for values returned by Prefill predictors. */
public final class PrefillPredictionBoundary {

    private PrefillPredictionBoundary() {
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

    /** Evaluate a group while retaining fractional milliseconds for planning. */
    public static double predictDecisionGroupMs(
            PrefillTimePredictor.Evaluator evaluator,
            PrefillBatchFeatures features) {
        return requireValidDecisionGroupMs(evaluator.predictBatchMs(features));
    }

    /** Evaluate committed work and convert it to lifecycle milliseconds. */
    public static long predictCommittedBatchMs(
            PrefillTimePredictor.Evaluator evaluator,
            PrefillBatchFeatures features) {
        return committedDecisionGroupMs(
                predictDecisionGroupMs(evaluator, features));
    }

    /** Convert an already planned group duration to lifecycle milliseconds. */
    public static long committedDecisionGroupMs(double predictedMs) {
        requireValidDecisionGroupMs(predictedMs);
        return predictedMs >= Long.MAX_VALUE
                ? Long.MAX_VALUE
                : (long) predictedMs;
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

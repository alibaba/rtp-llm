package org.flexlb.balance.prediction;

import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Prefill-time prediction contract.
 *
 * <p>One scheduling operation captures one immutable {@link Evaluator}. Online
 * learning may publish a replacement evaluator concurrently, but it cannot
 * change an evaluation already in progress.
 */
public interface PrefillTimePredictor {
    /** Absolute visibility estimate. Unknown preceding work deliberately produces no deadline. */
    public static long predictedVisibilityDeadlineMs(
            WorkSnapshot work, long unstartedMs, double lifetime, long nowMs) {
        if (unstartedMs < 0L || !Double.isFinite(lifetime) || lifetime < 1.0) {
            throw new IllegalArgumentException("invalid delivery prediction inputs");
        }
        var preceding = java.util.Objects.requireNonNull(work).totalRemainingWorkMsAt(nowMs);
        if (preceding.isEmpty()) {
            return 0L;
        }
        double scaled = Math.ceil(saturatedAdd(preceding.getAsLong(), unstartedMs) * lifetime);
        long duration = scaled >= Long.MAX_VALUE ? Long.MAX_VALUE : (long) scaled;
        return saturatedAdd(nowMs, saturatedAdd(duration, 10_000L));
    }

    /** Evaluate one request and require a non-negative duration. */
    public static long predictSingleRequestMs(
            PrefillTimePredictor.Evaluator evaluator, long totalTokens, long hitTokens) {
        long predictedMs = evaluator.estimateMs(totalTokens, hitTokens);
        if (predictedMs < 0L) {
            throw new InvalidPrefillPredictionException("SINGLE_REQUEST", predictedMs);
        }
        return predictedMs;
    }

    /** Evaluate committed work and convert it to lifecycle milliseconds. */
    public static long predictCommittedBatchMs(
            PrefillTimePredictor.Evaluator evaluator, PrefillBatchFeatures features) {
        return committedDecisionGroupMs(evaluator.predictBatchMs(features));
    }

    /** Convert an already planned group duration to lifecycle milliseconds. */
    public static long committedDecisionGroupMs(double predictedMs) {
        requireValidDecisionGroupMs(predictedMs);
        return predictedMs >= Long.MAX_VALUE ? Long.MAX_VALUE : (long) Math.ceil(predictedMs);
    }

    /** Require a finite, non-negative group duration. */
    public static double requireValidDecisionGroupMs(double predictedMs) {
        if (!Double.isFinite(predictedMs) || predictedMs < 0.0) {
            throw new InvalidPrefillPredictionException("DECISION_GROUP", predictedMs);
        }
        return predictedMs;
    }

    /** Capture the immutable model used by one scheduling operation. */
    Evaluator evaluator();

    /**
     * Publish learning from one completed batch when applicable.
     *
     * @return whether a replacement evaluator was published
     */
    LearningResult learn(PrefillBatchFeatures features, long predictedMs, long actualMs);

    interface Evaluator {
        /**
         * Identity of the immutable model snapshot behind this evaluator.
         *
         * <p>Independent endpoint predictors may return the same object only
         * when their predictions are guaranteed identical for equal inputs.
         * Projection uses this identity to reuse an equal singleton prediction
         * while it still evaluates every endpoint's queue and cache state.
         */
        default Object snapshotIdentity() {
            return this;
        }

        /** Estimate one request from its input and cache-hit token counts. */
        long estimateMs(long totalTokens, long hitTokens);

        /** Fresh state for one append-only planning operation on this immutable model. */
        default BatchPrediction newBatchPrediction() {
            java.util.List<PrefillBatchFeatures.Item> items = new java.util.ArrayList<>();
            return (seqLen, hitCache) -> {
                items.add(new PrefillBatchFeatures.Item(seqLen, hitCache));
                return predictBatchMs(new PrefillBatchFeatures(items));
            };
        }

        /** Estimate one payload-free batch. */
        double predictBatchMs(PrefillBatchFeatures features);
    }

    /**
     * Not shared or reused after a membership/model change; result retains fractional milliseconds.
     */
    @FunctionalInterface
    interface BatchPrediction {
        double append(long seqLen, long hitCache);
    }

    enum LearningResult {
        MODEL_UNCHANGED,
        MODEL_UPDATED
    }

    /**
     * Immutable, payload-free features retained for prediction and learning.
     */
    public record PrefillBatchFeatures(List<Item> items) {

        public PrefillBatchFeatures {
            items = List.copyOf(items);
        }

        /**
         * Materialize predictor features without depending on a scheduling item
         * type.
         */
        public static <T> PrefillBatchFeatures from(
                List<T> source,
                ToLongFunction<? super T> seqLen,
                ToLongFunction<? super T> hitCache) {
            List<Item> features = new ArrayList<>(source.size());
            for (T item : source) {
                features.add(new Item(seqLen.applyAsLong(item), hitCache.applyAsLong(item)));
            }
            return new PrefillBatchFeatures(features);
        }

        public int batchSize() {
            return items.size();
        }

        public record Item(long seqLen, long hitCache) implements ArithmeticFormula.Variables {
            public Item {
                checkArgument(seqLen >= 0L, "seqLen must be non-negative");
                checkArgument(
                        hitCache >= 0L && hitCache <= seqLen, "hitCache must be in [0, seqLen]");
            }

            @Override
            public double variable(int index) {
                return switch (index) {
                    case PrefillTimeFormula.IDX_INPUT_TOKENS -> seqLen;
                    case PrefillTimeFormula.IDX_HIT_CACHE_TOKENS -> hitCache;
                    case PrefillTimeFormula.IDX_COMPUTE_TOKENS -> seqLen - hitCache;
                    case PrefillTimeFormula.IDX_HAS_HIT_CACHE -> hitCache > 0L ? 1.0 : 0.0;
                    default -> 0.0;
                };
            }
        }
    }

    /**
     * A predictor returned a numeric value outside the scheduling contract.
     */
    public static final class InvalidPrefillPredictionException extends IllegalStateException {
        private final int memberIndex;

        InvalidPrefillPredictionException(String predictionKind, Number returnedValue) {
            super(
                    "Prefill predictor returned invalid "
                            + predictionKind
                            + " milliseconds: "
                            + returnedValue);
            memberIndex = 0;
        }

        private InvalidPrefillPredictionException(
                InvalidPrefillPredictionException cause, int index) {
            super(cause.getMessage(), cause);
            memberIndex = index;
        }

        /** Position in the ordered decision group; no request or input is retained by the failure. */
        public InvalidPrefillPredictionException atMember(int index) {
            return new InvalidPrefillPredictionException(this, index);
        }

        public int memberIndex() {
            return memberIndex;
        }
    }
}

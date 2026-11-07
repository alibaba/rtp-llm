package org.flexlb.balance.planner;

import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;

import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/** Ordered-prefix grouping shared by execution and prediction; implementations do not mutate queues. */
public interface GroupPlanner {
    String PREDICTED_EXECUTION_CAP = "predicted_execution_cap";
    String BATCH_FULL = "batch_full";
    String FIXED_WINDOW_TIMEOUT = "fixed_window_timeout";

    <T extends Input> Selection<T> select(
            Iterable<T> ordered, Constraints constraints, PrefixPrediction<T> predictor);

    String dispatchReason(Selection<?> selection, Constraints constraints, long nowMs);

    long nextCheckAtMs(Selection<?> selection, Constraints constraints);

    void validateConstraints(Constraints constraints);

    /**
     * Immutable input suitable for route-time projection. Ordering fields are
     * carried even though this planner deliberately requires its input to have
     * already been sorted by the queue's production comparator.
     */
    public record Item(
            String requestId,
            int priority,
            long enqueueSeq,
            long enqueuedAtMs,
            long expiresAtMs,
            PrefillBatchFeatures.Item features)
            implements Input, PrefillRequestQueue.QueueItem {

        public Item {
            Objects.requireNonNull(features, "features");
        }

        public Item(
                String requestId,
                int priority,
                long enqueueSeq,
                long enqueuedAtMs,
                long expiresAtMs,
                long seqLen,
                long hitCache) {
            this(
                    requestId,
                    priority,
                    enqueueSeq,
                    enqueuedAtMs,
                    expiresAtMs,
                    new PrefillBatchFeatures.Item(seqLen, hitCache));
        }

        @Override
        public long seqLen() {
            return features.seqLen();
        }

        public long hitCache() {
            return features.hitCache();
        }
    }

    /** Read-only request fields consumed by both live and projected planning. */
    public interface Input {
        long enqueuedAtMs();

        long seqLen();
    }

    /** Frozen policy and resource bounds for one planning operation. */
    public record Constraints(
            int maxRequests,
            long batchTokenCapacity,
            long batchKvCapacity,
            long predictedExecutionBudgetMs,
            long collectionWindowMs) {

        public Constraints {
            checkArgument(maxRequests >= 1, "maxRequests must be positive");
        }
    }

    /** Group selection before readiness is evaluated against a clock value. */
    public record Selection<T>(
            List<T> items,
            long paddedTokens,
            long kvTokens,
            long windowOpenedAtMs,
            boolean predictionBoundaryTriggered,
            OptionalDouble selectedPredictionMs) {

        public Selection {
            items = items == null ? List.of() : List.copyOf(items);
            validateSelectedPrediction(items, selectedPredictionMs);
        }

        public boolean fitsCompute(long capacity) {
            return capacity > 0L && paddedTokens < capacity;
        }

        public boolean fitsKv(long capacity) {
            return capacity == Long.MAX_VALUE || (capacity >= 0L && kvTokens <= capacity);
        }
    }

    /**
     * Owned by one selection: invoked once per visited prefix in append order.
     * The last tentative member may exceed the prediction budget and be excluded from the result.
     * Callbacks must not retain the mutable prefix list.
     */
    @FunctionalInterface
    public interface PrefixPrediction<T> {
        double append(T added, List<T> prefix);
    }

    public static boolean windowElapsed(
            long windowOpenedAtMs, long nowMs, long collectionWindowMs) {
        long boundedWindowMs = Math.max(0L, collectionWindowMs);
        return windowOpenedAtMs != Long.MAX_VALUE
                && nowMs >= windowOpenedAtMs
                && nowMs - windowOpenedAtMs >= boundedWindowMs;
    }

    public static long collectionDeadlineMs(long windowOpenedAtMs, long collectionWindowMs) {
        return saturatedAdd(windowOpenedAtMs, Math.max(0L, collectionWindowMs));
    }

    private static void validateSelectedPrediction(
            List<?> items, OptionalDouble selectedPredictionMs) {
        checkArgument(
                !items.isEmpty() || !selectedPredictionMs.isPresent(),
                "empty decision group cannot carry a prediction");
        if (selectedPredictionMs.isPresent()) {
            requireValidPrediction(selectedPredictionMs.getAsDouble());
        }
    }

    static double requireValidPrediction(double predictedMs) {
        checkArgument(
                Double.isFinite(predictedMs) && !(predictedMs < 0.0d),
                "group prediction must be finite and non-negative");
        return predictedMs;
    }
}

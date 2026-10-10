package org.flexlb.balance.planner;

import org.flexlb.balance.prediction.PrefillBatchFeatures;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;
import static com.google.common.math.LongMath.saturatedMultiply;

/**
 * Pure fixed-window decision-group planning.
 *
 * <p>The planner consumes an already ordered, immutable point-in-time view. It
 * performs no queue mutation, capacity reservation, clock read, or delivery.
 * The real batcher supplies live {@link Input}s; a
 * route-time projection instead uses {@link Item}s, including a
 * virtual incoming probe that has never entered the live queue.
 */
public final class GroupPlanner {

    /** Allocation hint only; {@link Constraints#maxRequests()} is authoritative. */
    private static final int INITIAL_SELECTION_CAPACITY = 32;

    public static final String PREDICTED_EXECUTION_CAP = "predicted_execution_cap";
    public static final String BATCH_FULL = "batch_full";
    public static final String FIXED_WINDOW_TIMEOUT = "fixed_window_timeout";

    private GroupPlanner() {
    }

    /**
     * Immutable input suitable for route-time projection. Ordering fields are
     * carried even though this planner deliberately requires its input to have
     * already been sorted by the queue's production comparator.
     */
    public record Item(
            long requestId,
            int priority,
            long enqueueSeq,
            long enqueuedAtMs,
            long expiresAtMs,
            PrefillBatchFeatures.Item features) implements Input {

        public Item {
            Objects.requireNonNull(features, "features");
        }

        public Item(long requestId, int priority, long enqueueSeq, long enqueuedAtMs,
                    long expiresAtMs, long seqLen, long hitCache) {
            this(requestId, priority, enqueueSeq, enqueuedAtMs, expiresAtMs,
                    new PrefillBatchFeatures.Item(seqLen, hitCache));
        }

        @Override
        public long seqLen() { return features.seqLen(); }

        public long hitCache() { return features.hitCache(); }
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
            return capacity == Long.MAX_VALUE
                    || (capacity >= 0L && kvTokens <= capacity);
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

    public static <T extends Input> Selection<T> selectWithPrediction(
            Iterable<T> orderedItems, Constraints constraints,
            PrefixPrediction<T> predictor) {
        Iterator<T> ordered = orderedItems.iterator();
        if (!ordered.hasNext()) {
            return new Selection<>(List.of(), 0L, 0L,
                    Long.MAX_VALUE, false, OptionalDouble.empty());
        }

        int maxRequests = constraints.maxRequests();
        T head = ordered.next();
        boolean mayGrow = maxRequests > 1 && ordered.hasNext();
        List<T> picked;
        if (mayGrow) {
            picked = new ArrayList<>(Math.min(maxRequests, INITIAL_SELECTION_CAPACITY));
            picked.add(head);
        } else {
            picked = List.of(head);
        }
        long headTokens = Math.max(0L, head.seqLen());
        long maxSeqLen = headTokens;
        long paddedTokens = headTokens;
        long kvTokens = headTokens;
        long windowOpenedAtMs = head.enqueuedAtMs();
        boolean predictionEnabled = predictor != null
                && constraints.predictedExecutionBudgetMs() > 0L;
        double selectedPredictionMs = 0.0;
        boolean predictionBoundaryTriggered = false;
        if (predictionEnabled) {
            selectedPredictionMs = requireValidPrediction(predictor.append(head, picked));
            predictionBoundaryTriggered = selectedPredictionMs >= constraints.predictedExecutionBudgetMs();
        }

        while (mayGrow && ordered.hasNext()
                && picked.size() < maxRequests
                && !predictionBoundaryTriggered) {
            T item = ordered.next();
            long itemTokens = Math.max(0L, item.seqLen());
            long nextMaxSeqLen = Math.max(maxSeqLen, itemTokens);
            long nextPaddedTokens = saturatedMultiply(nextMaxSeqLen, picked.size() + 1);
            long nextKvTokens = saturatedAdd(kvTokens, itemTokens);
            if (constraints.batchTokenCapacity() <= 0L
                    || nextPaddedTokens >= constraints.batchTokenCapacity()) {
                break;
            }
            if (constraints.batchKvCapacity() != Long.MAX_VALUE
                    && (constraints.batchKvCapacity() < 0L
                        || nextKvTokens > constraints.batchKvCapacity())) {
                break;
            }

            picked.add(item);
            if (predictionEnabled) {
                double predictedMs = requireValidPrediction(predictor.append(item, picked));
                if (predictedMs > constraints.predictedExecutionBudgetMs()) {
                    predictionBoundaryTriggered = true;
                    // The head is indivisible. An additional over-budget member
                    // stays queued for the following decision.
                    picked.remove(picked.size() - 1);
                    break;
                }
                selectedPredictionMs = predictedMs;
                predictionBoundaryTriggered = predictedMs >= constraints.predictedExecutionBudgetMs();
            }
            maxSeqLen = nextMaxSeqLen;
            paddedTokens = nextPaddedTokens;
            kvTokens = nextKvTokens;
            windowOpenedAtMs = Math.min(windowOpenedAtMs, item.enqueuedAtMs());
        }

        return new Selection<>(picked,
                paddedTokens, kvTokens, windowOpenedAtMs,
                predictionBoundaryTriggered,
                predictionEnabled ? OptionalDouble.of(selectedPredictionMs) : OptionalDouble.empty());
    }

    /** Return the dispatch reason at this clock, or null while the selected group must wait. */
    public static String dispatchReason(Selection<?> selection, Constraints constraints, long nowMs) {
        if (selection.predictionBoundaryTriggered()) {
            return PREDICTED_EXECUTION_CAP;
        }
        if (!selection.items().isEmpty() && selection.items().size() >= constraints.maxRequests()) {
            return BATCH_FULL;
        }
        return windowElapsed(selection.windowOpenedAtMs(), nowMs, constraints.collectionWindowMs())
                ? FIXED_WINDOW_TIMEOUT : null;
    }

    public static boolean windowElapsed(long windowOpenedAtMs,
                                        long nowMs,
                                        long collectionWindowMs) {
        long boundedWindowMs = Math.max(0L, collectionWindowMs);
        return windowOpenedAtMs != Long.MAX_VALUE
                && nowMs >= windowOpenedAtMs
                && nowMs - windowOpenedAtMs >= boundedWindowMs;
    }

    public static long collectionDeadlineMs(
            long windowOpenedAtMs, long collectionWindowMs) {
        return saturatedAdd(windowOpenedAtMs, Math.max(0L, collectionWindowMs));
    }

    private static void validateSelectedPrediction(
            List<?> items, OptionalDouble selectedPredictionMs) {
        checkArgument(!items.isEmpty() || !selectedPredictionMs.isPresent(),
                "empty decision group cannot carry a prediction");
        if (selectedPredictionMs.isPresent()) {
            requireValidPrediction(selectedPredictionMs.getAsDouble());
        }
    }

    private static double requireValidPrediction(double predictedMs) {
        checkArgument(Double.isFinite(predictedMs) && !(predictedMs < 0.0d),
                "group prediction must be finite and non-negative");
        return predictedMs;
    }

}

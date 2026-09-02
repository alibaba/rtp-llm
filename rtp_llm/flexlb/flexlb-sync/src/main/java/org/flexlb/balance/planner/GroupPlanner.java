package org.flexlb.balance.planner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.OptionalDouble;
import java.util.function.ToDoubleFunction;

/**
 * Pure fixed-window decision-group planning.
 *
 * <p>The planner consumes an already ordered, immutable point-in-time view. It
 * performs no queue mutation, capacity reservation, clock read, or delivery.
 * The real batcher supplies live items through an {@link ItemAccess}; a
 * route-time projection instead uses {@link Item}s, including a
 * virtual incoming probe that has never entered the live queue.
 */
public final class GroupPlanner {

    public static final String PREDICTED_EXECUTION_CAP = "predicted_execution_cap";
    public static final String BATCH_FULL = "batch_full";
    public static final String FIXED_WINDOW_TIMEOUT = "fixed_window_timeout";

    private static final ItemAccess<Item> ITEM_ACCESS =
            new ItemAccess<>() {
                @Override
                public long enqueuedAtMs(Item item) {
                    return item.enqueuedAtMs();
                }

                @Override
                public long seqLen(Item item) {
                    return item.seqLen();
                }
            };

    private GroupPlanner() {
    }

    /** Whether the selected group is ready now or still inside its window. */
    public enum WindowReadiness {
        READY,
        WAITING
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
            long seqLen,
            long hitCache) {

        public Item {
            if (seqLen < 0L) {
                throw new IllegalArgumentException(
                        "seqLen must be non-negative");
            }
            if (hitCache < 0L || hitCache > seqLen) {
                throw new IllegalArgumentException(
                        "hitCache must be in [0, seqLen]");
            }
        }
    }

    /** Compute and KV resource shape of one planned group. */
    public record Shape(
            int size,
            long maxSeqLen,
            long paddedTokens,
            long kvTokens) {

        public static Shape empty() {
            return new Shape(0, 0L, 0L, 0L);
        }

        public Shape add(long seqLen) {
            int nextSize = size + 1;
            long boundedSeqLen = Math.max(0L, seqLen);
            long nextMaxSeqLen = Math.max(maxSeqLen, boundedSeqLen);
            return new Shape(
                    nextSize,
                    nextMaxSeqLen,
                    saturatedMultiply(nextMaxSeqLen, nextSize),
                    saturatedAdd(kvTokens, boundedSeqLen));
        }

        public boolean fitsCompute(long capacity) {
            return capacity > 0L && paddedTokens < capacity;
        }

        public boolean fitsKv(long capacity) {
            return capacity == Long.MAX_VALUE
                    || (capacity >= 0L && kvTokens <= capacity);
        }

        private static long saturatedMultiply(long value, int multiplier) {
            if (value == 0L || multiplier == 0) {
                return 0L;
            }
            return value > Long.MAX_VALUE / multiplier
                    ? Long.MAX_VALUE : value * multiplier;
        }
    }

    /** Adapter that lets the pure planner operate on either live or projected items. */
    public interface ItemAccess<T> {
        long enqueuedAtMs(T item);

        long seqLen(T item);
    }

    /** Frozen policy and resource bounds for one planning operation. */
    public record Constraints(
            int maxRequests,
            long batchTokenCapacity,
            long batchKvCapacity,
            long predictedExecutionBudgetMs,
            long collectionWindowMs) {

        public Constraints {
            if (maxRequests < 1) {
                throw new IllegalArgumentException(
                        "maxRequests must be positive");
            }
        }
    }

    /** Group selection before readiness is evaluated against a clock value. */
    public record Selection<T>(
            List<T> items,
            Shape shape,
            long windowOpenedAtMs,
            boolean predictionBoundaryTriggered,
            OptionalDouble selectedPredictionMs) {

        public Selection {
            items = items == null ? List.of() : List.copyOf(items);
            validateSelectedPrediction(items, selectedPredictionMs);
        }
    }

    /** Complete pure plan: selected group, shape, dispatch reason, and window state. */
    public record Plan<T>(
            List<T> items,
            Shape shape,
            long windowOpenedAtMs,
            long collectionDeadlineMs,
            boolean predictionBoundaryTriggered,
            OptionalDouble selectedPredictionMs,
            WindowReadiness windowReadiness,
            String reason) {

        public Plan {
            items = items == null ? List.of() : List.copyOf(items);
            validateSelectedPrediction(items, selectedPredictionMs);
            if (windowReadiness == WindowReadiness.READY) {
                if (reason == null) {
                    throw new IllegalArgumentException(
                            "ready plan requires a reason");
                }
            } else if (reason != null) {
                throw new IllegalArgumentException("waiting plan must not have a reason");
            }
        }

        public boolean ready() {
            return windowReadiness == WindowReadiness.READY;
        }
    }

    /** Item adapter for immutable route-time {@link Item}s. */
    public static ItemAccess<Item> itemAccess() {
        return ITEM_ACCESS;
    }

    /**
     * Select the largest feasible homogeneous prefix. This is the production
     * FixedWindow picking rule without readiness or side effects.
     */
    public static <T> Selection<T> select(
            Iterable<T> orderedItems,
            ItemAccess<T> access,
            Constraints constraints,
            ToDoubleFunction<List<T>> predictor) {
        Iterator<T> ordered = orderedItems.iterator();
        if (!ordered.hasNext()) {
            return new Selection<>(List.of(), Shape.empty(),
                    Long.MAX_VALUE, false, OptionalDouble.empty());
        }

        int maxRequests = constraints.maxRequests();
        T head = ordered.next();
        Selection<T> singleton = selectSingleton(
                head, access, constraints, predictor);
        if (maxRequests == 1
                || singleton.predictionBoundaryTriggered()
                || !ordered.hasNext()) {
            return singleton;
        }

        List<T> picked = new ArrayList<>(Math.min(maxRequests, 32));
        picked.add(head);
        Shape shape = singleton.shape();
        long windowOpenedAtMs = singleton.windowOpenedAtMs();
        boolean predictionEnabled = predictor != null
                && constraints.predictedExecutionBudgetMs() > 0L;
        OptionalDouble selectedPredictionMs = singleton.selectedPredictionMs();
        boolean predictionBoundaryTriggered = false;

        while (ordered.hasNext()
                && picked.size() < maxRequests
                && !predictionBoundaryTriggered) {
            T item = ordered.next();

            Shape candidate = shape.add(access.seqLen(item));
            if (!candidate.fitsCompute(constraints.batchTokenCapacity())) {
                break;
            }
            if (!candidate.fitsKv(constraints.batchKvCapacity())) {
                break;
            }

            picked.add(item);
            if (predictionEnabled) {
                double predictedMs = validatedPrediction(predictor, picked);
                if (predictionGrowthLimitExceeded(
                        predictedMs, constraints.predictedExecutionBudgetMs())) {
                    predictionBoundaryTriggered = true;
                    // The head is indivisible. An additional over-budget member
                    // stays queued for the following decision.
                    picked.remove(picked.size() - 1);
                    break;
                }
                selectedPredictionMs = OptionalDouble.of(predictedMs);
                if (predictionDispatchBoundaryReached(
                        predictedMs, constraints.predictedExecutionBudgetMs())) {
                    shape = candidate;
                    windowOpenedAtMs = Math.min(
                            windowOpenedAtMs, access.enqueuedAtMs(item));
                    predictionBoundaryTriggered = true;
                    break;
                }
            }
            shape = candidate;
            windowOpenedAtMs = Math.min(
                    windowOpenedAtMs, access.enqueuedAtMs(item));
        }

        return new Selection<>(picked, shape, windowOpenedAtMs,
                predictionBoundaryTriggered, selectedPredictionMs);
    }

    /** Evaluate one selection against an explicit clock value. */
    public static <T> Plan<T> evaluateReadiness(
            Selection<T> selection,
            Constraints constraints,
            long nowMs) {
        long deadlineMs = collectionDeadlineMs(
                selection.windowOpenedAtMs(), constraints.collectionWindowMs());
        String reason = null;
        if (selection.predictionBoundaryTriggered()) {
            reason = PREDICTED_EXECUTION_CAP;
        } else if (!selection.items().isEmpty()
                && selection.items().size() >= constraints.maxRequests()) {
            reason = BATCH_FULL;
        } else if (windowElapsed(
                selection.windowOpenedAtMs(), nowMs, constraints.collectionWindowMs())) {
            reason = FIXED_WINDOW_TIMEOUT;
        }
        WindowReadiness readiness = reason == null
                ? WindowReadiness.WAITING : WindowReadiness.READY;
        return new Plan<>(
                selection.items(), selection.shape(), selection.windowOpenedAtMs(),
                deadlineMs, selection.predictionBoundaryTriggered(),
                selection.selectedPredictionMs(), readiness, reason);
    }

    /** Convenience entry point for projections whose clock is already frozen. */
    public static <T> Plan<T> plan(
            Iterable<T> orderedItems,
            ItemAccess<T> access,
            Constraints constraints,
            long nowMs,
            ToDoubleFunction<List<T>> predictor) {
        return evaluateReadiness(
                select(orderedItems, access, constraints, predictor), constraints, nowMs);
    }

    private static <T> Selection<T> selectSingleton(
            T item,
            ItemAccess<T> access,
            Constraints constraints,
            ToDoubleFunction<List<T>> predictor) {
        List<T> selected = List.of(item);
        Shape shape = Shape.empty().add(access.seqLen(item));
        long windowOpenedAtMs = access.enqueuedAtMs(item);
        if (predictor == null || constraints.predictedExecutionBudgetMs() <= 0L) {
            return new Selection<>(selected, shape, windowOpenedAtMs,
                    false, OptionalDouble.empty());
        }

        double predictedMs = validatedPrediction(predictor, selected);
        return new Selection<>(selected, shape, windowOpenedAtMs,
                predictionDispatchBoundaryReached(
                        predictedMs, constraints.predictedExecutionBudgetMs()),
                OptionalDouble.of(predictedMs));
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

    private static boolean predictionGrowthLimitExceeded(double predictedMs,
                                                          long thresholdMs) {
        return predictedMs > thresholdMs;
    }

    private static void validateSelectedPrediction(
            List<?> items, OptionalDouble selectedPredictionMs) {
        if (items.isEmpty() && selectedPredictionMs.isPresent()) {
            throw new IllegalArgumentException(
                    "empty decision group cannot carry a prediction");
        }
        if (selectedPredictionMs.isPresent()) {
            requireValidPrediction(selectedPredictionMs.getAsDouble());
        }
    }

    private static <T> double validatedPrediction(
            ToDoubleFunction<List<T>> predictor, List<T> items) {
        return requireValidPrediction(predictor.applyAsDouble(items));
    }

    private static double requireValidPrediction(double predictedMs) {
        if (!Double.isFinite(predictedMs) || predictedMs < 0.0d) {
            throw new IllegalArgumentException(
                    "group prediction must be finite and non-negative");
        }
        return predictedMs;
    }

    private static boolean predictionDispatchBoundaryReached(double predictedMs,
                                                              long thresholdMs) {
        return predictedMs >= thresholdMs;
    }

    private static long saturatedAdd(long left, long right) {
        return right > 0L && left > Long.MAX_VALUE - right
                ? Long.MAX_VALUE : left + right;
    }

}

package org.flexlb.balance.planner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.OptionalDouble;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;
import static com.google.common.math.LongMath.saturatedMultiply;

/**
 * Pure single-request and fixed-window group planning and readiness.
 *
 * <p>The planner consumes an already ordered, immutable point-in-time view. It
 * performs no queue mutation, capacity reservation, clock read, or delivery.
 * The real batcher supplies live {@link Input}s; a
 * route-time projection instead uses {@link Item}s, including a
 * virtual incoming probe that has never entered the live queue.
 */
public enum BuiltinGroupPlanner implements GroupPlanner {
    SINGLE,
    FIXED_WINDOW;

    public static BuiltinGroupPlanner from(org.flexlb.config.FlexlbConfig config) {
        return config.isQueue()
                        && config.decisionPolicy().getType()
                                == org.flexlb.config.DecisionPolicyConfig.Type.FIXED_WINDOW
                ? FIXED_WINDOW
                : SINGLE;
    }

    /** Allocation hint only; {@link Constraints#maxRequests()} is authoritative. */
    private static final int INITIAL_SELECTION_CAPACITY = 32;

    private static final int MAX_CACHED_SELECTION_SIZE = 4096;
    private static final ThreadLocal<ArrayList<?>> SELECTIONS =
            ThreadLocal.withInitial(() -> new ArrayList<>(INITIAL_SELECTION_CAPACITY));

    @SuppressWarnings(
            "unchecked") // The thread-local list is empty at acquisition and cleared before return.
    public <T extends Input> Selection<T> select(
            Iterable<T> orderedItems, Constraints constraints, PrefixPrediction<T> predictor) {
        validateConstraints(constraints);
        Iterator<T> ordered = orderedItems.iterator();
        if (!ordered.hasNext()) {
            return new Selection<>(
                    List.of(), 0L, 0L, Long.MAX_VALUE, false, OptionalDouble.empty());
        }

        int maxRequests = constraints.maxRequests();
        T head = ordered.next();
        boolean mayGrow = maxRequests > 1 && ordered.hasNext();
        List<T> picked;
        if (mayGrow) {
            var cached = SELECTIONS.get();
            // A predictor may select another group recursively; never share its caller's prefix.
            picked =
                    cached.isEmpty()
                            ? (ArrayList<T>) cached
                            : new ArrayList<>(Math.min(maxRequests, INITIAL_SELECTION_CAPACITY));
            picked.add(head);
        } else {
            picked = List.of(head);
        }
        try {
            return selectGroup(
                    ordered, constraints, this == SINGLE ? null : predictor, picked, mayGrow);
        } finally {
            if (mayGrow) {
                if (picked.size() >= MAX_CACHED_SELECTION_SIZE && picked == SELECTIONS.get()) {
                    SELECTIONS.remove();
                }
                picked.clear();
            }
        }
    }

    /** Packs one group; the caller owns and clears the borrowed prefix. */
    private static <T extends Input> Selection<T> selectGroup(
            Iterator<T> ordered,
            Constraints constraints,
            PrefixPrediction<T> predictor,
            List<T> picked,
            boolean mayGrow) {
        T head = picked.getFirst();
        int maxRequests = constraints.maxRequests();
        long headTokens = Math.max(0L, head.seqLen());
        long maxSeqLen = headTokens;
        long paddedTokens = headTokens;
        long kvTokens = headTokens;
        long windowOpenedAtMs = head.enqueuedAtMs();
        boolean predictionEnabled =
                predictor != null && constraints.predictedExecutionBudgetMs() > 0L;
        double selectedPredictionMs = 0.0;
        boolean predictionBoundaryTriggered = false;
        if (predictionEnabled) {
            selectedPredictionMs =
                    GroupPlanner.requireValidPrediction(predictor.append(head, picked));
            predictionBoundaryTriggered =
                    selectedPredictionMs >= constraints.predictedExecutionBudgetMs();
        }

        while (mayGrow
                && ordered.hasNext()
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
                double predictedMs =
                        GroupPlanner.requireValidPrediction(predictor.append(item, picked));
                if (predictedMs > constraints.predictedExecutionBudgetMs()) {
                    predictionBoundaryTriggered = true;
                    // The head is indivisible. An additional over-budget member
                    // stays queued for the following decision.
                    picked.remove(picked.size() - 1);
                    break;
                }
                selectedPredictionMs = predictedMs;
                predictionBoundaryTriggered =
                        predictedMs >= constraints.predictedExecutionBudgetMs();
            }
            maxSeqLen = nextMaxSeqLen;
            paddedTokens = nextPaddedTokens;
            kvTokens = nextKvTokens;
            windowOpenedAtMs = Math.min(windowOpenedAtMs, item.enqueuedAtMs());
        }

        List<T> selected =
                !mayGrow
                        ? picked
                        : switch (picked.size()) {
                            case 1 -> List.of(picked.getFirst());
                            case 2 -> List.of(picked.get(0), picked.get(1));
                            default -> picked;
                        };
        return new Selection<>(
                selected,
                paddedTokens,
                kvTokens,
                windowOpenedAtMs,
                predictionBoundaryTriggered,
                predictionEnabled
                        ? OptionalDouble.of(selectedPredictionMs)
                        : OptionalDouble.empty());
    }

    /** Return the dispatch reason at this clock, or null while the selected group must wait. */
    public String dispatchReason(Selection<?> selection, Constraints constraints, long nowMs) {
        if (selection.items().isEmpty()) {
            return null;
        }
        if (this == SINGLE) {
            return "single_request";
        }
        if (selection.predictionBoundaryTriggered()) {
            return PREDICTED_EXECUTION_CAP;
        }
        if (selection.items().size() >= constraints.maxRequests()) {
            return BATCH_FULL;
        }
        if (GroupPlanner.windowElapsed(
                selection.windowOpenedAtMs(), nowMs, constraints.collectionWindowMs())) {
            return FIXED_WINDOW_TIMEOUT;
        }
        checkArgument(
                GroupPlanner.collectionDeadlineMs(
                                selection.windowOpenedAtMs(), constraints.collectionWindowMs())
                        >= 0L,
                "collection deadline must be non-negative");
        return null;
    }

    /** Shared by live selection and the projection's single-member calculation. */
    public void validateConstraints(Constraints constraints) {
        checkArgument(
                this != SINGLE
                        || constraints.maxRequests() == 1
                                && constraints.collectionWindowMs() == 0L
                                && constraints.predictedExecutionBudgetMs() == 0L,
                "SINGLE requires one member and no collection or prediction budget");
    }

    /** Next policy wakeup for a group whose dispatchReason is null; empty groups have no timer. */
    public long nextCheckAtMs(Selection<?> selection, Constraints constraints) {
        return selection.items().isEmpty()
                ? Long.MAX_VALUE
                : GroupPlanner.collectionDeadlineMs(
                        selection.windowOpenedAtMs(), constraints.collectionWindowMs());
    }
}

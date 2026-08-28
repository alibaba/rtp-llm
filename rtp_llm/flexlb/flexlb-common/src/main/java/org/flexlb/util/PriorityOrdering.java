package org.flexlb.util;

import java.util.Comparator;

/**
 * Single source of truth for priority-based queue ordering across all
 * FlexLB queue layers (PR-B of the Luoli refactor).
 *
 * <p>The scheduler queue and eviction planner both delegate to this primitive
 * comparison, so priority behavior cannot drift between lifecycle boundaries.
 *
 * <p><b>Ordering rule (STRICT):</b>
 * <ol>
 *   <li>{@link Prioritized#priority()} <em>descending</em> (higher priority
 *       dispatched first).</li>
 *   <li>{@link Prioritized#enqueueSeq()} <em>ascending</em> (same-priority
 *       items are strictly first-in-first-out by enqueue order).</li>
 * </ol>
 *
 * <p>The previous third key — coarse admission deadline — has been
 * <em>removed</em>. The deadline was a weak signal that conflicted with FIFO
 * fairness under bursty arrivals and added complexity without measurable
 * benefit. Same-priority FIFO is now the sole tie-break after priority.
 *
 * <p>Callers that need a deterministic total order (e.g. the batcher queue
 * comparator) append a final {@code .thenComparingLong(...::requestId)} to
 * {@link #strict()}.
 */
public final class PriorityOrdering {

    /**
     * Strict priority-then-FIFO comparator for any {@link Prioritized} item.
     *
     * <p>Priority descending, then enqueue-sequence ascending. This is the
     * shared ordering primitive for the canonical per-worker scheduler
     * queues ({@code WorkerBatcher}).
     */
    public static final Comparator<Prioritized> STRICT = (left, right) -> compare(
            left.priority(), left.enqueueSeq(),
            right.priority(), right.enqueueSeq());

    /**
     * Allocation-free STRICT comparison over primitive ordering keys.
     */
    private static int compare(int leftPriority,
                               long leftEnqueueSeq,
                               int rightPriority,
                               long rightEnqueueSeq) {
        int priorityOrder = Integer.compare(rightPriority, leftPriority);
        return priorityOrder != 0
                ? priorityOrder : Long.compare(leftEnqueueSeq, rightEnqueueSeq);
    }

    /**
     * Allocation-free deterministic total order used by worker queues and
     * admission probes. Request id is consulted only after STRICT ties.
     */
    public static int compareWithRequestId(int leftPriority,
                                           long leftEnqueueSeq,
                                           long leftRequestId,
                                           int rightPriority,
                                           long rightEnqueueSeq,
                                           long rightRequestId) {
        int strictOrder = compare(leftPriority, leftEnqueueSeq,
                rightPriority, rightEnqueueSeq);
        return strictOrder != 0
                ? strictOrder : Long.compare(leftRequestId, rightRequestId);
    }

    /**
     * Returns {@link #STRICT} typed to a specific {@link Prioritized}
     * subtype, allowing further comparator chaining (e.g. adding a
     * deterministic {@code requestId} tie-break) without an unchecked cast.
     *
     * @param <T> the concrete Prioritized subtype
     * @return STRICT as a {@code Comparator<T>}
     */
    @SuppressWarnings("unchecked")
    public static <T extends Prioritized> Comparator<T> strict() {
        return (Comparator<T>) STRICT;
    }

    private PriorityOrdering() {}
}

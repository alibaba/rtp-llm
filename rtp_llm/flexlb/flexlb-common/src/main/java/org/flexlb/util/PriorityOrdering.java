package org.flexlb.util;

import java.util.Comparator;

/**
 * Orders requests by descending priority and ascending enqueue sequence. Callers that require a
 * total order can append their exact request identity as a final tie-breaker.
 */
public final class PriorityOrdering {
    private static final Comparator<String> REQUEST_ID_ORDER =
            Comparator.nullsFirst(Comparator.naturalOrder());

    /**
     * Strict priority-then-FIFO comparator for any {@link Prioritized} item.
     *
     * <p>Priority descending, then enqueue-sequence ascending. This is the
     * shared ordering primitive for the canonical per-worker scheduler
     * queues ({@code WorkerBatcher}).
     */
    public static final Comparator<Prioritized> STRICT =
            (left, right) ->
                    compare(
                            left.priority(), left.enqueueSeq(),
                            right.priority(), right.enqueueSeq());

    private PriorityOrdering() {}

    /**
     * Allocation-free deterministic total order used by worker queues and
     * admission probes. Request id is consulted only after STRICT ties.
     */
    public static int compareWithRequestId(
            int leftPriority,
            long leftEnqueueSeq,
            String leftRequestId,
            int rightPriority,
            long rightEnqueueSeq,
            String rightRequestId) {
        int strictOrder = compare(leftPriority, leftEnqueueSeq, rightPriority, rightEnqueueSeq);
        return strictOrder != 0
                ? strictOrder
                : REQUEST_ID_ORDER.compare(leftRequestId, rightRequestId);
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

    /**
     * Allocation-free STRICT comparison over primitive ordering keys.
     */
    private static int compare(
            int leftPriority, long leftEnqueueSeq, int rightPriority, long rightEnqueueSeq) {
        int priorityOrder = Integer.compare(rightPriority, leftPriority);
        return priorityOrder != 0 ? priorityOrder : Long.compare(leftEnqueueSeq, rightEnqueueSeq);
    }
}

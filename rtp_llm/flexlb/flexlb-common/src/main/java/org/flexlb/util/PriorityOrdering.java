package org.flexlb.util;

/**
 * Primitive ordering keys shared by worker queues and admission projections.
 *
 * <p><b>Ordering rule:</b>
 * <ol>
 *   <li>Priority <em>descending</em> (higher priority
 *       dispatched first).</li>
 *   <li>Enqueue sequence <em>ascending</em> (same-priority
 *       items are strictly first-in-first-out by enqueue order).</li>
 * </ol>
 *
 * <p>{@link #compareWithRequestId(int, long, long, int, long, long)} adds
 * request id as the final deterministic tie-break.
 */
public final class PriorityOrdering {

    /**
     * Allocation-free deterministic total order used by worker queues and
     * admission probes: priority descending, then enqueue sequence and
     * request id ascending.
     */
    public static int compareWithRequestId(int leftPriority,
                                           long leftEnqueueSeq,
                                           long leftRequestId,
                                           int rightPriority,
                                           long rightEnqueueSeq,
                                           long rightRequestId) {
        int priorityOrder = Integer.compare(rightPriority, leftPriority);
        if (priorityOrder != 0) { return priorityOrder; }
        int sequenceOrder = Long.compare(leftEnqueueSeq, rightEnqueueSeq);
        return sequenceOrder != 0
                ? sequenceOrder : Long.compare(leftRequestId, rightRequestId);
    }

    private PriorityOrdering() {}
}

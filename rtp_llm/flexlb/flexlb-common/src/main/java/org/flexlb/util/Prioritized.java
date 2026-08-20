package org.flexlb.util;

/**
 * Unified priority contract for all queueable items in the FlexLB routing
 * and scheduling layers.
 *
 * <p>Implemented by scheduler queue items so
 * {@link PriorityOrdering#STRICT} is the single ordering primitive for
 * canonical per-worker queues.
 *
 * <ul>
 *   <li>{@link #priority()} — normalized priority (1-100, higher = more
 *       important), sourced from {@code SchedulingMetadata}.</li>
 *   <li>{@link #enqueueSeq()} — monotonic enqueue sequence number used as the
 *       FIFO tie-break within the same priority level (earlier = first).</li>
 * </ul>
 *
 * @see PriorityOrdering
 */
public interface Prioritized {

    /**
     * Normalized priority in the range 1-100 (higher = more important).
     *
     * @return priority value 1-100
     */
    int priority();

    /**
     * Monotonic enqueue sequence number used as the same-priority FIFO
     * tie-break in {@link PriorityOrdering#STRICT}.
     *
     * <p>The value is assigned once at enqueue time and never mutated, so a re-offer
     * (e.g. retry / rescue) automatically sorts the item back to its
     * priority-correct position among same-priority items.
     *
     * @return the enqueue sequence number (earlier = ordered first)
     */
    long enqueueSeq();
}

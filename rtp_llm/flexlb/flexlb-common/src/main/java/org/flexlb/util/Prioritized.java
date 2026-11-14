package org.flexlb.util;

/**
 * Exposes normalized request priority and enqueue sequence for shared scheduling order. Higher
 * priorities run first; equal priorities retain FIFO order.
 */
public interface Prioritized {

    /**
     * Normalized priority in the range 1-100 (higher = more important).
     *
     * <p>On the legacy path (Auto-TPM off, budget = {@code null}) the
     * implementation may return 0 as a sentinel — legacy items never
     * participate in priority-based ordering.
     *
     * @return priority value 1-100, or 0 on the legacy path
     */
    int priority();

    /**
     * Monotonic enqueue sequence number used as the same-priority FIFO
     * tie-break in {@link PriorityOrdering#STRICT}.
     *
     * <p>For {@code BalanceContext} this is {@code sequenceId}; for
     * {@code BatchItem} this is {@code enqueuedAtMs}. In both cases the value
     * is assigned once at enqueue time and never mutated, so a re-offer
     * (e.g. retry / rescue) automatically sorts the item back to its
     * priority-correct position among same-priority items.
     *
     * @return the enqueue sequence number (earlier = ordered first)
     */
    long enqueueSeq();
}

package org.flexlb.util;

/**
 * Unified priority contract for all queueable items in the FlexLB routing
 * and scheduling layers.
 *
 * <p>Implemented by {@code BalanceContext} (the top-level request queue) and
 * {@code BatchItem} (the per-worker batcher queue), allowing
 * {@link PriorityOrdering#STRICT} to serve as the single ordering primitive
 * across both queue layers (PR-B of the Luoli refactor).
 *
 * <ul>
 *   <li>{@link #priority()} — normalized Auto-TPM priority (1-100, higher =
 *       more important), sourced from {@code ScheduleBudget} when Auto-TPM is
 *       active, or from {@code Request.getPriority()} on the legacy path.</li>
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

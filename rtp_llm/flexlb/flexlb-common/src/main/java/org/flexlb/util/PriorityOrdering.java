package org.flexlb.util;

import java.util.Comparator;

/**
 * Single source of truth for priority-based queue ordering across all
 * FlexLB queue layers (PR-B of the Luoli refactor).
 *
 * <p>Before this refactor the ordering logic was duplicated: the
 * Auto-TPM batcher comparator in {@code WorkerBatcher} and the probe
 * comparison in {@code PrefillQueueManager.ordersBefore} (removed by the
 * measured queue-age estimate rewrite) were hand-mirrored copies of the
 * same rule. The comparator now delegates to {@link #STRICT}, so any
 * future change to the ordering rule is made in exactly one place.
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
     * shared ordering primitive for both the top-level request queue
     * ({@code QueueManager}) and the per-worker batcher queue
     * ({@code WorkerBatcher}).
     */
    public static final Comparator<Prioritized> STRICT = Comparator
            .comparingInt(Prioritized::priority).reversed()
            .thenComparingLong(Prioritized::enqueueSeq);

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

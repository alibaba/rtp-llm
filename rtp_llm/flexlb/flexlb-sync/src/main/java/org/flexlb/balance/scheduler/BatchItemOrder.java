package org.flexlb.balance.scheduler;

import java.util.Comparator;

/**
 * Explicit pick-order comparators for {@link BatchItem} (Auto-TPM Queue MVP).
 *
 * <p>Design doc §8.1 mandates an explicit comparator instead of bit-encoding
 * priority into a sort key. Ordering dimensions:
 * <ol>
 *   <li>priority descending — higher QoS level is picked first</li>
 *   <li>enqueuedAtMs ascending — FIFO within the same priority</li>
 *   <li>requestId ascending — deterministic tie-break</li>
 * </ol>
 *
 * <p>The batcher-level deadline is the uniform constant
 * {@code flexlbBatchEnqueueDeadlineMs}, so a "deadline asc" dimension would
 * collapse into "arrival asc" and is intentionally not modeled.
 */
public final class BatchItemOrder {

    /** Priority-first pick order: priority desc, arrival asc, requestId asc. */
    public static final Comparator<BatchItem> PRIORITY_FIRST =
            Comparator.comparingInt(BatchItem::priority).reversed()
                    .thenComparingLong(BatchItem::enqueuedAtMs)
                    .thenComparingLong(BatchItem::requestId);

    private BatchItemOrder() {
    }
}

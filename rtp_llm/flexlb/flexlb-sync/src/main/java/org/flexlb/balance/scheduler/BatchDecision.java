package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * Result of one {@link FixedWindowBatcherAlgorithm#decide} cycle.
 *
 * <p>The algorithm owns its queue and removes picked / dropped items inside
 * {@code decide} before returning. The {@link WorkerBatcher} run loop
 * interprets the decision and executes the remaining side effects (metric
 * reporting, dispatch to the engine, settlement).
 *
 * <p>A {@code null} return from {@code decide} means no action this cycle
 * (park / engine backpressure); the batcher parks briefly and retries.
 */
public sealed interface BatchDecision {

    /**
     * Successful batch assembly — ready to dispatch.
     *
     * @param items           picked items in FIFO order (or priority-first order when
     *                        auto-TPM priority scheduling is enabled), never empty
     * @param reason          {@code "batch_full"} | {@code "fixed_window_timeout"}
     *                        | {@code "predict_threshold"}
     * @param headWaitMs      head item enqueue-to-now elapsed time
     * @param queueSizeBefore queue depth before the pick
     */
    record Dispatch(List<BatchItem> items,
                    String reason,
                    long headWaitMs,
                    int queueSizeBefore) implements BatchDecision {
    }

    /**
     * Head item must be dropped (expired or impossible to batch).
     *
     * @param item   the head item to remove and settle
     * @param cause  why the item is dropped
     * @param detail human-readable detail for logging
     */
    record Drop(BatchItem item,
                DropCause cause,
                String detail) implements BatchDecision {
    }

    /** Why the head item was dropped. */
    enum DropCause {
        QUEUE_DEADLINE_EXCEEDED,
        EXCEEDS_BATCH_TOKEN_CAPACITY
    }
}

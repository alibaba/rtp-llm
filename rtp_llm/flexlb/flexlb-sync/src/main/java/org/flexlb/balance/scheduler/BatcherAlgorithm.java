package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * Batching algorithm that owns its queue container and produces dispatch
 * decisions.
 *
 * <p>The algorithm is responsible for all queue operations: enqueuing
 * items in {@link #offer}, and removing picked / dropped items inside
 * {@link #decide}. The enclosing {@link WorkerBatcher} is a thin shell
 * that handles thread coordination and side-effect execution (metric
 * reporting, dispatch to the engine, settlement) but never touches the
 * queue directly.
 *
 * <p>A {@code null} return from {@link #decide} means no action this
 * cycle (park / engine backpressure); the batcher parks briefly and
 * retries.
 */
public interface BatcherAlgorithm {

    /** Enqueue an item into the algorithm's own container. */
    void offer(BatchItem item);

    /**
     * Produce the next dispatch / drop decision, removing the affected
     * items from the internal container.
     *
     * @return the decision for this cycle, or {@code null} to park
     */
    BatchDecision decide();

    /**
     * Estimated remaining wait time for the head of the queue before its
     * batch is dispatched. Returns 0 when the queue is empty.
     */
    long queueWaitMs();

    /** Current number of queued items. */
    int size();

    /** Clear the internal container. */
    void shutdown();

    /**
     * Drain all remaining items into the destination list. Used by
     * {@link WorkerBatcher#shutdown()} to settle queued items on close.
     * Default implementation is a no-op (implementations override as needed).
     */
    default void drainTo(List<BatchItem> dst) {}
}

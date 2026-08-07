package org.flexlb.balance.scheduler;

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
     * Effective strict padded-token limit for one batch, used by the default
     * {@link #check}. Implementations derive it from the static FlexLB config
     * and the latest worker-reported status (so the value may shrink between
     * offer and dispatch).
     */
    long batchTokenCapacity();

    /**
     * Pre-enqueue admission check (command-style query). Called by
     * {@link WorkerBatcher#offer} after the stopped / queue-full guards and
     * before {@link #offer} actually enqueues the item. Returns {@code null}
     * to admit the item; a non-null reject reason fails the offer via
     * {@link BatchItem#failOffer} without enqueueing.
     *
     * <p>The default implementation rejects a request whose own padded shape
     * can never fit the strict batch token capacity — such a request could
     * never be picked by any batch and would only occupy queue capacity until
     * it expires.
     *
     * <p>{@link #batchTokenCapacity()} is partly derived from worker-reported
     * status and may change between offer and dispatch, so {@link #decide}
     * implementations keep an equivalent head rejection as a fallback.
     */
    default String check(BatchItem item) {
        long capacity = batchTokenCapacity();
        if (!BatchShape.empty().add(item).fitsCompute(capacity)) {
            return "request seq_len=" + item.seqLen()
                    + " cannot fit strict padded batch token capacity=" + capacity;
        }
        return null;
    }

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
}

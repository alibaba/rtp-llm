package org.flexlb.balance.scheduler;

/**
 * Batching algorithm contract. One instance per {@link WorkerBatcher}.
 *
 * <p>Implementations encapsulate dispatch decision logic — when to
 * assemble a batch, how many items to pick, and when to wait.
 */
public interface BatcherAlgorithm {

    /**
     * Core decision loop. Called by {@link WorkerBatcher#runLoop()} each
     * iteration when the queue is non-empty.
     *
     * <p>On each call the implementation should make one of:
     * <ul>
     *   <li>Dispatch items via {@link BatcherContext#dispatch}</li>
     *   <li>Drop the head item via {@link BatcherContext#dropHead}
     *       (only for algorithms that support expiry)</li>
     *   <li>Park briefly (e.g. {@code TimeUnit.MILLISECONDS.sleep(1)})
     *       and return, letting the caller re-invoke</li>
     * </ul>
     */
    void processQueue(BatcherContext ctx) throws InterruptedException;

    /**
     * Compute the sort key used to order items in the per-worker
     * priority queue. Called by {@link WorkerBatcher#offer} before the
     * item is enqueued; the result is stored via
     * {@link BatchItem#setSortKey(long)}.
     *
     * <p>Different algorithms compute this differently:
     * <ul>
     *   <li>SLO-budget: SLO deadline = now + (sloMs - predMs - workerQueueMs)</li>
     *   <li>Fixed-window: FIFO (arrival timestamp)</li>
     * </ul>
     *
     * <p>The implementation can resolve SLO, predictor coefficients,
     * and worker queue depth from {@link BatcherContext#cfg()} and
     * {@link BatcherContext#prefillEp()}.
     */
    long computeSortKey(BatcherContext ctx, BatchItem item);

    /**
     * Hook called by {@link WorkerBatcher#offer} after the sort key is
     * computed and set. Gives the algorithm a chance to update arrival
     * statistics or perform lightweight bookkeeping.
     */
    default void onOffer(BatcherContext ctx, BatchItem item, long nowMs) {
    }

    /**
     * Estimated remaining wait time of the head request in the batcher
     * queue. Used by load-balancing strategies to compare workers without
     * leaking sort-key semantics.
     *
     * <p>Each algorithm computes this according to its own dispatch model:
     * <ul>
     *   <li>SLO-budget: remaining SLO slack = {@code sortKey - now}</li>
     *   <li>Fixed-window: remaining fixed window = {@code fixedWaitMs - elapsedMs}</li>
     * </ul>
     *
     * <p>The default implementation treats {@link BatchItem#sortKey()} as
     * a future deadline, which is correct for deadline-based algorithms.
     */
    default long headWaitMs(BatcherContext ctx) {
        BatchItem head = ctx.peek();
        if (head == null) {
            return 0;
        }
        return Math.max(0, head.sortKey() - ctx.now());
    }

    /**
     * Estimated time a new request would wait in the batcher queue before
     * its batch is dispatched to the engine. Used by load-balancing
     * strategies for worker selection scoring.
     *
     * <p>Unlike {@link #headWaitMs}, this accounts for the empty-queue
     * case: when the queue is empty, a new request starts a fresh batch
     * cycle and must wait for the dispatch trigger (e.g. fixed window
     * timeout).
     *
     * <p>Each algorithm defines this according to its dispatch model:
     * <ul>
     *   <li>Fixed-window: queue non-empty → remaining window;
     *       empty → full {@code fixedWaitMs} (new cycle).</li>
     *   <li>SLO-budget: remaining SLO slack of the head request.</li>
     * </ul>
     */
    default long queueWaitMs(BatcherContext ctx) {
        return headWaitMs(ctx);
    }

    /**
     * Hook called by {@link WorkerBatcher#shutdown} before the queue is drained.
     * Gives the algorithm a chance to clean up internal state.
     */
    default void onShutdown(BatcherContext ctx) {
    }
}

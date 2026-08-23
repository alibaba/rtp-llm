package org.flexlb.balance.scheduler;

/**
 * Queue decision-policy contract. One instance per {@link WorkerBatcher}.
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
     *   <li>Stage a decision group through {@link BatcherContext}</li>
     *   <li>Drop the head item via {@link BatcherContext#dropHead}
     *       (only for algorithms that support expiry)</li>
     *   <li>Park briefly (e.g. {@code TimeUnit.MILLISECONDS.sleep(1)})
     *       and return, letting the caller re-invoke</li>
     * </ul>
     */
    void processQueue(BatcherContext ctx) throws InterruptedException;

    /**
     * Hook called by {@link WorkerBatcher#offer} before enqueue. Gives the
     * algorithm a chance to update arrival statistics or perform lightweight
     * bookkeeping.
     */
    default void onOffer(BatcherContext ctx, BatchItem item, long nowMs) {
    }

    /**
     * Estimated time a new request would wait before its configured dispatch.
     */
    long queueWaitMs(BatcherContext ctx);

    /**
     * Hook called by {@link WorkerBatcher#shutdown} before the queue is drained.
     * Gives the algorithm a chance to clean up internal state.
     */
    default void onShutdown(BatcherContext ctx) {
    }
}

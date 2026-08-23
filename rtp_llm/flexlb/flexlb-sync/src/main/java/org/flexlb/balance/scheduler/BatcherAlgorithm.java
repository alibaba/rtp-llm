package org.flexlb.balance.scheduler;

/**
 * Queue decision-policy contract. One instance per {@link WorkerBatcher}.
 *
 * <p>Implementations encapsulate grouping and admission decisions — when to
 * form a group, how many items to propose, and when to wait.
 */
public interface BatcherAlgorithm {

    /**
     * Core decision loop. Called by {@link WorkerBatcher#runLoop()} each
     * iteration when the queue is non-empty.
     *
     * <p>On each call the implementation must return one typed outcome:
     * <ul>
     *   <li>Reserve hard capacity and deliver an admitted group through
     *       {@link BatcherContext}</li>
     *   <li>Drop the head item via {@link BatcherContext#dropHead}
     *       (only for algorithms that support expiry)</li>
     *   <li>Report the exact capacity resource, state generation, or deadline
     *       on which the worker must wait</li>
     * </ul>
     * The algorithm never sleeps or polls; {@link WorkerBatcher} owns all
     * condition waiting.
     */
    BatcherCycleResult processQueue(BatcherContext ctx);

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
     * Hook called by {@link WorkerBatcher#shutdown} after the active queue is drained.
     * Gives the algorithm a chance to clean up internal state.
     */
    default void onShutdown(BatcherContext ctx) {
    }
}

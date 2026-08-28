package org.flexlb.balance.scheduler;

/**
 * Queue decision-policy contract. One instance per {@link WorkerBatcher}.
 *
 * <p>Implementations encapsulate grouping and admission decisions — when to
 * form a group, how many items to propose, and when to wait.
 */
interface GroupPolicy {

    /**
     * Core decision loop. Called by {@link WorkerBatcher#runLoop()} each
     * iteration when the queue is non-empty.
     *
     * <p>On each call the implementation must return one typed outcome:
     * <ul>
     *   <li>Reserve hard capacity and deliver an admitted group through
     *       {@link BatcherContext}</li>
     *   <li>Drop the head item via {@link BatcherContext#dropHead}
     *       (only for policies that support expiry)</li>
     *   <li>Report the exact capacity resource, state generation, or deadline
     *       on which the worker must wait</li>
     * </ul>
     * The policy never sleeps or polls; {@link WorkerBatcher} owns all
     * condition waiting.
     */
    BatcherCycleResult processQueue(BatcherContext ctx);
}

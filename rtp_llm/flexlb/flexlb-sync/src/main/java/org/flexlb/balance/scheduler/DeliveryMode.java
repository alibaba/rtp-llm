package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;

/**
 * Immutable delivery choice captured when a request enters the priority scheduler.
 *
 * <p>The decision policy proposes an ordered candidate group. Capacity
 * admission turns a feasible prefix into the final group, and this value
 * determines whether the master sends that group through {@code EnqueueBatch}
 * or publishes route decisions for the frontend. Keeping the choice on
 * {@link BatchItem} prevents a live config mutation from changing the ownership
 * protocol of work already in the queue.
 */
enum DeliveryMode {
    BATCH_ENQUEUE,
    ROUTE_DECISION;

    static DeliveryMode from(FlexlbConfig config) {
        if (config == null) {
            throw new IllegalStateException("request dispatcher configuration is unavailable");
        }
        return config.isBatchDispatch()
                ? BATCH_ENQUEUE
                : ROUTE_DECISION;
    }
}

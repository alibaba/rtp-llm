package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.BatchItem;

/**
 * Owns one inflight registration until the prefill queue accepts the item.
 *
 * <p>The queue handoff is the transaction boundary: before it, closing the
 * guard rolls the registration back; after it, every terminal path belongs to
 * the normal batch-scheduler lifecycle and this guard becomes a no-op.
 */
final class InflightRegistration implements AutoCloseable {

    private final InflightRegistrar registrar;
    private final BatchItem item;
    private boolean rollbackRequired = true;

    private InflightRegistration(InflightRegistrar registrar, BatchItem item) {
        this.registrar = registrar;
        this.item = item;
    }

    static InflightRegistration tryRegister(InflightRegistrar registrar, BatchItem item) {
        return registrar.registerInflight(item)
                ? new InflightRegistration(registrar, item) : null;
    }

    void handoffToQueue() {
        rollbackRequired = false;
    }

    @Override
    public void close() {
        if (rollbackRequired) {
            rollbackRequired = false;
            registrar.unregisterInflight(item);
        }
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.enums.ScheduleModeEnum;

/**
 * Immutable delivery choice captured when a request enters the priority scheduler.
 *
 * <p>Scheduling decides <em>when</em> a group is ready. Delivery decides whether
 * the master sends that group through {@code EnqueueBatch}, or publishes each
 * route decision so the frontend can send the request. Keeping this choice on
 * {@link BatchItem} prevents a live configuration update from changing the
 * ownership protocol of an already admitted request.
 */
enum DeliveryMode {
    BATCH_ENQUEUE,
    ROUTE_DECISION;

    static DeliveryMode from(BalanceContext context) {
        if (context != null && context.getScheduleMode() == ScheduleModeEnum.QUEUE) {
            return ROUTE_DECISION;
        }
        return BATCH_ENQUEUE;
    }
}

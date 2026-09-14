package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.projection.WorkSnapshot;

/** One delivery attempt bound to an exact request; late callbacks cannot select another generation. */
public final class DeliveryClaim {
    final RequestSlot slot;
    final ScheduledRequest item;
    final DeliveryClaimKind kind;
    final long correlationId;
    boolean completed; // guarded by slot

    DeliveryClaim(RequestSlot slot, ScheduledRequest item, DeliveryClaimKind kind, long correlationId) {
        this.slot = slot; this.item = item; this.kind = kind; this.correlationId = correlationId;
    }
    public ScheduledRequest item() { return item; }
    public void begin(WorkSnapshot work, long unstartedMs) { slot.onDeliveryStarted(this, work, unstartedMs); }
    public void publishRoute(WorkSnapshot work, long unstartedMs) {
        if (kind != DeliveryClaimKind.ROUTE_DECISION) {
            throw new IllegalArgumentException("route delivery requires an exact route claim");
        }
        begin(work, unstartedMs);
        complete(DeliveryResult.delivered());
    }
    public void complete(DeliveryResult result) { slot.onDeliveryResult(this, result); }
}

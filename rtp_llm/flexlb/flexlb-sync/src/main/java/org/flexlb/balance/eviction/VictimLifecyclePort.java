package org.flexlb.balance.eviction;

import org.flexlb.balance.delivery.DeliveryItem;

/** Canonical terminal boundary for victims removed before Engine ownership. */
public interface VictimLifecyclePort {

    void finishYielded(DeliveryItem victim, String detail);

    void finishYieldedReservation(
            long requestId, long reservationToken, String detail);
}

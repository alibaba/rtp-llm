package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryItem;

import java.util.List;

/** Scheduler callbacks for facts already committed by an endpoint. */
public interface EndpointEventSink {

    void onStatusReduced(EndpointStatusReduction reduction);

    void onPrefillGenerationRetired(
            PrefillEndpoint endpoint,
            List<DeliveryItem> ownedItems);

    void onDecodeGenerationRetired(
            DecodeEndpoint endpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations);
}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/** Scheduler callbacks for facts already committed by an endpoint. */
public interface EndpointEventSink {

    void onStatusReduced(EndpointStatusReduction reduction);

    void onPrefillGenerationRetired(
            PrefillEndpoint endpoint,
            List<ScheduledRequest> ownedItems);

    void onDecodeGenerationRetired(
            DecodeEndpoint endpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations);

    void onQueuedItemExpired(ScheduledRequest exactItem);

    void onQueueOfferFailure(ScheduledRequest exactItem, Throwable cause);

    void onPreparedDeliveryFailure(ScheduledRequest exactItem, Throwable cause);
}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.WorkerStatus;

import java.util.List;

/** Scheduler callbacks for facts already committed by an endpoint. */
public interface EndpointEventSink {

    void onStatusReduced(EndpointStatusReduction reduction);

    /**
     * An engine-side observation has been (re)published for one prefill
     * endpoint, outside any worker-status lock. The default no-op keeps
     * existing sinks source-compatible; capacity consumers (the NAVI L2
     * feasible-domain gating) override it to receive an O(1) slot-free edge
     * signal — the observation itself is read-only here and must never be
     * traversed per task on this path.
     */
    default void onEngineObservationPublished(
            PrefillEndpoint endpoint,
            WorkerStatus.EngineObservation observation) {
    }

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

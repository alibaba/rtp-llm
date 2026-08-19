package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
/** Owns a Decode reservation until a prefill queue accepts the request. */
final class DecodeReservationOwnership implements AutoCloseable {

    private final DecodeEndpoint endpoint;
    private final long requestId;
    private boolean releaseRequired;

    private DecodeReservationOwnership(DecodeEndpoint endpoint,
                                       long requestId,
                                       boolean releaseRequired) {
        this.endpoint = endpoint;
        this.requestId = requestId;
        this.releaseRequired = releaseRequired;
    }

    /**
     * Own the current admission transaction's reservation until the request
     * is handed to a Prefill queue. The admission claim prevents request-id
     * reuse, so rollback may release the exact reservation.
     */
    static DecodeReservationOwnership own(
            DecodeEndpoint endpoint, long requestId) {
        return new DecodeReservationOwnership(endpoint, requestId, endpoint != null);
    }

    void handoffToQueue() {
        releaseRequired = false;
    }

    @Override
    public void close() {
        if (releaseRequired) {
            releaseRequired = false;
            endpoint.release(requestId);
        }
    }
}

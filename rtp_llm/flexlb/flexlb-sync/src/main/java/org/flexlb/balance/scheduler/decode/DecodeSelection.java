package org.flexlb.balance.scheduler.decode;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.dao.loadbalance.ServerStatus;

/**
 * The Decode worker chosen for one request.
 * status is the Decode address returned to the client; reservation is null until capacity is reserved.
 */
public record DecodeSelection(ServerStatus status,
                              DecodeEndpoint endpoint,
                              DecodeEndpoint.ReservationHandle reservation) {

    /**
     * Builds the Decode selection for one request from its route fields.
     * When the topology has no Decode, the route carries no Decode status, endpoint or reservation,
     * and this returns null.
     */
    public static DecodeSelection of(ServerStatus status,
                                     DecodeEndpoint endpoint,
                                     DecodeEndpoint.ReservationHandle reservation) {
        if (status == null && endpoint == null && reservation == null) {
            return null;
        }
        return new DecodeSelection(status, endpoint, reservation);
    }
}

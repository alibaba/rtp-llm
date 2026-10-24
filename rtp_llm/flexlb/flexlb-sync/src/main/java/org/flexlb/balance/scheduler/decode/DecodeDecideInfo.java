package org.flexlb.balance.scheduler.decode;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.Objects;

/**
 * Decode decision of one request: the requirement and the selected worker.
 * selection is null when the topology has no Decode; otherwise it holds the worker, and its reservation once
 * capacity is reserved.
 */
public record DecodeDecideInfo(DecodeRequirement requirement, DecodeSelection selection) {
    public DecodeDecideInfo {
        if (reservationOf(selection) != null
                && !Objects.equals(selection.reservation().requestId(), requirement.requestId())) {
            throw new IllegalArgumentException("Decode reservation belongs to another request");
        }
    }

    public static DecodeDecideInfo capture(BalanceContext context) {
        return new DecodeDecideInfo(DecodeRequirement.capture(context), null);
    }

    /**
     * A null selection keeps the current state.
     */
    public DecodeDecideInfo withSelection(DecodeSelection newSelection) {
        return newSelection == null ? this : new DecodeDecideInfo(requirement, newSelection);
    }

    /**
     * Absence is a topology choice; a partial selection is still an error.
     */
    public boolean isAbsent() {
        return selection == null;
    }

    public ServerStatus status() {
        return selection == null ? null : selection.status();
    }

    public DecodeEndpoint endpoint() {
        return selection == null ? null : selection.endpoint();
    }

    public DecodeEndpoint.ReservationHandle reservation() {
        return reservationOf(selection);
    }

    private static DecodeEndpoint.ReservationHandle reservationOf(DecodeSelection selection) {
        return selection == null ? null : selection.reservation();
    }
}

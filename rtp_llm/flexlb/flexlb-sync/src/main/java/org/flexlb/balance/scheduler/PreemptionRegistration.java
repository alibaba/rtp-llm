package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;

import java.util.Objects;

/** Exact victim/attempt identity. Resolution execution belongs to Runtime. */
public record PreemptionRegistration(
        RequestContext owner,
        long attemptToken,
        DecodeResources.ReservationHandle reservation,
        CancelTarget cancelTarget)
        implements ContextResult.GrantValue {
    /**
     * Intermediate Scheduler progress after one cancel reply was interpreted.
     *
     * <p>None of these values means that the request is closed or that the Decode ledger was
     * released. Those decisions require the matching Context and endpoint evidence.
     */
    public enum Reply {
        /** The reply was accepted as progress, but request and ledger settlement are pending. */
        WAITING,
        /** The remote worker did not own this work; this is not completion proof. */
        NOT_FOUND,
        /** The reply could not advance the matching attempt; this is not completion proof. */
        UNKNOWN
    }

    public PreemptionRegistration {
        Objects.requireNonNull(owner);
        Objects.requireNonNull(reservation);
        Objects.requireNonNull(cancelTarget);
        if (!cancelTarget.isRoutable()) {
            throw new IllegalArgumentException("victim has no routable Cancel target");
        }
    }
}

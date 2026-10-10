package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import java.util.Objects;

/** Exact victim/attempt identity. Resolution execution belongs to Runtime. */
public record PreemptionRegistration(RequestContext owner, long attemptToken,
                                     DecodeResources.ReservationHandle reservation,
                                     CancelTarget cancelTarget) implements ContextResult.GrantValue {
    public PreemptionRegistration {
        Objects.requireNonNull(owner); Objects.requireNonNull(reservation); Objects.requireNonNull(cancelTarget);
        if (!cancelTarget.isRoutable()) { throw new IllegalArgumentException("victim has no routable Cancel target"); }
    }
}

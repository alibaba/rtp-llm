package org.flexlb.balance.preemption;

/** Canonical lifecycle boundary for the Engine-Cancel preemption protocol. */
public interface PreemptionLifecyclePort {

    CancelTarget resolveCancelTarget(
            long requestId,
            long reservationToken);

    PreemptionClaim claimForPreemption(
            long requestId,
            long reservationToken,
            long attemptToken,
            String detail);

    boolean releasePreemptionClaim(PreemptionClaim claim);

    boolean markPreemptionCancelInFlight(PreemptionClaim claim);

    boolean markPreemptionCancelAccepted(PreemptionClaim claim);

    boolean markPreemptionNotFound(PreemptionClaim claim);

    boolean markPreemptionUnknown(PreemptionClaim claim);

    boolean finishTombstoned(
            PreemptionClaim claim,
            String detail);
}

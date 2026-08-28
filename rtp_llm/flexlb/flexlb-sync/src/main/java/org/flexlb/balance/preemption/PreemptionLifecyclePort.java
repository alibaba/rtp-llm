package org.flexlb.balance.preemption;

import java.util.Optional;

/** Canonical lifecycle boundary for the Engine-Cancel preemption protocol. */
public interface PreemptionLifecyclePort {

    /** Finite protocol updates accepted for one exact claim. */
    sealed interface Update permits Step, Tombstoned {
    }

    /** Updates which carry no additional terminal evidence. */
    enum Step implements Update {
        RELEASE,
        CANCEL_STARTED,
        CANCEL_ACCEPTED,
        CANCEL_NOT_FOUND,
        CANCEL_UNKNOWN
    }

    /** Authoritative Engine proof that the victim can no longer reappear. */
    record Tombstoned(String detail) implements Update {
    }

    /** Find the routable Prefill owner for one exact Decode reservation. */
    Optional<CancelTarget> findCancelTarget(
            long requestId,
            long reservationToken);

    /** Try to claim one exact victim generation for this attempt. */
    Optional<PreemptionClaim> tryClaim(
            long requestId,
            long reservationToken,
            long attemptToken,
            String detail);

    /**
     * Reduce one protocol update against the exact claim.
     *
     * @return {@code false} when the claim is stale or the update is illegal
     *         for its current state
     */
    boolean tryApplyUpdate(PreemptionClaim claim, Update update);
}

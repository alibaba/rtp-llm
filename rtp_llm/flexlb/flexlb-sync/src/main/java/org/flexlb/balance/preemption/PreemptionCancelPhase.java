package org.flexlb.balance.preemption;

/**
 * Canonical progress of one exact victim through the Engine-Cancel protocol.
 *
 * <p>This type deliberately excludes owner-specific terminal facts. A
 * scheduler registration may be settled, while an endpoint claim may transfer
 * its accounting to an Engine fence; neither fact is a Cancel protocol phase.
 */
public enum PreemptionCancelPhase {
    CLAIMED,
    CANCEL_IN_FLIGHT,
    CANCEL_REQUESTED,
    NOT_FOUND_STALE,
    CANCEL_UNKNOWN;

    public boolean canTransitionTo(PreemptionCancelPhase next) {
        if (next == null) {
            return false;
        }
        return switch (next) {
            case CLAIMED -> false;
            case CANCEL_IN_FLIGHT -> this == CLAIMED;
            case CANCEL_REQUESTED, NOT_FOUND_STALE -> this == CANCEL_IN_FLIGHT;
            case CANCEL_UNKNOWN -> this == CANCEL_IN_FLIGHT
                    || this == CANCEL_REQUESTED;
        };
    }

    public boolean isLocallyReleasable() {
        return this == CLAIMED || this == CANCEL_IN_FLIGHT;
    }

    public boolean isFenceTransferable() {
        return this == CLAIMED
                || this == CANCEL_IN_FLIGHT
                || this == NOT_FOUND_STALE;
    }

    public boolean acceptsTombstone() {
        return this == CANCEL_IN_FLIGHT
                || this == NOT_FOUND_STALE
                || this == CANCEL_UNKNOWN;
    }

    public boolean acceptsPriorityTerminal() {
        return this == CANCEL_IN_FLIGHT
                || this == CANCEL_REQUESTED
                || this == NOT_FOUND_STALE
                || this == CANCEL_UNKNOWN;
    }

    public boolean requiresOrdinaryReconciliation() {
        return this == NOT_FOUND_STALE || this == CANCEL_UNKNOWN;
    }
}

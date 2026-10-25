package org.flexlb.balance.endpoint.prefill;

/**
 * Exact request or batch ownership passed through admission and callback.
 */
public abstract class Reservation implements AutoCloseable {
    final PrefillState owner;
    /* guarded by PrefillState.lock */ LeaseState state = LeaseState.OPEN;
    final RequestEntry originalOwner;

    Reservation(PrefillState owner, RequestEntry originalOwner) {
        this.owner = owner;
        this.originalOwner = originalOwner;
    }

    /**
     * Roll back an OPEN lease. Committed capacity stays Registry-owned.
     */
    @Override
    public final void close() {
        owner.releaseOpenLease(this);
    }
}

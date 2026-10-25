package org.flexlb.balance.endpoint.prefill;

/**
 * Exact request or batch ownership passed through admission and callback.
 */
public abstract class Reservation implements AutoCloseable {
    public final PrefillState owner;
    /* guarded by PrefillState.lock */ public LeaseState state = LeaseState.OPEN;
    public final RequestEntry originalOwner;

    public Reservation(PrefillState owner, RequestEntry originalOwner) {
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

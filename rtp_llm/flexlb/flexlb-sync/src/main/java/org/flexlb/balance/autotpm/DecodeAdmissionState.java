package org.flexlb.balance.autotpm;

/**
 * Master-side lifecycle state of a decode reservation.
 *
 * <p>Models the three-phase progression of a request from the Master's
 * perspective once it has been routed to a decode endpoint:
 * <ol>
 *   <li>{@link #RESERVED_NOT_ACCEPTED} — Master has reserved decode capacity
 *       (slots + KV), but the Engine has not yet acknowledged the request.</li>
 *   <li>{@link #ACCEPTED_NOT_RUNNING} — Engine has acknowledged the request
 *       (TaskPhase = KV_ALLOCATED), but has not started generating yet.</li>
 *   <li>{@link #RUNNING} — Engine is actively decoding (TaskPhase = RUNNING).</li>
 * </ol>
 *
 * <h2>MVP eviction rule</h2>
 * Only {@link #RESERVED_NOT_ACCEPTED} and {@link #ACCEPTED_NOT_RUNNING} are
 * eligible for eviction. {@link #RUNNING} is <b>never</b> evicted in MVP.
 */
public enum DecodeAdmissionState {
    /** Master reserved decode capacity, Engine has not yet acknowledged. Eligible for eviction. */
    RESERVED_NOT_ACCEPTED,
    /** Engine acknowledged the request but has not started running. Eligible for eviction. */
    ACCEPTED_NOT_RUNNING,
    /** Engine is actively decoding the request. NEVER evicted in MVP. */
    RUNNING;

    /**
     * @return {@code true} if this state is eligible for priority eviction
     *         (i.e. not {@link #RUNNING}).
     */
    public boolean isEvictable() {
        return this != RUNNING;
    }

    /**
     * Map to the {@link PriorityCostFunction.VictimStage} used for cost computation.
     *
     * @return the corresponding victim stage
     */
    public PriorityCostFunction.VictimStage toVictimStage() {
        switch (this) {
            case RESERVED_NOT_ACCEPTED:
                return PriorityCostFunction.VictimStage.NOT_ACCEPTED;
            case ACCEPTED_NOT_RUNNING:
                return PriorityCostFunction.VictimStage.ACCEPTED_NOT_RUNNING;
            case RUNNING:
                return PriorityCostFunction.VictimStage.RUNNING;
            default:
                return PriorityCostFunction.VictimStage.NOT_ACCEPTED;
        }
    }
}

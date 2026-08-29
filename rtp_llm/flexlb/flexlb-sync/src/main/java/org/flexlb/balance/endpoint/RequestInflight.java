package org.flexlb.balance.endpoint;

import org.flexlb.balance.execution.TtlEvictor;
import org.flexlb.enums.DecodeTaskPhase;

/**
 * Tracks a single inflight decode request's KV reservation (Auto-TPM decode
 * admission view, design doc 10.1).
 *
 * <p>Phase note: entries are created as {@link DecodeTaskPhase#ENGINE_MAY_HAVE_SEEN}.
 * The endpoint's queued ownership bit projects them as
 * {@link DecodeTaskPhase#MASTER_QUEUED_NOT_DISPATCHED} in planning snapshots.
 * Once the engine confirms a request
 * (KV_ALLOCATED / RUNNING), {@code DecodeEndpoint.calibrate} removes the entry
 * and counts it in {@code confirmedRunningCount} instead — the accepted and
 * running layers are merged into a single confirmed count because the current
 * WorkerStatus report cannot reliably distinguish them. The three-phase enum
 * is kept for the Phase 5 accepted/running preemption interface.
 *
 * <p><b>Stage-2 L7 retirement (plan section 6):</b> this type was a record;
 * it is now a plain class so the queued-ownership sub-state — previously the
 * endpoint's separate layer-7 set — can live on the entry itself as the
 * mutable {@code masterQueued} flag flipped in place (the "L1 sub-state
 * projection" landing). In-place mutation (instead of record-instance
 * replacement) preserves instance identity for the identity-based
 * engine-lifecycle reservation set (L2) and for the dispatch-permit
 * reservation reference, and keeps the queued transition off the inflight
 * map. The immutable components keep their record-style accessor names; no
 * code relied on the generated value equals/hashCode.
 *
 * <p>Immutable fields carry the frozen reservation numerics:
 * {@code kvTokens} hard KV demand — the prompt's seqLen, used for
 * hard-capacity filtering (ensures the prompt itself fits);
 * {@code expectedKvTokens} conservative KV estimate — seqLen + maxNewTokens,
 * used for scoring / load balancing to account for generation-phase KV
 * growth; {@code createdAtMs} epoch-millis when this entry was created;
 * {@code priority} Auto-TPM normalized priority (30/40/50/60/70), 0 = no
 * priority (task40) — never evictable; {@code phase} shadow admission
 * phase; {@code reservationToken} endpoint-local monotonic identity for
 * this exact request reservation, zero is reserved for detached test /
 * snapshot values that were not admitted by an endpoint.
 */
public final class RequestInflight implements TtlEvictor.TtlTracked {

    private final long kvTokens;
    private final long expectedKvTokens;
    private final long createdAtMs;
    private final int priority;
    private final DecodeTaskPhase phase;
    private final long reservationToken;

    /**
     * Stage-2 L7 retirement: queued-ownership sub-state — the request sits
     * in a prefill queue, committed by the scheduler but not yet dispatched
     * to the engine. The entry flag is the incoming authority over the old
     * layer-7 set (which remains as the transitional dual-write mirror until
     * the harness retargets and the set storage is deleted).
     *
     * <p>Mutated only inside the endpoint admission lock; read lock-free by
     * the audit capture (weakly consistent — confirm-window territory).
     */
    private volatile boolean masterQueued;

    public RequestInflight(
            long kvTokens,
            long expectedKvTokens,
            long createdAtMs,
            int priority,
            DecodeTaskPhase phase,
            long reservationToken) {
        if (reservationToken < 0L) {
            throw new IllegalArgumentException(
                    "reservationToken must be non-negative");
        }
        this.kvTokens = kvTokens;
        this.expectedKvTokens = expectedKvTokens;
        this.createdAtMs = createdAtMs;
        this.priority = priority;
        this.phase = phase;
        this.reservationToken = reservationToken;
    }

    RequestInflight(
            long kvTokens,
            long expectedKvTokens,
            int priority,
            long reservationToken) {
        this(kvTokens, expectedKvTokens, System.currentTimeMillis(),
                priority, DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                reservationToken);
    }

    public long kvTokens() {
        return kvTokens;
    }

    public long expectedKvTokens() {
        return expectedKvTokens;
    }

    @Override
    public long createdAtMs() {
        return createdAtMs;
    }

    public int priority() {
        return priority;
    }

    public DecodeTaskPhase phase() {
        return phase;
    }

    public long reservationToken() {
        return reservationToken;
    }

    /** Whether this reservation is currently in the master queued phase. */
    public boolean masterQueued() {
        return masterQueued;
    }

    /**
     * Flip the queued sub-state on.  Returns false when it was already on
     * (idempotent no-op mirroring the old set-add semantics).  Production
     * callers hold the endpoint admission lock; public so reconciliation
     * tests can construct the sub-state (the stage-2 audit projection
     * reads it lock-free).
     */
    public boolean enterMasterQueued() {
        if (masterQueued) {
            return false;
        }
        masterQueued = true;
        return true;
    }

    /**
     * Flip the queued sub-state off.  Returns false when it was already off
     * (idempotent no-op mirroring the old set-remove semantics).  Production
     * callers hold the endpoint admission lock; public for symmetric test
     * construction.
     */
    public boolean leaveMasterQueued() {
        if (!masterQueued) {
            return false;
        }
        masterQueued = false;
        return true;
    }

    /**
     * Priority recorded when the caller carries no Auto-TPM priority: the
     * NO_PRIORITY sentinel (0) — such entries never participate in
     * priority mechanisms and are never selected as eviction victims.
     */
    static final int DEFAULT_PRIORITY = 0;

    /**
     * KV tokens returned to the pool when this entry is released: the hard
     * reservation ({@code kvTokens}) that was subtracted from
     * {@code realKvAvailable} (design doc 10.2).
     */
    public long releasableKvTokens() {
        return kvTokens;
    }
}

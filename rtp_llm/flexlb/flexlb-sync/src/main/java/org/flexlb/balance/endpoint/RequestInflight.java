package org.flexlb.balance.endpoint;

import org.flexlb.balance.execution.TtlEvictor;
import org.flexlb.enums.DecodeTaskPhase;

/**
 * One Decode request's endpoint-local KV reservation.
 *
 * <p>Phase note: entries are created as {@link DecodeTaskPhase#ENGINE_MAY_HAVE_SEEN}.
 * The endpoint's queued ownership bit projects them as
 * {@link DecodeTaskPhase#MASTER_QUEUED_NOT_DISPATCHED} in planning snapshots.
 * Once the engine confirms a request (KV_ALLOCATED / RUNNING), status
 * reconciliation removes this reservation and transfers it to the endpoint's
 * confirmed Engine-owned total. The layered registry still distinguishes
 * ACCEPTED_NOT_RUNNING from RUNNING; the aggregate is used only for capacity
 * accounting. The phase remains on detached snapshots used by priority
 * preemption planning.
 *
 * @param kvTokens         hard KV demand — the prompt's seqLen, used for
 *                         hard-capacity filtering (ensures the prompt itself fits)
 * @param expectedKvTokens conservative KV estimate — seqLen + maxNewTokens,
 *                         used for scoring / load balancing to account for
 *                         generation-phase KV growth
 * @param createdAtMs      epoch-millis when this entry was created
 * @param priority         Auto-TPM normalized priority (30/40/50/60/70);
 *                         0 = no priority (task40) — never evictable
 * @param phase            shadow admission phase
 * @param reservationToken endpoint-local monotonic identity for this exact
 *                         request reservation; zero is reserved for detached
 *                         test/snapshot values that were not admitted by an
 *                         endpoint
 */
public final class RequestInflight implements TtlEvictor.TtlTracked {
    private final long kvTokens;
    private final long expectedKvTokens;
    private final long createdAtMs;
    private final int priority;
    private final DecodeTaskPhase phase;
    private final long reservationToken;
    /** Guarded by the owning DecodeEndpoint admission lock. */
    private volatile boolean queued;
    /** Exact pre-delivery slot owner; guarded by the Decode admission lock. */
    private volatile DecodeEndpoint.EngineDispatchPermit dispatchPermit;

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

    /**
     * Priority recorded when the caller carries no Auto-TPM priority: the
     * NO_PRIORITY sentinel (0) — such entries never participate in
     * priority mechanisms and are never selected as eviction victims.
     */
    static final int DEFAULT_PRIORITY = 0;

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

    boolean queued() {
        return queued;
    }

    boolean markQueued() {
        if (queued) {
            return false;
        }
        queued = true;
        return true;
    }

    boolean clearQueued() {
        if (!queued) {
            return false;
        }
        queued = false;
        return true;
    }

    DecodeEndpoint.EngineDispatchPermit dispatchPermit() {
        return dispatchPermit;
    }

    void installDispatchPermit(DecodeEndpoint.EngineDispatchPermit permit) {
        if (dispatchPermit != null) {
            throw new IllegalStateException(
                    "Decode reservation already owns a dispatch permit");
        }
        dispatchPermit = java.util.Objects.requireNonNull(permit, "permit");
    }

    DecodeEndpoint.EngineDispatchPermit clearDispatchPermit() {
        DecodeEndpoint.EngineDispatchPermit current = dispatchPermit;
        dispatchPermit = null;
        return current;
    }

    /**
     * KV tokens returned to the pool when this entry is released: the hard
     * reservation ({@code kvTokens}) that was subtracted from
     * {@code realKvAvailable} (design doc 10.2).
     */
    public long releasableKvTokens() {
        return kvTokens;
    }
}

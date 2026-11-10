package org.flexlb.balance.endpoint;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeState.DispatchLease;
import org.flexlb.enums.DecodeTaskPhase;

/**
 * 保存一条 Decode 请求的预留、引擎状态和抢占关联；由 DecodeState 在节点锁内更新并清理。
 */
@Getter
@Setter
public final class DecodeRequestState {

    static final int DEFAULT_PRIORITY = 0;

    private volatile long publishedAtMs; // 0 表示尚未发布，不检查状态更新超时

    long kvTokens;
    /**
     * Before WorkerStatus confirmation: input tokens plus max_new_tokens.
     * After confirmation: actual KV tokens reported by the engine.
     */
    long kvBudgetTokens;
    final long createdAtMs;
    int priority;
    final long reservationToken;
    boolean priorityKnown = true;
    /**
     * Request phase used for capacity accounting; null excludes the request from capacity.
     */
    volatile DecodeTaskPhase phase = DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
    long lastSeenAtMs;
    boolean queued;
    /**
     * True after dispatch crosses the Master rollback boundary.
     */
    boolean engineLifecycleOwned;
    DispatchLease dispatchPermit;
    /**
     * Incoming instruction owner until dispatch crosses the rollback boundary.
     */
    volatile long returnedPreemptionToken;
    /**
     * Current protocol owners for this exact request generation.
     */
    public PreemptionClaim preemptionClaim;
    /**
     * Non-zero while stale WorkerStatus must not resurrect this request id.
     */
    long settledAtMs;

    public DecodeRequestState(long kvTokens, long kvBudgetTokens, int priority, long reservationToken) {
        if (reservationToken < 0L) {
            throw new IllegalArgumentException("reservationToken must be non-negative");
        }
        this.kvTokens = kvTokens;
        this.kvBudgetTokens = kvBudgetTokens;
        this.createdAtMs = System.currentTimeMillis();
        this.priority = priority;
        this.reservationToken = reservationToken;
    }

    long kvTokens() {
        return kvTokens;
    }

    long kvBudgetTokens() {
        return kvBudgetTokens;
    }

    CapacityRelease capacityRelease() {
        return new CapacityRelease(1L, kvTokens, kvBudgetTokens);
    }

    long createdAtMs() {
        return createdAtMs;
    }

    int priority() {
        return priority;
    }

    long reservationToken() {
        return reservationToken;
    }

    boolean queued() {
        return queued;
    }

    boolean countsTowardCapacity() {
        return phase != null;
    }

    boolean confirmed() {
        return phase != null && phase != DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
    }

    DecodeTaskPhase phase() {
        return phase;
    }

    boolean priorityKnown() {
        return priorityKnown;
    }

    long lastSeenAtMs() {
        return lastSeenAtMs;
    }

    void confirm(long engineKvTokens, DecodeTaskPhase phase, long observedAtMs) {
        kvTokens = Math.max(0L, engineKvTokens);
        kvBudgetTokens = kvTokens;
        this.phase = java.util.Objects.requireNonNull(phase, "phase");
        lastSeenAtMs = observedAtMs;
        queued = false;
        engineLifecycleOwned = false;
        dispatchPermit = null;
    }

    static DecodeRequestState untrackedConfirmed(long engineKvTokens, DecodeTaskPhase phase, long observedAtMs) {
        DecodeRequestState state = new DecodeRequestState(0L, 0L, DEFAULT_PRIORITY, 0L);
        state.priorityKnown = false;
        state.confirm(engineKvTokens, phase, observedAtMs);
        return state;
    }

    void refresh(DecodeTaskPhase phase, long observedAtMs) {
        if (!confirmed()) {
            throw new IllegalStateException("cannot refresh an unconfirmed Decode request");
        }
        this.phase = java.util.Objects.requireNonNull(phase, "phase");
        lastSeenAtMs = observedAtMs;
    }

    long lastSeenMs() {
        return lastSeenAtMs;
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

    DispatchLease dispatchPermit() {
        return dispatchPermit;
    }

    void installDispatchPermit(DispatchLease permit) {
        if (dispatchPermit != null) {
            throw new IllegalStateException("Decode reservation already owns a dispatch permit");
        }
        dispatchPermit = java.util.Objects.requireNonNull(permit, "permit");
    }

    DispatchLease clearDispatchPermit() {
        DispatchLease current = dispatchPermit;
        dispatchPermit = null;
        return current;
    }

    long releasableKvTokens() {
        return kvTokens;
    }

    boolean hasProtocolOwner() {
        return preemptionClaim != null;
    }

    /**
     * Excludes this request from capacity accounting while preserving its ID and reservation token.
     * Callers update the aggregate capacity counters under lock.
     */
    void stopCountingTowardCapacity() {
        phase = null;
    }
}

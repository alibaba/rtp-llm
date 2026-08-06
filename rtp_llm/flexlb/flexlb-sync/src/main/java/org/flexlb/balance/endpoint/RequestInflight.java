package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.enums.DecodeTaskPhase;

/**
 * Tracks a single inflight decode request's KV reservation (Auto-TPM decode
 * admission view, design doc 10.1).
 *
 * <p>Phase note: entries are created as {@link DecodeTaskPhase#RESERVED_NOT_ACCEPTED}
 * (Master shadow reservation). Once the engine confirms a request
 * (KV_ALLOCATED / RUNNING), {@code DecodeEndpoint.calibrate} removes the entry
 * and counts it in {@code confirmedRunningCount} instead — the accepted and
 * running layers are merged into a single confirmed count because the current
 * WorkerStatus report cannot reliably distinguish them. The three-phase enum
 * is kept for the Phase 5 accepted/running preemption interface.
 *
 * @param kvTokens         hard KV demand — the prompt's seqLen, used for
 *                         hard-capacity filtering (ensures the prompt itself fits)
 * @param expectedKvTokens conservative KV estimate — seqLen + maxNewTokens,
 *                         used for scoring / load balancing to account for
 *                         generation-phase KV growth
 * @param createdAtMs      epoch-millis when this entry was created
 * @param priority         Auto-TPM normalized priority (30/40/50/60/70);
 *                         0 = no priority (task40) — never evictable
 * @param deadlineMs       Auto-TPM admission deadline (epoch ms); 0 = unset
 * @param phase            decode admission phase; always
 *                         {@code RESERVED_NOT_ACCEPTED} in Phase 4
 */
public record RequestInflight(
        long kvTokens,
        long expectedKvTokens,
        long createdAtMs,
        int priority,
        long deadlineMs,
        DecodeTaskPhase phase
) implements InflightEvictor.TtlTracked {

    /**
     * Priority recorded when the caller carries no Auto-TPM priority: the
     * NO_PRIORITY sentinel (0, task40) — such entries never participate in
     * priority mechanisms and are never selected as eviction victims.
     */
    static final int DEFAULT_PRIORITY = 0;

    RequestInflight(long kvTokens, long expectedKvTokens) {
        this(kvTokens, expectedKvTokens, DEFAULT_PRIORITY, 0);
    }

    RequestInflight(long kvTokens, long expectedKvTokens, int priority, long deadlineMs) {
        this(kvTokens, expectedKvTokens, System.currentTimeMillis(),
                priority, deadlineMs, DecodeTaskPhase.RESERVED_NOT_ACCEPTED);
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

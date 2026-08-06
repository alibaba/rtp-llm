package org.flexlb.balance.endpoint;

import org.flexlb.enums.TaskPhase;

/**
 * Phase of an engine-accepted task in the second inflight layer
 * ({@code engineWork}), mapped from the engine-reported {@link TaskPhase}.
 *
 * <p>Mapping rules (no new engine-side reporting required):
 * <ul>
 *   <li>PENDING / RECEIVED → {@link #WAITING}</li>
 *   <li>KV_ALLOCATED → {@link #LOADING} for decode (remote KV loading);
 *       prefill has no loading concept, so it maps to {@link #WAITING}</li>
 *   <li>RUNNING → {@link #RUNNING}</li>
 * </ul>
 *
 * <p>Ordinal order (WAITING &lt; LOADING &lt; RUNNING) is meaningful: batch
 * phase aggregation takes the minimum across members (weakest-link rule).
 */
public enum EngineTaskPhase {

    /** Accepted by the engine, queued and not yet making progress. */
    WAITING,

    /** Decode only: remote KV cache loading in progress. */
    LOADING,

    /** Computing. */
    RUNNING;

    /**
     * Map an engine-reported phase for a prefill task. Prefill has no
     * loading stage, so KV_ALLOCATED counts as WAITING.
     * A {@code null} phase (absent in the report) is treated as WAITING.
     */
    public static EngineTaskPhase fromPrefill(TaskPhase phase) {
        return phase == TaskPhase.RUNNING ? RUNNING : WAITING;
    }

    /**
     * Map an engine-reported phase for a decode task.
     * A {@code null} phase (absent in the report) is treated as WAITING.
     */
    public static EngineTaskPhase fromDecode(TaskPhase phase) {
        if (phase == TaskPhase.RUNNING) {
            return RUNNING;
        }
        if (phase == TaskPhase.KV_ALLOCATED) {
            return LOADING;
        }
        return WAITING;
    }

    /** @return the weaker (earlier) of the two phases — batch aggregation rule. */
    public static EngineTaskPhase min(EngineTaskPhase a, EngineTaskPhase b) {
        return a.ordinal() <= b.ordinal() ? a : b;
    }
}

package org.flexlb.enums;

/**
 * Auto-TPM decode admission phase of one request tracked by the Master
 * (design doc 10.1). Ordinal order is the eviction stage order
 * (design doc 11.3): earlier phases are cheaper to evict.
 *
 * <p>A request still in the Master queue can be removed locally. Once its
 * dispatch may have reached Prefill, eviction must use the same Cancel
 * protocol as an accepted or running request.</p>
 */
public enum DecodeTaskPhase {

    /** Still owned exclusively by the Master queue; Engine cannot have seen it. */
    MASTER_QUEUED_NOT_DISPATCHED,

    /** Dispatch started or is in flight; original Prefill must handle Cancel. */
    ENGINE_MAY_HAVE_SEEN,

    /** Engine accepted the request and allocated KV, but has not started it. */
    ACCEPTED_NOT_RUNNING,

    /** Engine is running the request. Eviction requires Cancel and release confirmation. */
    RUNNING;

    /** Whether eviction must go through the original Prefill Cancel owner. */
    public boolean requiresEngineCancel() {
        return this == ENGINE_MAY_HAVE_SEEN
                || this == ACCEPTED_NOT_RUNNING
                || this == RUNNING;
    }

    /** Whether the phase is represented by Engine WorkerStatus. */
    public boolean isEngineConfirmed() {
        return this == ACCEPTED_NOT_RUNNING || this == RUNNING;
    }

    public boolean isMasterQueued() {
        return this == MASTER_QUEUED_NOT_DISPATCHED;
    }
}

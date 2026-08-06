package org.flexlb.enums;

/**
 * Auto-TPM decode admission phase of one request tracked by the Master
 * (design doc 10.1). Ordinal order is the eviction stage order
 * (design doc 11.3): earlier phases are cheaper to evict.
 *
 * <p><b>Reserved-only simplification:</b> Phase 4 only ever creates and
 * evicts {@link #RESERVED_NOT_ACCEPTED} entries (Master shadow reservation,
 * engine not yet confirmed); accepted/running requests are folded into a
 * single confirmed-running counter on the decode endpoint. The other two
 * phases are defined for the Phase 5 accepted/running layered view and
 * preemption interface but are never eviction candidates in this phase.
 */
public enum DecodeTaskPhase {

    /** Master shadow reservation; the engine has not confirmed the request. */
    RESERVED_NOT_ACCEPTED,

    /** Engine accepted (KV allocated) but not yet running. Phase 5 only. */
    ACCEPTED_NOT_RUNNING,

    /** Engine is running the request. Never evicted before Phase 5+. */
    RUNNING
}

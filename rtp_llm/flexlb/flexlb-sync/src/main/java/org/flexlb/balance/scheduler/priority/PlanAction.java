package org.flexlb.balance.scheduler.priority;

/**
 * Atomic step of an Auto-TPM admission plan.
 *
 * <p>MVP (Phase 1) only uses the two incoming-request actions. The eviction
 * actions are defined for later phases (prefill queue eviction, decode
 * reserved-only eviction) and are never produced yet.
 */
public enum PlanAction {

    /** Reserve decode slot/KV for the incoming request (done inside route()). */
    RESERVE_DECODE_FOR_INCOMING,

    /** Offer the incoming request to the target prefill batcher queue. */
    OFFER_PREFILL_FOR_INCOMING,

    /** Phase 2+: evict a lower-priority queued request from a prefill queue. */
    EVICT_PREFILL_QUEUED,

    /** Phase 4: evict a lower-priority reserved-only request from a decode endpoint. */
    EVICT_DECODE_RESERVED
}

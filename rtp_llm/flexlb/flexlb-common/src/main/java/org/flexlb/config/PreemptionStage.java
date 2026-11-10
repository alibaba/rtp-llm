package org.flexlb.config;

/**
 * Request ownership stages eligible for priority resource reclamation.
 */
public enum PreemptionStage {
    PREFILL_QUEUED,
    DECODE_RESERVED,
    DECODE_ENGINE_OWNED
}

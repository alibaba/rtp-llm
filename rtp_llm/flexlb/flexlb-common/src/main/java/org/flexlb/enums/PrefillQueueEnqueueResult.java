package org.flexlb.enums;

/**
 * Result of enqueuing a request at a selected Prefill endpoint.
 */
public enum PrefillQueueEnqueueResult {
    ACCEPTED,
    ENGINE_SUBQUEUE_LIMIT_REACHED,
    REJECTED
}

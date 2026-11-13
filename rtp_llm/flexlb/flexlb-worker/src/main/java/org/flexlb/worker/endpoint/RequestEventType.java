package org.flexlb.worker.endpoint;

/**
 * Identifies Worker execution, endpoint closure and confirmed remote cleanup observations.
 */
public enum RequestEventType {
    ACTIVE,
    ALLOCATED,
    COMPLETED,
    FAILED,
    PRIORITY_CANCELLED,
    ENDPOINT_CLOSED,
    // Cancellation confirms both Prefill and Decode resources were released.
    REMOTE_CLEANED,
    // Cancellation fences Prefill execution; Decode release is still unconfirmed.
    REMOTE_FENCED
}

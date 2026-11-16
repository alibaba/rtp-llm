package org.flexlb.worker.endpoint;

/**
 * Exposes the request identity and queue order used by preemption.
 */
public interface WorkerRequest {
    String getRequestId();

    long getPlacementSequence();
}

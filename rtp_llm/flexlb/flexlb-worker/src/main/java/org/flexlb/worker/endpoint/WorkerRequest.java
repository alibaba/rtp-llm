package org.flexlb.worker.endpoint;

/**
 * Identifies the request that owns a Decode resource wait.
 */
public interface WorkerRequest {
    String getRequestId();
}

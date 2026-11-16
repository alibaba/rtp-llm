package org.flexlb.worker.endpoint;

/**
 * Exposes scheduler retention and shutdown state for endpoint orphan cleanup.
 */
public interface WorkerRequests {

    boolean retainsIdentity(String requestId);

    boolean isClosed();
}

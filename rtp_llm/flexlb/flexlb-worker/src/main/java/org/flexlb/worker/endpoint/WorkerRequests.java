package org.flexlb.worker.endpoint;

/**
 * Looks up current request identities for Worker feedback and orphan cleanup.
 */
public interface WorkerRequests {
    WorkerRequest findActive(String requestId);

    boolean retainsIdentity(String requestId);

    boolean isClosed();
}

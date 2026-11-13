package org.flexlb.worker.endpoint;

/**
 * Exposes the identity, queue order and current route needed by Worker resource records.
 */
public interface WorkerRequest {
    String getRequestId();

    long getPlacementSequence();

    WorkerRequestSnapshot snapshot();

    WorkerEventSink getRequestScheduler();
}

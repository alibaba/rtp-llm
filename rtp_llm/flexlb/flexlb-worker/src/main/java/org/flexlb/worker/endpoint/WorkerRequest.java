package org.flexlb.worker.endpoint;

/**
 * Exposes the identity, queue order and current route needed by Worker resource records.
 */
public interface WorkerRequest {
    String getRequestId();

    DecodeEndpoint.ResourceWait currentResourceWait();

    boolean isResponseDone();

    long getPlacementSequence();

    WorkerRequestSnapshot snapshot();

    WorkerEventSink getRequestScheduler();
}

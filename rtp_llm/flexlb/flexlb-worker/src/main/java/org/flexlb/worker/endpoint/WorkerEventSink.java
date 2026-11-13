package org.flexlb.worker.endpoint;

/**
 * Receives exact Worker request observations after endpoint locks are released.
 */
public interface WorkerEventSink {
    void onWorkerRequestEvent(WorkerRequest request, WorkerRequestEvent event);
}

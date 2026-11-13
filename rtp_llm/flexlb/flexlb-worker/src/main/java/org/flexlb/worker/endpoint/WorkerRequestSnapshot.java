package org.flexlb.worker.endpoint;

/**
 * Provides the current routed request without exposing scheduler lifecycle fields.
 */
public interface WorkerRequestSnapshot {
    WorkerRoute route();
}

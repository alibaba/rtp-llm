package org.flexlb.worker.endpoint;

/**
 * Rolls back an exact request registration before the role's commit or worker-confirmation boundary.
 */
public interface SelectionRollback {
    boolean rollbackSelection(WorkerEndpoint.RequestRegistration registration);
}

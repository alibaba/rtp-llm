package org.flexlb.dao.master;

/**
 * Master-local lifecycle of one discovered worker generation.
 *
 * <p>Service discovery only creates a {@link #PROBING} generation. A worker is
 * schedulable only after a valid status response promotes it to {@link #READY}.
 * Once retirement starts the same {@code WorkerStatus} object can never become
 * ready again; a later recovery is represented by a new object generation.
 */
public enum WorkerLifecycleState {
    PROBING,
    READY,
    RETIRING,
    CLOSED
}

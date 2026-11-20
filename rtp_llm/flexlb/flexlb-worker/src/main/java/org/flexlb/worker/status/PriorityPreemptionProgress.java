package org.flexlb.worker.status;

/**
 * Engine-owned progress of a priority-preemption cancellation.
 */
public enum PriorityPreemptionProgress {
    NONE,
    CANCELING,
    CANCELED
}

package org.flexlb.enums;

/** Engine-owned progress of a priority-preemption cancellation. */
public enum PriorityPreemptionProgress {
    NONE,
    CANCELING,
    CANCELED;

    /** Merge observations monotonically; absent progress is treated as NONE. */
    public static PriorityPreemptionProgress merge(
            PriorityPreemptionProgress left, PriorityPreemptionProgress right) {
        if (left == CANCELED || right == CANCELED) {
            return CANCELED;
        }
        if (left == CANCELING || right == CANCELING) {
            return CANCELING;
        }
        return NONE;
    }
}

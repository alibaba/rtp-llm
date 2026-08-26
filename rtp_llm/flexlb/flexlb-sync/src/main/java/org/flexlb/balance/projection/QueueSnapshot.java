package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;

import java.util.Comparator;
import java.util.List;

/**
 * Immutable scheduling inputs captured for a route-time what-if projection.
 *
 * <p>The active items are already in production queue order. This object is
 * materialized from the canonical ownership registry at one linearization
 * point together with committed work and pending count.
 */
public record QueueSnapshot(
        long capturedAtMs,
        boolean queueScheduling,
        Comparator<GroupPlanner.Item> ordering,
        GroupPlanner.Constraints constraints,
        List<GroupPlanner.Item> activeItems,
        AdmissionBlock admissionBlock) {

    public QueueSnapshot {
        activeItems = List.copyOf(activeItems);
        if (admissionBlock != null) {
            if (activeItems.isEmpty()) {
                throw new IllegalArgumentException(
                        "admission block requires an ACTIVE head");
            }
            GroupPlanner.Item head = activeItems.getFirst();
            if (head.requestId() != admissionBlock.requestId()
                    || head.enqueueSeq() != admissionBlock.enqueueSeq()) {
                throw new IllegalArgumentException(
                        "admission block must identify the exact ACTIVE head");
            }
        }
    }

    /** Exact ACTIVE head whose current capacity rejection parks the worker. */
    public record AdmissionBlock(
            long requestId,
            long enqueueSeq,
            RouteProjection.AdmissionBlockSemantics semantics) {
    }

}

package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Immutable scheduling inputs captured for a route-time what-if projection.
 *
 * <p>The active items are already in production queue order. This object is
 * materialized from the endpoint's PrefillState at one linearization
 * point together with committed work and pending count.
 */
public record QueueSnapshot(
        long capturedAtMs,
        boolean queueScheduling,
        GroupingPolicy grouping,
        Comparator<GroupPlanner.Item> ordering,
        GroupPlanner.Constraints constraints,
        List<GroupPlanner.Item> activeItems,
        AdmissionBlock admissionBlock) {

    public QueueSnapshot {
        if (queueScheduling) { Objects.requireNonNull(grouping, "grouping"); }
        activeItems = List.copyOf(activeItems);
        if (admissionBlock != null) {
            checkArgument(!activeItems.isEmpty(), "admission block requires an ACTIVE head");
            GroupPlanner.Item head = activeItems.getFirst();
            checkArgument(head.requestId() == admissionBlock.requestId()
                    && head.enqueueSeq() == admissionBlock.enqueueSeq(),
                    "admission block must identify the exact ACTIVE head");
        }
    }

    /**
     * Exact ACTIVE head whose current capacity rejection parks the worker.
     * Null semantics mean the wait only limits delivery, not queue publication.
     */
    public record AdmissionBlock(
            long requestId,
            long enqueueSeq,
            RouteProjection.AdmissionBlockSemantics semantics) {
    }

}

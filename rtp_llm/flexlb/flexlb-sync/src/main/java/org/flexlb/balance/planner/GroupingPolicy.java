package org.flexlb.balance.planner;

import static com.google.common.base.Preconditions.checkArgument;

/** Pure grouping of ordered inputs. No clock reads, queue mutation or resource acquisition. */
public enum GroupingPolicy {
    SINGLE,
    FIXED_WINDOW;

    /**
     * The iterable is a frozen, bounded view for live planning and a lazy remaining-prefix view
     * for projection. It must not be copied wholesale. The predictor belongs to this call only.
     */
    public <T extends GroupPlanner.Input> GroupPlanner.Selection<T> select(
            Iterable<T> items, GroupPlanner.Constraints constraints,
            GroupPlanner.PrefixPrediction<T> predictor) {
        checkArgument(this != SINGLE
                || constraints.maxRequests() == 1
                && constraints.collectionWindowMs() == 0L
                && constraints.predictedExecutionBudgetMs() == 0L,
                "SINGLE requires one member and no collection or prediction budget");
        return GroupPlanner.selectWithPrediction(items, constraints, this == SINGLE ? null : predictor);
    }

    /** Null means that no group can dispatch at this clock. */
    public String dispatchReason(GroupPlanner.Selection<?> selection, GroupPlanner.Constraints constraints, long nowMs) {
        if (selection.items().isEmpty()) {
            return null;
        }
        String reason = this == SINGLE ? "single_request" : GroupPlanner.dispatchReason(selection, constraints, nowMs);
        checkArgument(reason != null
                || GroupPlanner.collectionDeadlineMs(selection.windowOpenedAtMs(), constraints.collectionWindowMs()) >= 0L,
                "collection deadline must be non-negative");
        return reason;
    }
}

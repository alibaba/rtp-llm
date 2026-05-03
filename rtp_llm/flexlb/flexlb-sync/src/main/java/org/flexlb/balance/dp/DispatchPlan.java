package org.flexlb.balance.dp;

import java.util.ArrayList;
import java.util.List;

/**
 * Output of {@link DispatchPlanner#plan}: zero or more {@link PrefillBatch}
 * ready for {@code Master.Enqueue}, plus zero or more {@link FailedRequest}
 * whose futures must be completed with a failure response.
 *
 * <p>The two lists are disjoint: every request from the input either lands in
 * exactly one batch or one failure entry, never both.
 */
public record DispatchPlan(List<PrefillBatch> batches,
                           List<FailedRequest> failures) {

    public static DispatchPlan of(List<PrefillBatch> batches) {
        return new DispatchPlan(batches, List.of());
    }

    public static DispatchPlan empty() {
        return new DispatchPlan(List.of(), List.of());
    }

    public static DispatchPlan allFailed(List<QueuedRequest> requests,
                                         org.flexlb.dao.loadbalance.StrategyErrorType reason,
                                         String message) {
        List<FailedRequest> fails = new ArrayList<>(requests.size());
        for (QueuedRequest qr : requests) {
            fails.add(new FailedRequest(qr, reason, message));
        }
        return new DispatchPlan(List.of(), fails);
    }
}

package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.StrategyErrorType;

/**
 * Per-request failure produced by {@link DispatchPlanner} (e.g. no decode worker
 * could be paired in the chosen group). The dispatcher completes the request's
 * future with a failure {@code Response} carrying {@link #reason()} and
 * {@link #message()}.
 */
public record FailedRequest(QueuedRequest request,
                            StrategyErrorType reason,
                            String message) {
}

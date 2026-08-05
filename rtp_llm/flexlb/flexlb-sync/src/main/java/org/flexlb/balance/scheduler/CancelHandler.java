package org.flexlb.balance.scheduler;

import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.CancelReasonPB;

/**
 * Functional interface for cancelling an inflight request via the cancel RPC chain.
 *
 * <p>Implemented by {@link org.flexlb.service.RouteService} so eviction-side
 * callers can be injected via {@code ObjectProvider} without a circular
 * dependency on the scheduler.
 */
@FunctionalInterface
public interface CancelHandler {
    /**
     * Cancel an inflight request with the given reason.
     *
     * @param requestId the request to cancel
     * @param reason    the cancel reason (e.g. {@code CANCEL_REASON_PRIORITY_PREEMPTED})
     */
    void cancel(long requestId, CancelReasonPB reason);
}

package org.flexlb.balance.scheduler;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RequestId;

/**
 * First cause asking the scheduling master to cancel a request.
 */
public enum CancelReason {
    CLIENT_CANCELLED("request cancelled by client"),
    DEADLINE_EXCEEDED("request deadline exceeded"),
    SHUTDOWN("request cancelled during shutdown"),
    PRIORITY_PREEMPTED("preempted by a higher-priority request");

    private final String message;

    CancelReason(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }

    /**
     * Build Engine cancellation without normalizing or rounding the business request ID.
     */
    public EngineRpcService.CancelRequestPB toRequest(String requestId) {
        var reason =
                switch (this) {
                    case CLIENT_CANCELLED ->
                            EngineRpcService.RequestCancelReasonPB
                                    .REQUEST_CANCEL_REASON_CLIENT_CANCELLED;
                    case DEADLINE_EXCEEDED ->
                            EngineRpcService.RequestCancelReasonPB
                                    .REQUEST_CANCEL_REASON_DEADLINE_EXCEEDED;
                    case SHUTDOWN ->
                            EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_SHUTDOWN;
                    case PRIORITY_PREEMPTED ->
                            EngineRpcService.RequestCancelReasonPB
                                    .REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED;
                };
        return EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(RequestId.toEngineRequestId(requestId))
                .setReason(reason)
                .build();
    }
}

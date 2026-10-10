package org.flexlb.balance.eviction;

import io.grpc.Context;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Sends cancellation to the request's original Prefill, which owns the P/D connection.
 * Ordinary cancellation checks the Engine capability before sending; priority
 * preemption keeps compatibility with engines implementing the original Cancel RPC.
 * Neither ACCEPTED nor a transport failure proves resource release. Cleanup RPCs
 * use a detached gRPC context so the caller disconnecting cannot cancel them.
 */
@Slf4j
@Component
public class GrpcEngineCancelChannel implements EngineCancelChannel {

    private final EngineGrpcClient engineGrpcClient;

    public GrpcEngineCancelChannel(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    @Override
    public CompletableFuture<CancelAck> cancel(CancelTarget target,
                                               long requestId,
                                               CancelReason reason,
                                               long timeoutMs) {
        if (target == null || !target.isRoutable()) {
            // No routable endpoint — report the transport-failure branch: the
            // intent never reached the engine, but release is still settled by
            // the WorkerStatus report (iron rule 4).
            log.debug("[request-cancel] cancel has no prefill control owner for request_id={}, not routed",
                    requestId);
            return CompletableFuture.completedFuture(CancelAck.FAILED);
        }

        try {
            EngineRpcService.CancelRequestPB requestPB =
                    EngineRpcService.CancelRequestPB.newBuilder()
                            .setRequestId(requestId)
                            .setReason(switch (reason) {
                                case CLIENT_CANCELLED -> EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_CLIENT_CANCELLED;
                                case DEADLINE_EXCEEDED -> EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_DEADLINE_EXCEEDED;
                                case SHUTDOWN -> EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_SHUTDOWN;
                                case PRIORITY_PREEMPTED -> EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED;
                            })
                            .build();

            // Fork the gRPC Context so that when the
            // caller is a server handler, the server call completing does not
            // cascade-cancel this in-flight outbound RPC.
            Context fork = Context.current().fork();
            Context previous = fork.attach();
            try {
                CompletableFuture<CancelAck> result;
                if (reason == CancelReason.PRIORITY_PREEMPTED) {
                    result = engineGrpcClient.cancelAsync(target.prefillIp(), target.prefillGrpcPort(), requestPB,
                            Math.max(1, timeoutMs)).thenApply(GrpcEngineCancelChannel::mapResponse);
                } else {
                    result = engineGrpcClient.getWorkerStatusAsync(target.prefillIp(), target.prefillGrpcPort(),
                                    EngineRpcService.StatusVersionPB.getDefaultInstance(), Math.max(1, timeoutMs))
                            .thenCompose(status -> {
                                if (!status.getSupportsRequestCleanup()) {
                                    return CompletableFuture.completedFuture(CancelAck.UNSUPPORTED);
                                }
                                // The continuation may run after the caller RPC has ended.
                                Context active = fork.attach();
                                try {
                                    return engineGrpcClient.cancelAsync(target.prefillIp(), target.prefillGrpcPort(),
                                            requestPB, Math.max(1, timeoutMs)).thenApply(GrpcEngineCancelChannel::mapResponse);
                                } finally {
                                    fork.detach(active);
                                }
                            });
                }
                return result
                        .exceptionally(t -> {
                            log.debug(
                                    "[request-cancel] cancel rpc failed for request_id={}: {}",
                                    requestId, t.getMessage());
                            return CancelAck.FAILED;
                        });
            } finally {
                fork.detach(previous);
            }
        } catch (RuntimeException | Error failure) {
            log.debug(
                    "[request-cancel] cancel setup failed for request_id={}: {}",
                    requestId, failure.getMessage());
            return CompletableFuture.failedFuture(failure);
        }
    }

    private static CancelAck mapResponse(EngineRpcService.CancelResponsePB response) {
        return switch (response.getStatus()) {
            case CANCEL_STATUS_ACCEPTED -> CancelAck.ACCEPTED;
            case CANCEL_STATUS_NOT_FOUND -> CancelAck.NOT_FOUND;
            case CANCEL_STATUS_TOMBSTONED -> response.getDecodeCleanupComplete()
                    ? CancelAck.REQUEST_CLEANED : CancelAck.REQUEST_FENCED;
            case CANCEL_STATUS_UNSPECIFIED, UNRECOGNIZED -> CancelAck.FAILED;
        };
    }

}

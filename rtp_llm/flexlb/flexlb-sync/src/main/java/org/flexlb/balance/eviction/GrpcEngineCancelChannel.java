package org.flexlb.balance.eviction;

import io.grpc.Context;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Sends cancellation to Decode, which owns the cancel registry and forwards the
 * terminal intent to the original Prefill. The Prefill address also lets Decode
 * establish a control-only cancellation before inference admission.
 */
@Slf4j
@Component
public class GrpcEngineCancelChannel implements EngineCancelChannel {

    private final EngineGrpcClient engineGrpcClient;

    public GrpcEngineCancelChannel(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    /** Decode implements the control-plane cancellation RPC. */
    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return true;
    }

    @Override
    public CompletableFuture<CancelAck> cancel(
            CancelTarget target, long requestId, long timeoutMs) {
        if (target == null || !target.isRoutable()) {
            // No routable endpoint — report the transport-failure branch: the
            // intent never reached the engine, but release is still settled by
            // the WorkerStatus report (iron rule 4).
            log.debug(
                    "[auto-tpm] cancel has no decode control owner for request_id={}, not routed",
                    requestId);
            return CompletableFuture.completedFuture(CancelAck.FAILED);
        }

        try {
            EngineRpcService.CancelRequestPB requestPB =
                    EngineRpcService.CancelRequestPB.newBuilder()
                            .setRequestId(requestId)
                            .setPrefillAddress(target.prefillAddress())
                            .build();

            // Fire-and-forget contract: fork the gRPC Context so that when the
            // caller is a server handler, the server call completing does not
            // cascade-cancel this in-flight outbound RPC.
            Context fork = Context.current().fork();
            Context previous = fork.attach();
            try {
                return engineGrpcClient
                        .cancelAsync(
                                target.decodeIp(),
                                target.decodeGrpcPort(),
                                requestPB,
                                Math.max(1, timeoutMs))
                        .thenApply(GrpcEngineCancelChannel::mapResponse)
                        .exceptionally(
                                t -> {
                                    log.debug(
                                            "[auto-tpm] cancel rpc failed for request_id={}: {}",
                                            requestId,
                                            t.getMessage());
                                    return CancelAck.FAILED;
                                });
            } finally {
                fork.detach(previous);
            }
        } catch (RuntimeException | Error failure) {
            log.debug(
                    "[auto-tpm] cancel setup failed for request_id={}: {}",
                    requestId,
                    failure.getMessage());
            return CompletableFuture.failedFuture(failure);
        }
    }

    private static CancelAck mapResponse(EngineRpcService.CancelResponsePB response) {
        return switch (response.getStatus()) {
            case CANCEL_STATUS_ACCEPTED -> CancelAck.ACCEPTED;
            case CANCEL_STATUS_NOT_FOUND -> CancelAck.NOT_FOUND;
            case CANCEL_STATUS_TOMBSTONED -> CancelAck.REQUEST_FENCED;
            case CANCEL_STATUS_UNSPECIFIED, UNRECOGNIZED -> CancelAck.FAILED;
        };
    }
}

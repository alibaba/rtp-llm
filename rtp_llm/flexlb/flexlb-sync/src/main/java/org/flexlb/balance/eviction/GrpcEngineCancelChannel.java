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
 * Production {@link EngineCancelChannel} that forwards cancel intents to the
 * victim's original Prefill worker via the engine gRPC
 * {@code RpcService.Cancel} method. The original Prefill is the authoritative
 * producer of typed {@code CANCELED}; Decode does not implement Cancel.
 *
 * <p>Transport behavior is independent from admission policy:
 * {@code preemption.allowedVictimStages} must include
 * {@code DECODE_ENGINE_OWNED} before the planner can use this transport.
 *
 * <p>Contract mirror of {@link EngineCancelChannel}: a cancel is an intent
 * injection only — settlement requires original-Prefill WorkerStatus carrying
 * typed {@code CANCELED+8429}. The engine Cancel RPC distinguishes ACCEPTED from NOT_FOUND;
 * these map to the matching local outcomes, while
 * any transport failure to {@code failed()} — never throws synchronously.
 */
@Slf4j
@Component
public class GrpcEngineCancelChannel implements EngineCancelChannel {

    private final EngineGrpcClient engineGrpcClient;

    public GrpcEngineCancelChannel(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    /**
     * The Decode argument is only the planning capability gate. The actual
     * destination is the original Prefill route carried by {@code target}.
     */
    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return true;
    }

    @Override
    public CompletableFuture<CancelAck> cancel(CancelTarget target,
                                               long requestId,
                                               long timeoutMs) {
        if (target == null || !target.isRoutable()) {
            // No routable endpoint — report the transport-failure branch: the
            // intent never reached the engine, but release is still settled by
            // the WorkerStatus report (iron rule 4).
            log.debug("[auto-tpm] cancel has no prefill control owner for request_id={}, not routed",
                    requestId);
            return CompletableFuture.completedFuture(CancelAck.FAILED);
        }

        try {
            EngineRpcService.CancelRequestPB requestPB =
                    EngineRpcService.CancelRequestPB.newBuilder()
                            .setRequestId(requestId)
                            .build();

            // Fire-and-forget contract: fork the gRPC Context so that when the
            // caller is a server handler, the server call completing does not
            // cascade-cancel this in-flight outbound RPC.
            Context fork = Context.current().fork();
            Context previous = fork.attach();
            try {
                return engineGrpcClient.cancelAsync(
                                target.prefillIp(),
                                target.prefillGrpcPort(),
                                requestPB,
                                Math.max(1, timeoutMs))
                        .thenApply(GrpcEngineCancelChannel::mapResponse)
                        .exceptionally(t -> {
                            log.debug(
                                    "[auto-tpm] cancel rpc failed for request_id={}: {}",
                                    requestId, t.getMessage());
                            return CancelAck.FAILED;
                        });
            } finally {
                fork.detach(previous);
            }
        } catch (RuntimeException | Error failure) {
            log.debug(
                    "[auto-tpm] cancel setup failed for request_id={}: {}",
                    requestId, failure.getMessage());
            return CompletableFuture.failedFuture(failure);
        }
    }

    private static CancelAck mapResponse(EngineRpcService.CancelResponsePB response) {
        return switch (response.getStatus()) {
            case CANCEL_STATUS_ACCEPTED -> CancelAck.ACCEPTED;
            case CANCEL_STATUS_NOT_FOUND -> CancelAck.NOT_FOUND;
            case CANCEL_STATUS_TOMBSTONED -> CancelAck.TOMBSTONED;
            case CANCEL_STATUS_UNSPECIFIED, UNRECOGNIZED -> CancelAck.FAILED;
        };
    }

}

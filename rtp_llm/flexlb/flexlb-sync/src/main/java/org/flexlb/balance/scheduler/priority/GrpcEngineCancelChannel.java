package org.flexlb.balance.scheduler.priority;

import io.grpc.Context;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Production {@link EngineCancelChannel} that forwards cancel intents to the
 * victim's original Prefill worker via the engine gRPC
 * {@code RpcService.Cancel} method. The original Prefill is the authoritative
 * producer of typed {@code CANCELED}; Decode does not implement Cancel.
 *
 * <p>Conditional wiring: the bean only exists when
 * {@code flexlb.auto-tpm.engine-cancel-enabled=true}. When absent, Spring wires
 * the {@link UnsupportedEngineCancelChannel} fallback and the accepted-eviction
 * gate stays dormant. When active, {@code @Primary} makes this bean win
 * resolution over the unsupported fallback (and over any test-only mock
 * channel whose own {@code @ConditionalOnProperty} is not set).
 *
 * <p>Contract mirror of {@link EngineCancelChannel}: a cancel is an intent
 * injection only — settlement requires original-Prefill WorkerStatus carrying
 * typed {@code CANCELED+8429}. The engine Cancel RPC distinguishes ACCEPTED from NOT_FOUND;
 * these map to the matching local outcomes, while
 * any transport failure to {@code failed()} — never throws synchronously.
 */
@Slf4j
@Component
@Primary
@ConditionalOnProperty(name = "flexlb.auto-tpm.engine-cancel-enabled", havingValue = "true")
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
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                   long requestId,
                                                   long timeoutMs) {
        if (target == null || !target.isRoutable()) {
            // No routable endpoint — report the transport-failure branch: the
            // intent never reached the engine, but release is still settled by
            // the WorkerStatus report (iron rule 4).
            log.warn("[auto-tpm] cancel has no prefill control owner for request_id={}, not routed",
                    requestId);
            return CompletableFuture.completedFuture(CancelOutcome.failed());
        }

        EngineRpcService.CancelRequestPB requestPB = EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(requestId)
                .build();

        // Fire-and-forget contract: fork the gRPC Context so that when the
        // caller is a server handler, the server call completing does not
        // cascade-cancel this in-flight outbound RPC ("io.grpc.Context was
        // cancelled without error").
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
                        log.warn("[auto-tpm] cancel rpc failed for request_id={}: {}",
                                requestId, t.getMessage());
                        return CancelOutcome.failed();
                    });
        } finally {
            fork.detach(previous);
        }
    }

    private static CancelOutcome mapResponse(EngineRpcService.CancelResponsePB response) {
        return switch (response.getStatus()) {
            case CANCEL_STATUS_ACCEPTED -> CancelOutcome.accepted();
            case CANCEL_STATUS_NOT_FOUND -> CancelOutcome.notFound();
            case CANCEL_STATUS_UNSPECIFIED, UNRECOGNIZED -> CancelOutcome.failed();
        };
    }

}

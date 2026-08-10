package org.flexlb.balance.scheduler.priority;

import io.grpc.Context;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Production {@link EngineCancelChannel} that forwards cancel intents to the
 * original Prefill lifecycle owner via the engine gRPC {@code RpcService.Cancel}
 * method.
 *
 * <p>Conditional wiring: the bean only exists when
 * {@code flexlb.auto-tpm.engine-cancel-enabled=true}. When absent, Spring wires
 * the {@link UnsupportedEngineCancelChannel} fallback and the accepted-eviction
 * gate stays dormant. When active, {@code @Primary} makes this bean win
 * resolution over the unsupported fallback (and over any test-only mock
 * channel whose own {@code @ConditionalOnProperty} is not set).
 *
 * <p>Contract mirror of {@link EngineCancelChannel}: a cancel is an intent
 * injection only — release confirmation remains the next WorkerStatus report
 * (iron rule 4). The engine Cancel RPC always answers ACCEPTED (intent
 * registration semantics), so a successful RPC maps to {@code accepted()} and
 * any transport failure to {@code failed()} — never throws synchronously.
 */
@Slf4j
@Component
@Primary
@ConditionalOnProperty(name = "flexlb.auto-tpm.engine-cancel-enabled", havingValue = "true")
public class GrpcEngineCancelChannel implements EngineCancelChannel {

    /** Single-call deadline (1s). */
    private static final long CANCEL_RPC_TIMEOUT_MS = 1000L;

    private final EngineGrpcClient engineGrpcClient;

    public GrpcEngineCancelChannel(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    /**
     * gRPC cancel is available for all workers (cancel always
     * targets the original Prefill owner, which is a gRPC endpoint).
     */
    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return true;
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                   long requestId,
                                                   CancelReason reason) {
        PrefillEndpoint lifecycleOwner = target.lifecycleOwner();
        if (lifecycleOwner == null) {
            // No recorded owner — the cancel cannot be routed. Report the
            // transport-failure branch: the intent never reached the engine,
            // but release is still settled by the WorkerStatus report, never
            // by this ack (iron rule 4).
            log.warn("[auto-tpm] cancel has no lifecycle owner for request_id={}, not routed",
                    requestId);
            return CompletableFuture.completedFuture(CancelOutcome.failed());
        }

        EngineRpcService.CancelRequestPB requestPB = EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(requestId)
                .setBatchId(target.batchId())
                .setReason(mapReason(reason))
                .build();

        // Fire-and-forget contract: fork the gRPC Context so that when the
        // caller is a server handler (e.g. the Frontend cancel entry point),
        // the server call completing does not cascade-cancel this in-flight
        // outbound RPC ("io.grpc.Context was cancelled without error").
        Context fork = Context.current().fork();
        Context previous = fork.attach();
        try {
            return engineGrpcClient.cancelAsync(
                            lifecycleOwner.getIp(),
                            lifecycleOwner.getGrpcPort(),
                            requestPB,
                            CANCEL_RPC_TIMEOUT_MS)
                    // The response body carries no decision-relevant fields — the
                    // RPC completing at all is the intent registration.
                    .thenApply(response -> CancelOutcome.accepted())
                    .exceptionally(t -> {
                        log.warn("[auto-tpm] cancel rpc failed for request_id={}: {}",
                                requestId, t.getMessage());
                        return CancelOutcome.failed();
                    });
        } finally {
            fork.detach(previous);
        }
    }

    // ==================== Proto mapping ====================

    private static EngineRpcService.EngineCancelReasonPB mapReason(CancelReason reason) {
        if (reason == null) {
            return EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_UNSPECIFIED;
        }
        return switch (reason) {
            case PRIORITY_PREEMPTED -> EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED;
            case USER_CANCELLED -> EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_USER_CANCELLED;
            case DEADLINE_EXCEEDED -> EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_DEADLINE_EXCEEDED;
            case ADMIN -> EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_ADMIN_CANCELLED;
        };
    }
}

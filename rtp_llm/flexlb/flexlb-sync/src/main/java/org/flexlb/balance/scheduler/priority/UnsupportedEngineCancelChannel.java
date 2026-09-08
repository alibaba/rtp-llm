package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;

import java.util.concurrent.CompletableFuture;

/**
 * Explicit no-op {@link EngineCancelChannel} for tests and isolated callers
 * that intentionally do not provide an engine Cancel transport. It is not a
 * Spring component: production always wires {@link GrpcEngineCancelChannel},
 * while the accepted-eviction business gate
 * ({@code autoTpmDecodeAcceptedEvictEnabled && channel.isSupported(...)})
 * controls whether the planner may invoke the injected transport. No fake gRPC stub —
 * {@link #cancel} only reports the unsupported branch defensively in case it
 * is ever reached despite the planning gate.
 */
public class UnsupportedEngineCancelChannel implements EngineCancelChannel {

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return false;
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                   long requestId,
                                                   long timeoutMs) {
        return CompletableFuture.completedFuture(CancelOutcome.unsupported());
    }
}

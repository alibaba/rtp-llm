package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Default {@link EngineCancelChannel} wired until the engine-side Cancel RPC
 * ships: every endpoint is unsupported, so the accepted-eviction gate
 * ({@code autoTpmDecodeAcceptedEvictEnabled && channel.isSupported(...)})
 * never opens and Phase 5 stays behaviorally dormant. No fake gRPC stub —
 * {@link #cancel} only reports the unsupported branch defensively in case it
 * is ever reached despite the planning gate.
 */
@Component
public class UnsupportedEngineCancelChannel implements EngineCancelChannel {

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return false;
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                   long requestId,
                                                   CancelReason reason) {
        return CompletableFuture.completedFuture(CancelOutcome.unsupported());
    }
}

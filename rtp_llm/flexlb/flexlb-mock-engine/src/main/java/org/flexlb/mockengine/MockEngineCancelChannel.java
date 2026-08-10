package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Test-only {@link EngineCancelChannel} backed by the in-process mock engine
 * cluster. Resolves the target {@link JavaMockEngineCluster.FastRpcService} by
 * the endpoint's gRPC port and drives the mock cancel behaviour: a live
 * request is removed and a CANCELLED completion surfaces in the next
 * WorkerStatus finished list, exactly like a real engine would report.
 * Mirrors the simplified engine contract: any cancel that reaches an engine
 * acks ACCEPTED (intent registration) — including cancels landing after
 * completion or for unknown ids — the observable effect lives in the mock
 * engine state, never in the ack.
 *
 * <p><b>Wiring:</b> this class is NOT a Spring component. Production contexts
 * keep {@code UnsupportedEngineCancelChannel}; tests inject this channel
 * explicitly, e.g.:
 * <pre>{@code
 *   Map<Integer, FastRpcService> services = ...; // port -> mock engine
 *   EngineCancelChannel channel = new MockEngineCancelChannel(services);
 * }</pre>
 */
public final class MockEngineCancelChannel implements EngineCancelChannel {

    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    public MockEngineCancelChannel(Map<Integer, JavaMockEngineCluster.FastRpcService> services) {
        this.services = services;
    }

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return services.containsKey(endpoint.getGrpcPort());
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target, long requestId, CancelReason reason) {
        DecodeEndpoint endpoint = target.decodeEndpoint();
        JavaMockEngineCluster.FastRpcService service = services.get(endpoint.getGrpcPort());
        if (service == null) {
            return CompletableFuture.completedFuture(CancelOutcome.unsupported());
        }
        try {
            // The mock engine applies the cancel side effects (removal +
            // CANCELLED WorkerStatus record for a live request; no-op for a
            // finished/unknown one); the ack is ACCEPTED either way.
            service.cancelRequest(requestId);
            return CompletableFuture.completedFuture(CancelOutcome.accepted());
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }
}

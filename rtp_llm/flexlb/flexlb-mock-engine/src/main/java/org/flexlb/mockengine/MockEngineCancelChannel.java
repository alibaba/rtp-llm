package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Test-only {@link EngineCancelChannel} backed by the in-process mock engine
 * cluster. Resolves the target {@link JavaMockEngineCluster.FastRpcService} by
 * the endpoint's gRPC port and drives the mock cancel behaviour: a live
 * request is removed and a CANCELLED completion surfaces in the next
 * WorkerStatus finished list, exactly like a real engine would report.
 * Mirrors the production engine contract (C++ Prefill Cancel): a live
 * request and its accepted-cancel tombstone return ACCEPTED; a request the
 * addressed Prefill has seen but already finished returns NOT_FOUND
 * (seen-but-terminal; the completion record stays deliverable from the
 * retain window); a rid the Prefill NEVER saw returns TOMBSTONED with the
 * ABSENT_FENCE tombstone installed (racing later Enqueues of that rid are
 * rejected with 8429); Decode rejects this RPC as unsupported.
 *
 * <p><b>Wiring:</b> this class is NOT a Spring component. Tests inject it
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
        return endpoint != null && services.containsKey(endpoint.getGrpcPort());
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target, long requestId,
                                                   long timeoutMs) {
        JavaMockEngineCluster.FastRpcService service = target == null
                ? null : services.get(target.prefillGrpcPort());
        if (service == null) {
            return CompletableFuture.completedFuture(CancelOutcome.unsupported());
        }
        try {
            // Deliberately inspect only the addressed Prefill. Scanning other
            // workers would hide an incorrect Prefill route in tests.
            JavaMockEngineCluster.CancelResult result = service.cancelRequest(requestId);
            if (result.found()) {
                return CompletableFuture.completedFuture(CancelOutcome.accepted());
            }
            // Production-faithful mapping (C++ Cancel handler): a request
            // this engine has seen but already finished answers NOT_FOUND
            // (the completion record stays deliverable from the retain
            // window); a never-seen rid answers TOMBSTONED — cancelRequest
            // installed the ABSENT_FENCE tombstone that rejects any racing
            // later Enqueue of that rid with the typed 8429.
            return CompletableFuture.completedFuture(
                    result.alreadyFinished()
                            ? CancelOutcome.notFound()
                            : CancelOutcome.tombstoned());
        } catch (UnsupportedOperationException e) {
            return CompletableFuture.completedFuture(CancelOutcome.failed());
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }
}

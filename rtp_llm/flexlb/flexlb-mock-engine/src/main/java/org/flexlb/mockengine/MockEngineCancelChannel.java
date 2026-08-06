package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Test-only {@link EngineCancelChannel} backed by the in-process mock engine
 * cluster. Resolves the target {@link JavaMockEngineCluster.FastRpcService} by
 * the endpoint's gRPC port and drives the mock cancel behaviour: on the found
 * branch the request is removed and a CANCELLED completion surfaces in the
 * next WorkerStatus finished list, exactly like a real engine would report.
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
    public CompletableFuture<CancelOutcome> cancel(DecodeEndpoint endpoint,
                                                   long requestId,
                                                   CancelReason reason) {
        JavaMockEngineCluster.FastRpcService service = services.get(endpoint.getGrpcPort());
        if (service == null) {
            return CompletableFuture.completedFuture(CancelOutcome.unsupportedEndpoint());
        }
        try {
            JavaMockEngineCluster.CancelResult result = service.cancelRequest(requestId);
            if (result.found()) {
                return CompletableFuture.completedFuture(
                        CancelOutcome.accepted(toTaskPhase(result.phase())));
            }
            if (result.alreadyFinished()) {
                return CompletableFuture.completedFuture(CancelOutcome.finishedBeforeCancel());
            }
            return CompletableFuture.completedFuture(CancelOutcome.notFound());
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }

    private static TaskPhase toTaskPhase(EngineRpcService.TaskPhase phase) {
        if (phase == null) {
            return null;
        }
        return switch (phase) {
            case TASK_PHASE_PENDING -> TaskPhase.PENDING;
            case TASK_PHASE_RECEIVED -> TaskPhase.RECEIVED;
            case TASK_PHASE_KV_ALLOCATED -> TaskPhase.KV_ALLOCATED;
            case TASK_PHASE_RUNNING, UNRECOGNIZED -> TaskPhase.RUNNING;
        };
    }
}

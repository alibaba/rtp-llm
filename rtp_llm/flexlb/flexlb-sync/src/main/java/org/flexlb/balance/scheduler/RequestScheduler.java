package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Request submission facade for DIRECT and QUEUE.
 *
 * <p>Both modes register one canonical request. DIRECT selects and commits on
 * ingress; QUEUE defers selection to the global ordered queue. Endpoint
 * batchers own only delivery after the selected route is committed.</p>
 */
@Component
public final class RequestScheduler {

    private final DefaultRouter router;
    private final EndpointRegistry endpointRegistry;
    private final RequestRegistry requestRegistry;
    private final GlobalQueueCoordinator globalQueue;

    @Autowired
    RequestScheduler(
            ConfigService configService,
            DefaultRouter router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestRegistry requestRegistry,
            PlacementAvailability placementAvailability) {
        Objects.requireNonNull(configService, "configService");
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.requestRegistry = Objects.requireNonNull(requestRegistry, "lifecycle");
        FlexlbConfig startupConfig = configService.loadBalanceConfig();
        this.globalQueue = startupConfig != null && startupConfig.isQueue()
                ? new GlobalQueueCoordinator(
                        configService,
                        Objects.requireNonNull(router, "router"),
                        Objects.requireNonNull(reporter, "reporter"),
                        Objects.requireNonNull(evictionManager, "evictionManager"),
                        this.requestRegistry,
                        Objects.requireNonNull(
                                placementAvailability, "placementAvailability"))
                : null;
    }

    /** Register once, then enter direct admission or the ordered global queue. */
    public CompletableFuture<Response> submit(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.INVALID_REQUEST, null));
        }
        FlexlbConfig requestConfig = context.getConfig();
        if (requestConfig.isQueue() && globalQueue == null) {
            return CompletableFuture.completedFuture(error(StrategyErrorType.DISPATCH_FAILED,
                    "QUEUE configuration was enabled after scheduler startup"));
        }
        CompletableFuture<Response> future = requestRegistry.register(context);
        context.setFuture(future);
        if (future.isDone()) {
            return future;
        }
        if (requestConfig.isDirect()) {
            submitDirect(context);
            return future;
        }
        try {
            if (!globalQueue.offer(context, future, context.getPriority())) {
                future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down"));
            }
        } catch (Throwable failure) {
            future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                    "Queue submission failed: " + failure.getMessage()));
        }
        return future;
    }

    private void submitDirect(BalanceContext context) {
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        try {
            PlacementResult<RouteAdmission, PlacementKey> selection = router.select(context);
            switch (selection.status()) {
                case SUCCESS -> {
                    try (RouteAdmission admission = selection.value()) {
                        var committed = admission.tryCommitDirectRoute(context, requestRegistry);
                        failure = switch (committed.status()) {
                            case SUCCESS -> {
                                var delivery = committed.value();
                                requestRegistry.publishRoute(delivery.claim(), delivery.precedingWork(), delivery.unstartedWorkMs());
                                yield null;
                            }
                            case REJECTED -> committed.rejection();
                            case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                            case BLOCKED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                        };
                    }
                }
                case BLOCKED -> failure = Response.error(selection.blocker().role().getErrorType());
                case REJECTED -> failure = selection.rejection();
                case CLOSED -> failure = Response.error(StrategyErrorType.REQUEST_CANCELLED);
            }
        } catch (RuntimeException selectionFailure) {
            Logger.warn("DIRECT admission failed: request_id={}", context.getRequestId(), selectionFailure);
        } finally {
            if (failure != null) {
                requestRegistry.publishDecisionResponseAsync(context.getRequestId(), context.getFuture(), failure);
            }
        }
    }

    public RequestState cancelRequest(
            long requestId,
            long expectedBatchId,
            CancelReason reason) {
        return requestRegistry.cancelRequest(requestId, expectedBatchId, reason);
    }

    public int getInflightSize() {
        return requestRegistry.liveRequestCount();
    }

    public int getQueuedRequestCount() {
        long queued = globalQueue == null ? 0L : globalQueue.size();
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    public List<RequestState> snapshotActiveRequests() {
        return requestRegistry.snapshotActiveRequests();
    }

    public RequestState getRequestState(long requestId, long expectedBatchId) {
        return requestRegistry.getRequestState(requestId, expectedBatchId);
    }

    public void closePlacement() {
        if (globalQueue != null) {
            globalQueue.close();
        }
    }

    private static Response error(StrategyErrorType type, String detail) {
        return Response.buildErrorResponse(type, detail);
    }
}

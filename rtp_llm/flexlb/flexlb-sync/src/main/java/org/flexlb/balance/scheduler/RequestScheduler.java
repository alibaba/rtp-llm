package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * Request submission facade for DIRECT and QUEUE.
 *
 * <p>Both modes register one canonical request. DIRECT selects and commits on
 * ingress; QUEUE defers selection to the Encoder or Generation ordered queue. Endpoint
 * batchers own only delivery after the selected route is committed.</p>
 */
@Component
public final class RequestScheduler {

    private final DefaultRouter router;
    private final EndpointRegistry endpointRegistry;
    private final RequestRegistry requestRegistry;
    private final GlobalQueueCoordinator globalQueue;
    private final EncoderQueueCoordinator encoderQueue;

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
        this.encoderQueue = startupConfig != null && startupConfig.isQueue() && router.hasEncoderRole()
                ? new EncoderQueueCoordinator(router, requestRegistry, startupConfig.isPriorityOrdering())
                : null;
    }

    /** Register once, then enter direct admission or the ordered global queue. */
    public CompletableFuture<Response> submit(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.INVALID_REQUEST, null));
        }
        Set<RoleType> requestedRoles = context.getRequestedRoles();
        if (requestedRoles != null && requestedRoles.contains(RoleType.ENCODER)
                && !router.hasEncoderRole()) {
            return CompletableFuture.completedFuture(error(StrategyErrorType.INVALID_REQUEST,
                    "Encoder role is not configured for this model"));
        }
        boolean encoderOnly = router.isEncoderOnly(context);
        if (encoderOnly) {
            context.setRequestPhase(RequestPhase.ENCODER);
        }
        FlexlbConfig requestConfig = context.getConfig();
        if (requestConfig.isQueue() && (encoderOnly ? encoderQueue == null : globalQueue == null)) {
            return CompletableFuture.completedFuture(error(StrategyErrorType.DISPATCH_FAILED,
                    "QUEUE configuration was enabled after scheduler startup"));
        }
        CompletableFuture<Response> future = requestRegistry.register(context);
        context.setFuture(future);
        if (!future.isDone()) {
            scheduleRegisteredRequest(context, future, requestConfig, encoderOnly);
        }
        return future;
    }

    private void scheduleRegisteredRequest(BalanceContext context, CompletableFuture<Response> future,
                                           FlexlbConfig requestConfig, boolean encoderOnly) {
        if (encoderOnly) {
            if (requestConfig.isQueue()) {
                enqueueEncoderRequest(context, future);
            } else {
                selectAndPublishEncoder(context);
            }
        } else if (requestConfig.isDirect()) {
            submitDirect(context);
        } else {
            enqueueGenerationRequest(context, future);
        }
    }

    private void enqueueEncoderRequest(BalanceContext context, CompletableFuture<Response> future) {
        try {
            if (!encoderQueue.offer(context, future)) {
                future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down"));
            }
        } catch (RuntimeException failure) {
            future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                    "Encoder queue submission failed: " + failure.getMessage()));
        }
    }

    private void enqueueGenerationRequest(BalanceContext context, CompletableFuture<Response> future) {
        try {
            if (!globalQueue.offer(context, future, context.getPriority())) {
                future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down"));
            }
        } catch (Throwable failure) {
            future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                    "Queue submission failed: " + failure.getMessage()));
        }
    }

    private void selectAndPublishEncoder(BalanceContext context) {
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        try {
            PlacementResult<SelectedRole, PlacementKey> selection = router.selectEncoder(context);
            context.setSchedulingDiagnostics(selection.diagnostics());
            switch (selection.status()) {
                case SUCCESS -> {
                    try (SelectedRole selected = selection.value()) {
                        Response response = new Response();
                        response.setSuccess(true);
                        response.setServerStatus(List.of(selected.serverStatus()));
                        try (var pin = selected.takeGenerationPin()) {
                            if (requestRegistry.claimEncoderRoute(context.getRequestId(), context.getFuture(), pin)
                                    && requestRegistry.publishEncoderRoute(
                                            context.getRequestId(), context.getFuture(), response)) {
                                failure = null;
                            } else {
                                failure = Response.error(StrategyErrorType.REQUEST_CANCELLED);
                            }
                        }
                    }
                }
                case BLOCKED -> failure = selection.failure() != null
                        ? selection.failure() : Response.error(StrategyErrorType.NO_ENCODER_WORKER);
                case REJECTED -> failure = selection.failure();
                case CLOSED -> failure = Response.error(StrategyErrorType.REQUEST_CANCELLED);
            }
        } catch (RuntimeException selectionFailure) {
            Logger.warn("Encoder DIRECT admission failed: request_id={}", context.getRequestId(), selectionFailure);
        } finally {
            if (failure != null) {
                requestRegistry.publishDecisionResponseAsync(context.getRequestId(), context.getFuture(), failure,
                        RequestPhase.ENCODER);
            }
        }
    }

    private void submitDirect(BalanceContext context) {
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        try {
            PlacementResult<RouteAdmission, PlacementKey> selection = router.select(context);
            context.setSchedulingDiagnostics(selection.diagnostics());
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
                            case REJECTED -> committed.failure();
                            case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                            case BLOCKED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                        };
                    }
                }
                case BLOCKED -> failure = selection.failure() != null
                        ? selection.failure() : Response.error(selection.blocker().role().getErrorType());
                case REJECTED -> failure = selection.failure();
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
            String requestId,
            long expectedBatchId,
            CancelReason reason) {
        return requestRegistry.cancelRequest(requestId, expectedBatchId, reason);
    }

    /**
     * Cancel the selected phase while preserving the other phase with the same request ID.
     */
    public RequestState cancelRequest(
            String requestId, long expectedBatchId, CancelReason reason, RequestPhase phase) {
        return requestRegistry.cancelRequest(requestId, expectedBatchId, reason, phase);
    }

    public int getInflightSize() {
        return requestRegistry.liveRequestCount();
    }

    public int getQueuedRequestCount() {
        long queued = (globalQueue == null ? 0L : globalQueue.size())
                + (encoderQueue == null ? 0L : encoderQueue.size());
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    public int getBlockedRequestCount() {
        return (globalQueue == null ? 0 : globalQueue.blockedSize())
                + (encoderQueue == null ? 0 : encoderQueue.blockedSize());
    }

    public List<RequestState> snapshotActiveRequests() {
        return requestRegistry.snapshotActiveRequests();
    }

    public RequestState getRequestState(String requestId, long expectedBatchId) {
        return requestRegistry.getRequestState(requestId, expectedBatchId);
    }

    /**
     * Read the Encoder or Generation lifecycle selected by phase.
     */
    public RequestState getRequestState(String requestId, long expectedBatchId, RequestPhase phase) {
        return requestRegistry.getRequestState(requestId, expectedBatchId, phase);
    }

    public void closePlacement() {
        if (globalQueue != null) {
            globalQueue.close();
        }
        if (encoderQueue != null) {
            encoderQueue.close();
        }
    }

    private static Response error(StrategyErrorType type, String detail) {
        return Response.buildErrorResponse(type, detail);
    }
}

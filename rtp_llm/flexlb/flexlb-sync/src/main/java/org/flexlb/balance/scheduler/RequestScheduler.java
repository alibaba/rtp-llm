package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * Request submission facade.
 *
 * <p>Each request registers once, then waits in the Encoder or Generation ordered queue for
 * selection. Endpoint batchers own only delivery after the selected route is committed.</p>
 */
@Component
public final class RequestScheduler {

    private final DefaultRouter router;
    private final EndpointRegistry endpointRegistry;
    private final RequestRegistry requestRegistry;
    private final GlobalQueueCoordinator globalQueue;
    private final EncoderQueueCoordinator encoderQueue;

    @Autowired
    RequestScheduler(ConfigService configService,
                     DefaultRouter router,
                     EndpointRegistry endpointRegistry,
                     BatchSchedulerReporter reporter,
                     EvictionManager evictionManager,
                     RequestRegistry requestRegistry,
                     PlacementAvailability placementAvailability,
                     QueueCapacity queueCapacity) {
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.requestRegistry = requestRegistry;
        FlexlbConfig startupConfig = configService.loadBalanceConfig();
        this.globalQueue = new GlobalQueueCoordinator(configService, router, reporter, evictionManager,
                this.requestRegistry, placementAvailability, queueCapacity);
        this.encoderQueue = router.hasEncoderRole()
                ? new EncoderQueueCoordinator(router, requestRegistry, startupConfig.isPriorityOrdering())
                : null;
    }

    /**
     * Register once, then enter the Encoder or Generation ordered queue.
     */
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
        CompletableFuture<Response> future = requestRegistry.register(context);
        context.setFuture(future);
        if (!future.isDone()) {
            if (encoderOnly) {
                enqueueEncoderRequest(context, future);
            } else {
                enqueueGenerationRequest(context, future);
            }
        }
        return future;
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
            switch (globalQueue.offer(context, future, context.getPriority())) {
                case ENQUEUED, COMPLETED -> { }
                case QUEUE_FULL -> future.complete(error(StrategyErrorType.QUEUE_FULL,
                        GlobalQueueCoordinator.QUEUE_FULL_REJECTION_MESSAGE));
                case QUEUE_FULL_HIGH_PRIORITY_REJECTED -> future.complete(error(
                        StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED,
                        GlobalQueueCoordinator.QUEUE_FULL_REJECTION_MESSAGE));
                case QUEUE_FULL_LOW_PRIORITY_REJECTED -> future.complete(error(
                        StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED,
                        GlobalQueueCoordinator.QUEUE_FULL_REJECTION_MESSAGE));
                case CLOSED -> future.complete(error(StrategyErrorType.DISPATCH_FAILED, "request scheduler is shutting down"));
            }
        } catch (Throwable failure) {
            future.complete(error(StrategyErrorType.DISPATCH_FAILED,
                    "Queue submission failed: " + failure.getMessage()));
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

    public int getTrackedRequestCount() {
        return requestRegistry.trackedRequestCount();
    }

    public int getQueuedRequestCount() {
        long queued = globalQueue.size() + (encoderQueue == null ? 0L : encoderQueue.size());
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
        return globalQueue.blockedSize() + (encoderQueue == null ? 0 : encoderQueue.blockedSize());
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
        globalQueue.close();
        if (encoderQueue != null) {
            encoderQueue.close();
        }
    }

    private static Response error(StrategyErrorType type, String detail) {
        return Response.buildErrorResponse(type, detail);
    }
}

package org.flexlb.balance.scheduler;

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
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Public QUEUE scheduling facade.
 *
 * <p>Ingress only registers the canonical lifecycle slot and appends the
 * request to the model's global ordered queue.  Endpoint selection, exact
 * reservation and publication happen together at the queue decision point;
 * endpoint batchers are delivery runtimes, not independent route selectors.</p>
 */
@Component
public final class RequestScheduler {

    private final ConfigService configService;
    private final EndpointRegistry endpointRegistry;
    private final RequestRegistry lifecycle;
    private final GlobalQueueCoordinator globalQueue;
    private final BatchSchedulerReporter reporter;

    @Autowired
    RequestScheduler(ConfigService configService,
                     DefaultRouter router,
                     EndpointRegistry endpointRegistry,
                     BatchSchedulerReporter reporter,
                     EvictionManager evictionManager,
                     RequestRegistry lifecycle,
                     PlacementAvailability placementAvailability) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.configService = Objects.requireNonNull(
                configService, "configService");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        FlexlbConfig startupConfig = this.configService.loadBalanceConfig();
        this.globalQueue = startupConfig != null && startupConfig.isQueue()
                ? new GlobalQueueCoordinator(
                        this.configService,
                        Objects.requireNonNull(router, "router"),
                        Objects.requireNonNull(reporter, "reporter"),
                        Objects.requireNonNull(evictionManager, "evictionManager"),
                        this.lifecycle,
                        Objects.requireNonNull(
                                placementAvailability, "placementAvailability"))
                : null;
    }

    /** Register once, then enqueue without doing route work on ingress. */
    public CompletableFuture<Response> submit(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.INVALID_REQUEST, null));
        }
        FlexlbConfig activeConfig;
        try {
            activeConfig = configService.loadBalanceConfig();
        } catch (Throwable failure) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Failed to load scheduler configuration: "
                            + failure.getMessage()));
        }
        if (activeConfig == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Scheduler configuration is unavailable"));
        }
        if (!activeConfig.isQueue()) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "RequestScheduler requires QUEUE configuration"));
        }
        if (globalQueue == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "QUEUE configuration was enabled after scheduler startup"));
        }
        int maxOutstanding = activeConfig.queueScheduler().getCapacity()
                .getMaxOutstandingRequestsGlobal();
        CompletableFuture<Response> future = lifecycle.register(
                context, maxOutstanding);
        if (future.isDone()) {
            return future;
        }
        try {
            if (!globalQueue.offer(context, future, context.getPriority())) {
                future.complete(error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                        "request scheduler is shutting down"));
            } else {
                try {
                    reporter.reportQueueEntry();
                } catch (RuntimeException failure) {
                    Logger.warn("Failed to report scheduler queue entry", failure);
                }
            }
        } catch (Throwable failure) {
            future.complete(error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Queue submission failed: " + failure.getMessage()));
        }
        return future;
    }

    public RequestState cancelRequest(
            String requestId,
            long expectedBatchId,
            CancelReason reason) {
        return lifecycle.cancelRequest(requestId, expectedBatchId, reason);
    }

    public int getInflightSize() {
        return lifecycle.liveRequestCount();
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void reportQueueState() {
        if (globalQueue != null) {
            try {
                reporter.reportSchedulerQueueSize(getQueuedRequestCount());
            } catch (RuntimeException failure) {
                Logger.warn("Failed to report scheduler queue size", failure);
            }
        }
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

    public int getBlockedRequestCount() {
        return globalQueue == null ? 0 : globalQueue.blockedSize();
    }

    public List<RequestState> snapshotActiveRequests() {
        return lifecycle.snapshotActiveRequests();
    }

    public RequestState getRequestState(String requestId, long expectedBatchId) {
        return lifecycle.getRequestState(requestId, expectedBatchId);
    }

    public boolean ownsRequestGeneration(String requestId) {
        return lifecycle.ownsRequestGeneration(requestId);
    }

    public void closePlacement() {
        if (globalQueue != null) {
            globalQueue.close();
        }
    }

    private static Response error(StrategyErrorType type, String detail) {
        return RequestRegistry.buildErrorResponse(type, detail);
    }
}

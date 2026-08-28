package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Public request-routing facade.
 *
 * <p>The facade owns no request lifecycle state. Endpoint callbacks, exact
 * request generations, delivery claims, fences, deadlines and publication are
 * canonicalized by {@link RequestLifecycleCoordinator}.
 */
@Component
public final class RequestScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final AdmissionFallback admissionFallback;
    private final RequestLifecycleCoordinator lifecycle;

    @Autowired
    RequestScheduler(
            ConfigService configService,
            Router router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            AdmissionFallback admissionFallback,
            RequestLifecycleCoordinator lifecycle) {
        this.configService = Objects.requireNonNull(
                configService, "configService");
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.admissionFallback = Objects.requireNonNull(
                admissionFallback, "admissionFallback");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
    }

    /** Route one request and hand its exact generation to the lifecycle owner. */
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
        int maxOutstanding = activeConfig.queueScheduler()
                .getCapacity().getMaxOutstandingRequestsGlobal();
        CompletableFuture<Response> future =
                lifecycle.register(context, maxOutstanding);
        if (future.isDone()) {
            return future;
        }

        try {
            routeRegistered(context, future, activeConfig);
        } catch (Throwable failure) {
            Logger.error(
                    "RequestScheduler submit failed for request id: {}",
                    context.getRequestId(), failure);
            future.complete(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + failure.getMessage()));
        }
        return future;
    }

    private void routeRegistered(
            BalanceContext context,
            CompletableFuture<Response> future,
            FlexlbConfig config) {
        RequestLifecycleCoordinator.AdmissionScope mutation =
                lifecycle.beginAdmission(context.getRequestId(), future);
        if (mutation == null) {
            return;
        }

        try (mutation) {
            if (!tryInstallDecodeAcceptanceGuard(context, future, config)) {
                mutation.close();
                if (!future.isDone()) {
                    AdmissionFailure failure =
                            AdmissionFailure.resourceExhausted();
                    int limit = config.queueScheduler().getLifecycle()
                            .getMaxDeliveredNotAcceptedRequestsGlobal();
                    future.complete(RequestLifecycleCoordinator
                            .buildAdmissionErrorResponse(
                                    failure,
                                    "post-success backpressure: "
                                            + "active_admissions="
                                            + lifecycle.decodeAcceptanceCount()
                                            + " limit=" + limit));
                }
                return;
            }
            QueueRoutingResult routing = router.routeForQueue(context);
            if (routing instanceof QueueRoutingResult.Rejected rejected) {
                mutation.close();
                if (!admissionFallback.tryAdmit(context, future)) {
                    future.complete(rejected.response());
                }
                return;
            }

            QueueRouteAdmission admission =
                    ((QueueRoutingResult.Admitted) routing).admission();
            StrategyErrorType failureType = null;
            String failureDetail = null;
            boolean tryFallback = false;
            try (admission) {
                if (lifecycle.isShuttingDown()) {
                    failureType = StrategyErrorType.BATCH_DISPATCH_FAILED;
                    failureDetail = "request scheduler is shutting down";
                } else {
                    BatchItem item = admission.buildItem(
                            context, future, System.currentTimeMillis());
                    context.setRouteSubmittedNanos(System.nanoTime());
                    if (!admission.commitTo(lifecycle, item, false)) {
                        tryFallback = lifecycle.isAdmissionOpen(
                                context.getRequestId(), future);
                    } else {
                        reportRouteSubmitted(context, item);
                    }
                }
            }

            if (tryFallback) {
                mutation.close();
                if (admissionFallback.tryAdmit(context, future)) {
                    return;
                }
                failureType = StrategyErrorType.BATCH_DISPATCH_FAILED;
                failureDetail =
                        "Worker scheduling queue rejected request";
            }
            if (failureType != null) {
                mutation.close();
                future.complete(error(failureType, failureDetail));
            }
        }
    }

    private boolean tryInstallDecodeAcceptanceGuard(
            BalanceContext context,
            CompletableFuture<Response> future,
            FlexlbConfig config) {
        return lifecycle.tryInstallDecodeAcceptanceGuard(
                context.getRequestId(),
                future,
                config.queueScheduler().getLifecycle()
                        .getMaxDeliveredNotAcceptedRequestsGlobal(),
                config.queueScheduler().getLifecycle()
                        .getDeliveredNotAcceptedTimeoutMs());
    }

    private void reportRouteSubmitted(
            BalanceContext context, BatchItem item) {
        try {
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Failed to record route-submit telemetry: request_id={}",
                    context.getRequestId(), telemetryFailure);
        }
    }

    public RequestLifecycleSnapshot cancelRequest(
            long requestId,
            long expectedBatchId,
            CancelReason reason) {
        return lifecycle.cancelRequest(requestId, expectedBatchId, reason);
    }

    public int getInflightSize() {
        return lifecycle.getInflightSize();
    }

    public int getQueuedRequestCount() {
        long queued = 0L;
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    public List<RequestLifecycleSnapshot> snapshotActiveRequests() {
        return lifecycle.snapshotActiveRequests();
    }

    public RequestLifecycleSnapshot getRequestState(
            long requestId, long expectedBatchId) {
        return lifecycle.getRequestState(requestId, expectedBatchId);
    }

    public boolean ownsRequestGeneration(long requestId) {
        return lifecycle.ownsRequestGeneration(requestId);
    }

    private static Response error(
            StrategyErrorType type, String detail) {
        return RequestLifecycleCoordinator.buildErrorResponse(type, detail);
    }
}

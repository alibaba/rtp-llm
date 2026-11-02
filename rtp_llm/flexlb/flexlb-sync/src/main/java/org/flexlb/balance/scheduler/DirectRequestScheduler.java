package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static org.flexlb.balance.scheduler.ContextResult.*;

/** Accepts DIRECT requests and completes one immediate endpoint selection and handoff. */
public final class DirectRequestScheduler extends AbstractRequestScheduler {

    /** Exact committed route and its frozen prediction, before response publication. */
    record RouteDelivery(DeliveryClaim claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        RouteDelivery {
            Objects.requireNonNull(claim, "claim");
            Objects.requireNonNull(precedingWork, "precedingWork");
        }
    }

    private final RequestWorkerSelector router;

    DirectRequestScheduler(RequestWorkerSelector router, SchedulerRuntime runtime, FlexlbConfig config,
                           DecodeResources.AdmissionCapacity capacity, RecentCacheKeyTraceReporter traceReporter) {
        super(runtime, config, capacity, traceReporter);
        this.router = Objects.requireNonNull(router, "router");
    }

    @Override
    public CompletableFuture<Response> submit(RequestContext context) {
        if (context != null && context.data().getRequest() != null) {
            var request = context.data().getRequest();
            if (request.isVitRouteOnly()) {
                var result = CompletableFuture.completedFuture(router.routeVit(context));
                context.data().setFuture(result);
                return result;
            }
            if (request.getSelectedVit() != null
                    && requests.getRequestState(context.data().getRequestId(), 0L) == null
                    && !router.selectedVitIsValid(context)) {
                return CompletableFuture.completedFuture(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
            }
        }
        if (context != null && context.data().getConfig().isQueue()) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                    "request configuration does not match scheduler mode"));
        }
        try {
            CompletableFuture<Response> future = register(context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
            if (future.isDone()) { return future; }
            AdmissionHandle operation = claimAdmissionHandle(context.data().getRequestId(), future);
            if (operation == null) { return future; }
            Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
            try {
                var selection = router.select(context, router.resolvePolicyGroup(context));
                context.data().setSchedulingDiagnostics(selection.diagnostics());
                switch (selection.status()) {
                    case SUCCESS -> {
                        PlacementResult<RouteDelivery, PlacementKey> committed = null;
                        try (RequestRoute routing = selection.value()) {
                            committed = commitDirectRoute(context, routing);
                        } catch (RuntimeException | Error cleanupFailure) {
                            if (committed == null || committed.status() != PlacementResult.Status.SUCCESS) { throw cleanupFailure; }
                            Logger.warn("DIRECT selection cleanup failed after handoff: request_id={}",
                                    context.data().getRequestId(), cleanupFailure);
                        }
                        failure = switch (committed.status()) {
                            case SUCCESS -> {
                                RouteDelivery delivery = committed.value();
                                operation.finish();
                                long now = System.currentTimeMillis();
                                long visibility = PrefillPredictionBoundary.predictedVisibilityDeadlineMs(delivery.precedingWork(),
                                        delivery.unstartedWorkMs(), context.data().getConfig().getRequestLifecycle().getDecision().getLifetime(), now);
                                onDeliveryEvidence(delivery.claim(), new RouteDelivered(delivery.claim(), visibility, now));
                                yield null;
                            }
                            case REJECTED -> committed.failure();
                            case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                            case BLOCKED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                        };
                    }
                    case BLOCKED -> failure = selection.failure() != null
                            ? selection.failure() : Response.error(selection.blocker().role().getErrorType());
                    case REJECTED -> failure = selection.failure();
                    case CLOSED -> failure = Response.error(StrategyErrorType.REQUEST_CANCELLED);
                }
            } catch (RuntimeException selectionFailure) {
                Logger.warn("DIRECT admission failed: request_id={}", context.data().getRequestId(), selectionFailure);
            } finally {
                operation.finish();
                if (failure != null) { publish(context, failure, false); }
            }
            return future;
        } catch (RuntimeException failure) {
            Response response = Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (context != null && context.data().getScheduler() == this) {
                publish(context, response, false);
                return context.data().getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    PlacementResult<RouteDelivery, PlacementKey> commitDirectRoute(RequestContext context, RequestRoute routing) {
        var result = ImmediateRouteAdmission.commit(this, context, routing, null);
        return switch (result.status()) {
            case SUCCESS -> PlacementResult.success(new RouteDelivery(result.value().claim(),
                    result.value().precedingWork(), result.value().unstartedWorkMs()));
            case BLOCKED -> PlacementResult.blocked(result.blocker());
            case CLOSED -> PlacementResult.closed();
            case REJECTED -> PlacementResult.rejected(result.failure());
        };
    }
}

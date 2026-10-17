package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.List;
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

    PlacementResult<RouteDelivery, PlacementKey> commitDirectRoute(
            RequestContext context, RequestRoute routing) {
        var request = context.data().getRequest();
        DecodeResources.ReservationHandle pendingDecodeRollback = routing.decodeEp() == null ? null
                : routing.decodeEp().tryReserveQueuedRequest(routing.decodePin(), routing.requestId(),
                        request.getHardKvTokens(), request.getExpectedKvTokens(), routing.priority(), null).handle();
        if (routing.decodeEp() != null && pendingDecodeRollback == null) {
            return PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
        }
        Throwable failure = null;
        RequestRoute acquired = null;
        var placement = context.snapshot().placement();
        var token = placement == null ? null : placement.operation();
        boolean delivered = false;
        try {
            RequestRoute route = acquired = RequestRoute.create(context, routing, pendingDecodeRollback);
            CapacityBoundary.Attempt<DeliveryDispatcher.Member> attempt = DeliveryDispatcher.Member.prepare(route, decodeAdmissionCapacity);
            if (!attempt.accepted()) {
                return switch (attempt.boundary().status()) {
                    case UNAVAILABLE -> PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
                    case OWNERSHIP_LOST -> PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                    case FAILED -> throw new IllegalStateException("route admission failed", attempt.boundary().cause());
                };
            }
            DeliveryDispatcher.Member member = attempt.value();
            try (member) {
                var reservationAttempt = new PrefillReservationAttempt(route);
                try {
                    PrefillState.CommittedHandoff handoff = null;
                    try {
                        PlacementResult.Status result = this.commitRoute(route, reservationAttempt);
                        if (result != PlacementResult.Status.SUCCESS) {
                            return result == PlacementResult.Status.CLOSED ? PlacementResult.closed()
                                    : PlacementResult.rejected(Response.error(reservationAttempt.failure()));
                        }
                        handoff = routing.prefillEp().commit(new PrefillState.CommitPlan(PrefillState.CommitKind.DIRECT, List.of(route),
                                List.of(reservationAttempt.reservation()), null, new long[0], null, System.currentTimeMillis()));
                        if (handoff == null) { return PlacementResult.rejected(Response.error(StrategyErrorType.DISPATCH_FAILED)); }
                        WorkSnapshot precedingWork = handoff.precedingWork().materialize();
                        var claim = this.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L,
                                member.decode());
                        if (claim == null) { return PlacementResult.closed(); }
                        delivered = true;
                        pendingDecodeRollback = null;
                        return PlacementResult.success(new RouteDelivery(claim, precedingWork, routing.prefillWorkMs()));
                    } finally {
                        Throwable cleanup = DeliveryDispatcher.closeCommitted(List.of(), handoff);
                        if (cleanup != null) { runtime.recordFailure(cleanup); }
                    }
                } finally {
                    if (reservationAttempt.reservation() != null) { routing.prefillEp().release(reservationAttempt.reservation(), null); }
                }
            }
        } catch (RuntimeException | Error cause) {
            failure = cause;
            throw cause;
        } finally {
            if (!delivered) {
                Throwable rollbackFailure = null;
                try { if (acquired != null) { acquired.prefillEp().release(null, acquired); } }
                catch (RuntimeException | Error problem) { rollbackFailure = problem; }
                try {
                    if (pendingDecodeRollback != null) {
                        var released = routing.decodeEp().release(pendingDecodeRollback, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                        if (released != DecodeResources.ReservationReleaseResult.RELEASED && released != DecodeResources.ReservationReleaseResult.STALE) {
                            throw new IllegalStateException("placement rollback retained Decode resources");
                        }
                    }
                } catch (RuntimeException | Error problem) { rollbackFailure = Failures.append(rollbackFailure, problem); }
                if (rollbackFailure == null && acquired != null && token != null) {
                    context.bindPlacement(token, acquired, PublicationResult.ROLLBACK_COMPLETED);
                }
                if (rollbackFailure != null) {
                    if (failure == null) { Failures.rethrow(rollbackFailure, "DIRECT rollback failed"); }
                    else { Failures.append(failure, rollbackFailure); }
                }
            }
        }
    }

    /** Holds acquisition output without mixing a failed publication with capacity pressure. */
    private static final class PrefillReservationAttempt implements Publication {
        private final RequestRoute route;
        private PrefillState.ReservationResult<PrefillState.RouteReservation> result;

        PrefillReservationAttempt(RequestRoute route) {
            this.route = route;
        }

        @Override
        public void publish() {
            result = route.prefillEp().reserveUnqueuedRoute(route.prefillPin(), route, route.prefillWorkMs());
        }

        @Override public boolean published() {
            return result != null && result.status() == PrefillState.CapacityStatus.ACQUIRED;
        }

        PrefillState.RouteReservation reservation() {
            return result == null ? null : result.reservation();
        }

        StrategyErrorType failure() {
            if (result == null) { return StrategyErrorType.REQUEST_CANCELLED; }
            return switch (result.status()) {
                case CAPACITY_FULL -> StrategyErrorType.RESOURCE_EXHAUSTED;
                case ENDPOINT_RETIRED -> StrategyErrorType.DISPATCH_FAILED;
                case REQUEST_ALREADY_RESERVED, REQUEST_NOT_ACTIVE, BATCH_ID_ALREADY_RESERVED ->
                        StrategyErrorType.RESOURCE_EXHAUSTED;
                case ACQUIRED -> throw new IllegalStateException("successful admission cannot be rejected");
            };
        }
    }

}

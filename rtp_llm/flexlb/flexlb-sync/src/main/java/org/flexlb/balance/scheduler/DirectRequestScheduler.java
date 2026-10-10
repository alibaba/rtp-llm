package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.util.Failures;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

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

    DirectRequestScheduler(RequestWorkerSelector router, SchedulerRuntime runtime, FlexlbConfig config) {
        super(runtime, config);
        this.router = Objects.requireNonNull(router, "router");
    }

    @Override
    public CompletableFuture<Response> submit(RequestContext context) {
        if (!runtime.isAccepting()) { return rejected(); }
        try {
            if (context != null && !context.getConfig().isDirect()) {
                return invalidMode();
            }
            CompletableFuture<Response> future = register(context, StrategyErrorType.BATCH_SLO_EXPIRED);
            if (!future.isDone()) {
                this.expirationTimer().scheduleInactivityDeadline(context);
                dispatchRegistered(context);
            }
            return future;
        } catch (RuntimeException failure) {
            return failSubmission(context, failure);
        }
    }

    private void dispatchRegistered(RequestContext context) {
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        AdmissionHandle operation = this.claimAdmissionHandle(context.getRequestId(), context.getFuture());
        if (operation == null) { return; }
        try {
            failure = selectAndCommit(context, operation);
        } catch (RuntimeException selectionFailure) {
            Logger.warn("DIRECT admission failed: request_id={}", context.getRequestId(), selectionFailure);
        } finally {
            operation.finish();
            if (failure != null) {
                this.publishDecisionResponseAsync(context.getRequestId(), context.getFuture(), failure);
            }
        }
    }

    PlacementResult<RouteDelivery, PlacementKey> commitDirectRoute(
            RequestContext context, RequestRoute routing) {
        DecodeResources.ReservationHandle pendingDecodeRollback = tryReserveDecode(routing, context.getRequirements());
        if (routing.decodeEp() != null && pendingDecodeRollback == null) {
            return PlacementResult.blocked(routing.decodePlacementKey());
        }
        Throwable failure = null;
        try {
            RequestRoute route = RequestRoute.create(context, routing, pendingDecodeRollback);
            initializeWorkerQueue(context, context.getEnqueueTime());
            CapacityBoundary.Attempt<DeliveryTransaction.Member> attempt = DeliveryTransaction.prepareMember(route);
            if (!attempt.accepted()) {
                return switch (attempt.boundary().status()) {
                    case UNAVAILABLE -> PlacementResult.blocked(routing.decodePlacementKey());
                    case OWNERSHIP_LOST -> PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                    case FAILED -> throw new IllegalStateException("route admission failed", attempt.boundary().cause());
                };
            }
            DeliveryTransaction.Member member = attempt.value();
            try (member) {
                var reservationAttempt = new PrefillReservationAttempt(route);
                try (PrefillEndpoint.RouteCommitAdmission routeCommit = routing.prefillEp().tryBeginRouteCommitAdmission()) {
                    if (routeCommit == null) {
                        return PlacementResult.rejected(Response.error(StrategyErrorType.DISPATCH_FAILED));
                    }
                    var members = List.of(member);
                    PrefillState.CommittedHandoff handoff = null;
                    try {
                        PlacementResult.Status result = this.commitRoute(route, reservationAttempt);
                        if (result != PlacementResult.Status.SUCCESS) {
                            return result == PlacementResult.Status.CLOSED ? PlacementResult.closed()
                                    : PlacementResult.rejected(Response.error(reservationAttempt.failure()));
                        }
                        handoff = routeCommit.commit(List.of(route), List.of(reservationAttempt.reservation()));
                        WorkSnapshot precedingWork = handoff.precedingWork().materialize();
                        var claim = this.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L,
                                member.decode());
                        if (claim == null) { return PlacementResult.closed(); }
                        pendingDecodeRollback = null;
                        return PlacementResult.success(new RouteDelivery(claim, precedingWork, routing.prefillWorkMs()));
                    } finally {
                        DeliveryTransaction.closeCommitted(members, handoff);
                    }
                } finally {
                    if (reservationAttempt.reservation() != null) { routing.prefillEp().rollbackReservation(reservationAttempt.reservation()); }
                }
            }
        } catch (RuntimeException | Error cause) {
            failure = cause;
            throw cause;
        } finally {
            if (pendingDecodeRollback != null) {
                try {
                    routing.decodeEp().release(pendingDecodeRollback, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                } catch (RuntimeException | Error cleanupFailure) {
                    if (failure == null) { throw cleanupFailure; }
                    Failures.append(failure, cleanupFailure);
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

    private Response selectAndCommit(RequestContext context, AdmissionHandle operation) {
        PlacementResult<RequestRoute, PlacementKey> selection = router.select(context, router.resolvePolicyGroup(context));
        context.setSchedulingDiagnostics(selection.diagnostics());
        return switch (selection.status()) {
            case SUCCESS -> {
                PlacementResult<RouteDelivery, PlacementKey> committed = null;
                try (RequestRoute routing = selection.value()) {
                    committed = commitDirectRoute(context, routing);
                } catch (RuntimeException | Error cleanupFailure) {
                    if (committed == null || committed.status() != PlacementResult.Status.SUCCESS) {
                        throw cleanupFailure;
                    }
                    // Delivery ownership is already transferred; cleanup cannot retract publication.
                    Logger.warn("DIRECT selection cleanup failed after handoff: request_id={}",
                            context.getRequestId(), cleanupFailure);
                }
                yield switch (committed.status()) {
                    case SUCCESS -> {
                        RouteDelivery delivery = committed.value();
                        operation.finish();
                        this.publishRoute(delivery.claim(), delivery.precedingWork(), delivery.unstartedWorkMs());
                        yield null;
                    }
                    case REJECTED -> committed.failure();
                    case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                    case BLOCKED -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                };
            }
            case BLOCKED -> selection.failure() != null
                    ? selection.failure() : Response.error(selection.blocker().role().getErrorType());
            case REJECTED -> selection.failure();
            case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
        };
    }
}

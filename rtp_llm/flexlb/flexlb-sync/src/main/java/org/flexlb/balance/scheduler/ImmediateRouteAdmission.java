package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Failures;

import java.util.List;

import static org.flexlb.balance.scheduler.ContextResult.*;

/** One nonblocking reservation and handoff; failed admission retains no worker queue assignment. */
final class ImmediateRouteAdmission {
    private ImmediateRouteAdmission() { }

    record Delivery(DeliveryClaim claim, WorkSnapshot precedingWork, long unstartedWorkMs, DeliveryDispatcher.Batch batch) { }

    static PlacementResult<Delivery, PlacementKey> commit(
            AbstractRequestScheduler owner, RequestContext context, RequestRoute routing,
            DecodeResources.ReservationHandle adopted) {
        var request = context.data().getRequest();
        DecodeResources.ReservationHandle pendingDecodeRollback = adopted != null ? adopted : routing.decodeEp() == null ? null
                : routing.decodeEp().tryReserveQueuedRequest(routing.decodePin(), routing.requestId(),
                        request.getHardKvTokens(), request.getExpectedKvTokens(), routing.priority(), context.data().getConfig().isQueue() ? owner.decodeAdmissionCapacity : null).handle();
        if (routing.decodeEp() != null && pendingDecodeRollback == null) {
            return PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
        }
        Throwable failure = null;
        RequestRoute acquired = null;
        var placement = context.snapshot().placement();
        var token = placement == null ? null : placement.operation();
        boolean delivered = false;
        boolean batch = context.data().getConfig().getDispatcher().requiresGenerateInput();
        DeliveryExecution.DeliveryPermit execution = null;
        try {
            if (batch) {
                var attempt = owner.runtime.deliveries().tryReserveDelivery();
                if (!attempt.accepted()) {
                    if (attempt.boundary().unavailable()) { return PlacementResult.blocked(PlacementKey.anyGroup(routing.prefill().getRole())); }
                    throw new IllegalStateException("batch delivery admission failed", attempt.boundary().cause());
                }
                execution = attempt.value();
            }
            RequestRoute route = acquired = RequestRoute.create(context, routing, pendingDecodeRollback);
            CapacityBoundary.Attempt<DeliveryDispatcher.Member> attempt = DeliveryDispatcher.Member.prepare(route, owner.decodeAdmissionCapacity);
            if (!attempt.accepted()) {
                return switch (attempt.boundary().status()) {
                    case UNAVAILABLE -> PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
                    case OWNERSHIP_LOST -> PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                    case FAILED -> throw new IllegalStateException("route admission failed", attempt.boundary().cause());
                };
            }
            DeliveryDispatcher.Member member = attempt.value();
            Throwable memberFailure = null;
            try {
                long batchId = batch ? owner.runtime.deliveries().nextBatchId() : 0L;
                var reservationAttempt = new PrefillReservationAttempt(route, batchId, owner.runtime.deliveries().maxInflightBatches);
                try {
                    PrefillState.CommittedHandoff handoff = null;
                    try {
                        PlacementResult.Status result = owner.commitRoute(route, reservationAttempt);
                        if (result != PlacementResult.Status.SUCCESS) {
                            return result == PlacementResult.Status.CLOSED ? PlacementResult.closed()
                                    : context.data().getConfig().isQueue() && reservationAttempt.result != null
                                            && reservationAttempt.result.status() == PrefillState.CapacityStatus.CAPACITY_FULL
                                    ? PlacementResult.blocked(routing.placementKey(RoleType.PREFILL))
                                    : PlacementResult.rejected(Response.error(reservationAttempt.failure()));
                        }
                        handoff = routing.prefillEp().commit(new PrefillState.CommitPlan(
                                batch ? PrefillState.CommitKind.UNQUEUED_BATCH : PrefillState.CommitKind.DIRECT, List.of(route),
                                batch ? List.of() : List.of((PrefillState.RouteReservation) reservationAttempt.reservation()),
                                batch ? (PrefillState.BatchReservation) reservationAttempt.reservation() : null,
                                batch ? new long[]{routing.prefillWorkMs()} : new long[0], null, System.currentTimeMillis()));
                        if (handoff == null) { return PlacementResult.rejected(Response.error(StrategyErrorType.DISPATCH_FAILED)); }
                        WorkSnapshot precedingWork = handoff.precedingWork().materialize();
                        if (batch) {
                            var command = new DeliveryDispatcher.Batch(batchId, List.of(member), precedingWork,
                                    routing.prefillWorkMs(), routing.prefillEp().predictor.evaluator(), "cold_dispatch",
                                    handoff.remainingQueueDepth(), handoff, execution);
                            delivered = true;
                            pendingDecodeRollback = null;
                            return PlacementResult.success(new Delivery(null, precedingWork, routing.prefillWorkMs(), command));
                        }
                        var claim = owner.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L,
                                member.decode());
                        if (claim == null) { return PlacementResult.closed(); }
                        delivered = true;
                        pendingDecodeRollback = null;
                        return PlacementResult.success(new Delivery(claim, precedingWork, routing.prefillWorkMs(), null));
                    } finally {
                        Throwable cleanup = batch && delivered ? null : DeliveryDispatcher.closeCommitted(List.of(), handoff);
                        if (cleanup != null) { owner.runtime.recordFailure(cleanup); }
                    }
                } finally {
                    if (reservationAttempt.reservation() != null) { routing.prefillEp().release(reservationAttempt.reservation(), null); }
                }
            } catch (RuntimeException | Error cause) {
                memberFailure = cause;
                throw cause;
            } finally {
                if (!batch || !delivered) {
                    Throwable cleanup = Failures.close(member);
                    if (memberFailure != null) { Failures.append(memberFailure, cleanup); }
                    else { Failures.rethrow(cleanup, "Decode admission cleanup failed"); }
                }
            }
        } catch (RuntimeException | Error cause) {
            failure = cause;
            throw cause;
        } finally {
            if (!delivered) {
                Throwable rollbackFailure = execution == null ? null : Failures.close(execution);
                try { if (acquired != null) { acquired.prefillEp().release(null, acquired); } }
                catch (RuntimeException | Error problem) { rollbackFailure = Failures.append(rollbackFailure, problem); }
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
                    if (failure == null) { Failures.rethrow(rollbackFailure, "immediate admission rollback failed"); }
                    else { Failures.append(failure, rollbackFailure); }
                }
            }
        }
    }

    /** Holds acquisition output without mixing a failed publication with capacity pressure. */
    private static final class PrefillReservationAttempt implements AbstractRequestScheduler.Publication {
        private final RequestRoute route;
        private final long batchId;
        private final int maxBatches;
        private PrefillState.ReservationResult<? extends PrefillState.Reservation> result;

        PrefillReservationAttempt(RequestRoute route, long batchId, int maxBatches) {
            this.route = route;
            this.batchId = batchId;
            this.maxBatches = maxBatches;
        }

        @Override
        public void publish() {
            result = batchId == 0L ? route.prefillEp().reserveUnqueuedRoute(route.prefillPin(), route, route.prefillWorkMs())
                    : route.prefillEp().reserveUnqueuedBatch(route.prefillPin(), route, batchId, maxBatches);
        }

        @Override public boolean published() {
            return result != null && result.status() == PrefillState.CapacityStatus.ACQUIRED;
        }

        PrefillState.Reservation reservation() {
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

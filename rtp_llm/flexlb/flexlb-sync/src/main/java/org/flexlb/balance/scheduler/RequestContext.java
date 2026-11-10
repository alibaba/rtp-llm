package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.CloseFacts;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryFacts;
import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineFacts;
import static org.flexlb.balance.scheduler.ContextResult.Identity;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionFacts;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.RoutingCompletion;
import static org.flexlb.balance.scheduler.ContextResult.RoutingPhase;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;

/**
 * Tracks atomic request facts, routing phases and whether a route can execute or only needs cleanup.
 * Scheduler settles results and Dispatcher performs transport outside this lock.
 */
public final class RequestContext {
    private static final AtomicLong INCARNATIONS = new AtomicLong();
    private static final Snapshot EMPTY =
            new Snapshot(
                    null,
                    0L,
                    null,
                    null,
                    RoutingPhase.NONE,
                    null,
                    new EngineFacts(false, 0L, false, 0L, false, null),
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    new CloseFacts(true, true, false, false, false),
                    new ExpirationTimer.Plan(0L, null, null, null),
                    null,
                    0L,
                    0L,
                    0L);

    private final RequestData data;

    private Identity key;
    private RouteResult route;
    private RoutingContext routingContext;
    private RoutingPhase routingPhase = RoutingPhase.NONE;
    private DeliveryFacts delivery;
    private EngineFacts engine = EMPTY.engine();
    private PreemptionFacts preemption;
    private CancellationFacts cancellation;
    private End outcome;
    private EndEvidence pendingEnd;
    private ContextResult.RouteFailure routeFailure;
    private ResponseCandidate selected;
    private CloseFacts close = EMPTY.close();
    private ExpirationTimer.Plan deadlines = EMPTY.deadlines();
    private RequestState terminalRecord;
    private Snapshot state;

    private long createdAtMs;
    private long lastWorkerAtMs;
    private long absoluteDeadline;
    private boolean requestDeadlineConsumed;
    private long inactivityMs;

    private StrategyErrorType timeoutError;

    public RequestContext(FlexlbConfig config) {
        data = new RequestData(config);
        state = EMPTY;
    }

    public RequestData data() {
        return data;
    }

    public synchronized void register(
            String requestId,
            long nowMs,
            long expiresAtMs,
            long inactivityTimeoutMs,
            StrategyErrorType timeoutError) {
        if (key != null) {
            return;
        }
        if (inactivityTimeoutMs <= 0L) {
            throw new IllegalArgumentException("positive inactivity timeout required");
        }
        this.timeoutError = Objects.requireNonNull(timeoutError, "timeout error required");
        key =
                new Identity(
                        requestId,
                        data.getOriginalNumericRequestId(),
                        data.getPhase(),
                        INCARNATIONS.incrementAndGet());
        createdAtMs = lastWorkerAtMs = nowMs;
        absoluteDeadline = expiresAtMs;
        inactivityMs = inactivityTimeoutMs;
        commitSnapshot(nowMs);
    }

    public synchronized Snapshot snapshot() {
        return state;
    }

    /**
     * Starts worker selection for a request that has no route and no other routing work in progress.
     */
    public synchronized RoutingContext beginRouteSelection() {
        var before = state;
        if (!canBeginRouting() || route != null) {
            return null;
        }
        routingContext = new RoutingContext(this, null);
        data.setSchedulingDiagnostics(null);
        routingPhase = RoutingPhase.SELECTING;
        commitSnapshot(before.updatedAtMs());
        return routingContext;
    }

    /**
     * Starts preempting the selected route before it is handed to an engine; resources are released later.
     */
    public synchronized RoutingContext beginRoutePreemption(RouteResult routeToPreempt) {
        Objects.requireNonNull(routeToPreempt, "routeToPreempt");
        var before = state;
        if (!canBeginRouting()
                || route != routeToPreempt
                || delivery != null
                || engine.decodeAccepted()) {
            return null;
        }
        routingContext = new RoutingContext(this, routeToPreempt);
        data.setSchedulingDiagnostics(null);
        routingPhase = RoutingPhase.REMOVING_ROUTE;
        commitSnapshot(before.updatedAtMs());
        return routingContext;
    }

    /**
     * Publish only after the endpoint accepted this route. Cancellation can reject this handoff.
     */
    public synchronized boolean publishRoute(RoutingContext exact, RouteResult acquired) {
        Objects.requireNonNull(acquired);
        if (!canPublishRoute(exact)
                || acquired.ctx() != this
                || !Objects.equals(acquired.requestId(), key.requestId())) return false;
        acceptRoute(acquired);
        commitSnapshot(state.updatedAtMs());
        return true;
    }

    public synchronized Snapshot completeRouting(
            RoutingContext exact, RoutingCompletion result, long nowMs) {
        if (routingContext == null || routingContext != exact) {
            return null;
        }
        com.google.common.base.Preconditions.checkArgument(
                result == RoutingCompletion.FAILED
                        || (exact.routeToWithdraw() == null
                                ? result == RoutingCompletion.FINISHED
                                : result == RoutingCompletion.RESTORED
                                        || result == RoutingCompletion.WITHDRAWN),
                "completion does not match the routing context");
        routingContext = null;
        routingPhase = RoutingPhase.NONE;
        if (result == RoutingCompletion.WITHDRAWN) {
            route = null;
            delivery = null;
            routeFailure = null;
            engine = new EngineFacts(false, 0L, false, 0L, false, null);
            close = close.withSettlement(true, true);
        }
        boolean rolledBack = route == null && close.prefillSettled() && close.decodeSettled();
        if (rolledBack && pendingEnd != null) {
            pendingEnd = pendingEnd.withRoute(null);
        }
        boolean admissionFailure =
                result == RoutingCompletion.FAILED && outcome == null && cancellation == null;
        if (admissionFailure && pendingEnd != null && pendingEnd.kind() == EndKind.SHUTDOWN) {
            pendingEnd = null;
        }
        // Retirement of a rolled-back preparation must not end the request: no route
        // was delivered, so the scheduler can select another endpoint. Keep all other endings.
        boolean unpublishedRollback = rolledBack && delivery == null && !engine.decodeAccepted();
        if (unpublishedRollback && pendingEnd != null && pendingEnd.kind() == EndKind.RETIREMENT) {
            pendingEnd = null;
        }
        return commitSnapshot(nowMs);
    }

    public synchronized Snapshot recordDeliveryResult(DeliveryEvidence evidence) {
        if (!(evidence instanceof TransportResult) && !(evidence instanceof RouteDelivered))
            throw new IllegalArgumentException("expected delivery completion");
        if (!matchesDelivery(evidence.delivery())) return null;
        if (evidence instanceof RouteDelivered delivered) {
            if (outcome != null) return null;
            if (delivery.identity().kind() != Delivery.SendMode.ROUTE_DECISION)
                throw new IllegalArgumentException(
                        "route completion requires route delivery identity");
            delivery = delivery.withSend(SendState.DELIVERED);
            engine = engine.predict(delivered.delivery(), delivered.predictedAtMs());
        } else {
            var transport = (TransportResult) evidence;
            var recorded =
                    delivery.recordTransport(
                            transport.result(),
                            cancellation == null
                                    ? CancelReason.CLIENT_CANCELLED
                                    : cancellation.reason());
            if (recorded == delivery) return state;
            delivery = recorded;
            if (transport.result().failed())
                retainEnd(EndEvidence.deliveryFailure(route, transport));
        }
        return commitSnapshot(evidence.nowMs());
    }

    public synchronized Snapshot recordDeliveryPrediction(Prediction prediction) {
        if (!matchesDelivery(prediction.delivery()) || outcome != null) return null;
        engine = engine.predict(prediction.delivery(), prediction.predictedAtMs());
        return commitSnapshot(prediction.nowMs());
    }

    public synchronized Snapshot recordRemoteCleanup(ContextResult.CleanupAcknowledgement cleanup) {
        if (!matchesDelivery(cleanup.delivery())) return null;
        var kind = cleanup.kind();
        delivery =
                delivery.mergeReleaseProof(
                        false,
                        kind == ContextResult.CancelEvidenceKind.CLEANED
                                || kind == ContextResult.CancelEvidenceKind.FENCED,
                        kind == ContextResult.CancelEvidenceKind.CLEANED);
        return commitSnapshot(cleanup.nowMs());
    }

    public synchronized Snapshot recordPreparationFailure(PreparationFailure failure) {
        if (failure.delivery() == null) {
            if (route != failure.route() || outcome != null || delivery != null) return null;
        } else {
            if (!matchesDelivery(failure.delivery())) return null;
            if (delivery.send() == SendState.SENDING)
                throw new IllegalStateException("preparation failure crossed send boundary");
            if (delivery.send() == SendState.NOT_STARTED)
                delivery = delivery.withSend(SendState.NOT_SENT);
        }
        retainEnd(EndEvidence.deliveryFailure(route, failure));
        return commitSnapshot(failure.nowMs());
    }

    public synchronized Snapshot recordWorkerRequestEvent(WorkerRequestEvent requestEvent) {
        if (!matches(requestEvent.route(), requestEvent.reservation(), requestEvent.role())
                || terminalRecord != null) {
            return null;
        }
        boolean isDecode = requestEvent.role() == RoleType.DECODE;
        engine = engine.observe(requestEvent, route.decodeEndpoint() != null);
        lastWorkerAtMs = Math.max(lastWorkerAtMs, requestEvent.nowMs());
        if (preemption != null
                && !preemption.finished()
                && !preemption.restorationObserved()
                && (isDecode
                        ? requestEvent.type() == WorkerRequestEventType.ALLOCATED
                        : requestEvent.type() == WorkerRequestEventType.ACTIVE)) {
            preemption = preemption.withRestorationObserved(true);
        }
        if (!requestEvent.terminal()) return commitSnapshot(requestEvent.nowMs());
        recordWorkerResourceRelease(requestEvent.role());
        boolean ending =
                isDecode
                        || requestEvent.role() == RoleType.PDFUSION
                        || requestEvent.role() == RoleType.ENCODER
                        || requestEvent.type() == WorkerRequestEventType.FAILED
                        || requestEvent.type() == WorkerRequestEventType.PRIORITY_CANCELLED
                                && preemption != null
                                && !preemption.finished()
                                && preemption.phase().acceptsPriorityTerminal();
        if (outcome == null && ending) retainEnd(requestEvent.end());
        return commitSnapshot(requestEvent.nowMs());
    }

    public synchronized Snapshot recordEndpointRetirement(
            ContextResult.EndpointRetirement retired) {
        if (!isCurrentEndpointRetirement(retired)) return null;
        boolean isDecode = retired.role() == RoleType.DECODE;
        recordWorkerResourceRelease(retired.role());
        boolean ending =
                isDecode
                        || retired.role() == RoleType.ENCODER
                        || delivery == null && !engine.decodeAccepted() && preemption == null;
        if (routingContext != null
                && ending
                && (!isDecode || delivery == null && !engine.decodeAccepted())) {
            // An unsent route can still roll back and retry. Keep retirement as the
            // pending end, below any other end already accepted for this routing work.
            if (outcome == null && pendingEnd == null) pendingEnd = retired.end();
            return commitSnapshot(retired.nowMs());
        }
        if (outcome == null && ending) retainEnd(retired.end());
        return commitSnapshot(retired.nowMs());
    }

    public synchronized boolean requestCancel(Cancellation request) {
        var before = state;
        if (key == null) {
            return false;
        }
        Objects.requireNonNull(request.reason());
        long batchId = delivery == null ? 0L : delivery.identity().batchId();
        if (request.expectedBatchId() != 0L && request.expectedBatchId() != batchId) {
            return false;
        }
        if (outcome != null) {
            return false;
        }
        if (cancellation != null) {
            return false;
        }
        if (request.reason() == CancelReason.DEADLINE_EXCEEDED && delivery != null) {
            return false;
        }
        boolean beforeSend = delivery == null;
        if (request.frontend() && (selected != null || !beforeSend)) {
            return false;
        }
        StrategyErrorType error =
                Objects.requireNonNull(request.errorType(), "cancellation error required");
        cancellation =
                new CancellationFacts(
                        request.reason(),
                        error,
                        request.detail() == null
                                ? request.reason().getMessage()
                                : request.detail());
        if (delivery != null && !DeliveryFacts.releasable(delivery)) {
            var old = delivery;
            var send =
                    old.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                                    && old.send() == SendState.NOT_STARTED
                            ? SendState.ROUTE_CLOSED
                            : old.send();
            delivery = old.abandon(send, request.reason());
        }
        retainEnd(
                EndEvidence.failure(
                        route, EndKind.FRONTEND, cancellation.errorType(), cancellation.detail()));
        return commitSnapshot(request.nowMs()) != before;
    }

    public synchronized Snapshot recordDeadline(
            ExpirationTimer.Token exact, long nowMs, CancellationFacts diagnosis) {
        if (exact.owner() != this
                || deadlines.token(exact.kind()) != exact
                || terminalRecord != null) {
            return null;
        }
        if (nowMs < exact.atMs()) {
            deadlines = deadlines.without(exact.kind());
            return commitSnapshot(nowMs);
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            engine = engine.expireVisibility();
            return commitSnapshot(nowMs);
        }
        boolean inactive = exact.kind() == ExpirationTimer.Kind.INACTIVITY;
        if (inactive) {
            deadlines = deadlines.without(exact.kind());
            long activity =
                    Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
            if (nowMs - activity < inactivityMs) return commitSnapshot(nowMs);
            close = close.withExpired(true);
        } else {
            requestDeadlineConsumed = true;
            if (outcome != null || delivery != null && routingContext == null)
                return commitSnapshot(nowMs);
        }
        String detail = inactive ? "request inactive" : "request scheduling deadline exceeded";
        if (cancellation == null && outcome == null) {
            cancellation =
                    new CancellationFacts(
                            CancelReason.DEADLINE_EXCEEDED,
                            diagnosis == null ? timeoutError : diagnosis.errorType(),
                            diagnosis == null ? detail : diagnosis.detail());
        }
        if (inactive || routingContext != null || delivery == null && !engine.decodeAccepted()) {
            retainEnd(EndEvidence.failure(route, EndKind.INACTIVITY, timeoutError, detail));
        }
        return commitSnapshot(nowMs);
    }

    public synchronized PreemptionRegistration beginPreemption(
            DecodeResources.RequestReservation exact, long attempt, CancelTarget target) {
        var before = state;
        if (outcome != null
                || route == null
                || cancellation != null
                || preemption != null
                || !Objects.equals(route.decodeReservation(), exact)) {
            return null;
        }
        if (routingContext != null) {
            return null;
        }
        if (delivery != null
                && delivery.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                && !delivery.acknowledged()) {
            return null;
        }
        var registration = new PreemptionRegistration(this, attempt, exact, target);
        preemption =
                new PreemptionFacts(
                        registration, PreemptionCancelPhase.CLAIMED, false, false, null, false);
        commitSnapshot(before.updatedAtMs());
        return registration;
    }

    public synchronized boolean canReconcilePreemption(PreemptionRegistration exact) {
        if (preemption == null || preemption.identity() != exact || terminalRecord != null) {
            return false;
        }
        if (preemption.finished()) {
            return false;
        }
        var terminal = preemption.pendingTerminal();
        boolean authoritative =
                terminal != null
                        && terminal.kind() == EndKind.WORKER
                        && (terminal.role() == RoleType.DECODE
                                || terminal.role() == RoleType.PDFUSION
                                || !terminal.success()
                                || engine.decodeAccepted());
        boolean priorityProof =
                terminal != null
                        && terminal.kind() == EndKind.PRIORITY
                        && preemption.phase().acceptsPriorityTerminal();
        boolean observed = preemption.restorationObserved();
        boolean eligible =
                authoritative
                        || priorityProof
                        || preemption.phase() == PreemptionCancelPhase.NOT_FOUND_STALE
                                && (terminal == null
                                        ? (state.canConfirmDelivery() && !delivery.acknowledged())
                                                || observed
                                        : observed || delivery != null && delivery.decodeProof());
        return eligible;
    }

    public synchronized boolean completePreemption(
            PreemptionRegistration exact, PreemptionReport report, long nowMs) {
        var before = state;
        if (preemption == null || preemption.identity() != exact || terminalRecord != null) {
            return false;
        }
        var previous = preemption;
        if (previous.finished() && report.phase() != null) {
            return false;
        }
        var next = report.phase() == null ? previous.phase() : report.phase();
        if (report.phase() != null
                && next != previous.phase()
                && !previous.phase().canTransitionTo(next)) {
            return false;
        }
        var terminal = report.terminal() == null ? previous.pendingTerminal() : report.terminal();
        preemption = previous.recordReport(next, terminal, report.ledgerCommitted());
        if (report.ledgerCommitted()) {
            if (delivery != null && terminal != null) {
                delivery =
                        delivery.mergeReleaseProof(
                                terminal.kind() == EndKind.WORKER,
                                terminal.prefillSettled(),
                                report.decodeSettled());
            }
            close =
                    close.mergeSettlement(
                            terminal != null && terminal.prefillSettled(),
                            report.decodeSettled() || terminal != null && terminal.decodeSettled());
            if (terminal != null) {
                retainEnd(terminal);
            } else {
                preemption = null;
            }
        } else if (report.releaseAttempt() && next.isLocallyReleasable()) {
            preemption = null;
            if (terminal != null) {
                retainEnd(terminal);
            }
        }
        return commitSnapshot(nowMs) != before;
    }

    public synchronized boolean selectResponse(ResponseCandidate candidate) {
        if (candidate.owner() != this) {
            return false;
        }
        if (key == null) {
            return false;
        }
        if (selected != null) {
            return selected == candidate;
        }
        if (terminalRecord != null && candidate.kind() != ResponseKind.TERMINAL) {
            return false;
        }
        if (candidate.kind() == ResponseKind.ACK) {
            if (!confirmDelivery(candidate.nowMs())) return false;
        } else if (candidate.kind() == ResponseKind.TERMINAL && outcome == null
                || candidate.kind() == ResponseKind.CANCELLATION && cancellation == null) {
            return false;
        }
        selected = candidate;
        commitSnapshot(candidate.nowMs());
        return true;
    }

    public synchronized Snapshot beginClose(EndEvidence evidence, long nowMs) {
        var before = state;
        if (key == null) {
            return before;
        }
        if (terminalRecord != null) {
            return before;
        }
        if (evidence != null && evidence.route() != null && evidence.route() != route) {
            return before;
        }
        if (evidence != null) {
            retainEnd(evidence);
        }
        EndKind ending = pendingEnd == null ? null : pendingEnd.kind();
        if (waitsForPlacement(ending) || outcome == null && waitsForPreemption(ending)) {
            return commitSnapshot(nowMs);
        }
        if (outcome == null && pendingEnd == null) {
            return before;
        }
        if (outcome == null) {
            outcome = classify(pendingEnd);
            close = close.beginEnding(pendingEnd, engine.prefillCompletedAtMs() > 0L);
            if (delivery != null) {
                var old = delivery;
                boolean success =
                        pendingEnd.kind() == EndKind.WORKER
                                && pendingEnd.success()
                                && cancellation == null;
                var abandoned =
                        success || DeliveryFacts.releasable(delivery)
                                ? old.abandonment()
                                : old.abandonment() != null
                                        ? old.abandonment()
                                        : cancellation == null
                                                ? CancelReason.CLIENT_CANCELLED
                                                : cancellation.reason();
                delivery = old.recordEnd(abandoned, success);
            }
            if (preemption != null
                    && (close.expired()
                            || pendingEnd.kind() == EndKind.RETIREMENT
                                    && pendingEnd.decodeSettled())) {
                preemption = preemption.withFinished(true);
            }
            pendingEnd = null;
        }
        return commitSnapshot(nowMs);
    }

    public synchronized DecodeResources.ReleaseReason releaseReason() {
        if (outcome == null
                || terminalRecord != null
                || routingContext != null
                || !DeliveryFacts.releasable(delivery)) {
            return null;
        }
        return delivery != null && delivery.send() == SendState.NOT_SENT
                ? DecodeResources.ReleaseReason.NOT_SENT
                : delivery != null && delivery.prefillProof() && delivery.decodeProof()
                        ? DecodeResources.ReleaseReason.REMOTE_CLEANUP
                        : close.expired()
                                ? DecodeResources.ReleaseReason.EXPIRED
                                : DecodeResources.ReleaseReason.COUNTERPART_FINISHED;
    }

    /**
     * Validate the exact route and record expiry before Dispatcher transfers dispatch capacity.
     * Dispatcher retains this monitor until it records the accepted delivery.
     */
    synchronized boolean validateDelivery(RouteResult exact, RoutingContext selecting, long nowMs) {
        if (selecting == null
                ? route != exact || !state.preparedDelivery()
                : !canPublishRoute(selecting)
                        || exact.ctx() != this
                        || !Objects.equals(exact.requestId(), key.requestId())) return false;
        if (absoluteDeadline > 0L && nowMs >= absoluteDeadline
                || nowMs - lastWorkerAtMs >= inactivityMs) {
            if (cancellation == null) {
                cancellation =
                        new CancellationFacts(
                                CancelReason.DEADLINE_EXCEEDED,
                                timeoutError,
                                "request expired before delivery");
            }
            retainEnd(
                    EndEvidence.failure(
                            route, EndKind.INACTIVITY, timeoutError, cancellation.detail()));
            commitSnapshot(nowMs);
            return false;
        }
        return true;
    }

    synchronized boolean canPublishRoute(RoutingContext exact) {
        return key != null
                && outcome == null
                && cancellation == null
                && pendingEnd == null
                && route == null
                && routingContext != null
                && routingContext == exact
                && exact.requestContext() == this
                && routingPhase == RoutingPhase.SELECTING;
    }

    /**
     * Only the exact route's cleanup may advance its resource release facts.
     */
    synchronized void recordResourcesReleased(
            RouteResult exact, boolean prefillReleased, boolean decodeReleased, long nowMs) {
        if (route != exact || outcome == null || terminalRecord != null) return;
        close = close.mergeSettlement(prefillReleased, decodeReleased);
        completeClose(nowMs);
    }

    /**
     * Record completed stop/deadline notifications independently of resource release.
     */
    synchronized void recordCloseNotifications(boolean attempted, boolean completed, long nowMs) {
        if (outcome == null || terminalRecord != null) return;
        close =
                new CloseFacts(
                        close.prefillSettled(),
                        close.decodeSettled(),
                        close.notificationsDone()
                                || completed && (preemption == null || preemption.finished()),
                        close.expired(),
                        close.notificationsAttempted() || attempted);
        completeClose(nowMs);
    }

    /**
     * Record an unsent route failure once. Active routing finishes before withdrawal starts.
     */
    synchronized boolean failUnsentRoute(
            ContextResult.RouteFailure failure, DeliveryEvidence evidence) {
        if (route != failure.route()) return false;
        var beforeDelivery = delivery;
        if (evidence instanceof TransportResult transport
                && transport.result().status() == Delivery.Result.Status.NOT_SENT
                && delivery != null
                && delivery.identity() == transport.delivery()
                && (delivery.send() == SendState.SENDING
                        || delivery.send() == SendState.NOT_STARTED))
            delivery = delivery.withSend(SendState.NOT_SENT);
        if (!canWithdrawRoute()) {
            delivery = beforeDelivery;
            return false;
        }
        if (routeFailure == null) routeFailure = failure;
        commitSnapshot(System.currentTimeMillis());
        return true;
    }

    synchronized RoutingContext beginFailedRouteWithdrawal() {
        if (routeFailure == null
                || routingContext != null
                || preemption != null
                || !canWithdrawRoute()) return null;
        var operation = new RoutingContext(this, route);
        routingContext = operation;
        routingPhase = RoutingPhase.REMOVING_ROUTE;
        commitSnapshot(System.currentTimeMillis());
        return operation;
    }

    synchronized ContextResult.RouteFailure unsentDeliveryFailure(DeliveryEvidence evidence) {
        RouteResult failedRoute;
        String detail;
        if (evidence instanceof PreparationFailure preparation && preparation.delivery() == null) {
            failedRoute = preparation.route();
            detail = preparation.detail();
        } else if (evidence instanceof TransportResult transport
                && transport.result().status() == Delivery.Result.Status.NOT_SENT) {
            if (delivery == null || delivery.identity() != transport.delivery()) return null;
            failedRoute = transport.delivery().route();
            detail = "Transport confirmed that the request was not sent";
        } else return null;
        if (route != failedRoute) return null;
        return new ContextResult.RouteFailure(
                failedRoute,
                failedRoute.selectedEndpointsAvailable()
                        ? ContextResult.RouteFailureReason.PREPARATION_FAILED
                        : ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                detail);
    }

    /**
     * Keep failed local cleanup reachable by the request's asynchronous cleanup task.
     * This does not permit delivery or publish a selection result.
     */
    synchronized void retainRouteForCleanup(RoutingContext exact, RouteResult unreleased) {
        if (routingContext == null || routingContext != exact || route != null) return;
        route = unreleased;
        close = close.withSettlement(false, unreleased.decodeReservation() == null);
        retainEnd(
                EndEvidence.failure(
                        unreleased,
                        EndKind.FAILURE,
                        StrategyErrorType.DISPATCH_FAILED,
                        "routing rollback failed"));
        commitSnapshot(state.updatedAtMs());
    }

    /**
     * Claim completion before resource cleanup, keeping routingContext visible until cleanup finishes.
     */
    synchronized boolean beginRoutingCompletion(RoutingContext exact) {
        if (routingContext == null || routingContext != exact) return false;
        if (routingPhase == RoutingPhase.SELECTING) {
            routingPhase = RoutingPhase.FINISHING_SELECTION;
        } else if (routingPhase == RoutingPhase.REMOVING_ROUTE) {
            routingPhase = RoutingPhase.FINISHING_ROUTE_REMOVAL;
        } else return false;
        commitSnapshot(state.updatedAtMs());
        return true;
    }

    synchronized void recordRoutingDiagnostics(
            RoutingContext exact, java.util.Map<String, Object> diagnostics) {
        if (routingContext != null
                && routingContext == exact
                && (routingPhase == RoutingPhase.SELECTING
                        || routingPhase == RoutingPhase.REMOVING_ROUTE)
                && cancellation == null) {
            data.setSchedulingDiagnostics(
                    diagnostics == null ? null : java.util.Map.copyOf(diagnostics));
        }
    }

    /**
     * Accept a direct route together with its committed delivery, without publishing a provisional route.
     */
    synchronized void recordDirectDeliveryStarted(
            RoutingContext selecting, Delivery.Member member, long nowMs) {
        com.google.common.base.Preconditions.checkState(
                canPublishRoute(selecting),
                "direct delivery requires its active selection under the Context lock");
        acceptRoute(member.route());
        recordDeliveryStarted(member, nowMs);
    }

    /**
     * Record the delivery after Dispatcher transfers capacity while holding this monitor.
     * No provisional request facts escape to the transport layer.
     */
    void recordDeliveryStarted(Delivery.Member member, long nowMs) {
        com.google.common.base.Preconditions.checkState(
                Thread.holdsLock(this), "delivery commit requires Context lock");
        delivery =
                new DeliveryFacts(
                        member,
                        member.kind() == Delivery.SendMode.BATCH_ENQUEUE
                                ? SendState.SENDING
                                : SendState.NOT_STARTED,
                        false,
                        null,
                        false,
                        false,
                        route.decodeEndpoint() == null || route.decodeReservation() == null,
                        nowMs);
        commitSnapshot(nowMs);
    }

    synchronized boolean isCurrentEndpointRetirement(ContextResult.EndpointRetirement retired) {
        if (!matches(retired.route(), retired.reservation(), retired.role())
                || terminalRecord != null
                || retired.endpoint() == null) return false;
        return switch (retired.role()) {
            case DECODE -> route.decodeEndpoint() == retired.endpoint();
            case ENCODER -> route.encoderEndpoint() == retired.endpoint();
            case PREFILL, PDFUSION -> route.prefillEndpoint() == retired.endpoint();
            default -> false;
        };
    }

    /**
     * A transport receipt does not select or publish the frontend response.
     */
    synchronized boolean confirmDelivery(long nowMs) {
        if (!state.canConfirmDelivery()) return false;
        if (preemption != null && !preemption.finished() && !delivery.acknowledged()) return false;
        long activity = Math.max(lastWorkerAtMs, delivery.startedAtMs());
        if (nowMs - activity >= inactivityMs) {
            cancellation =
                    new CancellationFacts(
                            CancelReason.DEADLINE_EXCEEDED,
                            timeoutError,
                            "request inactive before confirmation");
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            timeoutError,
                            "request inactive before confirmation"));
            commitSnapshot(nowMs);
            return false;
        }
        if (!delivery.acknowledged()) {
            delivery = delivery.withAcknowledged(true);
        }
        commitSnapshot(nowMs);
        return true;
    }

    /**
     * Publisher capacity is reserved by Scheduler before freezing the outcome.
     */
    synchronized boolean readyToClose(EndKind ending) {
        return !waitsForPlacement(ending) && !waitsForPreemption(ending);
    }

    private boolean matchesDelivery(Delivery.Member exact) {
        return delivery != null && delivery.identity() == exact && terminalRecord == null;
    }

    private void acceptRoute(RouteResult acquired) {
        route = acquired;
        data.setSchedulingDiagnostics(null);
        close = close.withSettlement(false, acquired.decodeReservation() == null);
    }

    private void completeClose(long nowMs) {
        if (delivery != null
                && delivery.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                && delivery.send() == SendState.NOT_STARTED) {
            delivery = delivery.withSend(SendState.ROUTE_CLOSED);
        }
        if (preemption != null
                && close.decodeSettled()
                && (close.expired()
                        || delivery == null
                        || delivery.send() == SendState.NOT_SENT
                        || delivery.abandonment() != null
                                && delivery.prefillProof()
                                && delivery.decodeProof())) {
            preemption = preemption.withFinished(true);
        }
        if (close.prefillSettled()
                && close.decodeSettled()
                && close.notificationsDone()
                && DeliveryFacts.releasable(delivery)
                && routingContext == null
                && (preemption == null || preemption.finished())) {
            terminalRecord =
                    new RequestState(
                            key.requestId(),
                            key.originalNumericRequestId(),
                            key.phase(),
                            outcome.phase(),
                            delivery == null ? Delivery.SendMode.NONE : delivery.identity().kind(),
                            delivery == null ? 0L : delivery.identity().batchId(),
                            createdAtMs,
                            nowMs,
                            outcome.detail());
            route = null;
            preemption = null;
            pendingEnd = null;
        }
        commitSnapshot(nowMs);
    }

    private Snapshot commitSnapshot(long nowMs) {
        long inactivityAtMs =
                com.google.common.math.LongMath.saturatedAdd(
                        Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs()),
                        inactivityMs);
        if (key != null) {
            deriveDeadlines(inactivityAtMs);
        }

        // An unchanged observation preserves snapshot identity and does not allocate a wrapper.
        if (Objects.equals(key, state.key())
                && createdAtMs == state.createdAtMs()
                && route == state.route()
                && Objects.equals(routingContext, state.routingContext())
                && routingPhase == state.routingPhase()
                && Objects.equals(delivery, state.delivery())
                && Objects.equals(engine, state.engine())
                && Objects.equals(preemption, state.preemption())
                && Objects.equals(cancellation, state.cancellation())
                && Objects.equals(outcome, state.outcome())
                && Objects.equals(pendingEnd, state.pendingEnd())
                && Objects.equals(routeFailure, state.routeFailure())
                && Objects.equals(selected, state.selectedResponse())
                && Objects.equals(close, state.close())
                && Objects.equals(deadlines, state.deadlines())
                && Objects.equals(terminalRecord, state.terminalRecord())
                && lastWorkerAtMs == state.lastWorkerAtMs()) {
            return state;
        }
        state =
                new Snapshot(
                        key,
                        createdAtMs,
                        route,
                        routingContext,
                        routingPhase,
                        delivery,
                        engine,
                        preemption,
                        cancellation,
                        outcome,
                        pendingEnd,
                        routeFailure,
                        selected,
                        close,
                        deadlines,
                        terminalRecord,
                        lastWorkerAtMs,
                        Math.max(state.updatedAtMs(), nowMs),
                        inactivityAtMs);
        return state;
    }

    private void deriveDeadlines(long inactivityAtMs) {
        long requestAt =
                outcome == null
                                && !requestDeadlineConsumed
                                && (delivery == null || !delivery.acknowledged())
                        ? absoluteDeadline
                        : 0L;
        long visibilityAt = outcome == null ? engine.visibilityAtMs() : 0L;
        boolean settled =
                outcome != null
                        && terminalRecord == null
                        && close.prefillSettled()
                        && close.decodeSettled()
                        && close.notificationsDone()
                        && DeliveryFacts.releasable(delivery)
                        && routingContext == null
                        && (preemption == null || preemption.finished());
        long inactivityAt =
                terminalRecord != null || close.expired() || settled ? 0L : inactivityAtMs;
        deadlines = deadlines.schedule(this, requestAt, visibilityAt, inactivityAt);
    }

    private boolean canBeginRouting() {
        return key != null
                && outcome == null
                && cancellation == null
                && pendingEnd == null
                && routingContext == null
                && preemption == null;
    }

    private boolean canWithdrawRoute() {
        return outcome == null
                && cancellation == null
                && pendingEnd == null
                && selected == null
                && preemption == null
                && !engine.decodeAccepted()
                && !engine.prefillObserved()
                && (delivery == null
                        || delivery.identity().kind() == Delivery.SendMode.BATCH_ENQUEUE
                                && (delivery.send() == SendState.NOT_STARTED
                                        || delivery.send() == SendState.NOT_SENT));
    }

    private void recordWorkerResourceRelease(RoleType role) {
        boolean isDecode = role == RoleType.DECODE;
        if (delivery != null) {
            delivery =
                    delivery.mergeReleaseProof(
                            isDecode || role == RoleType.PDFUSION, !isDecode, isDecode);
        }
        close = close.mergeSettlement(!isDecode, isDecode);
    }

    private boolean waitsForPlacement(EndKind ending) {
        return routingContext != null
                && !(ending == EndKind.DELIVERY_FAILURE && cancellation == null);
    }

    private void retainEnd(EndEvidence evidence) {
        if (outcome != null) {
            close = close.retainEnd(evidence);
            return;
        }
        if (routingContext != null) {
            // A preparation failure can settle before routing returns; other ends wait for
            // it.
            boolean immediateFailure =
                    evidence.kind() == EndKind.DELIVERY_FAILURE && cancellation == null;
            boolean previousFailure =
                    pendingEnd != null
                            && pendingEnd.kind() == EndKind.DELIVERY_FAILURE
                            && cancellation == null;
            if (immediateFailure
                    || !previousFailure
                            && (pendingEnd == null
                                    || pendingEnd.kind() == EndKind.RETIREMENT
                                    || evidence.kind() == EndKind.WORKER
                                            && pendingEnd.kind() != EndKind.WORKER
                                    || evidence.kind() == EndKind.RETIREMENT
                                            && pendingEnd.kind() != EndKind.WORKER)) {
                pendingEnd = evidence;
            }
            return;
        }
        if (waitsForPreemption(evidence.kind())) {
            var previous = preemption.pendingTerminal();
            boolean better =
                    previous == null
                            || evidence.kind() == EndKind.WORKER
                                    && previous.kind() != EndKind.WORKER;
            preemption = preemption.withPendingTerminal(better ? evidence : previous);
            return;
        }
        if (pendingEnd == null
                || evidence.kind() == EndKind.WORKER && pendingEnd.kind() != EndKind.WORKER) {
            pendingEnd = evidence;
        }
    }

    private boolean waitsForPreemption(EndKind ending) {
        return preemption != null
                && !preemption.finished()
                && !(ending == EndKind.DELIVERY_FAILURE && cancellation == null)
                && ending != EndKind.FRONTEND
                && ending != EndKind.INACTIVITY
                && ending != EndKind.RETIREMENT;
    }

    private End classify(EndEvidence evidence) {
        boolean cancelled =
                cancellation != null && (evidence.kind() != EndKind.FAILURE || delivery == null);
        if (cancelled || evidence.kind() == EndKind.INACTIVITY) {
            var cause =
                    cancellation == null ? CancelReason.DEADLINE_EXCEEDED : cancellation.reason();
            String detail = cancellation == null ? evidence.detail() : cancellation.detail();
            return new End(
                    evidence.kind(),
                    cause == CancelReason.DEADLINE_EXCEEDED
                            ? RequestState.Phase.TIMED_OUT
                            : RequestState.Phase.CANCELLED,
                    cancellation == null ? timeoutError : cancellation.errorType(),
                    detail);
        }
        if (evidence.kind() == EndKind.WORKER && evidence.success()
                || evidence.kind() == EndKind.FRONTEND && evidence.success()) {
            String detail =
                    evidence.detail() == null || evidence.detail().isBlank()
                            ? "request completed"
                            : evidence.detail();
            return new End(evidence.kind(), RequestState.Phase.COMPLETED, null, detail);
        }
        var error =
                evidence.errorType() != null
                        ? evidence.errorType()
                        : evidence.kind() == EndKind.PRIORITY
                                ? StrategyErrorType.PRIORITY_PREEMPTED
                                : evidence.kind() == EndKind.WORKER
                                        ? StrategyErrorType.WORKER_EXECUTION_FAILED
                                        : StrategyErrorType.DISPATCH_FAILED;
        String detail =
                evidence.detail() == null || evidence.detail().isBlank()
                        ? evidence.kind() == EndKind.WORKER
                                ? "worker error code " + evidence.errorCode()
                                : evidence.kind() == EndKind.PRIORITY
                                        ? "priority victim canceled by worker"
                                        : "request ended"
                        : evidence.detail();
        return new End(
                evidence.kind(),
                evidence.kind() == EndKind.PRIORITY
                        ? RequestState.Phase.CANCELLED
                        : RequestState.Phase.FAILED,
                error,
                detail);
    }

    private boolean matches(
            RouteResult exact, DecodeResources.RequestReservation reservation, RoleType role) {
        if (route == null
                || route != exact
                || exact.ctx() != this
                || role == null
                || role != RoleType.DECODE && role != RoleType.ENCODER && !role.supportsPrefill()) {
            return false;
        }
        return role != RoleType.DECODE || Objects.equals(route.decodeReservation(), reservation);
    }
}

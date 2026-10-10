package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.CleanupNext;
import org.flexlb.balance.scheduler.RequestContext.CleanupPass;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.balance.scheduler.RequestContext.DeliveryPublication;
import org.flexlb.balance.scheduler.RequestContext.PendingPrefillRetirement;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestContext.PublicationKind;
import org.flexlb.balance.scheduler.RequestContext.RequestFuture;
import org.flexlb.balance.scheduler.RequestContext.RequestEndDecision;
import org.flexlb.balance.scheduler.RequestContext.ResponseCompletion;
import org.flexlb.balance.scheduler.RequestContext.ResponseResult;
import org.flexlb.balance.scheduler.ExpirationTimer.DecisionDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.RequestDeadline;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/** Shared request protocol; concrete schedulers own their mode-specific placement algorithm. */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    private static final AtomicLong WORKER_ENQUEUE_SEQUENCE = new AtomicLong();

    static void initializeWorkerQueue(RequestContext context, long nowMs) {
        context.initializeWorkerQueue(nowMs, WORKER_ENQUEUE_SEQUENCE::incrementAndGet);
    }

    protected final RequestRepository requests;
    protected final SchedulerRuntime runtime;
    protected final FlexlbConfig config;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;
    private final ResponseCompletionExecutor responseCompletions;
    private final RequestContinuationExecutor continuations;
    private final ExpirationTimer expirationTimer;
    private final Object admissionQuiescenceMonitor = new Object();
    private int inFlightAdmissionHandles;
    private final RequestSchedulerReporter requestReporter;

    protected AbstractRequestScheduler(SchedulerRuntime runtime, FlexlbConfig config) {
        this.runtime = Objects.requireNonNull(runtime);
        this.config = Objects.requireNonNull(config);
        this.requests = runtime.requests();
        this.recentCacheKeyTraceReporter = runtime.recentCacheKeyTraceReporter();
        this.responseCompletions = runtime.responseCompletions();
        this.continuations = runtime.continuations();
        this.expirationTimer = runtime.timer();
        this.requestReporter = runtime.requestReporter();
    }

    /** Stop mode-specific placement before the runtime drains shared work. */
    void closePlacement() { }

    private boolean isCurrentContext(RequestContext exact) {
        return exact != null && exact.scheduler() == this && requests.isCurrent(exact);
    }
    RequestContext findRequestContext(long requestId) {
        RequestContext exact = requests.findActive(requestId);
        return exact != null && exact.scheduler() == this ? exact : null;
    }
    private List<RequestContext> ownedRequests() {
        return requests.snapshotActive().stream().filter(context -> context.scheduler() == this).toList();
    }
    final void recordFailure(Throwable cause) { runtime.recordFailure(cause); }

    @Override
    public final RequestState cancel(long requestId, long batchId, CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        RequestContext context = requests.findActive(requestId);
        if (context != null) {
            return context.scheduler() == this ? cancelRequest(context, batchId, reason) : null;
        }
        RequestRepository.TerminalRecord terminal = requests.findTerminal(requestId);
        return terminal != null && terminal.owner() == this && terminal.state().matchesBatch(batchId)
                ? terminal.state() : null;
    }

    protected final CompletableFuture<Response> register(RequestContext context, StrategyErrorType expiredError) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        var config = context.getConfig();
        FlexlbTrace.setScheduleAttribute(context.getTraceContext(), FlexlbTrace.SCHEDULE_MODE,
                config.getDispatcher().requiresGenerateInput() ? "BATCH" : config.getScheduler().getType().name());
        if (config.getDispatcher().requiresGenerateInput()
                && !context.hasGenerateInput()) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST, "missing serialized generate_input for batch dispatch"));
        }
        CompletableFuture<Response> future = registerRequest(context, expiredError);
        if (!(context.getFuture() instanceof RequestContext.RequestFuture)) { context.setFuture(future); }
        registerResponseCallback(context, future);
        return future;
    }

    protected final CompletableFuture<Response> failSubmission(RequestContext context, RuntimeException failure) {
        Response response = Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
        if (context == null || context.scheduler() != this) {
            return CompletableFuture.completedFuture(response);
        }
        terminateLocallyAndPublishResponse(context, response);
        return context.getFuture();
    }

    protected static CompletableFuture<Response> rejected() {
        return CompletableFuture.completedFuture(Response.buildErrorResponse(
                StrategyErrorType.DISPATCH_FAILED, "request scheduler is not accepting new requests"));
    }

    protected static CompletableFuture<Response> invalidMode() {
        return CompletableFuture.completedFuture(Response.buildErrorResponse(
                StrategyErrorType.DISPATCH_FAILED, "request configuration does not match scheduler mode"));
    }

    public AdmissionHandle claimQueuedRoute(DecodeEndpoint endpoint, DecodeResources.ReservationHandle victim, int incomingPriority) {
        return null;
    }

    public void completeWithdrawal(AdmissionHandle withdrawal, boolean committed) {
        throw new IllegalStateException("scheduler does not own queued withdrawals");
    }

    /** The resolved request retains its identity through archival; Context rejects retired routes. */
    public void onPrefillStatus(RequestContext context, PrefillEndpoint source, RoleType role,
                                PrefillState.PrefillRequestStatus requestStatus) {
        try {
            if (context != null && context.scheduler() == this) {
                submitContinuation(context, acceptPrefillStatus(context, source, role, requestStatus, System.currentTimeMillis()));
            }
        } catch (Throwable failure) {
            logEndpointFailure("Prefill status", failure);
        }
    }

    public void onDecodeStatus(RequestContext context, DecodeEndpoint source,
                               DecodeResources.DecodeRequestStatus requestStatus) {
        try {
            if (context != null && context.scheduler() == this) {
                submitContinuation(context, acceptDecodeStatus(context, source, requestStatus, System.currentTimeMillis()));
            }
        } catch (Throwable failure) {
            logEndpointFailure("Decode status", failure);
        }
    }

    private void submitContinuation(RequestContext context, Runnable work) {
        if (work != null) {
            continuations.submit(context, work);
        }
    }

    private static void logEndpointFailure(String event, Throwable failure) {
        try {
            Logger.error("Endpoint event isolated: event={}", event, failure);
        } catch (Throwable ignored) {
            // Diagnostics cannot prevent the remaining updates from being processed.
        }
    }

    public void onPrefillGenerationRetired(PrefillEndpoint source, RequestRoute exact) {
        if (source == null || exact == null) { return; }
        try {
            RequestContext context = exact.ctx();
            if (!isCurrentContext(context)) { return; }
            DeliveryClaim delivery = context.delivery();
            if (delivery != null && delivery.item == exact) {
                if (delivery.recordEndpointRetirement(source)) { settleDelivery(delivery); }
            }
            submitContinuation(context, acceptPrefillRetirement(context, source, exact));
        } catch (Throwable failure) {
            logEndpointFailure("Prefill retirement", failure);
        }
    }

    public void onDecodeGenerationRetired(RequestContext context, DecodeEndpoint source,
                                          DecodeResources.ReservationHandle exact) {
        if (source == null || exact == null) { return; }
        try {
            if (!isCurrentContext(context)) { return; }
            DeliveryClaim delivery = context.delivery();
            if (delivery != null && Objects.equals(delivery.item.decodeReservation(), exact)) {
                if (delivery.recordEndpointRetirement(source)) { settleDelivery(delivery); }
            }
            Runnable work;
            synchronized (context) {
                work = context.ownsDecodeReservationLocked(source, exact)
                        ? advanceRequestEndLocked(context, context.route(), DeferredTerminal.decodeGenerationRetired(
                                "Decode endpoint generation retired: generation=" + exact.endpointGenerationId()))
                        : null;
            }
            submitContinuation(context, work);
        } catch (Throwable failure) {
            logEndpointFailure("Decode retirement", failure);
        }
    }

    public void enqueueInactivityDeadline(RequestContext ctx, ExpirationTimer.InactivityDeadline exact, long nowMs, Runnable rearm) {
        Runnable effect;
        synchronized (ctx) {
            if (!ctx.consumeInactivityDeadlineLocked(exact)) { return; }
            effect = decideInactivityLocked(ctx, nowMs, null);
        }
        continuations.submit(ctx, () -> {
            try { execute(ctx, effect); }
            finally { rearm.run(); }
        });
    }

    private CompletableFuture<Response> registerRequest(RequestContext context, StrategyErrorType expiredError) {
        if (context.requestExpired(System.currentTimeMillis())) {
            return CompletableFuture.completedFuture(buildErrorResponse(expiredError, "request scheduling deadline has expired before placement"));
        }
        RequestFuture response = new RequestFuture((completion, value, failure, interrupt) ->
                completeExternal(context, completion, value, failure, interrupt));
        var result = requests.register(context, this, response);
        if (result != RequestRepository.RegistrationResult.REGISTERED) {
            return CompletableFuture.completedFuture(buildErrorResponse(
                    result == RequestRepository.RegistrationResult.CLOSED ? StrategyErrorType.DISPATCH_FAILED : StrategyErrorType.INVALID_REQUEST,
                    result == RequestRepository.RegistrationResult.DUPLICATE_ID
                            ? "duplicate request_id: " + context.getRequestId() : "request registration rejected: " + result));
        }
        context.setEnqueueTime(System.currentTimeMillis());
        if (requests.isClosed()) {
            completeError(response, StrategyErrorType.DISPATCH_FAILED, "request scheduler is shutting down");
        } else if (context.requestExpired(System.currentTimeMillis())) {
            cancelRequest(context, 0L, CancelReason.DEADLINE_EXCEEDED);
        }
        return response;
    }

    public boolean isAdmissionOpen(long requestId, CompletableFuture<?> future) {
        if (requests.isClosed()) {
            return false;
        }
        RequestContext requestContext = findRequestContext(requestId);
        if (requestContext == null || !requestContext.ownsFuture(future)) {
            return false;
        }
        synchronized (requestContext) {
            return requests.isCurrent(requestContext) && requestContext.isOpen();
        }
    }

    public AdmissionHandle claimAdmissionHandle(long requestId, CompletableFuture<?> future) {
        if (!enterAdmissionHandleGate()) {
            return null;
        }
        boolean transferred = false;
        try {
            RequestContext requestContext = findRequestContext(requestId);
            if (requestContext == null || !requestContext.ownsFuture(future)) {
                return null;
            }
            AdmissionHandle handle;
            synchronized (requestContext) {
                handle = requests.isCurrent(requestContext) ? requestContext.beginAdmission((operation, response) -> finishAdmission(requestContext, operation, response)) : null;
            }
            transferred = handle != null;
            return handle;
        } finally {
            if (!transferred) {
                exitAdmissionHandleGate();
            }
        }
    }

    boolean enterAdmissionHandleGate() {
        synchronized (admissionQuiescenceMonitor) {
            if (requests.isClosed()) {
                return false;
            }
            checkState(inFlightAdmissionHandles != Integer.MAX_VALUE, "admission handle counter overflow");
            inFlightAdmissionHandles++;
            return true;
        }
    }

    void exitAdmissionHandleGate() {
        synchronized (admissionQuiescenceMonitor) {
            checkState(inFlightAdmissionHandles > 0, "admission handle counter underflow");
            inFlightAdmissionHandles--;
            if (inFlightAdmissionHandles == 0) {
                admissionQuiescenceMonitor.notifyAll();
            }
        }
    }

    void awaitAdmissionMutations() {
        boolean interrupted = false;
        synchronized (admissionQuiescenceMonitor) {
            while (inFlightAdmissionHandles != 0) {
                try {
                    admissionQuiescenceMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    public void onQueuedItemPreempted(RequestRoute victim, RequestRoute incoming) {
        try {
            RequestContext context = findRouteContext(victim);
            if (context != null) {
                recordSchedulingFailure(context, StrategyErrorType.PRIORITY_PREEMPTED,
                        "preempted by higher-priority request " + incoming.requestId());
            }
            requestReporter.reportVictim(victim.priority(), incoming.priority(), "prefill_queued", "prefill_inflight_requests");
        } catch (Throwable failure) {
            logEndpointFailure("Queued preemption request_id=" + victim.requestId(), failure);
        }
    }

    public Optional<PreemptionRegistration> tryClaim(DecodeResources.ReservationHandle exact, long attemptToken, String detail) {
        Objects.requireNonNull(exact, "exact reservation");
        RequestContext context = findRequestContext(exact.requestId());
        return context == null ? Optional.empty()
                : Optional.ofNullable(context.tryInstallPreemption(exact, attemptToken, detail));
    }

    public void onQueuedItemExpired(RequestRoute exact) {
        RequestContext requestContext = findRouteContext(exact);
        if (requestContext != null) {
            cancelRequest(requestContext, 0L, CancelReason.DEADLINE_EXCEEDED);
        }
    }

    void onQueuedItemControl(RequestRoute exact) {
        RequestContext requestContext = findRouteContext(exact);
        if (requestContext != null) {
            cancelInWorkerQueue(requestContext, exact);
        }
    }

    public void onQueueOfferFailure(RequestRoute exact, Throwable error) {
        RequestContext requestContext = findRouteContext(exact);
        if (requestContext != null) {
            cancelInWorkerQueue(requestContext, exact);
            recordSchedulingFailure(requestContext, StrategyErrorType.DISPATCH_FAILED, "Worker scheduling queue rejected request: " + (error == null ? "endpoint publication failed" : error.getMessage()));
        }
    }

    private RequestContext findRouteContext(RequestRoute item) {
        RequestContext context = item.ctx();
        synchronized (context) {
            return isCurrentContext(context) && context.ownsActiveRoute(item) ? context : null;
        }
    }

    private static void completeError(CompletableFuture<Response> future, StrategyErrorType errorType, String message) {
        if (future.isDone()) {
            return;
        }
        future.complete(buildErrorResponse(errorType, message));
    }

    public boolean publishDecisionResponseAsync(long requestId, CompletableFuture<Response> future, Response response) {
        RequestContext requestContext = findRequestContext(requestId);
        return requestContext != null && requestContext.ownsFuture(future) && terminateLocallyAndPublishResponse(requestContext, response);
    }

    void closeOutstandingAndTerminalize() {
        checkState(requests.isClosed(), "admission must close before terminal shutdown");
        List<TerminalAction> actions = new ArrayList<>();
        for (RequestContext requestContext : ownedRequests()) {
            synchronized (requestContext) {
                TerminalAction action = claimFinalizationLocked(requestContext, requestContext.decideShutdownLocked());
                if (action != null) { actions.add(action); }
            }
        }
        Throwable failure = null;
        for (TerminalAction action : actions) {
            failure = Failures.run(failure, () -> executeFinalization(action));
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    ExpirationTimer expirationTimer() { return expirationTimer; }

    /** Register successful-route cache tracing on Future completion. */
    void registerResponseCallback(RequestContext context, CompletableFuture<Response> future) {
        future.whenComplete((response, failure) -> {
            if (failure != null) { return; }
            try {
                if (response != null && response.isSuccess()) { recentCacheKeyTraceReporter.report(context); }
            } catch (RuntimeException completionHandlerFailure) {
                Logger.warn("Route completion side effect failed: request_id={}", context.getRequestId(), completionHandlerFailure);
            }
        });
    }

    interface Publication {
        void publish();
        boolean published();
    }

    static DecodeResources.ReservationHandle tryReserveDecode(
            RequestRoute routing, RequestRequirements requirements) {
        com.google.common.base.Preconditions.checkArgument(requirements.requestId() == routing.requestId(),
                "request inputs do not match selected route");
        DecodeEndpoint endpoint = routing.decodeEp();
        if (endpoint == null) { return null; }
        DecodeResources.AdmissionCapacity capacity = switch (requirements.mode()) {
            case IMMEDIATE -> null;
            case WAIT_AT_PLACEMENT, PREEMPT_AT_PLACEMENT -> requirements.capacity();
        };
        return endpoint.tryReserveQueuedRequest(routing.decodePin(), requirements.requestId(),
                requirements.hardKvTokens(), requirements.expectedKvTokens(), requirements.priority(), capacity);
    }

    PlacementResult.Status commitRoute(RequestRoute exact, Publication publication) {
        Objects.requireNonNull(publication, "publication");
        if (requests.isClosed() || exact == null || exact.future().isDone()) {
            return PlacementResult.Status.CLOSED;
        }
        RequestContext ctx = exact.ctx();
        synchronized (ctx) {
            if (!isCurrentContext(ctx) || !ctx.bindRoute(exact)) {
                return PlacementResult.Status.CLOSED;
            }

        }
        // Only failed publication rolls back the binding. Later failures retain its exact owner.
        Throwable failure = null;
        try {
            publication.publish();
        } catch (RuntimeException | Error publicationFailure) {
            failure = publicationFailure;
        }
        boolean published = publication.published();
        if (published) {
            synchronized (ctx) { ctx.confirmRoutePublication(exact); }
            if (failure == null) { exact.prefillEp().signalRouteReady(); }
            else { Failures.run(failure, exact.prefillEp()::signalRouteReady); }
        } else {
            try {
                synchronized (ctx) { ctx.rejectRoutePublication(exact); }
            } catch (RuntimeException | Error rollbackFailure) {
                if (failure == null) { throw rollbackFailure; }
                Failures.append(failure, rollbackFailure);
            }
        }
        Failures.rethrow(failure, "route publication failed");
        return published ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED;
    }

    /**
     * Close one admission and settle all facts retained while it owned the request.
     */
    void finishAdmission(RequestContext ctx, AdmissionHandle exact, Response failureResponse) {
        Throwable failure = null;
        try {
            try {
                Runnable effect = null;
                boolean cleanupPending;
                RequestRoute routeToCancel;
                RequestRoute restoredRoute = null;
                synchronized (ctx) {
                    if (ctx.admission() == exact) {
                        restoredRoute = ctx.finishAdmission(exact);
                        effect = requestEndEffectsLocked(ctx, ctx.settleAdmissionLocked(exact, failureResponse,
                                pendingWorkerQueueCancellationLocked(ctx) != null), null);
                    }
                    cleanupPending = ctx.hasCleanup();
                    routeToCancel = cleanupPending || effect != null ? null : pendingWorkerQueueCancellationLocked(ctx);
                }
                if (restoredRoute != null) {
                    restoredRoute.prefillEp().signalRouteReady();
                }
                if (routeToCancel != null) { scheduleWorkerQueueCancellation(ctx, routeToCancel); }
                if (effect != null) { execute(ctx, effect); }
                else if (cleanupPending) { resumeCleanup(ctx); }
            } catch (Throwable settlementFailure) {
                failure = settlementFailure;
            }
            // A failed settlement must not strand the expiry watch or the admission gate.
            failure = Failures.run(failure, () -> expirationTimer.scheduleInactivityDeadline(ctx));
            // Drain must see the failure before the last admission gate opens.
            if (failure != null) {
                runtime.recordFailure(failure);
            }
        } finally {
            exitAdmissionHandleGate();
        }
        Failures.rethrow(failure, "request cleanup failed");
    }

    /**
     * Atomically arbitrate expiry against the one endpoint-ownership handoff.
     */
    public DeliveryClaim claimDelivery(RequestRoute exact, DeliveryClaimKind kind, long correlationId, DecodeEndpoint.EngineDispatchPermit permit) {
        RequestContext ctx = exact.ctx();
        Runnable expired;
        synchronized (ctx) {
            checkArgument(kind != DeliveryClaimKind.NONE
                    && (kind != DeliveryClaimKind.BATCH_ENQUEUE
                    || correlationId > 0L),
                    "invalid delivery identity");
            if (!ownsPreparedDeliveryLocked(ctx, exact)) {
                return null;
            }
            // Read time under the same lock as the claim. Timer execution can
            // lag, so an unconsumed timer is not proof that this item is live.
            long nowMs = System.currentTimeMillis();
            if (ctx.requestInactiveLocked(nowMs)) {
                expired = decideInactivityLocked(ctx, nowMs, null);
            } else if (exact.requestExpired(nowMs)) {
                recordCancellationLocked(ctx, CancelReason.DEADLINE_EXCEEDED, "request scheduling deadline exceeded before delivery");
                expired = finalizationEffects(claimFinalizationLocked(ctx, ctx.decideCancellationTerminationLocked()), null);
            } else {
                if (permit != null) {
                    checkArgument(permit.belongsTo(exact.decodeEp(), exact.decodeReservation()),
                            "Decode permit belongs to another route reservation");
                    switch (permit.dispatch()) {
                        case TRANSFERRED -> { }
                        case OWNERSHIP_LOST -> throw new IllegalStateException("endpoint ownership lost for request " + ctx.getRequestId());
                        case ENDPOINT_RETIRED -> throw new IllegalStateException("Decode endpoint generation retired: request_id=" + ctx.getRequestId());
                    }
                }
                return ctx.beginDelivery(exact, kind, correlationId, nowMs);
            }
        }
        execute(ctx, expired);
        return null;
    }

    /** Check the exact send identity immediately before adding the member to the RPC. */
    boolean tryStartSend(DeliveryClaim claim) {
        RequestContext ctx = claim.item.ctx();
        CancelReason refused;
        synchronized (ctx) {
            if (!claim.awaitingSendLocked()) { return false; }
            refused = claim.startSendLocked(System.currentTimeMillis());
        }
        if (refused != null) { abandonDelivery(claim, refused); }
        return refused == null;
    }

    void completeDelivery(DeliveryClaim claim, DeliveryResult result) {
        claim.recordSendResult(result);
        if (result.failed()) { abandonDelivery(claim, CancelReason.CLIENT_CANCELLED); }
        try {
            Runnable work = acceptDeliveryResult(claim, result);
            if (work != null) { submitContinuation(claim.item.ctx(), work); }
        } finally { settleDelivery(claim); }
    }

    void abandonDelivery(DeliveryClaim claim, CancelReason reason) {
        boolean start;
        synchronized (claim.item.ctx()) {
            if (!claim.recordAbandonmentLocked(reason)) { return; }
            start = claim.kind == DeliveryClaimKind.BATCH_ENQUEUE && !claim.settled.isDone();
        }
        if (start) { runtime.startDeliveryCleanup(claim); }
        settleDelivery(claim);
    }

    void acceptDeliveryCleanup(DeliveryClaim claim, org.flexlb.balance.eviction.EngineCancelChannel.CancelAck ack) {
        claim.acceptCleanupAck(ack);
        settleDelivery(claim);
    }

    /** Queue local cleanup before releasing the evidence barrier awaited by shutdown. */
    void settleDelivery(DeliveryClaim claim) {
        RequestContext ctx = claim.item.ctx();
        boolean ready;
        boolean cleanupPending;
        synchronized (ctx) {
            ready = claim.readyToSettleLocked();
            cleanupPending = ctx.hasCleanup() && claim.kind == DeliveryClaimKind.BATCH_ENQUEUE;
        }
        if (ready) {
            try { if (cleanupPending) { enqueueCleanup(ctx); } }
            finally { claim.settled.complete(null); }
        }
    }

    /**
     * Queue publication makes an exact item claimable even while admission still pins its resources.
     */
    boolean ownsPreparedDeliveryLocked(RequestContext ctx, RequestRoute exact) {
        return isCurrentContext(ctx) && ctx.ownsPreparedDeliveryLocked(exact);
    }

    private DecisionDeadline applyDeliveryPredictionLocked(RequestContext ctx, WorkSnapshot work, long predictedMs, long nowMs) {
        ctx.recordDeliveryPredictionLocked(work, predictedMs, nowMs);
        RequestRoute route = ctx.route();
        if (route.decodeEp() != null && route.decodeEp().isAcceptedByEngine(route.decodeReservation())) {
            ctx.recordWorkerActivityLocked(nowMs);
            return ctx.markDecodeAcceptedLocked();
        }
        return null;
    }

    public void setDeliveryPrediction(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        RequestContext ctx = claim.item.ctx();
        DecisionDeadline obsolete;
        synchronized (ctx) {
            if (!ctx.ownsActiveRoute(claim.item)) {
                return;
            }
            obsolete = applyDeliveryPredictionLocked(ctx, work, predictedMs, System.currentTimeMillis());
        }
        ExpirationTimer.releaseDecisionDeadline(obsolete);
        expirationTimer.scheduleDecisionDeadline(ctx);
    }

    public void publishRoute(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        RequestContext ctx = claim.item.ctx();
        Runnable effect;
        DecisionDeadline obsolete;
        synchronized (ctx) {
            if (!ctx.ownsActiveRoute(claim.item)) {
                return;
            }
            checkArgument(ctx.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION,
                    "route publication requires a route claim");
            obsolete = applyDeliveryPredictionLocked(ctx, work, predictedMs, System.currentTimeMillis());
            effect = acknowledgeDeliveryLocked(ctx, null);
        }
        claim.completeRoute();
        settleDelivery(claim);
        executeEngineEffects(ctx, effect, obsolete);
    }

    private Runnable acceptDeliveryResult(DeliveryClaim claim, DeliveryResult result) {
        RequestContext ctx = claim.item.ctx();
        Objects.requireNonNull(result, "delivery result");
        Runnable acknowledgement;
        synchronized (ctx) {
            if (!ctx.acceptDeliveryClaim(claim) || ctx.hasTerminalAction() || ctx.hasCleanup()) { return null; }
            if (result.failed()) {
                PublicationPermit permit = selectDeliveryFailureLocked(ctx, claim.item, result.status(), "Delivery failed: " + detailOf(result.cause()));
                ResponseResult selected = ctx.selectedResponse();
                return () -> publishFailureAndCleanUp(ctx, permit, selected);
            }
            if (!ctx.ownsActiveRoute(claim.item)) {
                return null;
            }
            if (result.status() == DeliveryResult.Status.DELIVERED) {
                ctx.setAckAtMs(System.currentTimeMillis());
                ctx.setAckAtNanos(System.nanoTime());
            } else if (!ctx.decodeAccepted()) {
                ctx.markAwaitingConfirmationLocked("Delivery outcome uncertain: " + detailOf(result.cause()));
                return null;
            }
            acknowledgement = acknowledgeDeliveryLocked(ctx, null);
        }
        return acknowledgement == null ? null : () -> execute(ctx, acknowledgement);
    }

    public void failDeliveryPreparation(RequestRoute exact, Throwable cause) {
        RequestContext ctx = exact.ctx();
        PublicationPermit permit;
        ResponseResult selected;
        synchronized (ctx) {
            if (!ownsPreparedDeliveryLocked(ctx, exact)) {
                return;
            }
            permit = selectDeliveryFailureLocked(ctx, exact, DeliveryResult.Status.NOT_SENT, "Delivery preparation failed: " + detailOf(cause));
            selected = ctx.selectedResponse();
        }
        publishFailureAndCleanUp(ctx, permit, selected);
    }

    private void publishFailureAndCleanUp(RequestContext ctx, PublicationPermit permit, ResponseResult selected) {
        Throwable failure = Failures.run(null, permit == null ? null : () -> submitResponse(permit, selected));
        failure = Failures.run(failure, () -> resumeCleanup(ctx));
        Failures.rethrow(failure, "request cleanup failed");
    }

    private static String detailOf(Throwable cause) {
        if (cause == null) {
            return "unknown delivery failure";
        }
        String message = cause.getMessage();
        return message == null || message.isBlank() ? cause.getClass().getSimpleName() : message;
    }

    // ── Worker 事实与 Endpoint 退出：接收、核验、推进 ──

    Runnable acceptPrefillStatus(RequestContext ctx, PrefillEndpoint source, RoleType role,
                                 PrefillState.PrefillRequestStatus requestStatus, long nowMs) {
        Runnable work;
        DecisionDeadline obsolete;
        boolean resume;
        RequestRoute routeToCancel;
        synchronized (ctx) {
            if (!ctx.ownsPrefillRouteLocked(source, requestStatus.route())) {
                return null;
            }
            PreemptionRegistration previous = ctx.preemption();
            boolean cleaning = ctx.hasCleanup();
            work = applyPrefillStatusLocked(ctx, role, requestStatus, nowMs);
            obsolete = cleaning ? null : ctx.detachObsoleteDecisionDeadlineLocked();
            resume = previous != null && ctx.preemption() == null && ctx.hasCleanup() && work == null;
            routeToCancel = previous == null ? null : pendingWorkerQueueCancellationLocked(ctx);
            if (work == null && obsolete == null && !resume && routeToCancel == null
                    && ctx.decisionDeadlineAtMs().isEmpty()) {
                return null;
            }
        }
        return () -> {
            if (resume) {
                resumeCleanup(ctx);
            } else {
                executeEngineEffects(ctx, work, obsolete);
            }
            if (routeToCancel != null) {
                scheduleWorkerQueueCancellation(ctx, routeToCancel);
            }
        };
    }

    private Runnable applyPrefillStatusLocked(RequestContext ctx, RoleType role, PrefillState.PrefillRequestStatus requestStatus, long nowMs) {
        ctx.requireContextLock("Prefill request status reduction");
        boolean cleaning = ctx.hasCleanup();
        if (!ctx.consumePrefillStatusLocked(role, requestStatus.kind(), nowMs)) {
            return () -> resumeCleanup(ctx);
        }
        DecodeEndpoint capacityRelease = null;
        PreemptionRegistration claim = ctx.preemptionForPrefillStatusLocked(requestStatus.kind());
        Runnable transition = switch(requestStatus.kind()) {
            case ACTIVE -> {
                    if (claim != null) {
                        DecodeEndpoint decode = requestStatus.route().decodeEp();
                        if (decode == null || decode.reconcilePreemptionResources(claim.attemptToken(),
                                DecodeResources.PreemptionUpdate.active(requestStatus.route().decodeReservation()))) {
                            ctx.detachPreemptionOwnerLocked(claim);
                            capacityRelease = decode;
                            yield cleaning ? () -> resumeCleanup(ctx) : null;
                        }
                    }
                    yield null;
                }
            case COMPLETED -> role == RoleType.PDFUSION
                    ? advanceRequestEndLocked(ctx, requestStatus.route(), DeferredTerminal.worker(
                            WorkerTerminalSource.PREFILL_ENDPOINT, true, requestStatus.errorCode())) : null;
            case FAILED ->
                advanceRequestEndLocked(ctx, requestStatus.route(), DeferredTerminal.worker(WorkerTerminalSource.PREFILL_ENDPOINT, false, requestStatus.errorCode()));
            case PRIORITY_CANCELED -> {
                DecodeEndpoint decode = requestStatus.route().decodeEp();
                if (claim == null || !decode.reconcilePreemptionResources(claim.attemptToken(),
                                DecodeResources.PreemptionUpdate.canceled(requestStatus.route().decodeReservation()))) {
                    yield null;
                }
                capacityRelease = decode;
                yield requestEndEffectsLocked(ctx, ctx.decidePreemptedRequestEndLocked(claim, "priority victim canceled by worker", true), claim);
            }
        };
        if (!cleaning) { ctx.reconcileDecisionEvidenceLocked(); }
        if (capacityRelease == null) { return transition; }
        return capacityReleaseEffects(capacityRelease, transition, "Prefill request status continuation failed");
    }

    Runnable acceptDecodeStatus(RequestContext ctx, DecodeEndpoint source, DecodeResources.DecodeRequestStatus requestStatus, long nowMs) {
        Runnable work = null;
        DecisionDeadline obsolete = null;
        boolean capacityChanged = false;
        synchronized (ctx) {
            if (!ctx.ownsDecodeReservationLocked(source, requestStatus.reservation())) {
                return null;
            }
            obsolete = ctx.recordDecodeProgressLocked(requestStatus.kind(), nowMs);
            if (requestStatus.kind() == DecodeResources.DecodeRequestStatus.Kind.TERMINAL) {
                work = advanceRequestEndLocked(ctx, ctx.route(), DeferredTerminal.worker(
                        WorkerTerminalSource.DECODE_ENDPOINT, requestStatus.errorCode() == 0L, requestStatus.errorCode()));
                obsolete = ctx.detachObsoleteDecisionDeadlineLocked();
            } else {
                // Membership and allocation remain separate facts. Only allocation reconciles a NOT_FOUND claim.
                RequestEndDecision decision = ctx.decodeAllocationReconciliationLocked(requestStatus.allocationObserved());
                if (decision != null) {
                    PreemptionRegistration claim = decision.reconciliation();
                    capacityChanged = source.reconcilePreemptionResources(claim.attemptToken(), decision.terminal() == null
                            ? DecodeResources.PreemptionUpdate.active(requestStatus.reservation())
                            : DecodeResources.PreemptionUpdate.finished(requestStatus.reservation()));
                    if (capacityChanged) {
                        work = advanceAfterDecodeReconciliationLocked(ctx, claim, decision.terminal(), claim, true);
                    }
                }
            }
            if (requestStatus.kind() == DecodeResources.DecodeRequestStatus.Kind.ACTIVE
                    && work == null && obsolete == null && !capacityChanged
                    && ctx.decisionDeadlineAtMs().isEmpty()) {
                return null;
            }
        }
        Runnable effect = work;
        DecisionDeadline deadline = obsolete;
        boolean publishCapacity = capacityChanged;
        return () -> {
            Throwable notificationFailure = publishCapacity ? Failures.run(null, source::publishCapacityRelease) : null;
            try {
                DeliveryClaim delivery = ctx.delivery();
                if (delivery != null && delivery.recordDecodeTerminalStatus(source, requestStatus)) { settleDelivery(delivery); }
                executeEngineEffects(ctx, effect, deadline);
            } catch (Throwable failure) {
                throw Failures.propagate(Failures.append(failure, notificationFailure), "Decode request status continuation failed");
            }
            Failures.rethrow(notificationFailure, "Decode capacity publication failed");
        };
    }

    Runnable decideInactivityLocked(RequestContext ctx, long nowMs, PreemptionRegistration signal) {
        if (ctx.expireCleanup(nowMs)) {
            return () -> resumeCleanup(ctx);
        }
        if (!ctx.ownsActiveGenerationLocked() || !ctx.requestInactiveLocked(nowMs)) {
            return null;
        }
        String message = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
        recordCancellationLocked(ctx, CancelReason.DEADLINE_EXCEEDED, message);
        if (ctx.admission() != null) {
            ctx.retainAdmissionExpiry();
            return null;
        }
        return requestEndEffectsLocked(ctx, ctx.decideRequestEndLocked(DeferredTerminal.inactivityExpired(message)), signal);
    }

    Runnable advanceRequestEndLocked(RequestContext ctx, RequestRoute expected, DeferredTerminal event) {
        PreemptionRegistration signal = ctx.preemption();
        return requestEndEffectsLocked(ctx, ctx.acceptRequestEndLocked(expected, event), signal);
    }

    Runnable reconcilePreemptionLocked(RequestContext ctx, PreemptionRegistration exact, boolean transportUnknown, PreemptionRegistration signal) {
        return requestEndEffectsLocked(ctx, ctx.decidePreemptionReconciliationLocked(exact, transportUnknown), signal);
    }

    private Runnable requestEndEffectsLocked(RequestContext ctx, RequestEndDecision decision, PreemptionRegistration signal) {
        ctx.requireContextLock("request end execution");
        if (decision == null) { return null; }
        switch (decision.kind()) {
            case CLEANUP -> { return () -> resumeCleanup(ctx); }
            case CONFIRM_DELIVERY -> { return acknowledgeDeliveryLocked(ctx, signal); }
            case FINALIZE -> { return finalizationEffects(claimFinalizationLocked(ctx, decision), signal); }
            case RECONCILE -> { }
        }
        PreemptionRegistration exact = decision.reconciliation();
        DeferredTerminal terminal = decision.terminal();
        RequestRoute active = ctx.activeRoute();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        // Decode terminal facts already committed its ledger; all other evidence must reconcile it first.
        boolean capacityChanged = decode != null && !(terminal != null && terminal.decodeTerminalAlreadyApplied());
        if (capacityChanged && !decode.reconcilePreemptionResources(exact.attemptToken(), terminal != null
                ? DecodeResources.PreemptionUpdate.finished(active.decodeReservation())
                : DecodeResources.PreemptionUpdate.active(active.decodeReservation()))) { return null; }
        Runnable work = advanceAfterDecodeReconciliationLocked(ctx, exact, terminal,
                signal, capacityChanged);
        return capacityChanged ? capacityReleaseEffects(decode, work, "preemption continuation failed") : work;
    }

    private static Runnable capacityReleaseEffects(DecodeEndpoint source, Runnable effect, String failureMessage) {
        return () -> {
            Throwable failure = Failures.run(null, source::publishCapacityRelease);
            failure = Failures.run(failure, effect);
            Failures.rethrow(failure, failureMessage);
        };
    }

    private Runnable advanceAfterDecodeReconciliationLocked(RequestContext ctx, PreemptionRegistration claim,
            DeferredTerminal terminal, PreemptionRegistration signal, boolean decodeSettled) {
        return requestEndEffectsLocked(ctx, ctx.applyPreemptionReconciliationLocked(claim, terminal, decodeSettled), signal);
    }

    Runnable acknowledgeDeliveryLocked(RequestContext ctx, PreemptionRegistration signal) {
        ctx.requireContextLock("delivery acknowledgement");
        if (!ctx.acceptDeliveryConfirmationLocked()) { return null; }
        PreemptionRegistration blocked = ctx.preemption();
        if (blocked != null) { return reconcilePreemptionLocked(ctx, blocked, false, null); }
        // A delayed timer continuation cannot let an expired silent request publish a late ACK.
        long nowMs = System.currentTimeMillis();
        if (ctx.requestInactiveLocked(nowMs)) {
            return decideInactivityLocked(ctx, nowMs, signal);
        }
        PublicationPermit permit = requirePublicationPermitLocked(ctx, PublicationKind.DELIVERY);
        try {
            DeliveryPublication publication = ctx.acknowledgeDelivery(permit, nowMs);
            return deliveryEffects(ctx, publication, signal);
        } catch (RuntimeException | Error failure) {
            permit.abandonIfUnused();
            throw failure;
        }
    }

    void executeEngineEffects(RequestContext ctx, Runnable work, DecisionDeadline obsolete) {
        ExpirationTimer.releaseDecisionDeadline(obsolete);
        Throwable failure = Failures.run(null, () -> expirationTimer.scheduleDecisionDeadline(ctx));
        failure = Failures.run(failure, () -> execute(ctx, work));
        Failures.rethrow(failure, "request cleanup failed");
    }

    private Runnable acceptPrefillRetirement(RequestContext ctx, PrefillEndpoint source, RequestRoute exact) {
        String detail = "Prefill endpoint generation retired: " + source.ipPort() + "#" + source.getStatus().getGenerationId();
        synchronized (ctx) {
            return requestEndEffectsLocked(ctx, ctx.acceptPrefillRetirementLocked(new PendingPrefillRetirement(source, exact, detail)), null);
        }
    }

    /**
     * Return the resulting request snapshot; accepting cancellation does not imply immediate cleanup.
     */
    protected void onCancellationRecorded(RequestContext exact) { }

    public void onResponseUndeliverable(RequestContext exact) {
        if (exact == null || exact.scheduler() != this) { return; }
        DeliveryClaim delivery = exact.delivery();
        if (delivery != null) { abandonDelivery(delivery, CancelReason.CLIENT_CANCELLED); }
        cancelRequest(exact, 0L, CancelReason.CLIENT_CANCELLED);
    }

    final boolean recordCancellationLocked(RequestContext ctx, CancelReason reason, String message) {
        ctx.requireContextLock("record cancellation");
        Objects.requireNonNull(reason, "reason");
        if (!ctx.ownsActiveGenerationLocked() || ctx.cancellationReason() != null) { return false; }
        ctx.recordCancellationLocked(reason, message, cancellationDiagnosticsLocked(ctx, reason, message));
        return true;
    }

    protected Map<String, Object> cancellationDiagnosticsLocked(RequestContext ctx, CancelReason reason, String message) {
        return null;
    }

    protected boolean awaitingGlobalQueueCancellationLocked(RequestContext ctx) { return false; }

    protected RequestRoute pendingWorkerQueueCancellationLocked(RequestContext ctx) { return null; }

    final RequestState cancelRequest(RequestContext ctx, long expectedBatchId, CancelReason reason) {
        return cancelRequest(ctx, expectedBatchId, reason, null);
    }

    /** Records the first cancellation, then asks the resource owner to settle the request. */
    private RequestState cancelRequest(RequestContext ctx, long expectedBatchId, CancelReason reason,
                                       RequestDeadline deadline) {
        Objects.requireNonNull(reason, "reason");
        TerminalAction action = null;
        RequestState result;
        RequestRoute routeToCancel;
        synchronized (ctx) {
            // Validate the cancellation target: exact timer, or expected batch (0 means any batch).
            if (deadline != null) {
                if (!ctx.consumeRequestDeadline(deadline) || !ctx.isOpen()) {
                    return null;
                }
            } else if (expectedBatchId != 0L && ctx.batchId() != expectedBatchId) {
                return null;
            }

            // A scheduling timeout cannot cancel delivery, except while admission is still running.
            // Only the exact scheduling timer can use that admission exception.
            boolean deadlineDuringAdmission = deadline != null && ctx.admission() != null;
            if (reason == CancelReason.DEADLINE_EXCEEDED && ctx.deliveryClaimKind() != DeliveryClaimKind.NONE
                    && !deadlineDuringAdmission) {
                return deadline == null ? ctx.snapshot() : null;
            }

            // Repeated cancellation only reads the state; the first caller owns cancellation work.
            String message = deadlineDuringAdmission
                    ? "request scheduling deadline exceeded during admission" : reason.getMessage();
            if (!recordCancellationLocked(ctx, reason, message)) {
                return deadline == null ? ctx.snapshot() : null;
            }

            // Queued requests are settled by their queue owner. Otherwise try to settle now;
            // an admission or delivery still in progress can defer that finalization.
            routeToCancel = pendingWorkerQueueCancellationLocked(ctx);
            if (routeToCancel == null && !awaitingGlobalQueueCancellationLocked(ctx)) {
                action = claimFinalizationLocked(ctx, ctx.decideCancellationTerminationLocked());
            }
            result = deadline == null ? ctx.snapshot() : null;
        }

        // Notify delivery and queue owners outside the context lock; publish completion last.
        DeliveryClaim delivery = ctx.delivery();
        if (delivery != null) {
            abandonDelivery(delivery, reason);
            if (action == null) {
                synchronized (ctx) {
                    action = claimFinalizationLocked(ctx, ctx.decideFinalizationLocked(null,
                            TerminalOutcome.cancellation(reason, reason.getMessage()), Response.copyOf(ctx.cancellationResponse()), true));
                }
            }
        }
        onCancellationRecorded(ctx);
        if (routeToCancel != null) {
            scheduleWorkerQueueCancellation(ctx, routeToCancel);
        }
        executeFinalization(action);
        return result;
    }

    void scheduleWorkerQueueCancellation(RequestContext ctx, RequestRoute exact) {
        if (!exact.prefillEp().signalQueuedControl(exact)) {
            continuations.submit(ctx, () -> cancelInWorkerQueue(ctx, exact));
        }
    }

    void cancelInWorkerQueue(RequestContext ctx, RequestRoute exact) {
        TerminalAction action;
        synchronized (ctx) {
            if (exact == null || pendingWorkerQueueCancellationLocked(ctx) != exact) {
                return;
            }
            action = claimFinalizationLocked(ctx, ctx.decideCancellationTerminationLocked());
        }
        executeFinalization(action);
    }

    void recordSchedulingFailure(RequestContext ctx, StrategyErrorType error, String detail) {
        Runnable work;
        synchronized (ctx) {
            work = advanceRequestEndLocked(ctx, ctx.activeRoute(), DeferredTerminal.failure(error, detail));
        }
        execute(ctx, work);
    }

    // ── 调度期限：安装与到期 ──

    public void onSchedulingDeadline(RequestContext ctx, RequestDeadline exact) {
        cancelRequest(ctx, 0L, CancelReason.DEADLINE_EXCEEDED, Objects.requireNonNull(exact, "deadline"));
    }

    // ── 抢占协议：注册、进展、释放与完成 ──

    public boolean updatePreemption(PreemptionRegistration claim, PreemptionCancelPhase next) {
        RequestContext ctx = claim.owner;
        if (next == null) { return false; }
        Runnable work;
        synchronized (ctx) {
            if (!isCurrentContext(ctx) || !ctx.advancePreemption(claim, next)) { return false; }
            work = requestEndEffectsLocked(ctx, ctx.decidePreemptionProgressLocked(claim), claim);
        }
        execute(ctx, work);
        return true;
    }

    public boolean releasePreemption(PreemptionRegistration claim) {
        RequestContext ctx = claim.owner;
        Runnable work;
        boolean cleanupPending;
        RequestRoute routeToCancel;
        synchronized (ctx) {
            if (!isCurrentContext(ctx) || !ctx.releasePreemptionLocked(claim)) { return false; }
            cleanupPending = ctx.hasCleanup();
            work = cleanupPending ? null : reconcilePreemptionLocked(ctx, claim, false, claim);
            routeToCancel = cleanupPending || work != null ? null : pendingWorkerQueueCancellationLocked(ctx);
        }
        if (cleanupPending) { resumeCleanup(ctx); } else { execute(ctx, work); }
        if (routeToCancel != null) { scheduleWorkerQueueCancellation(ctx, routeToCancel); }
        return true;
    }

    /** Decode's exact resource CAS precedes request completion and publishes capacity outside the request monitor. */
    public boolean onPreemptionCleanupProven(PreemptionRegistration claim, DecodeEndpoint endpoint,
            DecodeResources.ReservationHandle reservation, String detail) {
        checkArgument(claim.scheduler() == this && claim.requestId() == reservation.requestId(),
                "cleanup proof belongs to another preemption victim");
        return endpoint.updatePreemption(claim.attemptToken(), DecodeResources.PreemptionUpdate.fenced(reservation))
                && completePreemption(claim, detail);
    }

    boolean completePreemption(PreemptionRegistration claim, String detail) {
        RequestContext ctx = claim.owner;
        Runnable work;
        synchronized (ctx) {
            if (!isCurrentContext(ctx)) { return false; }
            RequestEndDecision decision = ctx.decidePreemptedRequestEndLocked(claim, detail, false);
            if (decision == null) { return false; }
            work = requestEndEffectsLocked(ctx, decision, claim);
        }
        execute(ctx, work);
        return true;
    }

    /**
     * The only close gate: no event may discard an outstanding cleanup obligation.
     */

    void commitTerminalRecord(RequestContext ctx, TerminalAction action) {
        ExpirationTimer.DetachedDeadlines deadlines;
        synchronized (ctx) { deadlines = ctx.detachDeadlines(); }
        deadlines.release();
        RequestState terminal;
        synchronized (ctx) {
            try { terminal = ctx.finishTerminal(action); }
            catch (RuntimeException | Error failure) {
                if (action.publication() != null) { action.publication().abandonIfUnused(); }
                throw failure;
            }
        }
        requests.archive(ctx, terminal);
    }

    void executeFinalization(TerminalAction action) {
        if (action == null) { return; }
        Throwable failure = Failures.run(null, () -> finishTerminal(action));
        failure = Failures.run(failure, () -> publishTerminal(action));
        Failures.rethrow(failure, "request finalization failed");
    }

    private void publishTerminal(TerminalAction action) {
        if (action.publication() != null && action.response() != null) {
            submitResponse(action.publication(), selectPublication(action.requestContext(), action.publication(),
                    ResponseCompletion.RESPONSE, action.response(), null, false));
        }
    }

    private PublicationPermit finishTerminal(TerminalAction action) {
        RequestContext context = action.requestContext();
        context.requireCleanupOwner(action);
        Throwable cleanupFailure = null;
        cleanupFailure = Failures.run(cleanupFailure, () -> action.terminalResources().release());
        cleanupFailure = Failures.run(cleanupFailure, action.preemption() == null ? null : () -> action.preemption().signalResolution(new VictimResolution(context.getRequestId(), VictimResolution.Outcome.REQUEST_END)));
        if (cleanupFailure != null) { recordFailure(cleanupFailure); }
        DeliveryClaim delivery = context.delivery();
        boolean successfulWorker = action.event() != null && action.event().kind() == DeferredTerminal.Kind.WORKER
                && action.event().workerSuccessful();
        if (delivery != null) {
            if (successfulWorker && context.cancellationReason() == null && !delivery.cleanupRequired()) {
                if (delivery.recordWorkerCompletion(action.item())) { settleDelivery(delivery); }
            } else {
                abandonDelivery(delivery, context.cancellationReason() == null ? CancelReason.CLIENT_CANCELLED : context.cancellationReason());
            }
        }
        boolean archive;
        boolean batchDelivery;
        synchronized (context) {
            if (cleanupFailure == null) { context.finishTerminalEffectsLocked(action); }
            archive = context.claimArchiveLocked(action);
            batchDelivery = context.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE;
        }
        if (archive) {
            commitTerminalRecord(context, action);
        } else if (batchDelivery) {
            enqueueCleanup(context);
        } else {
            Throwable releaseFailure = Failures.run(null, () -> resumeCleanup(context));
            if (releaseFailure != null) { recordFailure(releaseFailure); }
        }
        return action.publication();
    }

    private static void execute(RequestContext ctx, Runnable effect) {
        if (effect == null) { return; }
        ctx.requireOutsideContextLock("request effects");
        effect.run();
    }

    Runnable finalizationEffects(TerminalAction action, PreemptionRegistration signal) {
        if (action == null) { return null; }
        return () -> {
            Throwable failure = Failures.run(null, () -> executeFinalization(action));
            failure = Failures.run(failure, signal == null ? null
                    : () -> signal.signalResolution(new VictimResolution(action.requestContext().getRequestId(), VictimResolution.Outcome.REQUEST_END)));
            Failures.rethrow(failure, "request cleanup failed");
        };
    }

    Runnable deliveryEffects(RequestContext ctx, DeliveryPublication delivery, PreemptionRegistration signal) {
        return () -> {
            Throwable failure = Failures.run(null, () -> submitDeliveryResponse(delivery));
            failure = Failures.run(failure, signal == null ? null
                    : () -> signal.signalResolution(new VictimResolution(ctx.getRequestId(), VictimResolution.Outcome.DELIVERY_RESUMED)));
            Failures.rethrow(failure, "request cleanup failed");
        };
    }

    /** Keep ACK selection on the response execution queue, so newer terminal facts can win. */
    private void submitDeliveryResponse(DeliveryPublication delivery) {
        try {
            delivery.item().ctx().requireOutsideContextLock("delivery response submission");
            try {
                if (delivery.requestDeadline() != null) { delivery.requestDeadline().cancel(); }
            } catch (Throwable failure) {
                Logger.error("Delivery deadline cancellation failed request_id={}", delivery.item().requestId(), failure);
            }
            responseCompletions.submit(delivery.publication().registration, () -> {
                try {
                    if (delivery.batchEnqueueStartedAtMs() > 0L && delivery.item().ctx().getAckAtMs() > 0L) {
                        runtime.deliveryReporter().reportLatency(DeliveryMetricsReporter.Latency.DISPATCH_ACK,
                                RoleType.PREFILL.name(),
                                delivery.item().prefillEp().getIp(),
                                Math.max(0L, delivery.item().ctx().getAckAtMs() - delivery.batchEnqueueStartedAtMs()));
                    }
                } catch (Throwable failure) {
                    Logger.error("Delivery ACK reporting failed request_id={}", delivery.item().requestId(), failure);
                }
                return completeFutureResult(delivery.publication(), selectPublication(delivery.item().ctx(), delivery.publication(),
                        ResponseCompletion.RESPONSE, delivery.response(), null, false));
            });
        } catch (RuntimeException | Error failure) {
            delivery.publication().abandonIfUnused();
            throw failure;
        }
    }

    static ResponseResult selectPublication(RequestContext ctx, PublicationPermit permit, ResponseCompletion completion, Response response, Throwable failure, boolean mayInterruptIfRunning) {
        ctx.requireOutsideContextLock("response selection");
        checkArgument(permit.requestContext == ctx
                && (completion == ResponseCompletion.RESPONSE
                || permit.kind == PublicationKind.TERMINAL),
                "incompatible publication permit");
        permit.consumeForSelection();
        try {
            synchronized (ctx) {
                return ctx.claimPublicationResultLocked(permit.kind, completion, response, failure, mayInterruptIfRunning);
            }
        } catch (RuntimeException | Error selectionFailure) {
            permit.closePublication();
            throw selectionFailure;
        }
    }

    /**
     * Single-use response selection attempt for an exact request and publication kind.
     * Consuming this permit does not select a winner; selectedResponse records that decision.
     * The associated execution registration keeps shutdown waiting until completion or abandonment.
     */
    static final class PublicationPermit {

        final ResponseCompletionExecutor.CompletionRegistration registration;

        final RequestContext requestContext;

        final PublicationKind kind;

        private final AtomicBoolean consumed = new AtomicBoolean();

        PublicationPermit(ResponseCompletionExecutor.CompletionRegistration registration, RequestContext requestContext, PublicationKind kind) {
            this.registration = Objects.requireNonNull(registration);
            this.requestContext = requestContext;
            this.kind = kind;
        }

        void closePublication() { registration.close(); }

        /**
         * Abandon a permit only when no other submitter consumed it.
         */
        void abandonIfUnused() {
            if (consumed.compareAndSet(false, true)) {
                closePublication();
            }
        }

        void consumeForSelection() {
            if (!consumed.compareAndSet(false, true)) {
                throw new IllegalStateException("response selection permit already consumed for request " + requestContext.getRequestId());
            }
        }
    }

    static boolean completeFutureResult(PublicationPermit permit, ResponseResult result) {
        RequestContext context = permit.requestContext;
        context.requireOutsideContextLock("response completion");
        if (result == null) { return false; }
        var future = context.future();
        return switch (result.completion()) {
            case RESPONSE -> future.completeOwned(result.response());
            case FAILURE -> future.completeExceptionallyOwned(result.failure());
            case CANCELLATION -> future.cancelOwned(result.interrupt());
        };
    }

    private void submitResponse(PublicationPermit permit, ResponseResult result) {
        try {
            permit.requestContext.requireOutsideContextLock("response submission");
            responseCompletions.submit(permit.registration, () -> completeFutureResult(permit, result));
        } catch (RuntimeException | Error failure) {
            permit.closePublication();
            throw failure;
        }
    }

    boolean completeResponseNow(PublicationPermit permit, ResponseResult result) {
        return responseCompletions.completeNow(permit.registration, () -> completeFutureResult(permit, result));
    }

    // ── 响应：本地结束、结果仲裁与发布交接 ──
    boolean terminateLocallyAndPublishResponse(RequestContext ctx, Response response) {
        PublicationPermit permit = terminateLocallyAndAcquirePublication(ctx, responseOutcome(response));
        if (permit == null) {
            return false;
        }
        submitResponse(permit, selectPublication(ctx, permit, ResponseCompletion.RESPONSE, response, null, false));
        return true;
    }

    boolean completeExternal(RequestContext ctx, ResponseCompletion completion, Response response, Throwable error, boolean interrupt) {
        ctx.requireOutsideContextLock("external Future completion");
        TerminalOutcome outcome = switch(completion) {
            case RESPONSE ->
                responseOutcome(response);
            case FAILURE ->
                {
                    Objects.requireNonNull(error, "error");
                    yield TerminalOutcome.fail("external future failure" + (error.getMessage() == null ? "" : ": " + error.getMessage()));
                }
            case CANCELLATION ->
                TerminalOutcome.cancel(CancelReason.CLIENT_CANCELLED.getMessage());
        };
        PublicationPermit permit = terminateLocallyAndAcquirePublication(ctx, outcome);
        return permit != null && completeResponseNow(permit, selectPublication(ctx, permit, completion, response, error, interrupt));
    }

    private static TerminalOutcome responseOutcome(Response response) {
        String detail = response != null && response.getErrorMessage() != null ? response.getErrorMessage() : "external future completion";
        return response != null && !response.isSuccess() ? TerminalOutcome.fail(detail) : TerminalOutcome.complete(detail);
    }

    private PublicationPermit terminateLocallyAndAcquirePublication(RequestContext ctx, TerminalOutcome transition) {
        TerminalAction action;
        synchronized (ctx) {
            if (ctx.future().isDone() || ctx.selectedResponse() != null) {
                return null;
            }
            if (ctx.cancellationReason() != null || !ctx.canFinalizeBeforeExecutionLocked()) {
                return null;
            }
            if (transition.phase() == RequestState.Phase.COMPLETED
                    && ctx.deliveryClaimKind() == DeliveryClaimKind.NONE) {
                return null;
            }
            action = claimFinalizationLocked(ctx, ctx.decideFinalizationLocked(null, transition, null, true));
        }
        if (action == null) { return null; }
        try {
            return finishTerminal(action);
        } catch (RuntimeException | Error failure) {
            if (action.publication() != null) { Failures.run(failure, action.publication()::abandonIfUnused); }
            throw failure;
        }
    }

    PublicationPermit requirePublicationPermitLocked(RequestContext ctx, PublicationKind kind) {
        ctx.requireContextLock("publication registration");
        var registration = responseCompletions.tryRegister();
        if (registration == null) {
            throw new IllegalStateException("frontend publication is closed for request " + ctx.getRequestId());
        }
        return new PublicationPermit(registration, ctx, kind);
    }

    /** Decide and register under one request monitor; failed registration leaves terminal ownership untouched. */
    TerminalAction claimFinalizationLocked(RequestContext ctx, RequestEndDecision decision) {
        ctx.requireContextLock("terminal ownership");
        if (decision == null) { return null; }
        PublicationPermit permit = decision.requestPublication() ? requirePublicationPermitLocked(ctx, PublicationKind.TERMINAL) : null;
        try {
            return ctx.commitFinalizationLocked(decision, permit);
        } catch (RuntimeException | Error failure) {
            if (permit != null) { permit.abandonIfUnused(); }
            throw failure;
        }
    }

    PublicationPermit selectDeliveryFailureLocked(RequestContext ctx, RequestRoute exact, DeliveryResult.Status source, String detail) {
        ctx.requireContextLock("delivery failure selection");
        if (!ctx.ownsActiveRoute(exact) || ctx.hasCleanup()) { return null; }
        PublicationPermit permit = ctx.selectedResponse() == null && !ctx.future().isDone()
                ? requirePublicationPermitLocked(ctx, PublicationKind.TERMINAL) : null;
        try {
            ctx.recordDeliveryFailureLocked(source, detail, permit != null);
            if (permit != null) { permit.consumeForSelection(); }
            return permit;
        } catch (RuntimeException | Error failure) {
            if (permit != null) { permit.abandonIfUnused(); }
            throw failure;
        }
    }

    record Settlement(boolean prefillSettled, boolean decodeSettled, Throwable failure) { }

    static Settlement releaseResources(RequestRoute exact, DecodeResources.ReleaseReason releaseReason,
                             org.flexlb.balance.delivery.DeliveryResult.Status source,
                             boolean prefillSettled, boolean decodeSettled) {
        if (exact == null) { return new Settlement(true, true, null); }
        Throwable failure = null;
        try {
            if (!prefillSettled) { exact.prefillEp().releaseRequest(exact); }
            prefillSettled = true;
        } catch (Throwable problem) { failure = problem; }
        try {
            if (!decodeSettled) {
                if (exact.decodeEp() == null || exact.decodeReservation() == null) { decodeSettled = true; }
                else if (releaseReason != null) {
                    DecodeResources.ReservationReleaseResult released = Objects.requireNonNull(
                            exact.decodeEp().release(exact.decodeReservation(), releaseReason), "Decode release result");
                    decodeSettled = released == DecodeResources.ReservationReleaseResult.RELEASED
                            || released == DecodeResources.ReservationReleaseResult.STALE;
                } else {
                    checkArgument(source == DeliveryResult.Status.NOT_SENT || source == DeliveryResult.Status.PREFILL_REJECTED,
                            "expected a definite request failure");
                    if (source == DeliveryResult.Status.NOT_SENT) { exact.decodeEp().release(exact.decodeReservation(), DecodeResources.ReleaseReason.NOT_SENT); }
                    decodeSettled = !exact.decodeEp().hasOwnedResources(exact.decodeReservation());
                }
            }
        } catch (Throwable problem) { failure = Failures.append(failure, problem); }
        return new Settlement(prefillSettled, decodeSettled, failure);
    }

    /** Resume the one request cleanup owner when delivery release evidence becomes complete. */
    void enqueueCleanup(RequestContext ctx) {
        RequestContext.CleanupQueue queued;
        synchronized (ctx) { queued = ctx.tryQueueCleanupLocked(); }
        if (queued == null) { return; }
        try {
            submitContinuation(ctx, () -> {
                synchronized (ctx) { ctx.releaseCleanupQueueLocked(queued); }
                resumeCleanup(ctx);
            });
        } catch (RuntimeException | Error failure) {
            synchronized (ctx) { ctx.releaseCleanupQueueLocked(queued); }
            throw failure;
        }
    }

    void resumeCleanup(RequestContext ctx) {
        ctx.requireOutsideContextLock("request cleanup");
        DeliveryClaim delivery;
        boolean failedDelivery;
        synchronized (ctx) {
            if (!ctx.hasCleanup()) { return; }
            delivery = ctx.delivery();
            failedDelivery = ctx.cleanupSource() != null;
        }
        if (delivery != null && failedDelivery) {
            abandonDelivery(delivery, ctx.cancellationReason() == null ? CancelReason.CLIENT_CANCELLED : ctx.cancellationReason());
        }
        Throwable error = null;
        try {
            while (true) {
                CleanupPass pass;
                synchronized (ctx) { pass = ctx.beginCleanup(); }
                if (pass == null) { break; }
                error = Failures.run(error, pass.requestDeadline() == null ? null : pass.requestDeadline()::cancel);
                error = Failures.run(error, () -> ExpirationTimer.releaseDecisionDeadline(pass.decisionDeadline()));
                var settlement = releaseResources(pass.route(), pass.releaseReason(), pass.source(),
                        pass.prefillSettled(), pass.decodeSettled());
                error = Failures.append(error, settlement.failure());
                TerminalAction completed;
                TerminalAction start;
                synchronized (ctx) {
                    CleanupNext next = ctx.finishCleanup(pass, settlement.prefillSettled(), settlement.decodeSettled());
                    if (next == CleanupNext.STALE) { break; }
                    if (next == CleanupNext.REPEAT) { continue; }
                    completed = ctx.completedCleanupActionLocked();
                    start = completed == null ? claimFinalizationLocked(ctx, ctx.decideCleanupCompletionLocked()) : null;
                }
                if (completed != null) {
                    commitTerminalRecord(ctx, completed);
                } else {
                    executeFinalization(start);
                }
                break;
            }
        } catch (Throwable problem) { error = Failures.append(error, problem); }
        Failures.rethrow(error, "request cleanup failed");
    }

}

/**
 * Endpoint that published the terminal request status; Decode statuses follow its ledger update.
 */
enum WorkerTerminalSource {
    PREFILL_ENDPOINT, DECODE_ENDPOINT
}

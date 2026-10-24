package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;
import static org.flexlb.balance.scheduler.ContextResult.*;

/**
 * The single QUEUE admission owner for one FlexLB model.
 *
 * <p>This queue orders the start of placement attempts for the model. A request is
 * selected from the complete live candidate fleet before the endpoint runtime
 * receives it. A bounded set of in-flight requests controls how many
 * independent routes can be prepared together; request collection and grouping
 * remain exclusively endpoint-runtime concerns. The queue lock protects only
 * index operations; route projection and RPCs are never performed while it is
 * held.</p>
 *
 * <p>Ready requests receive planning slots in FIFO/priority order. A request
 * that cannot be admitted waits for a relevant capacity event; it does not fence
 * later requests from trying the same worker with different resource demands.
 * Planning is bounded and parallel; completed plans are published without waiting
 * for earlier planners. Arrivals and wakeups do not invalidate in-flight work.
 * WorkerBatcher owns SINGLE/FIXED_WINDOW grouping.</p>
 */
public final class QueuedRequestScheduler extends AbstractRequestScheduler implements AutoCloseable {

    private static final int MIN_PLANNER_THREADS = 1;

    private final RequestWorkerSelector router;
    private final DeliveryMetricsReporter reporter;
    private final DecodeCapacityAcquirer decodeCapacity;
    private final PlacementAvailability availability;
    private final QueueExecutionSettings queueSettings;
    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final OrderedRequestQueue orderedQueue;
    private final PlacementCapacityWaitQueue waitingRequests;
    /** Entries exist only while the scheduler owns global placement. */
    private final Map<RequestContext, GlobalQueueEntry> queuedEntries = new IdentityHashMap<>();
    private final ArrayDeque<QueueEvent> events = new ArrayDeque<>();
    private final Set<Plan> pendingPreemptions = Collections.newSetFromMap(new IdentityHashMap<>());
    /** Waiting on local preemption slots, independent of physical Worker capacity edges. */
    private final Set<GlobalQueueEntry> preemptionQuotaWaiters =
            Collections.newSetFromMap(new IdentityHashMap<>());
    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ArrayDeque<Plan> completedPlans = new ArrayDeque<>();
    private final ArrayDeque<RequestContext> controlInbox = new ArrayDeque<>();
    /** Published by the decision thread; timeout readers never acquire the queue lock. */
    private volatile Map<String, Object> latestQueueWaitSnapshot = Map.of("cause", "waiting for placement");
    private final ExecutorService planners;
    private final Thread decisionThread;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            key -> postEvent(new QueueEvent(EventKind.CAPACITY, null, null, key));

    QueuedRequestScheduler(FlexlbConfig config, RequestWorkerSelector router,
            DeliveryMetricsReporter reporter, DecodeCapacityAcquirer decodeCapacity,
            SchedulerRuntime runtime, PlacementAvailability availability,
            DecodeResources.AdmissionCapacity capacity, RecentCacheKeyTraceReporter traceReporter) {
        super(runtime, config, capacity, traceReporter);
        this.queueSettings = QueueExecutionSettings.capture(config);
        this.router = Objects.requireNonNull(router, "router");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.decodeCapacity = Objects.requireNonNull(
                decodeCapacity, "decodeCapacity");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.orderedQueue = new OrderedRequestQueue(queueSettings.priorityOrdering());
        this.waitingRequests = new PlacementCapacityWaitQueue(queueSettings.priorityOrdering(), availability);
        this.scanBudgetMultiplier = config.queueScheduler().getScanBudgetMultiplier();

        this.plannerCount = Math.max(MIN_PLANNER_THREADS,
                config.getInternalRuntime()
                        .getQueuePlannerThreads());
        this.planners = Executors.newFixedThreadPool(this.plannerCount,
                Thread.ofPlatform().daemon().name("flexlb-global-planner-", 1).factory());
        this.decisionThread = new Thread(this::runDecisionLoop,
                "flexlb-global-decision");
        this.decisionThread.setDaemon(true);
        this.decisionThread.setUncaughtExceptionHandler((thread, failure) -> {
            Logger.error("Global queue decision thread failed", failure);
            close();
        });
        availability.addListener(availabilityListener);
    }

    /** Start scheduler workers before accepting requests. */
    void start() {
        decisionThread.start();
    }

    /** Latest queue wait cause and counters; this snapshot is not specific to the current request. */
    Map<String, Object> getLatestQueueWaitSnapshot() {
        return latestQueueWaitSnapshot;
    }

    @Override
    protected void queueControl(RequestContext exact) {
        postEvent(new QueueEvent(EventKind.CONTROL, null, exact, null));
    }

    public void completeWithdrawal(AdmissionHandle withdrawal, boolean committed) {
        checkArgument(withdrawal.owner().data().getScheduler() == this, "foreign withdrawal");
        Throwable failure = null;
        RequestContext context = withdrawal.owner();
        // Retain the exact queued route while the placement token owns withdrawal.
        RequestRoute item = withdrawal.withdrawingRoute();
        try {
            if (committed) {
                // Decode replacement has committed. Never take the Prefill lock under the context monitor.
                if (!item.prefillEp().control(item, WorkerBatcher.Control.REMOVE)) {
                    throw new IllegalStateException("withdrawn route is no longer queued: " + context.data().getRequestId());
                }
                synchronized (context) {
                    context.bindPlacement(withdrawal.token, item, PublicationResult.WITHDRAWN);
                }
            }
        } catch (Throwable detachFailure) {
            failure = Failures.append(failure, detachFailure);
            try {
                withdrawal.terminate(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, "queued route withdrawal failed"));
            } catch (Throwable terminalFailure) {
                failure = Failures.append(failure, terminalFailure);
            }
        } finally {
            try {
                withdrawal.finish();
                var state = context.snapshot();
                if (committed && state.phase() == Phase.OPEN && state.cancellation() == null && !context.data().getFuture().isDone() && !requeue(item)) {
                    item.future().complete(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, "scheduler closed during route withdrawal"));
                }
            } catch (Throwable closeFailure) {
                failure = Failures.append(failure, closeFailure);
                try {
                    item.future().complete(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, "queued route requeue failed"));
                } catch (Throwable terminalFailure) {
                    failure = Failures.append(failure, terminalFailure);
                }
            }
        }
        Failures.rethrow(failure, "queued route withdrawal failed");
    }

    public AdmissionHandle claimQueuedRoute(DecodeEndpoint endpoint, DecodeResources.ReservationHandle victim, int incomingPriority) {
        if (!runtime.registerAdmission()) { return null; }
        boolean transferred = false;
        PlacementToken token = null;
        RequestContext context = requests.findActive(victim.requestId());
        try {
            if (context == null || context.data().getScheduler() != this) { return null; }
            synchronized (context) {
                var state = context.snapshot();
                RequestRoute item = state.route();
                // Context serializes the exact queue claim with delivery permit transfer.
                if (item == null || item.priority() >= incomingPriority
                        || item.decodeEp() != endpoint || !Objects.equals(item.decodeReservation(), victim)
                        || !requests.isCurrent(context) || state.phase() != Phase.OPEN
                        || state.cancellation() != null || state.delivery() != null || state.placement() != null
                        || state.preemption() != null || state.engine().decodeAccepted()
                        || context.data().getFuture().isDone() || item.requestExpired(System.currentTimeMillis())
                        || !endpoint.ownership(victim).withdrawable()) { return null; }
                var claim = context.beginPlacement(item);
                if (!(claim instanceof Granted<PlacementToken> granted)) { return null; }
                token = granted.value();
                AdmissionHandle handle = new AdmissionHandle(token, this);
                transferred = true;
                return handle;
            }
        } finally {
            if (!transferred) {
                if (token != null) { context.completePlacement(token, PublicationResult.FAILED, System.currentTimeMillis()); }
                runtime.finishAdmission();
            }
        }
    }

    @Override
    protected Map<String, Object> cancellationDiagnostics(RequestContext ctx, CancelReason reason, String message) {
        if (reason != CancelReason.DEADLINE_EXCEEDED) { return null; }
        var state = ctx.snapshot();
        if (state.delivery() != null) { return Map.of("cause", message); }
        return state.route() != null ? state.route().prefillEp().getLatestQueueWaitSnapshot() : latestQueueWaitSnapshot;
    }

    boolean resolveControl(RequestContext context, boolean shuttingDown) {
        if (context == null || context.data().getScheduler() != this || !requests.isCurrent(context)) { return false; }
        var state = context.snapshot();
        if (state.route() == null && state.placement() == null && state.cancellation() != null) {
            var cancelled = state.cancellation();
            if (cancelled.operation() != null) {
                context.completeCancel(cancelled.operation(), new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN,
                        false, false), System.currentTimeMillis());
            }
            closeRequest(context, state.pendingEnd() != null ? state.pendingEnd()
                    : new EndEvidence(null, EndKind.FRONTEND, null, false, 0L,
                            cancelled.errorType(), cancelled.detail(), false, false), null);
        }
        if (shuttingDown) {
            closeRequest(context, new EndEvidence(state.route(), EndKind.SHUTDOWN, null, false, 0L,
                    StrategyErrorType.DISPATCH_FAILED, "request scheduler is shutting down", false, false), null);
        }
        return true;
    }

    /** Enqueue without selecting an endpoint on the ingress thread. */
    @Override
    public CompletableFuture<Response> submit(RequestContext context) {
        return submit(context, () -> {});
    }

    @Override
    public CompletableFuture<Response> submit(RequestContext context, Runnable onRegistered) {
        Objects.requireNonNull(onRegistered, "onRegistered");
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
        if (context != null && context.data().getConfig().isDirect()) {
            return CompletableFuture.completedFuture(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                    "request configuration does not match scheduler mode"));
        }
        if (closed.get()) {
            return CompletableFuture.completedFuture(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                    "request scheduler is not accepting new requests"));
        }
        try {
            if (context != null && (context.data().getRequestExpiresAtMs() <= 0L
                    || System.currentTimeMillis() >= context.data().getRequestExpiresAtMs())) {
                context.data().setSchedulingDiagnostics(latestQueueWaitSnapshot);
            }
            CompletableFuture<Response> future = register(context, onRegistered, StrategyErrorType.RESOURCE_EXHAUSTED);
            var state = context == null ? null : context.snapshot();
            if (!future.isDone() && state.phase() == Phase.OPEN && state.cancellation() == null) {
                if (!trySubmitRegistered(context)) { resolveControl(context, true); }
            } else if (context != null && context.data().getScheduler() == this && context.data().getFuture() == future) {
                resolveControl(context, false);
            }
            return future;
        } catch (RuntimeException failure) {
            Response response = buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (context != null && context.data().getScheduler() == this) {
                publish(context, response, false);
                return context.data().getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    boolean trySubmitRegistered(RequestContext context) {
        try {
            Objects.requireNonNull(context, "context");
            CompletableFuture<Response> future = Objects.requireNonNull(context.data().getFuture(), "registered future");
            var state = context.snapshot();
            if (future.isDone() || state.phase() != Phase.OPEN || state.cancellation() != null) { return true; }
            GlobalQueueEntry entry = new GlobalQueueEntry(context,
                    router.resolvePolicyGroup(context));
            if (!postEvent(new QueueEvent(EventKind.SUBMIT, entry, null, null))) { return false; }
            future.whenComplete((ignored, failure) -> queueControl(context));
            return true;
        } catch (RuntimeException failure) {
            // Once registration has been handed here, this mode owns execution failures.
            if (!publish(context, Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                            "Queue submission failed: " + failure.getMessage()), false)) {
                resolveControl(context, false);
            }
            return true;
        }
    }

    /** A withdrawal is not a new request: keep its sequence, context, Future and absolute deadline. */
    public boolean requeue(RequestRoute previous) {
        if (previous.future().isDone()) { return true; }
        RequestContext context = previous.ctx();
        GlobalQueueEntry entry = new GlobalQueueEntry(context, router.resolvePolicyGroup(context));
        entry.sequence = context.data().getPlacementSequence();
        return postEvent(new QueueEvent(EventKind.REQUEUE, entry, null, null));
    }

    private boolean postEvent(QueueEvent event) {
        lock.lock();
        try {
            if (closed.get()) { return false; }
            events.addLast(event);
            changed.signal();
            return true;
        } finally { lock.unlock(); }
    }

    private void consumeEventsLocked() {
        QueueEvent event;
        while ((event = events.pollFirst()) != null) {
            switch (event.kind) {
                case CAPACITY -> waitingRequests.capacityChanged(event.key);
                case CONTROL -> {
                    controlInbox.addLast(event.context);
                }
                case SUBMIT, REQUEUE -> {
                    GlobalQueueEntry entry = event.entry;
                    checkState(queuedEntries.putIfAbsent(entry.context, entry) == null,
                            "duplicate global queue identity");
                    if (event.kind == EventKind.REQUEUE) {
                        entry.context.data().setPlanType("");
                        entry.context.data().setPlanCost(0L);
                        entry.context.data().setVictimCount(0);
                        orderedQueue.restore(entry);
                    } else {
                        orderedQueue.add(entry);
                        if (entry.context.data().getPlacementSequence() == 0L) { entry.context.data().setPlacementSequence(entry.sequence); }
                    }
                    if (entry.context.data().getFuture().isDone() || (requests.isCurrent(entry.context) && entry.context.snapshot().cancellation() != null)) {
                        controlInbox.addLast(entry.context);
                    }
                }
            }
        }
    }

    private record ControlAction(RequestContext context, GlobalQueueEntry entry) { }

    private void runControlActions(List<ControlAction> readyControls) {
        for (ControlAction control : readyControls) {
            RequestContext context = control.context();
            GlobalQueueEntry entry = control.entry();
            try {
                resolveControl(context, false);
                var state = context.snapshot();
                if (entry != null && context.data().getScheduler() == this && requests.isCurrent(context)
                        && state.phase() == Phase.OPEN && state.route() == null && state.placement() == null
                        && state.preemption() == null && state.cancellation() == null && !context.data().getFuture().isDone()) {
                    lock.lock();
                    try {
                        if (!closed.get() && queuedEntries.get(context) == entry && entry.removed) {
                            orderedQueue.restore(entry);
                            changed.signal();
                        }
                    } finally {
                        lock.unlock();
                    }
                } else if (entry != null) { removeRequest(entry); }
            } catch (Throwable failure) {
                Logger.error("Global queue control failed: request_id={}",
                        context.data().getRequestId(), failure);
                publish(context, buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                        "global queue control failed"), false);
            }
        }
        readyControls.clear();
    }

    private void runDecisionLoop() {
        boolean drained = false;
        try {
            List<ControlAction> controls = new ArrayList<>();
            while (true) {
                if (closed.get() && !drained) { drainOnClose(); drained = true; }
                Plan completed;
                List<GlobalQueueEntry> candidates;
                lock.lock();
                try {
                    if (closed.get() && inFlight.isEmpty() && pendingPreemptions.isEmpty()) { break; }
                    consumeEventsLocked();
                    RequestContext controlled;
                    while ((controlled = controlInbox.pollFirst()) != null) {
                        GlobalQueueEntry entry = queuedEntries.get(controlled);
                        if (entry != null) {
                            orderedQueue.remove(entry);
                            waitingRequests.remove(entry);
                        }
                        controls.add(new ControlAction(controlled, entry));
                    }
                    completed = completedPlans.pollFirst();
                } finally { lock.unlock(); }
                runControlActions(controls);
                if (completed != null) { processCompletedPlan(completed); }
                lock.lock();
                try {
                    // Control effects can restore an older entry, and a completed route
                    // can publish capacity. Both precede selecting the next planning slots.
                    consumeEventsLocked();
                    RequestContext controlled;
                    while ((controlled = controlInbox.pollFirst()) != null) {
                        GlobalQueueEntry entry = queuedEntries.get(controlled);
                        if (entry != null) {
                            orderedQueue.remove(entry);
                            waitingRequests.remove(entry);
                        }
                        controls.add(new ControlAction(controlled, entry));
                    }
                    int slots = plannerCount - inFlight.size();
                    candidates = closed.get() || slots == 0 ? List.of() : planningCandidates(slots);
                    inFlight.addAll(candidates);
                } finally { lock.unlock(); }
                runControlActions(controls);
                candidates.forEach(this::submitPlan);
                lock.lock();
                try {
                    // Arrivals and plan publication use this same lock. Check work again
                    // after effects; a signal while effects ran must never be lost.
                    while (events.isEmpty() && controlInbox.isEmpty() && completedPlans.isEmpty()
                            && (!closed.get() || drained && (!inFlight.isEmpty() || !pendingPreemptions.isEmpty()))
                            && (inFlight.size() == plannerCount || closed.get()
                                || (!orderedQueue.hasUnscannedRequests() && !waitingRequests.hasReadyRequests()))) {
                        try { changed.await(); }
                        catch (InterruptedException interruption) {
                            if (closed.get()) { Thread.currentThread().interrupt(); }
                        }
                    }
                } finally { lock.unlock(); }
            }
        } finally {
            closed.set(true);
            availability.removeListener(availabilityListener);
            planners.shutdown();
            drainOnClose();
        }
    }

    private void submitPlan(GlobalQueueEntry entry) {
        try {
            planners.execute(() -> {
                Plan plan = new Plan(entry, availability.sequence());
                try {
                    if (!entry.removed && !entry.context.data().getFuture().isDone()) {
                        if (entry.context.data().getConfig().getDispatcher().requiresGenerateInput()) {
                            entry.context.data().prepareGenerateInput();
                        }
                        if (entry.context.data().getRequestExpiresAtMs() <= 0L || System.currentTimeMillis() >= entry.context.data().getRequestExpiresAtMs()) {
                            cancel(entry.context, 0L, CancelReason.DEADLINE_EXCEEDED);
                        } else {
                            plan.handle = claimAdmissionHandle(entry.context.data().getRequestId(), entry.context.data().getFuture());
                            if (plan.handle != null) {
                                plan.result = Objects.requireNonNull(router.select(entry.context, entry.routingGroup), "placement result");
                                plan.handle.recordDiagnostics(plan.result.diagnostics());
                                plan.waitKey = plan.result.blocker();
                            }
                        }
                    }
                } catch (Throwable failure) { plan.planningFailure = failure; }
                finally { publishPlan(plan); }
            });
        } catch (Throwable failure) {
            publishPlan(Plan.failure(entry, failure, availability.sequence()));
        }
    }

    private void publishPlan(Plan plan) {
        lock.lock();
        try {
            completedPlans.addLast(plan);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void processCompletedPlan(Plan plan) {
        boolean retry = false;
        boolean completingPreemption = plan.preemptionCompleted;
        Map<String, Object> diagnostics = plan.result.diagnostics();
        try {
            boolean abandoned = closed.get() || completingPreemption && (plan.entry.removed || plan.entry.context.data().getFuture().isDone());
            Outcome outcome = Outcome.DONE;
            Throwable commitFailure = null;
            try {
                if (abandoned) {
                    settlePreemptionResult(plan, false);
                } else {
                    Failures.rethrow(plan.planningFailure, "placement planning failed");
                    outcome = commit(plan);
                }
            } catch (Throwable failure) { commitFailure = failure; }
            finally {
                if (abandoned || completingPreemption || !pendingPreemptions.contains(plan)) {
                    try { plan.close(); }
                    catch (Throwable failure) { runtime.recordFailure(failure); Logger.warn("Failed to close global queue route plan", failure); }
                    finally { commitFailure = Failures.run(commitFailure, plan::finishAdmission); }
                }
            }
            Failures.rethrow(commitFailure, "placement commit failed");
            retry = outcome == Outcome.REPLAN
                    || (outcome == Outcome.BLOCKED && !park(plan, diagnostics));
        } catch (Throwable failure) {
            removeRequest(plan.entry);
            completeDecisionResponse(plan.entry, buildErrorResponse(
                    StrategyErrorType.DISPATCH_FAILED,
                    "Placement failed: " + failure.getMessage()));
            Logger.error("Global queue commit failed: request_id={}",
                    plan.entry.context.data().getRequestId(), failure);
        } finally {
            // Plan ownership has settled before this slot becomes eligible again.
            lock.lock();
            try {
                inFlight.remove(plan.entry);
                if (completingPreemption) {
                    pendingPreemptions.remove(plan);
                    for (GlobalQueueEntry waiter : preemptionQuotaWaiters) {
                        waitingRequests.remove(waiter);
                        if (!waiter.removed && !waiter.context.data().getFuture().isDone()) { orderedQueue.markRequestReadyForRetry(waiter); }
                    }
                    preemptionQuotaWaiters.clear();
                }
                if (retry && (!plan.entry.removed && !plan.entry.context.data().getFuture().isDone())) {
                    orderedQueue.markRequestReadyForRetry(plan.entry);
                }
            } finally {
                lock.unlock();
            }
        }
    }

    /** Caller holds lock; every examined entry counts against the scan budget. */
    private List<GlobalQueueEntry> planningCandidates(int slots) {
        waitingRequests.resumeReady(slots, orderedQueue::markRequestReadyForRetry);
        return orderedQueue.scanForPlanningCandidates(
                slots, (int) Math.min(Integer.MAX_VALUE, Math.ceil(slots * scanBudgetMultiplier)), entry -> {
                    if (entry.context.data().getFuture().isDone()) {
                        removeRequest(entry);
                        return false;
                    }
                    return !entry.removed && !inFlight.contains(entry)
                            && pendingPreemptions.stream().noneMatch(plan -> plan.entry == entry)
                            && !waitingRequests.isWaiting(entry);
                });
    }

    private Outcome commit(Plan plan) {
        GlobalQueueEntry entry = plan.entry;
        boolean reclaimed = plan.preemptionCompleted;
        String rejection = null;
        try {
            if (reclaimed) {
                if (plan.preemptionFailure != null || plan.preemptionResult == null) {
                    rejection = "Decode eviction control failed before commit";
                } else if (!plan.preemptionResult.committed()) {
                    rejection = plan.preemptionResult.detail();
                } else if (!settlePreemptionResult(plan, true)) {
                    rejection = "Decode generation retired before canonical placement";
                }
            }
            if (rejection == null && (reclaimed || (!entry.removed && !entry.context.data().getFuture().isDone()))) {
                PlacementResult<RequestRoute, PlacementKey> result = plan.result;
                switch (result.status()) {
                    case REJECTED -> {
                        plan.finishAdmission();
                        completeDecisionResponse(entry, result.failure());
                    }
                    case CLOSED -> { }
                    case BLOCKED -> { return Outcome.BLOCKED; }
                    case SUCCESS -> {
                        RequestRoute routing = result.value();
                        var publication = enqueueRoute(plan);
                        if (reclaimed && publication.status() != PlacementResult.Status.SUCCESS) {
                            rejection = "selected Prefill capacity changed before canonical placement";
                        } else if (publication.status() == PlacementResult.Status.BLOCKED) {
                            plan.waitKey = publication.blocker();
                            WorkerEndpoint blocked = routing.blockedEndpointIfCurrent(plan.waitKey);
                            if (blocked == null) { return Outcome.REPLAN; }
                            if (!tryPriorityRescue(plan, blocked)) { return Outcome.BLOCKED; }
                            return Outcome.PENDING;
                        }
                    }
                }
            }
        } catch (RuntimeException | Error failure) {
            if (!reclaimed) { throw failure; }
            rejection = "Decode eviction placement failed: " + failure.getMessage();
        }
        if (rejection != null) {
            plan.handle.terminate(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED, rejection));
        }
        removeRequest(entry);
        return Outcome.DONE;
    }

    private boolean tryPriorityRescue(Plan plan, WorkerEndpoint blockedEndpoint) {
        GlobalQueueEntry entry = plan.entry;
        if (!queueSettings.priorityOrdering() || entry.context.data().getFuture().isDone()) { return false; }
        if (pendingPreemptions.size() >= plannerCount) {
            preemptionQuotaWaiters.add(entry);
            return false;
        }
        var execution = decodeCapacity.tryReclaim(entry.context, blockedEndpoint);
        if (execution == null) {
            return false;
        }
        pendingPreemptions.add(plan);
        execution.whenComplete((result, failure) -> {
            plan.preemptionResult = result;
            plan.preemptionFailure = failure;
            plan.preemptionCompleted = true;
            publishPlan(plan);
        });
        return true;
    }

    private boolean settlePreemptionResult(Plan plan, boolean adopt) {
        var result = plan.preemptionResult;
        if (result == null || !result.committed()) { return false; }
        plan.preemptionResult = null;
        RequestRoute route = plan.result.value();
        if (adopt) {
            boolean adopted = false;
            Throwable failure = null;
            try {
                if (route.decodeEp() == null || result.reservation().requestId() != route.requestId()
                        || plan.pendingDecodeRollback != null) { return false; }
                if (!route.decodeEp().markQueued(route.decodePin(), result.reservation()).status().applied()) { return false; }
                plan.pendingDecodeRollback = result.reservation();
                adopted = true;
                return true;
            } catch (RuntimeException | Error cause) {
                failure = cause;
                throw cause;
            } finally {
                if (!adopted && route.decodeEp() != null) {
                    try {
                        route.decodeEp().release(result.reservation(), DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                    } catch (RuntimeException | Error cleanupFailure) {
                        if (failure == null) { throw cleanupFailure; }
                        Failures.append(failure, cleanupFailure);
                    }
                }
            }
        }
        try {
            route.decodeEp().release(result.reservation(), DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
        } catch (RuntimeException | Error failure) {
            runtime.recordFailure(failure);
            throw failure;
        }
        return false;
    }

    PlacementResult<RequestRoute, PlacementKey> enqueueRoute(Plan plan) {
        RequestContext context = plan.entry.context;
        RequestRoute routing = plan.result.value();
        if (plan.pendingDecodeRollback == null) {
            var request = context.data().getRequest();
            plan.pendingDecodeRollback = routing.decodeEp() == null ? null
                    : routing.decodeEp().tryReserveQueuedRequest(routing.decodePin(), routing.requestId(),
                            request.getHardKvTokens(), request.getExpectedKvTokens(), routing.priority(), decodeAdmissionCapacity).handle();
            if (routing.decodeEp() != null && plan.pendingDecodeRollback == null) {
                return PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
            }
        }
        RequestRoute route = RequestRoute.create(context, routing, plan.pendingDecodeRollback);
        plan.acquired = route;
        var placement = context.snapshot().placement();
        plan.placementToken = placement == null ? null : placement.operation();
        context.data().setRouteSubmittedNanos(System.nanoTime());
        PlacementResult<RequestRoute, PlacementKey> result = switch (this.commitRoute(route, new QueuePublication(plan, route))) {
            case SUCCESS -> PlacementResult.success(route);
            case BLOCKED -> PlacementResult.blocked(routing.placementKey(RoleType.PREFILL));
            case CLOSED -> PlacementResult.closed();
            case REJECTED -> throw new IllegalStateException("route commit cannot reject with a response");
        };
        if (result.status() == PlacementResult.Status.SUCCESS) {
            try {
                reporter.reportLatency(DeliveryMetricsReporter.Latency.ROUTE_SUBMIT,
                        route.prefill().getRole().name(), route.prefillEp().getIp(),
                        System.currentTimeMillis() - context.data().getStartTime());
            } catch (Throwable failure) { Logger.warn("Failed to record route-submit telemetry", failure); }
        }
        return result;
    }

    private final class QueuePublication implements Publication {
        private final Plan plan;
        private final RequestRoute route;
        private boolean published;

        private QueuePublication(Plan plan, RequestRoute route) {
            this.plan = plan;
            this.route = route;
        }

        @Override public void publish() {
            RequestRoute routing = plan.result.value();
            if (routing.prefillEp().offerPinned(routing.prefillPin(), route, queueSettings)) {
                published = true;
                plan.published = true;
                plan.pendingDecodeRollback = null;
            }
        }

        @Override public boolean published() { return published; }
    }

    private boolean park(Plan plan, Map<String, Object> diagnostics) {
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (!(!plan.entry.removed && !plan.entry.context.data().getFuture().isDone())) { return true; }
            parked = waitingRequests.park(plan.entry, plan.waitKey, plan.availabilitySequence);
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        Map<Integer, Integer> priorityCounts = new LinkedHashMap<>();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) { priorityCounts.put(priority, counts[priority]); }
        }
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("cause", plan.waitKey.role().name() + " placement unavailable");
        details.put("role", plan.waitKey.role().name());
        if (plan.waitKey.group() != null) { details.put("group", plan.waitKey.group()); }
        if (plan.waitKey.endpoint() != null) { details.put("endpoint", plan.waitKey.endpoint()); }
        details.put("capturedAtMs", System.currentTimeMillis());
        details.put("queueDepth", depth);
        details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
        if (diagnostics != null) {
            details.put("decision", diagnostics);
        }
        latestQueueWaitSnapshot = Collections.unmodifiableMap(details);
        return parked;
    }

    private void removeRequest(GlobalQueueEntry entry) {
        boolean acquire = !lock.isHeldByCurrentThread();
        if (acquire) { lock.lock(); }
        try {
            orderedQueue.remove(entry);
            waitingRequests.remove(entry);
            preemptionQuotaWaiters.remove(entry);
            queuedEntries.remove(entry.context, entry);
            changed.signal();
        } finally { if (acquire) { lock.unlock(); } }
    }

    private void completeDecisionResponse(GlobalQueueEntry entry, Response response) {
        try { publish(entry.context, response, false); }
        catch (Throwable failure) {
            Logger.error("Global queue response publication failed: request_id={}",
                    entry.context.data().getRequestId(), failure);
        }
    }

    private void drainOnClose() {
        List<GlobalQueueEntry> abandoned;
        lock.lock();
        try {
            Set<GlobalQueueEntry> all = Collections.newSetFromMap(new IdentityHashMap<>());
            all.addAll(orderedQueue.drain());
            all.addAll(queuedEntries.values());
            waitingRequests.clear();
            preemptionQuotaWaiters.clear();
            for (QueueEvent event : events) {
                if (event.kind == EventKind.SUBMIT || event.kind == EventKind.REQUEUE) {
                    all.add(event.entry);
                }
            }
            abandoned = List.copyOf(all);
            events.clear();
            queuedEntries.clear();
            controlInbox.clear();
        } finally {
            lock.unlock();
        }
        for (GlobalQueueEntry entry : abandoned) {
            try {
                if (!resolveControl(entry.context, true)) {
                    completeDecisionResponse(entry, buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "request scheduler is shutting down"));
                }
            } catch (Throwable failure) {
                Logger.error("Global queue close settlement failed: request_id={}",
                        entry.context.data().getRequestId(), failure);
            }
        }
    }

    @Override
    protected void closePlacement() { close(); }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            availability.removeListener(availabilityListener);
            lock.lock();
            try { changed.signal(); } finally { lock.unlock(); }
            // Late planner results release their own route ownership.
            planners.shutdown();
        }
        // Every caller observes the same drain, including a second shutdown owner.
        boolean interrupted = false;
        if (Thread.currentThread() != decisionThread) {
            while (decisionThread.isAlive()) {
                try {
                    decisionThread.join();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        while (!planners.isTerminated()) {
            try {
                planners.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }

    private enum Outcome {
        DONE,
        BLOCKED,
        REPLAN,
        PENDING
    }

    private enum EventKind { SUBMIT, REQUEUE, CONTROL, CAPACITY }
    private record QueueEvent(EventKind kind, GlobalQueueEntry entry, RequestContext context, PlacementKey key) { }

    static final class Plan implements AutoCloseable {
        private final GlobalQueueEntry entry;
        private AdmissionHandle handle;
        private DecodeResources.ReservationHandle pendingDecodeRollback;
        private RequestRoute acquired;
        private PlacementToken placementToken;
        private boolean published;
        private Throwable rollbackFailure;
        private Throwable planningFailure;
        private volatile boolean preemptionCompleted;
        private org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult preemptionResult;
        private Throwable preemptionFailure;
        // The plan owns selection pins and uncommitted Decode capacity.
        private PlacementResult<RequestRoute, PlacementKey> result;
        private final long availabilitySequence;
        // Retained after selection pins close so the request can be parked.
        private PlacementKey waitKey;

        private Plan(GlobalQueueEntry entry, long availabilitySequence) {
            this.entry = entry;
            this.availabilitySequence = availabilitySequence;
            this.result = PlacementResult.closed();
        }

        static Plan failure(GlobalQueueEntry entry, Throwable failure, long availabilitySequence) {
            Plan plan = new Plan(entry, availabilitySequence);
            plan.planningFailure = failure;
            return plan;
        }

        void finishAdmission() {
            AdmissionHandle owned = handle;
            handle = null;
            if (owned != null) {
                if (rollbackFailure == null) { owned.finish(); }
                else { owned.terminate(buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, "placement rollback failed")); }
            }
        }

        @Override
        public void close() {
            RequestRoute routing = result == null ? null : result.value();
            result = null;
            DecodeResources.ReservationHandle rollback = pendingDecodeRollback;
            pendingDecodeRollback = null;
            try (routing) {
                try {
                    if (rollback != null) {
                        var released = routing.decodeEp().release(rollback, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                        checkState(released == DecodeResources.ReservationReleaseResult.RELEASED
                                || released == DecodeResources.ReservationReleaseResult.STALE, "placement rollback retained Decode resources");
                    }
                    if (!published && acquired != null && placementToken != null) {
                        acquired.ctx().bindPlacement(placementToken, acquired, PublicationResult.ROLLBACK_COMPLETED);
                    }
                } catch (RuntimeException | Error failure) {
                    rollbackFailure = failure;
                    throw failure;
                }
            }
        }
    }
}

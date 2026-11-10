package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.RoutingCompletion;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/**
 * Selects queued requests and hands accepted routes to Prefill delivery.
 * Capacity changes and timers resume waiting requests; failed routes and Decode preemption
 * return through this scheduler's retry and cleanup paths.
 * The background driver runs global selection and delegates Prefill events and deadlines to its delivery scheduler.
 */
public class QueuedRequestScheduler extends AbstractRequestScheduler {
    private static final int MIN_PLANNER_THREADS = 1;

    private final RequestRouteSelector router;
    private final DecodePreemption decodePreemption;
    private final CapacityNotifier capacityNotifier;
    private final QueueExecutionSettings queueSettings;

    private final boolean priorityOrdering;
    private final int encoderRequestLimit;
    private final int plannerCount;
    private final double scanBudgetMultiplier;

    private final ReentrantLock lock = new ReentrantLock();
    private final SchedulingQueue orderedQueue;
    private final CapacityWaitQueue waitingRequests;

    /**
     * Entries exist only while the scheduler owns global placement.
     */
    private final Map<RequestContext, GlobalQueueEntry> queuedEntries = new IdentityHashMap<>();

    private final Set<WorkerEndpoint> unavailableEndpoints =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final Map<GlobalQueueEntry, CompletableFuture<Void>> pendingPreemptions =
            new IdentityHashMap<>();

    /**
     * Waiting on local preemption slots, independent of physical Worker capacity edges.
     */
    private final Set<GlobalQueueEntry> preemptionQuotaWaiters =
            Collections.newSetFromMap(new IdentityHashMap<>());

    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ExecutorService planners;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final CapacityNotifier.Listener capacityListener =
            new CapacityNotifier.Listener() {
                @Override
                public void onCapacityChanged(CapacityKey key) {
                    lock.lock();
                    try {
                        if (!closed.get()) waitingRequests.capacityChanged(key);
                    } finally {
                        lock.unlock();
                        requestRun();
                    }
                }

                @Override
                public void onEndpointChanged(WorkerEndpoint endpoint) {
                    QueuedRequestScheduler.this.onEndpointChanged(endpoint);
                }
            };
    private final ThreadLocal<Boolean> planningThread = new ThreadLocal<>();

    private boolean started;

    private final java.util.concurrent.atomic.AtomicLong notifications =
            new java.util.concurrent.atomic.AtomicLong();
    private final Thread schedulerThread =
            Thread.ofPlatform().daemon().name("flexlb-scheduling").unstarted(this::run);
    private final PrefillDeliveryScheduler prefillDelivery;

    public QueuedRequestScheduler(
            FlexlbConfig config,
            RequestRouteSelector router,
            DecodePreemption decodePreemption,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryCoordinator dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            CapacityNotifier capacityNotifier,
            SchedulingQueue queue,
            GroupPlanner grouping,
            EndpointRegistry endpoints) {
        super(
                requests,
                tasks,
                deadlines,
                dispatcher,
                responseCompletions,
                deliveryReporter,
                cacheTrace,
                endpoints);
        this.priorityOrdering = config.isPriorityOrdering();
        this.queueSettings = QueueExecutionSettings.capture(config, grouping);
        this.encoderRequestLimit = config.getDispatcher().getMaxInflightPerEncoderWorker();
        this.router = Objects.requireNonNull(router, "router");
        this.decodePreemption = Objects.requireNonNull(decodePreemption, "decodePreemption");
        this.capacityNotifier = Objects.requireNonNull(capacityNotifier, "capacityNotifier");
        this.orderedQueue = Objects.requireNonNull(queue, "queue");
        this.waitingRequests = new CapacityWaitQueue(queue.order(), capacityNotifier);
        this.scanBudgetMultiplier = config.queueScheduler().getScanBudgetMultiplier();

        this.plannerCount =
                Math.max(MIN_PLANNER_THREADS, config.getInternalRuntime().getQueuePlannerThreads());
        var factory = Thread.ofPlatform().daemon().name("flexlb-global-planner-", 1).factory();
        this.planners =
                Executors.newFixedThreadPool(
                        this.plannerCount,
                        queueScheduler ->
                                factory.newThread(
                                        () -> {
                                            planningThread.set(true);
                                            try {
                                                queueScheduler.run();
                                            } finally {
                                                planningThread.remove();
                                            }
                                        }));
        this.prefillDelivery =
                new PrefillDeliveryScheduler(
                        queueSettings,
                        dispatcher,
                        planners,
                        plannerCount,
                        this::requestRun,
                        this::recordRequestFailure);
        capacityNotifier.addListener(capacityListener);
    }

    @Override
    public int queuedRequestCount() {
        int pending = 0;
        for (var request : repository.snapshotActive()) {
            var state = request.snapshot();
            if (request.data().getRequestScheduler() == this
                    && state.phase() == ContextResult.Phase.OPEN
                    && state.delivery() == null) pending++;
        }
        return pending;
    }

    @Override
    public void start() {
        lock.lock();
        try {
            if (started || closed.get()) return;
            started = true;
            schedulerThread.start();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public boolean shouldRetryRouteFailure(ContextResult.RouteFailure failure) {
        return !closed.get()
                && failure.reason() == ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE;
    }

    @Override
    public void scheduleRequest(RequestContext context) {
        if (!checkRequestMode(context, true)) return;
        if (!enqueue(context)) shutdownRequest(context);
    }

    @Override
    public void onEndpointChanged(WorkerEndpoint endpoint) {
        var status = Objects.requireNonNull(endpoint, "endpoint").getStatus();
        lock.lock();
        try {
            if (closed.get()) return;
            if (!endpoint.acceptsNewRequests()) unavailableEndpoints.add(endpoint);
            waitingRequests.capacityChanged(
                    CapacityKey.exact(
                            status.getRole(),
                            status.topologySnapshot().group(),
                            endpoint.ipPort()));
        } finally {
            lock.unlock();
        }
        if (endpoint instanceof PrefillEndpoint prefill) prefillDelivery.endpointChanged(prefill);
        requestRun();
    }

    @Override
    public int blockedRequestCount() {
        lock.lock();
        try {
            return waitingRequests.size()
                    + (int)
                            preemptionQuotaWaiters.stream()
                                    .filter(entry -> !waitingRequests.contains(entry))
                                    .count();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void close() {
        Throwable failure = Failures.run(null, decodePreemption::stopAdmission);
        failure =
                Failures.run(
                        failure,
                        () -> {
                            stop();
                            drainOnClose();
                            checkState(
                                    Thread.currentThread() == schedulerThread
                                            ? awaitPlanners(Long.MAX_VALUE)
                                            : awaitStopped(Long.MAX_VALUE),
                                    "scheduler cannot wait for its own scheduling tasks");
                            List<CompletableFuture<Void>> preemptions;
                            lock.lock();
                            try {
                                preemptions = List.copyOf(pendingPreemptions.values());
                            } finally {
                                lock.unlock();
                            }
                            for (var pending : preemptions) pending.join();
                            prefillDelivery.clearAfterTasksStopped();
                        });
        failure = Failures.run(failure, decodePreemption::close);
        Failures.rethrow(failure, "scheduler failed to stop preemption");
    }

    @Override
    public void stop() {
        closed.set(true);
        prefillDelivery.stop();
        requestRun();
        capacityNotifier.removeListener(capacityListener);
        planners.shutdown();
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (Boolean.TRUE.equals(planningThread.get())) return false;
        if (schedulerThread == Thread.currentThread()) return false;
        boolean interrupted = false;
        try {
            while (schedulerThread.isAlive()) {
                long remaining = remainingNanos(deadlineNanos);
                if (remaining <= 0) return false;
                try {
                    if (!schedulerThread.join(java.time.Duration.ofNanos(remaining))) return false;
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt();
        }
        return awaitPlanners(deadlineNanos);
    }

    @Override
    public boolean allowsRoutePreemption() {
        return true;
    }

    public RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            RequestContext context,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        return planRoute(context, selection, false).decision();
    }

    @Override
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.QUEUE_TIMEOUT;
    }

    @Override
    protected int encoderRequestLimit() {
        return encoderRequestLimit;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        lock.lock();
        try {
            var entry = queuedEntries.get(request);
            if (entry != null) removeRequest(entry);
        } finally {
            lock.unlock();
        }
    }

    @Override
    protected boolean onRouteFailure(
            ContextResult.RouteFailure failure, DeliveryEvidence evidence) {
        var route = failure.route();
        var ctx = route.ctx();
        if (ctx.data().getRequestScheduler() != this
                || !repository.isCurrent(ctx)
                || !shouldRetryRouteFailure(failure)) return false;

        if (!ctx.failUnsentRoute(failure, evidence)) return false;
        withdrawFailedRoute(ctx);
        return true;
    }

    @Override
    protected void processPendingRequestActions(RequestContext ctx) {
        if (ctx.snapshot().routeFailure() != null) {
            withdrawFailedRoute(ctx);
        }
        super.processPendingRequestActions(ctx);
    }

    @Override
    protected Map<String, Object> describeWait(
            RequestContext request, CancelReason reason, String message) {
        var state = request.snapshot();
        if (reason != CancelReason.DEADLINE_EXCEEDED
                || state.delivery() != null
                || state.routingContext() != null
                || state.route() == null
                || state.route().encoderEndpoint() != null) {
            return super.describeWait(request, reason, message);
        }
        var details = describeWaitAtStage(request, state, "PREFILL_QUEUE");
        var observation =
                prefillDelivery.getLatestQueueWaitSnapshot(state.route().prefillEndpoint());
        if (observation != null) {
            // This is the observed queue head, which may be a different request.
            details.put("queueObservation", observation);
        }
        return completeWaitDiagnostics(message, details);
    }

    /**
     * First admission and a released route share one queue insertion. Restored requests keep
     * their original sequence and absolute deadline; replacing old membership is atomic.
     */
    boolean enqueue(RequestContext context) {
        if (context.data().getFuture().isDone()) return true;
        var entry = new GlobalQueueEntry(context, router.resolvePolicyGroup(context));
        entry.sequence = context.data().getPlacementSequence();
        boolean restored = entry.sequence != 0L;
        lock.lock();
        try {
            if (closed.get()) return false;
            var state = context.snapshot();
            if (context.data().getFuture().isDone()
                    || state.phase() != Phase.OPEN
                    || state.cancellation() != null) return true;
            if (restored) {
                var previous = queuedEntries.get(context);
                if (previous != null) removeRequest(previous);
            }
            checkState(
                    queuedEntries.putIfAbsent(context, entry) == null,
                    "duplicate global queue identity");
            if (restored) {
                context.data().setPlanType("");
                context.data().setPlanCost(0L);
                context.data().setPreemptedRequestCount(0);
                orderedQueue.restore(entry);
            } else {
                orderedQueue.add(entry);
                context.data().setPlacementSequence(entry.sequence);
            }
            return true;
        } finally {
            lock.unlock();
            requestRun();
        }
    }

    RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            RoutingContext routingContext,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        return executeQueuedRouting(routingContext, selection, false).decision();
    }

    /**
     * Publish the already registered route into its Prefill queue.
     * Record successful queue admission before notifying consumers, so notification failure
     * cannot revoke the published route or its resources.
     */
    RoutingResult<RouteResult, CapacityKey> enqueueRoute(RouteResult route) {
        var context = route.ctx();
        checkState(
                route.decodeEndpoint() == null || route.decodeReservation() != null,
                "selected route requires Decode registration");
        context.data().setRouteSubmittedNanos(System.nanoTime());
        var routingContext = currentRoutingContext(route);
        if (routingContext == null) return RoutingResult.closed();
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(RoutingResult.closed());
        boolean enqueued =
                route.prefillEndpoint().acceptQueuedRoute(route, queueSettings, routingContext);
        if (enqueued) return RoutingResult.success(route);
        if (!context.canPublishRoute(routingContext)) return RoutingResult.closed();
        return route.validateEndpoints(
                RoutingResult.blocked(
                        route.capacityKey(RoleType.PREFILL),
                        Response.buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                "Prefill queue rejected publication at "
                                        + route.prefillEndpoint().getIp()),
                        Map.of(
                                "cause",
                                "Prefill queue rejected publication",
                                "endpoint",
                                route.prefillEndpoint().getIp())));
    }

    /**
     * Notify consumers after the request accepted its queued route; notification failure cannot revoke it.
     */
    void notifyQueuedRoute(RouteResult route) {
        var failure =
                Failures.run(
                        null, () -> route.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT));
        try {
            deliveryReporter.reportLatency(
                    DeliveryMetricsReporter.Latency.ROUTE_SUBMIT,
                    route.prefillEndpoint().getStatus().getRole().name(),
                    route.prefillEndpoint().getIp(),
                    System.currentTimeMillis() - route.ctx().data().getStartTime());
        } catch (Throwable telemetryFailure) {
            Logger.warn("Failed to record route-submit telemetry", telemetryFailure);
        }
        Failures.rethrow(failure, "route publication failed");
    }

    @Override
    void completeRouting(
            RoutingContext routingContext, RoutingCompletion result, Response failure) {
        if (routingContext.routeToWithdraw() == null) {
            super.completeRouting(routingContext, result, failure);
            return;
        }
        var ctx = routingContext.requestContext();
        checkState(ctx.data().getRequestScheduler() == this, "foreign routing context");
        if (!ctx.beginRoutingCompletion(routingContext)) return;
        var previous = routingContext.routeToWithdraw();
        var routeFailure = ctx.snapshot().routeFailure();
        var withdrawingDelivery = ctx.snapshot().delivery();
        Throwable problem = null;
        if (result == RoutingCompletion.WITHDRAWN) {
            try {
                if (routeFailure == null) {
                    checkState(
                            previous.prefillEndpoint().releaseQueuedRequest(previous),
                            "withdrawn route is no longer queued: %s",
                            ctx.data().getRequestId());
                }
                var delivery = ctx.snapshot().delivery();
                if (delivery != null && delivery.identity().route() == previous)
                    delivery.identity().close();
                Failures.rethrow(
                        releaseRouteResources(
                                previous, DecodeResources.ReleaseReason.UNSENT_ROUTE_ROLLBACK),
                        "route withdrawal cleanup failed");
            } catch (Throwable detachFailure) {
                problem = detachFailure;
                result = RoutingCompletion.FAILED;
                failure =
                        buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                "queued route withdrawal failed");
            }
        }
        var after = ctx.completeRouting(routingContext, result, System.currentTimeMillis());
        checkState(after != null, "routing completion lost its request context");
        if (result == RoutingCompletion.WITHDRAWN && withdrawingDelivery != null)
            withdrawingDelivery.identity().complete();
        if (after.route() == previous && failure == null) {
            problem =
                    Failures.run(
                            problem,
                            () -> previous.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT));
        }
        finishRoutingRequest(ctx, after, result, failure, problem);
    }

    private void run() {
        try {
            while (!closed.get()) {
                long observed = notifications.get();
                long nextCheck = scheduleAt(System.nanoTime());
                if (closed.get()) break;
                // A blocking operation inside scheduleAt() may consume an unpark permit.
                if (notifications.get() != observed) continue;
                if (nextCheck == Long.MAX_VALUE) {
                    java.util.concurrent.locks.LockSupport.park(this);
                } else {
                    long remaining = nextCheck - System.nanoTime();
                    if (remaining > 0)
                        java.util.concurrent.locks.LockSupport.parkNanos(this, remaining);
                }
                Thread.interrupted();
            }
        } catch (RuntimeException | Error failure) {
            recordRequestFailure(failure);
            stop();
            // The driver may close its planner pool without waiting for its own thread.
            Throwable cleanup = Failures.run(null, this::close);
            if (cleanup != null) recordRequestFailure(cleanup);
        }
    }

    private long scheduleAt(long nowNanos) {
        if (closed.get()) return Long.MAX_VALUE;
        withdrawUnavailableRoutes();
        long queueSchedulingCheck = prefillDelivery.scheduleAt(nowNanos);
        List<GlobalQueueEntry> candidates;
        lock.lock();
        try {
            int slots = plannerCount - inFlight.size();
            candidates = closed.get() || slots == 0 ? List.of() : planningCandidates(slots);
            inFlight.addAll(candidates);
        } finally {
            lock.unlock();
        }
        candidates.forEach(this::submitPlan);
        lock.lock();
        try {
            if (closed.get() && inFlight.isEmpty() && pendingPreemptions.isEmpty()) {
                return Long.MAX_VALUE;
            }
            boolean ready =
                    !closed.get()
                            && inFlight.size() < plannerCount
                            && (orderedQueue.hasUnscannedRequests()
                                    || waitingRequests.hasReadyRequests());
            return ready ? nowNanos : queueSchedulingCheck;
        } finally {
            lock.unlock();
        }
    }

    private void withdrawUnavailableRoutes() {
        Set<WorkerEndpoint> endpoints;
        lock.lock();
        try {
            if (unavailableEndpoints.isEmpty()) return;
            endpoints = Collections.newSetFromMap(new IdentityHashMap<>());
            endpoints.addAll(unavailableEndpoints);
            unavailableEndpoints.clear();
        } finally {
            lock.unlock();
        }
        for (var request : repository.snapshotActive()) {
            var route = request.snapshot().route();
            if (route != null && route.usesAnyEndpoint(endpoints))
                onRouteFailure(
                        route,
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        "Selected Endpoint no longer accepts requests");
        }
    }

    /**
     * Caller holds lock; every examined entry counts against the scan budget.
     */
    private List<GlobalQueueEntry> planningCandidates(int slots) {
        waitingRequests.resumeReady(slots, orderedQueue::markRequestReadyForRetry);
        return orderedQueue.scanForPlanningCandidates(
                slots,
                (int) Math.min(Integer.MAX_VALUE, Math.ceil(slots * scanBudgetMultiplier)),
                entry -> {
                    if (entry.context.data().getFuture().isDone()) {
                        removeRequest(entry);
                        return false;
                    }
                    return !entry.removed
                            && !inFlight.contains(entry)
                            && !pendingPreemptions.containsKey(entry)
                            && !waitingRequests.isWaiting(entry);
                });
    }

    private void removeRequest(GlobalQueueEntry entry) {
        lock.lock();
        try {
            orderedQueue.remove(entry);
            waitingRequests.remove(entry);
            preemptionQuotaWaiters.remove(entry);
            queuedEntries.remove(entry.context, entry);
            requestRun();
        } finally {
            lock.unlock();
        }
    }

    private void requestRun() {
        notifications.incrementAndGet();
        java.util.concurrent.locks.LockSupport.unpark(schedulerThread);
    }

    private boolean awaitPlanners(long deadlineNanos) {
        if (Boolean.TRUE.equals(planningThread.get())) return false;
        boolean interrupted = Thread.interrupted();
        try {
            while (!planners.isTerminated()) {
                long remaining = remainingNanos(deadlineNanos);
                if (remaining <= 0) return false;
                try {
                    if (!planners.awaitTermination(remaining, TimeUnit.NANOSECONDS)) return false;
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
            return true;
        } finally {
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private static long remainingNanos(long deadline) {
        return deadline == Long.MAX_VALUE ? Long.MAX_VALUE : deadline - System.nanoTime();
    }

    private void submitPlan(GlobalQueueEntry entry) {
        try {
            planners.execute(() -> place(entry));
        } catch (Throwable failure) {
            try {
                if (!rejectedDuringStop(failure)) placementFailed(entry, failure);
            } finally {
                finishSchedulingTask(entry, SchedulingTask.ROUTING, false);
            }
        }
    }

    private void place(GlobalQueueEntry entry) {
        RoutingPlan plan = null;
        boolean retry = false;
        long observedCapacity = capacityNotifier.sequence();
        try {
            var data = entry.context.data();
            if (closed.get()) return;
            if (entry.removed || data.getFuture().isDone()) {
                removeRequest(entry);
                return;
            }
            if (data.getConfig().deliveryType(data.getPhase())
                    == org.flexlb.config.DispatcherConfig.Type.BATCH) data.prepareGenerateInput();
            if (data.getRequestExpiresAtMs() <= 0L
                    || System.currentTimeMillis() >= data.getRequestExpiresAtMs()) {
                cancel(entry.context, 0L, CancelReason.DEADLINE_EXCEEDED);
                removeRequest(entry);
                return;
            }
            plan =
                    planRoute(
                            entry.context,
                            () -> router.select(entry.context, entry.routingGroup),
                            true);
            if (plan.decision() != null)
                retry = finishDecision(entry, plan.decision(), observedCapacity);
        } catch (Throwable failure) {
            placementFailed(entry, failure);
        } finally {
            finishSchedulingTask(entry, SchedulingTask.ROUTING, retry);
        }
        if (plan != null && plan.continuation() != null) plan.continuation().run();
    }

    /**
     * Plan a queued route, retaining the same operation when the policy starts asynchronous recovery.
     * Completed decisions follow publication or cleanup; deferred work starts after the planner releases its slot.
     */
    private RoutingPlan planRoute(
            RequestContext context,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection,
            boolean allowDecodePreemption) {
        var operation = tryBeginRouteSelection(context);
        return operation == null
                ? new RoutingPlan(RoutingResult.closed(), null)
                : executeQueuedRouting(operation, selection, allowDecodePreemption);
    }

    private RoutingPlan executeQueuedRouting(
            RoutingContext routingContext,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection,
            boolean allowDecodePreemption) {
        var request = routingContext.requestContext();
        checkState(
                request.snapshot().routingContext() == routingContext,
                "routing requires its exact active admission");
        RouteResult route = null;
        RoutingResult<RouteResult, CapacityKey> result = null;
        ContextResult.RouteDelivered delivered = null;
        Throwable failure = null;
        boolean routeAccepted = false;
        Runnable continuation = null;
        try {
            result = Objects.requireNonNull(selection.get(), "routing result");
            checkState(
                    result.value() == null || result.value().ctx() == request,
                    "route belongs to another request");
            route = result.value();
            request.recordRoutingDiagnostics(routingContext, result.diagnostics());
            if (request.data().getFuture().isDone()
                    || !request.canPublishRoute(routingContext)
                    || closed.get()) {
                result = RoutingResult.closed();
            } else if (result.status() == RoutingResult.Status.BLOCKED && allowDecodePreemption) {
                continuation = startOrWaitForDecodePreemption(routingContext, result.blocker());
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                result = route.validateEndpoints(result);
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                if (route.prefillEndpoint() != null) {
                    result = enqueueRoute(route);
                } else {
                    var publication = dispatcher.dispatchSelectedRoute(route);
                    if (publication.status() == RoutingResult.Status.SUCCESS)
                        delivered = publication.value();
                    else result = publication.asFailure();
                }
                routeAccepted = result.status() == RoutingResult.Status.SUCCESS;
            }
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            if (continuation == null) {
                Throwable cleanup;
                try {
                    cleanup = finishUnpublishedRoute(routeAccepted ? null : route, routingContext);
                } catch (RuntimeException | Error completionFailure) {
                    cleanup = completionFailure;
                }
                if (cleanup != null) recordRequestFailure(cleanup);
                if (failure != null) Failures.append(failure, cleanup);
                else if (delivered == null) Failures.rethrow(cleanup, "placement cleanup failed");
            }
        }
        if (routeAccepted && delivered == null) notifyQueuedRoute(route);
        if (delivered != null) onDeliveryResult(delivered);
        return continuation == null
                ? new RoutingPlan(result, null)
                : new RoutingPlan(null, continuation);
    }

    /**
     * Starts Decode preemption or records a second wake condition for a capacity waiter.
     * The returned action subscribes to an already started result after this planner exits.
     */
    private Runnable startOrWaitForDecodePreemption(RoutingContext operation, CapacityKey blocker) {
        var blocked = router.blockedEndpoint(blocker);
        if (!(blocked instanceof DecodeEndpoint endpoint)
                || !priorityOrdering
                || operation.requestContext().data().getFuture().isDone()) return null;
        GlobalQueueEntry entry;
        lock.lock();
        try {
            entry = queuedEntries.get(operation.requestContext());
            if (closed.get() || entry == null || entry.removed) return null;
            preemptionQuotaWaiters.remove(entry);
            if (pendingPreemptions.size() >= plannerCount) {
                preemptionQuotaWaiters.add(entry);
                return null;
            }
            pendingPreemptions.put(entry, new CompletableFuture<>());
        } finally {
            lock.unlock();
        }
        boolean started = false;
        try {
            var result = decodePreemption.tryPreempt(entry.context, endpoint);
            if (result == null) return null;
            Runnable resume = () -> resumeAfterPreemption(entry, operation, endpoint, result);
            started = true;
            return resume;
        } finally {
            if (!started) finishSchedulingTask(entry, SchedulingTask.PREEMPTION, false);
        }
    }

    private void resumeAfterPreemption(
            GlobalQueueEntry entry,
            RoutingContext operation,
            DecodeEndpoint endpoint,
            java.util.concurrent.CompletionStage<DecodePreemption.PreemptionResult> pending) {
        pending.whenCompleteAsync(
                (result, failure) -> {
                    boolean retry = false;
                    long observedCapacity = capacityNotifier.sequence();
                    try {
                        var placement =
                                resumeRoute(
                                        operation, endpoint, result, failure, entry.routingGroup);
                        retry = finishDecision(entry, placement, observedCapacity);
                    } catch (Throwable problem) {
                        placementFailed(entry, problem);
                    } finally {
                        finishSchedulingTask(entry, SchedulingTask.PREEMPTION, retry);
                    }
                },
                work -> {
                    try {
                        planners.execute(work);
                    } catch (java.util.concurrent.RejectedExecutionException rejected) {
                        work.run();
                    }
                });
    }

    /**
     * Resume a resource acquisition under the original operation, including obsolete late results.
     * The request's scheduling policy receives no resource cleanup or operation-ending obligations.
     */
    private RoutingResult<RouteResult, CapacityKey> resumeRoute(
            RoutingContext operation,
            DecodeEndpoint endpoint,
            org.flexlb.balance.preemption.DecodePreemption.PreemptionResult result,
            Throwable failure,
            String routingGroup) {
        var request = operation.requestContext();
        if (failure != null) recordRequestFailure(failure);
        if (failure != null || result == null || !result.committed()) {
            failRouting(
                    operation,
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Decode preemption did not commit: "
                                    + (failure != null
                                            ? failure.getClass().getSimpleName()
                                                    + ": "
                                                    + failure.getMessage()
                                            : result == null
                                                    ? "control returned no result"
                                                    : result.detail())));
            return RoutingResult.closed();
        }
        return executeQueuedRouting(
                        operation,
                        () -> {
                            if (closed.get()
                                    || request.data().getFuture().isDone()
                                    || !request.canPublishRoute(operation)) {
                                RoleSelection.Decode.release(
                                        endpoint,
                                        result.reservation(),
                                        DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                                return RoutingResult.closed();
                            }
                            return router.select(
                                    request, routingGroup, endpoint, result.reservation());
                        },
                        false)
                .decision();
    }

    private boolean finishDecision(
            GlobalQueueEntry entry,
            RoutingResult<RouteResult, CapacityKey> result,
            long observedCapacity) {
        return switch (result.status()) {
            case RETRY -> true;
            case BLOCKED -> parkOrRetry(entry, result, observedCapacity);
            case REJECTED, SUCCESS, CLOSED -> {
                if (result.status() == RoutingResult.Status.REJECTED)
                    completeDecisionResponse(entry, result.failure());
                removeRequest(entry);
                yield false;
            }
        };
    }

    /**
     * Registers a capacity wait, or returns true when a new capacity event requires a retry.
     */
    private boolean parkOrRetry(
            GlobalQueueEntry entry,
            RoutingResult<RouteResult, CapacityKey> result,
            long observedCapacity) {
        var waitKey = result.blocker();
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (entry.removed || entry.context.data().getFuture().isDone()) {
                return false;
            }
            parked = waitingRequests.park(entry, waitKey, observedCapacity);
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        synchronized (entry.context) {
            var state = entry.context.snapshot();
            if (state.cancellation() == null
                    && state.route() == null
                    && state.phase() == Phase.OPEN) {
                entry.context
                        .data()
                        .setSchedulingDiagnostics(
                                RequestSchedulerReporter.placementWait(
                                        waitKey.role().name(),
                                        waitKey.group(),
                                        waitKey.endpoint(),
                                        depth,
                                        counts,
                                        result.failure() == null
                                                ? null
                                                : result.failure().getErrorMessage(),
                                        result.diagnostics()));
            }
        }
        return !parked;
    }

    private void completeDecisionResponse(GlobalQueueEntry entry, Response response) {
        try {
            publish(entry.context, response, false);
        } catch (Throwable failure) {
            Logger.error(
                    "Global queue response publication failed: request_id={}",
                    entry.context.data().getRequestId(),
                    failure);
        }
    }

    private void placementFailed(GlobalQueueEntry entry, Throwable failure) {
        if (closed.get()) recordRequestFailure(failure);
        removeRequest(entry);
        completeDecisionResponse(
                entry,
                buildErrorResponse(
                        StrategyErrorType.DISPATCH_FAILED,
                        "Placement failed: "
                                + failure.getClass().getSimpleName()
                                + ": "
                                + failure.getMessage()));
        Logger.error(
                "Global queue placement failed: request_id={}",
                entry.context.data().getRequestId(),
                failure);
    }

    /**
     * Returns the completed task's slot and publishes retries under the queue lock.
     * Preemption completion is signalled outside the lock, after its queue effects are visible.
     */
    private void finishSchedulingTask(
            GlobalQueueEntry entry, SchedulingTask completedTask, boolean retry) {
        CompletableFuture<Void> completion = null;
        lock.lock();
        try {
            if (completedTask == SchedulingTask.PREEMPTION)
                completion = pendingPreemptions.remove(entry);
            else inFlight.remove(entry);
            if (!closed.get()) {
                if (retry && !entry.removed && !entry.context.data().getFuture().isDone())
                    orderedQueue.markRequestReadyForRetry(entry);
                if (pendingPreemptions.size() < plannerCount) retryPreemptionWaiters();
            }
            requestRun();
        } finally {
            lock.unlock();
        }
        if (completion != null) completion.complete(null);
    }

    /**
     * Keep notifications for running planners until their result has been handled;
     * otherwise their later capacity wait can overwrite the retry.
     */
    private void retryPreemptionWaiters() {
        if (closed.get()) return;
        var waiters = preemptionQuotaWaiters.iterator();
        while (waiters.hasNext()) {
            var waiter = waiters.next();
            if (inFlight.contains(waiter)) continue;
            waiters.remove();
            waitingRequests.remove(waiter);
            if (!waiter.removed && !waiter.context.data().getFuture().isDone())
                orderedQueue.markRequestReadyForRetry(waiter);
        }
    }

    private boolean rejectedDuringStop(Throwable failure) {
        return closed.get()
                && planners.isShutdown()
                && failure instanceof java.util.concurrent.RejectedExecutionException;
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
            abandoned = List.copyOf(all);
            queuedEntries.clear();
        } finally {
            requestRun();
            lock.unlock();
        }
        for (GlobalQueueEntry entry : abandoned) {
            try {
                if (!shutdownRequest(entry.context)) {
                    completeDecisionResponse(
                            entry,
                            buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "request scheduler is shutting down"));
                }
            } catch (Throwable failure) {
                Logger.error(
                        "Global queue close settlement failed: request_id={}",
                        entry.context.data().getRequestId(),
                        failure);
            }
        }
    }

    private boolean shutdownRequest(RequestContext context) {
        if (context.data().getRequestScheduler() != this || !repository.isCurrent(context)) {
            return false;
        }
        closeRequest(
                context,
                new EndEvidence(
                        context.snapshot().route(),
                        EndKind.SHUTDOWN,
                        null,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down",
                        false,
                        false));
        return true;
    }

    private void withdrawFailedRoute(RequestContext ctx) {
        if (repository.isClosed()) return;
        var routingContext = ctx.beginFailedRouteWithdrawal();
        if (routingContext == null) return;
        completeRouting(routingContext, RoutingCompletion.WITHDRAWN);
    }

    private enum SchedulingTask {
        ROUTING,
        PREEMPTION
    }

    /**
     * A finished routing decision or the work that resumes the same operation after releasing a planner slot.
     */
    private record RoutingPlan(
            RoutingResult<RouteResult, CapacityKey> decision, Runnable continuation) {
        private RoutingPlan {
            com.google.common.base.Preconditions.checkArgument(
                    (decision == null) != (continuation == null),
                    "routing has one decision or continuation");
        }
    }
}

package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.CapacityWaitQueue;
import org.flexlb.balance.queue.GlobalQueueEntry;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.worker.endpoint.Response.buildErrorResponse;

/**
 * Selects queued requests and returns committed routes using a bounded pool of planner threads.
 * Capacity changes and released preemption slots resume waiting requests; asynchronous preemption
 * results return through the same route selection and cleanup paths.
 * Closing removes queued requests and waits for planners and preemption result cleanup to finish.
 */
public class QueuedRequestScheduler extends AbstractRequestScheduler {
    private static final int MIN_PLANNER_THREADS = 1;

    private final RequestRouteSelector router;
    private final DecodePreemption decodePreemption;
    private final CapacityNotifier capacityNotifier;

    private final boolean priorityOrdering;
    private final int plannerCount;
    private final double scanBudgetMultiplier;

    private final ReentrantLock lock = new ReentrantLock();
    private final SchedulingQueue orderedQueue;
    private final CapacityWaitQueue waitingRequests;

    private final SchedulingRequests schedulingRequests = new SchedulingRequests();

    /**
     * Waiting on local preemption slots, independent of physical Worker capacity edges.
     */
    private final Set<GlobalQueueEntry> preemptionQuotaWaiters =
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

    public QueuedRequestScheduler(
            FlexlbConfig config,
            RequestRouteSelector router,
            DecodePreemption decodePreemption,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            CapacityNotifier capacityNotifier,
            SchedulingQueue queue,
            EndpointRegistry endpoints) {
        super(
                requests,
                tasks,
                deadlines,
                engineCancellation,
                responseCompletions,
                deliveryReporter,
                cacheTrace,
                endpoints);
        this.priorityOrdering = config.isPriorityOrdering();
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
        capacityNotifier.addListener(capacityListener);
    }

    @Override
    public int queuedRequestCount() {
        return repository.pendingRequestCount();
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
            waitingRequests.capacityChanged(
                    CapacityKey.exact(
                            status.getRole(),
                            status.topologySnapshot().group(),
                            endpoint.ipPort()));
        } finally {
            lock.unlock();
        }
        requestRun();
    }

    @Override
    public int blockedRequestCount() {
        lock.lock();
        try {
            return schedulingRequests.waitingRequestCount();
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
                                preemptions = schedulingRequests.pendingPreemptions();
                            } finally {
                                lock.unlock();
                            }
                            for (var pending : preemptions) pending.join();
                        });
        failure = Failures.run(failure, decodePreemption::close);
        Failures.rethrow(failure, "scheduler failed to stop preemption");
    }

    @Override
    public void stop() {
        closed.set(true);
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
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.QUEUE_TIMEOUT;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        lock.lock();
        try {
            var entry = schedulingRequests.find(request);
            if (entry != null) removeRequest(entry);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Starts Decode preemption or records a second wake condition for a capacity waiter.
     * The returned action subscribes to an already started result after this planner exits.
     */
    @Override
    protected Runnable startOrWaitForDecodePreemption(
            SelectionAttempt operation, CapacityKey blocker) {
        var blocked = router.blockedEndpoint(blocker);
        if (!(blocked instanceof DecodeEndpoint endpoint)
                || !priorityOrdering
                || operation.requestContext().getFuture().isDone()) return null;
        GlobalQueueEntry entry;
        lock.lock();
        try {
            entry = schedulingRequests.find(operation.requestContext());
            if (closed.get() || entry == null || entry.isRemoved()) return null;
            preemptionQuotaWaiters.remove(entry);
            if (!schedulingRequests.tryStartPreemption(entry, plannerCount)) {
                preemptionQuotaWaiters.add(entry);
                return null;
            }
        } finally {
            lock.unlock();
        }
        boolean started = false;
        try {
            var result = decodePreemption.tryPreempt(entry.context(), endpoint);
            if (result == null) return null;
            Runnable resume = () -> resumeAfterPreemption(entry, operation, endpoint, result);
            started = true;
            return resume;
        } finally {
            if (!started) finishSchedulingTask(entry, SchedulingTask.PREEMPTION, false);
        }
    }

    @Override
    protected boolean routeSelectionStopped() {
        return closed.get();
    }

    /**
     * First admission and a released route share one queue insertion. Restored requests keep
     * their original sequence and absolute deadline; replacing old membership is atomic.
     */
    boolean enqueue(RequestContext context) {
        if (context.getFuture().isDone()) return true;
        var entry =
                new GlobalQueueEntry(
                        context,
                        router.resolvePolicyGroup(context),
                        context.getPlacementSequence());
        boolean restored = context.getPlacementSequence() != 0L;
        lock.lock();
        try {
            if (closed.get()) return false;
            var state = context.snapshot();
            if (context.getFuture().isDone()
                    || state.phase() != Phase.OPEN
                    || state.cancellationReason() != null) return true;
            if (restored) {
                var previous = schedulingRequests.find(context);
                if (previous != null) removeRequest(previous);
            }
            schedulingRequests.register(entry);
            if (restored) {
                context.recordPreemptionStatistics(0L, 0);
                orderedQueue.restore(entry);
            } else {
                orderedQueue.add(entry);
                context.setPlacementSequence(entry.sequence());
            }
            return true;
        } finally {
            lock.unlock();
            requestRun();
        }
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
        long queueSchedulingCheck = Long.MAX_VALUE;
        List<GlobalQueueEntry> candidates;
        lock.lock();
        try {
            int slots = plannerCount - schedulingRequests.decisionCount();
            candidates = closed.get() || slots == 0 ? List.of() : planningCandidates(slots);
            schedulingRequests.startDecisions(candidates);
        } finally {
            lock.unlock();
        }
        candidates.forEach(this::submitPlan);
        lock.lock();
        try {
            if (closed.get() && !schedulingRequests.hasWork()) {
                return Long.MAX_VALUE;
            }
            boolean ready =
                    !closed.get()
                            && schedulingRequests.decisionCount() < plannerCount
                            && (orderedQueue.hasUnscannedRequests()
                                    || waitingRequests.hasReadyRequests());
            return ready ? nowNanos : queueSchedulingCheck;
        } finally {
            lock.unlock();
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
                    if (entry.context().getFuture().isDone()) {
                        removeRequest(entry);
                        return false;
                    }
                    return !entry.isRemoved()
                            && !schedulingRequests.isBusy(entry)
                            && !waitingRequests.isWaiting(entry);
                });
    }

    private void removeRequest(GlobalQueueEntry entry) {
        lock.lock();
        try {
            orderedQueue.remove(entry);
            waitingRequests.remove(entry);
            preemptionQuotaWaiters.remove(entry);
            schedulingRequests.remove(entry);
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
        RequestResult plan = null;
        boolean retry = false;
        long observedCapacity = capacityNotifier.sequence();
        try {
            var data = entry.context();
            if (closed.get()) return;
            if (entry.isRemoved() || data.getFuture().isDone()) {
                removeRequest(entry);
                return;
            }
            if (data.getRequestExpiresAtMs() <= 0L
                    || System.currentTimeMillis() >= data.getRequestExpiresAtMs()) {
                cancel(entry.context(), CancelReason.DEADLINE_EXCEEDED);
                removeRequest(entry);
                return;
            }
            plan =
                    planRoute(
                            entry.context(),
                            () -> router.select(entry.context(), entry.routingGroup()),
                            true);
            if (plan.preemptionContinuation() == null)
                retry = finishDecision(entry, plan, observedCapacity);
        } catch (Throwable failure) {
            placementFailed(entry, failure);
        } finally {
            finishSchedulingTask(entry, SchedulingTask.ROUTING, retry);
        }
        if (plan != null && plan.preemptionContinuation() != null)
            plan.preemptionContinuation().run();
    }

    private boolean finishDecision(
            GlobalQueueEntry entry, RequestResult result, long observedCapacity) {
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
            GlobalQueueEntry entry, RequestResult result, long observedCapacity) {
        var waitKey = result.blocker();
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (entry.isRemoved() || entry.context().getFuture().isDone()) {
                return false;
            }
            parked = waitingRequests.park(entry, waitKey, observedCapacity);
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        synchronized (entry.context()) {
            var state = entry.context().snapshot();
            if (state.cancellationReason() == null
                    && state.route() == null
                    && state.phase() == Phase.OPEN) {
                entry.context()
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
            publish(entry.context(), response, false);
        } catch (Throwable failure) {
            Logger.error(
                    "Global queue response publication failed: request_id={}",
                    entry.context().getRequestId(),
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
                entry.context().getRequestId(),
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
                completion = schedulingRequests.finishPreemption(entry);
            else schedulingRequests.finishDecision(entry);
            if (!closed.get()) {
                if (retry && !entry.isRemoved() && !entry.context().getFuture().isDone())
                    orderedQueue.markRequestReadyForRetry(entry);
                if (schedulingRequests.preemptionCount() < plannerCount) retryPreemptionWaiters();
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
            if (schedulingRequests.isDeciding(waiter)) continue;
            waiters.remove();
            waitingRequests.remove(waiter);
            if (!waiter.isRemoved() && !waiter.context().getFuture().isDone())
                orderedQueue.markRequestReadyForRetry(waiter);
        }
    }

    private boolean rejectedDuringStop(Throwable failure) {
        return closed.get()
                && planners.isShutdown()
                && failure instanceof java.util.concurrent.RejectedExecutionException;
    }

    private void resumeAfterPreemption(
            GlobalQueueEntry entry,
            SelectionAttempt operation,
            DecodeEndpoint endpoint,
            java.util.concurrent.CompletionStage<RequestResult> pending) {
        pending.whenCompleteAsync(
                (result, failure) -> {
                    boolean retry = false;
                    long observedCapacity = capacityNotifier.sequence();
                    try {
                        var placement =
                                resumeRoute(
                                        operation, endpoint, result, failure, entry.routingGroup());
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
    private RequestResult resumeRoute(
            SelectionAttempt operation,
            DecodeEndpoint endpoint,
            org.flexlb.balance.schedule.RequestResult result,
            Throwable failure,
            String routingGroup) {
        var request = operation.requestContext();
        if (failure != null) recordRequestFailure(failure);
        if (failure == null && result != null && result.status() == RoutingResult.Status.BLOCKED) {
            return executeRouteSelection(
                    operation, () -> router.select(request, routingGroup, null, null), false);
        }
        if (failure != null || result == null || !result.committed()) {
            request.getRequestScheduler()
                    .finishSelectionWithoutCommit(
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
            return RequestResult.closed();
        }
        return executeRouteSelection(
                operation,
                () -> {
                    if (closed.get()
                            || request.getFuture().isDone()
                            || !request.canPublishRoute(operation)) {
                        RoleSelection.releaseDecode(
                                endpoint,
                                result.reservation(),
                                WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                        return RequestResult.closed();
                    }
                    return router.select(request, routingGroup, endpoint, result.reservation());
                },
                false);
    }

    private void drainOnClose() {
        List<GlobalQueueEntry> abandoned;
        lock.lock();
        try {
            Set<GlobalQueueEntry> all = Collections.newSetFromMap(new IdentityHashMap<>());
            all.addAll(orderedQueue.drain());
            all.addAll(schedulingRequests.drainRegistered());
            waitingRequests.clear();
            preemptionQuotaWaiters.clear();
            abandoned = List.copyOf(all);
        } finally {
            requestRun();
            lock.unlock();
        }
        for (GlobalQueueEntry entry : abandoned) {
            try {
                if (!shutdownRequest(entry.context())) {
                    completeDecisionResponse(
                            entry,
                            buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "request scheduler is shutting down"));
                }
            } catch (Throwable failure) {
                Logger.error(
                        "Global queue close settlement failed: request_id={}",
                        entry.context().getRequestId(),
                        failure);
            }
        }
    }

    private boolean shutdownRequest(RequestContext context) {
        if (context.getRequestScheduler() != this || !repository.isCurrent(context)) {
            return false;
        }
        closeRequest(
                context,
                new RequestEndEvent(
                        context.snapshot().route(),
                        EndKind.SHUTDOWN,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down"));
        return true;
    }

    private enum SchedulingTask {
        ROUTING,
        PREEMPTION
    }
}

package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.CapacityWaitQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Failures;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.StrategyErrorType;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkState;

/**
 * Scans queued requests on notification or a fixed interval and submits bounded, concurrent routing tasks.
 * Preemption owns resource waits; its ready checks and result handling use the same execution slots.
 * Closing ends queued requests and waits for all accepted tasks and resource cleanup before stopping the pool.
 */
public class QueuedRequestScheduler extends AbstractRequestScheduler {
    private final RequestRouteSelector router;
    private final RequestPreemption preemption;
    private final SchedulingQueue orderedQueue;
    private final CapacityWaitQueue waitingRequests;
    private final CapacityNotifier capacityNotifier;

    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final long capacityCheckIntervalNanos;

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final Map<RequestContext, SchedulingTask> executing = new IdentityHashMap<>();
    private final ThreadLocal<Boolean> planningThread = new ThreadLocal<>();
    private final ExecutorService planners;
    private final Thread schedulerThread;
    private final CapacityNotifier.Listener capacityListener;

    private boolean started;
    private volatile boolean stopped;

    public QueuedRequestScheduler(
            FlexlbConfig config,
            RequestRouteSelector router,
            RequestPreemption preemption,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            CapacityNotifier capacityNotifier,
            SchedulingQueue queue) {
        super(
                requests,
                tasks,
                deadlines,
                engineCancellation,
                responseCompletions,
                deliveryReporter,
                cacheTrace);
        this.router = router;
        this.preemption = preemption;
        this.orderedQueue = queue;
        this.waitingRequests = new CapacityWaitQueue(queue.order());
        this.capacityListener =
                endpoint -> {
                    preemption.requestCheck(endpoint);
                    lock.lock();
                    try {
                        waitingRequests.capacityChanged();
                        changed.signal();
                    } finally {
                        lock.unlock();
                    }
                };
        this.capacityNotifier = capacityNotifier;
        this.plannerCount = Math.max(1, config.getInternalRuntime().getQueuePlannerThreads());
        this.scanBudgetMultiplier = config.queueScheduler().getScanBudgetMultiplier();
        this.capacityCheckIntervalNanos =
                TimeUnit.MILLISECONDS.toNanos(config.queueScheduler().getCapacityCheckIntervalMs());
        this.planners =
                Executors.newFixedThreadPool(
                        plannerCount,
                        Thread.ofPlatform().daemon().name("flexlb-global-planner-", 1).factory());
        this.schedulerThread =
                Thread.ofPlatform().daemon().name("flexlb-scheduling").unstarted(this::run);
        preemption.setWakeup(this::requestRun);
        capacityNotifier.addListener(capacityListener);
    }

    @Override
    public void start() {
        lock.lock();
        try {
            if (started || stopped) return;
            started = true;
            schedulerThread.start();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void scheduleRequest(RequestContext request) {
        if (!checkRequestMode(request, true)) return;
        if (!enqueue(request)) shutdownRequest(request);
    }

    @Override
    public int queuedRequestCount() {
        return repository.pendingRequestCount();
    }

    @Override
    public int blockedRequestCount() {
        Set<RequestContext> waiting = preemption.waitingRequests();
        lock.lock();
        try {
            int count = orderedQueue.size();
            for (var request : executing.keySet()) if (orderedQueue.contains(request)) count--;
            for (var request : waiting)
                if (!executing.containsKey(request) && orderedQueue.contains(request)) count--;
            return count;
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void close() {
        stop();
        checkState(awaitStopped(Long.MAX_VALUE), "scheduler cannot wait for its own task");
        preemption.close();
    }

    @Override
    public void stop() {
        stopped = true;
        capacityNotifier.removeListener(capacityListener);
        preemption.stopAdmission();
        List<RequestContext> abandoned;
        lock.lock();
        try {
            abandoned = orderedQueue.drain();
            waitingRequests.clear();
            changed.signalAll();
            if (!started) planners.shutdown();
        } finally {
            lock.unlock();
        }
        Throwable failure = null;
        for (var request : abandoned)
            failure = Failures.run(failure, () -> shutdownRequest(request));
        if (failure != null) recordRequestFailure(failure);
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (Boolean.TRUE.equals(planningThread.get()) || Thread.currentThread() == schedulerThread)
            return false;
        boolean interrupted = false;
        try {
            while (schedulerThread.isAlive()) {
                long remaining =
                        (deadlineNanos == Long.MAX_VALUE
                                ? Long.MAX_VALUE
                                : deadlineNanos - System.nanoTime());
                if (remaining <= 0) return false;
                try {
                    if (!schedulerThread.join(java.time.Duration.ofNanos(remaining))) return false;
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
            while (!planners.isTerminated()) {
                long remaining =
                        (deadlineNanos == Long.MAX_VALUE
                                ? Long.MAX_VALUE
                                : deadlineNanos - System.nanoTime());
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

    @Override
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.QUEUE_TIMEOUT;
    }

    @Override
    protected boolean routeSelectionStopped() {
        return stopped;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        preemption.cancel(request);
        lock.lock();
        try {
            orderedQueue.remove(request);
            waitingRequests.remove(request);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    boolean enqueue(RequestContext request) {
        if (request.isResponseDone()) return true;
        if (request.getPlacementSequence() == 0L)
            request.setRoutingGroup(router.resolvePolicyGroup(request));
        var state = request.snapshot();
        lock.lock();
        try {
            if (stopped) return false;
            if (request.isResponseDone()
                    || state.phase() != RequestContext.Phase.OPEN
                    || state.cancellationReason() != null) return true;
            if (!orderedQueue.contains(request)) {
                if (request.getPlacementSequence() == 0L) orderedQueue.add(request);
                else orderedQueue.restore(request);
            }
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    private void run() {
        long nextCheck = System.nanoTime();
        try {
            while (true) {
                try {
                    long now = System.nanoTime();
                    boolean periodicCheck = now - nextCheck >= 0;
                    if (periodicCheck) {
                        preemption.requestPeriodicCheck();
                        nextCheck = now + capacityCheckIntervalNanos;
                    }
                    var ready = preemption.readyWork();
                    var waiting = preemption.waitingRequests();
                    boolean preemptionIdle = preemption.isIdle();
                    List<SchedulingTask> tasks;
                    lock.lock();
                    try {
                        if (stopped && executing.isEmpty() && preemptionIdle) return;
                        if (periodicCheck && !stopped) {
                            waitingRequests.capacityChanged();
                            orderedQueue.beginScan();
                        }
                        tasks = chooseTasks(ready, waiting);
                        if (tasks.isEmpty()) {
                            if (!stopped
                                    && executing.size() < plannerCount
                                    && orderedQueue.hasUnscannedRequests()) continue;
                            long remaining = nextCheck - System.nanoTime();
                            if (remaining > 0) changed.awaitNanos(remaining);
                            continue;
                        }
                    } finally {
                        lock.unlock();
                    }
                    for (var task : tasks) submit(task);
                } catch (InterruptedException ignored) {
                    // Check shutdown and newly ready work after interruption.
                } catch (RuntimeException | Error failure) {
                    org.flexlb.util.Logger.error("Queue scheduling failed", failure);
                    recordRequestFailure(failure);
                    stop();
                }
            }
        } finally {
            planners.shutdown();
        }
    }

    private List<SchedulingTask> chooseTasks(
            List<RequestPreemption.Work> ready, Set<RequestContext> waiting) {
        int slots = plannerCount - executing.size();
        List<SchedulingTask> tasks = List.of();
        for (var work : ready) {
            if (tasks.size() == slots) break;
            var request = work.selection().requestContext();
            if (executing.containsKey(request)) continue;
            var task = new SchedulingTask(request, work.action(), work.selection());
            executing.put(request, task);
            if (tasks.isEmpty()) tasks = new ArrayList<>();
            tasks.add(task);
        }
        int remaining = slots - tasks.size();
        if (!stopped && remaining > 0) {
            waitingRequests.resumeReady(
                    remaining, request -> !executing.containsKey(request), orderedQueue::retry);
            var candidates =
                    orderedQueue.scanForPlanningCandidates(
                            remaining,
                            (int)
                                    Math.min(
                                            Integer.MAX_VALUE,
                                            Math.ceil(remaining * scanBudgetMultiplier)),
                            request ->
                                    !executing.containsKey(request)
                                            && !waiting.contains(request)
                                            && !waitingRequests.isWaiting(request));
            for (var request : candidates) {
                var task = new SchedulingTask(request, null, null);
                executing.put(request, task);
                if (tasks.isEmpty()) tasks = new ArrayList<>(candidates.size());
                tasks.add(task);
            }
        }
        return tasks;
    }

    private void submit(SchedulingTask task) {
        try {
            planners.execute(() -> runTask(task));
        } catch (RuntimeException | Error failure) {
            if (!stopped) recordRequestFailure(failure);
            var stoppingFailure = Failures.run(null, this::stop);
            if (stoppingFailure != null) recordRequestFailure(stoppingFailure);
            runTask(task);
        }
    }

    private void runTask(SchedulingTask task) {
        planningThread.set(true);
        try {
            if (task.action != null) task.action.run();
            else selectRoute(task.request);
        } catch (Throwable failure) {
            org.flexlb.util.Logger.error(
                    "Request routing failed: request_id={}", task.request.getRequestId(), failure);
            recordRequestFailure(failure);
            var response =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Routing failed: " + failure.getMessage());
            if (task.selection != null) finishSelectionWithoutCommit(task.selection, response);
            publish(task.request, response, false);
            removeRequest(task.request);
        } finally {
            planningThread.remove();
            lock.lock();
            try {
                executing.remove(task.request, task);
                if (stopped
                        || orderedQueue.hasUnscannedRequests()
                        || waitingRequests.hasReadyRequests()) changed.signal();
            } finally {
                lock.unlock();
            }
        }
    }

    private void selectRoute(RequestContext request) {
        if (stopped || request.isResponseDone()) {
            removeRequest(request);
            return;
        }
        if (System.currentTimeMillis() >= request.getRequestExpiresAtMs()) {
            cancel(request, CancelReason.DEADLINE_EXCEEDED);
            return;
        }
        var selection = tryBeginRouteSelection(request);
        if (selection == null) return;
        boolean waiting = false;
        boolean selectionReturned = false;
        try {
            var result =
                    executeRouteSelection(
                            selection, () -> router.select(request, request.getRoutingGroup()));
            selectionReturned = true;
            if (result.status() == SelectionResult.Status.BLOCKED) {
                var endpoint = router.blockedEndpoint(result.blocker());
                waiting =
                        endpoint != null
                                && preemption.start(
                                        selection,
                                        endpoint,
                                        acquired -> resumeSelection(selection, acquired));
            }
            if (waiting) {
                lock.lock();
                try {
                    waitingRequests.remove(request);
                } finally {
                    lock.unlock();
                }
                recordCapacityWait(request, result);
            } else handleResult(request, result);
        } finally {
            if (selectionReturned
                    && !waiting
                    && java.util.Objects.equals(request.currentSelectionAttempt(), selection))
                finishSelectionWithoutCommit(selection, null);
        }
    }

    private void resumeSelection(SelectionAttempt selection, RoleSelection acquired) {
        var request = selection.requestContext();
        boolean selectionReturned = true;
        try {
            if (stopped || request.isResponseDone() || !request.canPublishRoute(selection)) {
                if (acquired != null && acquired.registration() != null)
                    ((org.flexlb.worker.endpoint.SelectionRollback) acquired.endpoint())
                            .rollbackSelection(acquired.registration());
                return;
            }
            selectionReturned = false;
            var result =
                    executeRouteSelection(
                            selection,
                            () ->
                                    acquired == null
                                            ? router.select(request, request.getRoutingGroup())
                                            : router.select(
                                                    request, request.getRoutingGroup(), acquired));
            selectionReturned = true;
            handleResult(request, result);
        } finally {
            if (selectionReturned
                    && java.util.Objects.equals(request.currentSelectionAttempt(), selection))
                finishSelectionWithoutCommit(selection, null);
        }
    }

    private void handleResult(
            RequestContext request, SelectionResult<RequestRoute, CapacityKey> result) {
        switch (result.status()) {
            case SUCCESS -> removeQueuedRequest(request);
            case REJECTED -> {
                publish(request, result.failure(), false);
                removeQueuedRequest(request);
            }
            case CLOSED -> {
                if (request.isResponseDone()
                        || request.snapshot().phase() != RequestContext.Phase.OPEN)
                    removeQueuedRequest(request);
            }
            case BLOCKED -> {
                lock.lock();
                try {
                    if (!stopped && orderedQueue.contains(request))
                        waitingRequests.park(request, result.blocker());
                } finally {
                    lock.unlock();
                }
                recordCapacityWait(request, result);
            }
            case RETRY -> {}
        }
    }

    private void removeQueuedRequest(RequestContext request) {
        lock.lock();
        try {
            orderedQueue.remove(request);
            waitingRequests.remove(request);
        } finally {
            lock.unlock();
        }
    }

    private void recordCapacityWait(
            RequestContext request, SelectionResult<RequestRoute, CapacityKey> result) {
        int depth;
        int[] counts;
        lock.lock();
        try {
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        var key = result.blocker();
        request.setSchedulingDiagnostics(
                RequestSchedulerReporter.placementWait(
                        key.role().name(),
                        key.group(),
                        key.endpoint(),
                        depth,
                        counts,
                        result.failure() == null ? null : result.failure().getErrorMessage(),
                        result.diagnostics()));
    }

    private void requestRun() {
        lock.lock();
        try {
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void shutdownRequest(RequestContext request) {
        if (request.getRequestScheduler() != this || !repository.isCurrent(request)) return;
        closeRequest(
                request,
                new RequestContext.RequestEndEvent(
                        request.snapshot().route(),
                        RequestContext.EndCause.SCHEDULER_SHUTDOWN,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down"));
    }

    private record SchedulingTask(
            RequestContext request, Runnable action, SelectionAttempt selection) {}
}

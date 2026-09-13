package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

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
final class GlobalQueueCoordinator implements AutoCloseable {

    private static final int MIN_PLANNER_THREADS = 1;

    private final DefaultRouter router;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestRegistry lifecycle;
    private final PlacementAvailability availability;
    private final boolean priorityOrdering;
    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final ConfigService configService;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final OrderedRequestQueue orderedQueue;
    private final PlacementWaitQueue waitingRequests;
    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ArrayDeque<Plan> completedPlans = new ArrayDeque<>();
    private final ExecutorService planners;
    private final Thread decisionThread;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            this::onAvailabilityChanged;

    GlobalQueueCoordinator(
            ConfigService configService,
            DefaultRouter router,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestRegistry lifecycle,
            PlacementAvailability availability) {
        ConfigService checkedConfig = Objects.requireNonNull(
                configService, "configService");
        this.configService = checkedConfig;
        this.router = Objects.requireNonNull(router, "router");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.evictionManager = Objects.requireNonNull(
                evictionManager, "evictionManager");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.priorityOrdering = resolvePriorityOrdering(checkedConfig);
        this.orderedQueue = new OrderedRequestQueue(priorityOrdering);
        this.waitingRequests = new PlacementWaitQueue(priorityOrdering, availability);
        this.scanBudgetMultiplier = checkedConfig.loadBalanceConfig().queueScheduler().getScanBudgetMultiplier();

        AtomicInteger plannerId = new AtomicInteger();
        ThreadFactory plannerFactory = task -> {
            Thread thread = new Thread(task,
                    "flexlb-global-planner-" + plannerId.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
        this.plannerCount = Math.max(MIN_PLANNER_THREADS,
                checkedConfig.loadBalanceConfig().getInternalRuntime()
                        .getQueuePlannerThreads());
        this.planners = Executors.newFixedThreadPool(this.plannerCount, plannerFactory);
        this.decisionThread = new Thread(this::runDecisionLoop,
                "flexlb-global-decision");
        this.decisionThread.setDaemon(true);
        this.decisionThread.setUncaughtExceptionHandler((thread, failure) -> {
            Logger.error("Global queue decision thread failed", failure);
            close();
        });
        availability.addListener(availabilityListener);
        decisionThread.start();
    }

    /** Enqueue without selecting an endpoint on the ingress thread. */
    boolean offer(
            BalanceContext context,
            CompletableFuture<Response> future,
            int priority) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(future, "future");
        GlobalQueueEntry entry = new GlobalQueueEntry(context, future, normalizePriority(priority),
                router.resolvePolicyGroup(context));
        lock.lock();
        try {
            if (closed.get()) {
                return false;
            }
            orderedQueue.add(entry);
            // Completion removes this exact request from the ordering and wait
            // indexes without scanning the backlog.
            future.whenComplete((ignored, failure) -> removeRequest(entry));
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    int size() {
        lock.lock();
        try {
            return orderedQueue.size();
        } finally {
            lock.unlock();
        }
    }

    private void runDecisionLoop() {
        try {
            while (!closed.get()) {
                Plan completed = pollCompletedPlan();
                if (completed != null) {
                    processCompletedPlan(completed);
                }
                // Reuse released slots even when other completed plans are buffered.
                claimPlanningSlots().forEach(this::submitPlan);
                awaitIfNoWork();
            }
        } finally {
            closed.set(true);
            availability.removeListener(availabilityListener);
            planners.shutdown();
            drainOnClose();
        }
    }

    private Plan pollCompletedPlan() {
        lock.lock();
        try {
            return completedPlans.pollFirst();
        } finally {
            lock.unlock();
        }
    }

    private List<GlobalQueueEntry> claimPlanningSlots() {
        lock.lock();
        try {
            int slots = plannerCount - inFlight.size();
            if (closed.get() || slots == 0) {
                return List.of();
            }
            List<GlobalQueueEntry> candidates = planningCandidates(slots);
            inFlight.addAll(candidates);
            return candidates;
        } finally {
            lock.unlock();
        }
    }

    private void awaitIfNoWork() {
        lock.lock();
        try {
            // Check under the same lock as arrivals and result publication: a signal
            // between processing and this check must not leave completed work asleep.
            while (!closed.get() && completedPlans.isEmpty()
                    && (inFlight.size() == plannerCount
                        || (!orderedQueue.hasUnscannedRequests()
                            && !waitingRequests.hasReadyRequests()))) {
                awaitChanged();
            }
        } finally {
            lock.unlock();
        }
    }

    private void submitPlan(GlobalQueueEntry entry) {
        try {
            planners.execute(() -> {
                Plan result;
                try {
                    result = plan(entry);
                } catch (Throwable failure) {
                    result = Plan.failure(entry, failure, availability.sequence());
                }
                publishPlan(result);
            });
        } catch (Throwable failure) {
            publishPlan(Plan.failure(entry, failure, availability.sequence()));
        }
    }

    private void publishPlan(Plan plan) {
        lock.lock();
        try {
            if (!closed.get()) {
                completedPlans.addLast(plan);
                changed.signal();
                return;
            }
        } finally {
            lock.unlock();
        }
        // Shutdown may finish before a slow planner. The producer then owns cleanup.
        closePlan(plan);
    }

    private void processCompletedPlan(Plan plan) {
        boolean retry = false;
        try {
            if (!closed.get()) {
                Outcome outcome = commit(plan);
                retry = outcome == Outcome.REPLAN
                        || (outcome == Outcome.BLOCKED && !park(plan));
            }
        } catch (Throwable failure) {
            removeRequest(plan.entry);
            completeDecisionResponse(plan.entry, error(
                    StrategyErrorType.DISPATCH_FAILED,
                    "Placement failed: " + failure.getMessage()));
            Logger.error("Global queue commit failed: request_id={}",
                    plan.entry.context.getRequestId(), failure);
        } finally {
            // Release ownership before making this request eligible again.
            closePlan(plan);
            lock.lock();
            try {
                inFlight.remove(plan.entry);
                if (retry && isQueued(plan.entry)) {
                    orderedQueue.markRequestReadyForRetry(plan.entry);
                }
            } finally {
                lock.unlock();
            }
        }
    }

    private static void closePlan(Plan plan) {
        try {
            plan.close();
        } catch (Throwable failure) {
            Logger.warn("Failed to close global queue route plan", failure);
        }
    }

    /** Caller holds lock; every examined entry counts against the scan budget. */
    private List<GlobalQueueEntry> planningCandidates(int slots) {
        waitingRequests.resumeReady(slots, orderedQueue::markRequestReadyForRetry);
        return orderedQueue.scanForPlanningCandidates(
                slots, calculateScanBudget(slots), entry -> {
                    if (entry.future.isDone()) {
                        removeRequestUnderLock(entry);
                        return false;
                    }
                    return !entry.removed && !inFlight.contains(entry)
                            && !waitingRequests.isWaiting(entry);
                });
    }

    private Plan plan(GlobalQueueEntry entry) {
        long availabilitySequence = availability.sequence();
        if (entry.removed || entry.future.isDone()) {
            return Plan.done(entry, availabilitySequence);
        }
        // The decision point is the final hard-deadline gate. A delayed timer
        // must not allow an already expired request to scan the fleet or be
        // published to an endpoint queue.
        if (entry.context.requestExpired(System.currentTimeMillis())) {
            lifecycle.cancelRequest(
                    entry.context.getRequestId(),
                    0L,
                    CancelReason.DEADLINE_EXCEEDED);
            return Plan.done(entry, availabilitySequence);
        }
        AdmissionMutation mutation = lifecycle.claimAdmissionMutation(
                entry.context.getRequestId(), entry.future);
        if (mutation == null) {
            return Plan.done(entry, availabilitySequence);
        }
        try {
            // A wakeup grants a fresh fleet-wide decision, not an obligation
            // to return to the endpoint which published the capacity event.
            PlacementResult<RouteAdmission, PlacementKey> result =
                    router.select(entry.context, entry.routingGroup);
            if (result.status() == PlacementResult.Status.SUCCESS) {
                return Plan.success(
                        entry, mutation, result.value(), availabilitySequence);
            }
            mutation.close();
            return Plan.result(entry, result, availabilitySequence);
        } catch (Throwable failure) {
            mutation.close();
            return Plan.failure(entry, failure, availabilitySequence);
        }
    }

    private Outcome commit(Plan plan) {
        GlobalQueueEntry entry = plan.entry;
        if (entry.removed || entry.future.isDone()) {
            plan.close();
            removeRequest(entry);
            return Outcome.DONE;
        }
        if (plan.failure != null) {
            removeRequest(entry);
            completeDecisionResponse(entry, error(
                    StrategyErrorType.DISPATCH_FAILED,
                    "Placement failed: " + plan.failure.getMessage()));
            return Outcome.DONE;
        }
        PlacementResult<RouteAdmission, PlacementKey> result = plan.result;
        if (result.status() == PlacementResult.Status.REJECTED) {
            removeRequest(entry);
            completeDecisionResponse(entry, result.rejection());
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.BLOCKED) {
            return Outcome.BLOCKED;
        }
        if (result.status() == PlacementResult.Status.CLOSED) {
            removeRequest(entry);
            return Outcome.DONE;
        }
        RouteAdmission admission = plan.admission;
        try {
            PlacementResult<ScheduledRequest, PlacementKey> publication =
                    admission.tryEnqueue(entry.context, entry.future, lifecycle);
            if (publication.status() == PlacementResult.Status.SUCCESS) {
                removeRequest(entry);
                reportRouteSubmitted(entry.context, publication.value());
                return Outcome.DONE;
            }
            if (publication.status() == PlacementResult.Status.REJECTED
                    || publication.status() == PlacementResult.Status.CLOSED) {
                removeRequest(entry);
                return Outcome.DONE;
            }
            plan.blockedOn(publication.blocker());
            boolean staleSelection = admission.blockedEndpointChanged();
            if (!staleSelection && tryPriorityRescue(plan, admission.blockedEndpoint())) {
                removeRequest(entry);
                return Outcome.DONE;
            }
            return staleSelection ? Outcome.REPLAN : Outcome.BLOCKED;
        } finally {
            plan.close();
        }
    }

    private boolean tryPriorityRescue(Plan plan, WorkerEndpoint blockedEndpoint) {
        GlobalQueueEntry entry = plan.entry;
        if (!priorityOrdering || entry.future.isDone()) {
            return false;
        }
        plan.closeMutation();
        RouteAdmission admission = plan.admission();
        if (admission == null || !evictionManager.tryAdmit(
                entry.context, entry.future, admission, blockedEndpoint)) {
            return false;
        }
        // A successful rescue consumes the exact route synchronously or
        // transfers it to an asynchronous preemption transaction.
        plan.takeAdmission();
        return true;
    }

    private static boolean isQueued(GlobalQueueEntry entry) {
        return !entry.removed && !entry.future.isDone();
    }

    private boolean park(Plan plan) {
        lock.lock();
        try {
            return !isQueued(plan.entry)
                    || waitingRequests.park(plan.entry, plan.waitKey(), plan.availabilitySequence);
        } finally {
            lock.unlock();
        }
    }

    private void removeRequest(GlobalQueueEntry entry) {
        if (entry.removed) {
            return;
        }
        lock.lock();
        try {
            removeRequestUnderLock(entry);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Remove from both indexes atomically. Publication and completion callbacks
     * share this boundary and return any active retry opportunity to its domain.
     * Caller holds {@link #lock}.
     */
    private void removeRequestUnderLock(GlobalQueueEntry entry) {
        if (orderedQueue.remove(entry)) {
            waitingRequests.remove(entry);
            changed.signal();
        }
    }

    private int calculateScanBudget(int candidates) {
        return (int) Math.min(Integer.MAX_VALUE, Math.ceil(candidates * scanBudgetMultiplier));
    }

    private static boolean resolvePriorityOrdering(ConfigService configService) {
        return configService.loadBalanceConfig().isPriorityOrdering();
    }

    private void awaitChanged() {
        try {
            changed.await();
        } catch (InterruptedException interruption) {
            if (closed.get()) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void signal() {
        lock.lock();
        try {
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void onAvailabilityChanged(PlacementAvailability.Event event) {
        lock.lock();
        try {
            waitingRequests.capacityChanged(event.key());
            // Other planning work may progress even when no parked entry
            // matches this exact edge.
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void completeDecisionResponse(GlobalQueueEntry entry, Response response) {
        try {
            lifecycle.publishDecisionResponseAsync(
                    entry.context.getRequestId(), entry.future, response);
        } catch (Throwable failure) {
            Logger.error(
                    "Global queue response publication failed: request_id={}",
                    entry.context.getRequestId(), failure);
        }
    }

    private void reportRouteSubmitted(
            BalanceContext context,
            ScheduledRequest item) {
        try {
            reporter.reportRouteSubmitTimeMs(
                    item.prefill().getRole().name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (Throwable failure) {
            Logger.warn("Failed to record route-submit telemetry", failure);
        }
    }

    private void drainOnClose() {
        List<GlobalQueueEntry> abandoned;
        List<Plan> completed;
        lock.lock();
        try {
            abandoned = orderedQueue.drain();
            waitingRequests.clear();
            completed = List.copyOf(completedPlans);
            completedPlans.clear();
            inFlight.clear();
        } finally {
            lock.unlock();
        }
        completed.forEach(GlobalQueueCoordinator::closePlan);
        for (GlobalQueueEntry entry : abandoned) {
            completeDecisionResponse(entry, error(
                    StrategyErrorType.DISPATCH_FAILED,
                    "request scheduler is shutting down"));
        }
    }

    private static int normalizePriority(int priority) {
        return PriorityNormalizer.isValid(priority)
                ? priority : PriorityNormalizer.DEFAULT_PRIORITY;
    }

    private static Response error(StrategyErrorType type, String detail) {
        return RequestRegistry.buildErrorResponse(type, detail);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        availability.removeListener(availabilityListener);
        signal();
        // Every submitted task must run: late results release their own route ownership.
        planners.shutdown();
        if (Thread.currentThread() != decisionThread) {
            try {
                decisionThread.join(decisionThreadJoinTimeoutMs());
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        }
        awaitPlannerTermination();
    }

    private void awaitPlannerTermination() {
        try {
            planners.awaitTermination(
                    decisionThreadJoinTimeoutMs(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
        }
    }

    private long decisionThreadJoinTimeoutMs() {
        return Math.max(0L, configService.loadBalanceConfig()
                .getInternalRuntime()
                .getQueueDecisionThreadJoinTimeoutMs());
    }

    private enum Outcome {
        DONE,
        BLOCKED,
        REPLAN
    }

    private static final class Plan implements AutoCloseable {
        private final GlobalQueueEntry entry;
        private AdmissionMutation mutation;
        private RouteAdmission admission;
        private final PlacementResult<RouteAdmission, PlacementKey> result;
        private final Throwable failure;
        private final long availabilitySequence;
        // Retained after admission closes so the request can be parked.
        private PlacementKey waitKey;

        private Plan(
                GlobalQueueEntry entry,
                AdmissionMutation mutation,
                RouteAdmission admission,
                PlacementResult<RouteAdmission, PlacementKey> result,
                Throwable failure,
                long availabilitySequence) {
            this.entry = entry;
            this.mutation = mutation;
            this.admission = admission;
            this.result = result;
            this.failure = failure;
            this.availabilitySequence = availabilitySequence;
            if (result != null && result.status() == PlacementResult.Status.BLOCKED) {
                waitKey = result.blocker();
            }
        }

        static Plan success(GlobalQueueEntry entry, AdmissionMutation mutation,
                            RouteAdmission admission,
                            long availabilitySequence) {
            return new Plan(entry, mutation, admission,
                    PlacementResult.success(admission), null,
                    availabilitySequence);
        }

        static Plan result(GlobalQueueEntry entry,
                           PlacementResult<RouteAdmission, PlacementKey> result,
                           long availabilitySequence) {
            return new Plan(entry, null, null, result, null,
                    availabilitySequence);
        }

        static Plan done(GlobalQueueEntry entry, long availabilitySequence) {
            return new Plan(entry, null, null,
                    PlacementResult.closed(), null, availabilitySequence);
        }

        static Plan failure(GlobalQueueEntry entry, Throwable failure,
                            long availabilitySequence) {
            return new Plan(entry, null, null, null, failure,
                    availabilitySequence);
        }

        PlacementKey waitKey() {
            return waitKey;
        }

        void blockedOn(PlacementKey key) {
            waitKey = Objects.requireNonNull(key, "waitKey");
        }

        RouteAdmission takeAdmission() {
            RouteAdmission owned = admission;
            admission = null;
            return owned;
        }

        RouteAdmission admission() {
            return admission;
        }

        void closeMutation() {
            AdmissionMutation owned = mutation;
            mutation = null;
            if (owned != null) {
                owned.close();
            }
        }

        @Override
        public void close() {
            RouteAdmission ownedAdmission = admission;
            admission = null;
            if (ownedAdmission != null) {
                try {
                    ownedAdmission.close();
                } catch (Throwable failure) {
                    Logger.warn("Failed to close abandoned route plan", failure);
                }
            }
            closeMutation();
        }
    }
}

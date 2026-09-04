package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Objects;
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
 * <p>This queue is the model's ordered placement boundary. A request is
 * selected from the complete live candidate fleet before the endpoint runtime
 * receives it. A bounded planning frontier controls how many
 * independent routes can be prepared together; request collection and grouping
 * remain exclusively endpoint-runtime concerns. The queue lock protects only
 * index operations; route projection and RPCs are never performed while it is
 * held.</p>
 *
 * <p>Plans are committed in queue order unless the exact endpoint selected by
 * one request is locally full. In that case the request is parked against that
 * endpoint and a later plan may commit only when it does not use the parked
 * endpoint. A selector-level miss has no concrete endpoint and therefore still
 * stops its ordered suffix. Planning concurrency is bounded by aggregate
 * delivery credits, while {@link WorkerBatcher} remains the sole SINGLE or
 * FIXED_WINDOW group owner. This keeps cache/KV
 * projections adjacent to each exact reservation while retaining planner
 * parallelism where the policy permits it.</p>
 */
final class GlobalQueueCoordinator implements AutoCloseable {

    private static final int MIN_PLANNER_THREADS = 1;
    private static final int MIN_PLANNING_FRONTIER_SIZE = 1;

    private final DefaultRouter router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestRegistry lifecycle;
    private final PlacementAvailability availability;
    private final boolean priorityOrdering;
    private final int plannerCount;
    private final ConfigService configService;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final OrderedRequestQueue orderedQueue;
    private final BlockedRequestIndex blockedRequests =
            new BlockedRequestIndex();
    private final ExecutorService planners;
    private final Thread decisionThread;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            this::onAvailabilityChanged;

    GlobalQueueCoordinator(
            ConfigService configService,
            DefaultRouter router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestRegistry lifecycle,
            PlacementAvailability availability) {
        ConfigService checkedConfig = Objects.requireNonNull(
                configService, "configService");
        this.configService = checkedConfig;
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.evictionManager = Objects.requireNonNull(
                evictionManager, "evictionManager");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.priorityOrdering = resolvePriorityOrdering(checkedConfig);
        this.orderedQueue = new OrderedRequestQueue(priorityOrdering);

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
        GlobalQueueEntry entry = new GlobalQueueEntry(context, future, normalizePriority(priority));
        lock.lock();
        try {
            if (closed.get()) {
                return false;
            }
            orderedQueue.add(entry);
            // Completion unlinks this exact intrusive node in O(1), so
            // cancellation never scans the queue or leaves historical nodes
            // behind a blocked head.
            future.whenComplete((ignored, failure) -> markCompleted(entry));
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
                List<GlobalQueueEntry> frontier;
                try {
                    frontier = nextPlanningFrontier();
                } catch (Throwable failure) {
                    Logger.error(
                            "Global queue planning-frontier capture failed",
                            failure);
                    continue;
                }
                if (frontier.isEmpty()) {
                    continue;
                }
                PlanningPipeline plans = new PlanningPipeline(
                        frontier, plannerCount);
                boolean restartFrontier = false;
                for (int planIndex = 0; planIndex < plans.size(); planIndex++) {
                    Plan plan = plans.awaitNext();
                    if (closed.get()) {
                        closePlan(plan);
                        plans.closeSubmitted();
                        break;
                    }
                    if (!isQueued(plan.entry)) {
                        closePlan(plan);
                        continue;
                    }
                    if (hasHigherPriorityEntry(plan.entry)) {
                        // A priority arrival may race a captured planner
                        // frontier. Ordering must not depend on CPU count or
                        // how many low-priority plans happened to be in flight.
                        closePlan(plan);
                        plans.closeSubmitted();
                        restartFrontier = true;
                        break;
                    }
                    if (parkIfConflicting(plan)) {
                        closePlan(plan);
                        continue;
                    }
                    Outcome outcome;
                    try {
                        outcome = commit(plan);
                    } catch (Throwable failure) {
                        closePlan(plan);
                        remove(plan.entry);
                        completeDecisionResponse(plan.entry, error(
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                "Placement failed: " + failure.getMessage()));
                        Logger.error("Global queue commit failed: request_id={}",
                                plan.entry.context.getRequestId(), failure);
                        continue;
                    }
                    if (outcome == Outcome.REPLAN) {
                        // The winner became stale at endpoint-local commit.
                        // Replan this ordered frontier before consuming its
                        // suffix. Otherwise a later request which selected the
                        // same endpoint could overtake the older request before
                        // its exact blocker has been published.
                        plans.closeSubmitted();
                        restartFrontier = true;
                        break;
                    }
                    if (outcome == Outcome.BLOCKED) {
                        PlacementKey blocker = plan.blocker();
                        if (blocker == null) {
                            throw new IllegalStateException(
                                    "blocked placement has no capacity domain");
                        }
                        WorkerEndpoint blockedEndpoint = plan.blockedEndpoint();
                        if (!parkIfCapacityUnchanged(
                                plan, blocker, blockedEndpoint)) {
                            // The capacity edge may have been published after
                            // this planner captured its snapshot but before
                            // the atomic park. Retry against the newer state.
                            closePlan(plan);
                            plans.closeSubmitted();
                            restartFrontier = true;
                            break;
                        }
                        if (blockedEndpoint == null) {
                            // There is no exact engine identity to compare. Keep
                            // strict frontier semantics for this case.
                            closePlan(plan);
                            plans.closeSubmitted();
                            break;
                        }
                        closePlan(plan);
                        // The next plan can use another engine immediately. Plans
                        // on this same engine are parked as well by the conflict
                        // check above, so no repeated failed reservation occurs.
                        continue;
                    }
                }
                if (restartFrontier) {
                    signal();
                }
            }
        } finally {
            drainOnClose();
        }
    }

    private Plan awaitPlan(
            CompletableFuture<Plan> future,
            GlobalQueueEntry entry) {
        try {
            return future.join();
        } catch (Throwable failure) {
            Logger.error("Global queue planner failed: request_id={}",
                    entry.context.getRequestId(), failure);
            return Plan.failure(entry, failure, availability.sequence());
        }
    }

    private static void closePlan(Plan plan) {
        try {
            plan.close();
        } catch (Throwable failure) {
            Logger.warn("Failed to close global queue route plan", failure);
        }
    }

    private List<GlobalQueueEntry> nextPlanningFrontier() {
        while (!closed.get()) {
            long capacitySequence;
            lock.lock();
            try {
                orderedQueue.pruneCompletedHeads();
                if (orderedQueue.peekHead() == null) {
                    awaitChanged();
                    continue;
                }
                blockedRequests.clearStaleFrontier();
                capacitySequence = availability.sequence();
            } finally {
                lock.unlock();
            }

            // Endpoint credit aggregation may inspect every Prefill endpoint.
            // Keep it outside the global ordering lock so ingress/cancel never
            // waits behind a 750-worker fleet scan.
            int frontierSize = planningFrontierSize();

            lock.lock();
            try {
                orderedQueue.pruneCompletedHeads();
                if (orderedQueue.peekHead() == null) {
                    continue;
                }
                blockedRequests.clearStaleFrontier();
                if (frontierSize <= 0
                        && availability.sequence() != capacitySequence) {
                    // A capacity edge raced with the advisory scan. Recompute
                    // instead of sleeping after the wakeup has linearized.
                    continue;
                }
                List<GlobalQueueEntry> frontier = orderedQueue.snapshotPrefix(
                        frontierSize,
                        this::isEligible,
                        blockedRequests.frontier());
                if (frontier.isEmpty()) {
                    // Every queued entry is parked, or the first otherwise
                    // eligible entry is the selector-level frontier. A
                    // capacity event, a cancellation, or a newly inserted
                    // higher-priority request will wake the coordinator.
                    awaitChanged();
                    continue;
                }
                return frontier;
            } finally {
                lock.unlock();
            }
        }
        return List.of();
    }

    private boolean isEligible(GlobalQueueEntry entry) {
        return !entry.removed && !entry.future.isDone()
                && !blockedRequests.isExactBlocked(entry);
    }

    /**
     * Bounds speculative work to the planner pool. A suffix is submitted only
     * as earlier plans are consumed, so a blocked head can abandon at most the
     * in-flight planner set rather than the complete planning frontier.
     */
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
            PlacementResult<QueueRouteAdmission, PlacementKey> result =
                    router.routeForQueue(entry.context);
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
            remove(entry);
            return Outcome.DONE;
        }
        if (plan.failure != null) {
            remove(entry);
            completeDecisionResponse(entry, error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Placement failed: " + plan.failure.getMessage()));
            return Outcome.DONE;
        }
        PlacementResult<QueueRouteAdmission, PlacementKey> result = plan.result;
        if (result.status() == PlacementResult.Status.REJECTED) {
            remove(entry);
            completeDecisionResponse(entry, result.rejection());
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.BLOCKED) {
            entry.blockedKey = result.blocker();
            return Outcome.BLOCKED;
        }
        if (result.status() == PlacementResult.Status.CLOSED) {
            remove(entry);
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.LIMIT_REACHED) {
            remove(entry);
            completeAcceptanceLimit(entry);
            return Outcome.DONE;
        }
        QueueRouteAdmission admission = plan.admission;
        try {
            PlacementResult<ScheduledRequest, PlacementKey> publication =
                    admission.tryPublish(entry.context, entry.future, lifecycle);
            if (publication.status() == PlacementResult.Status.SUCCESS) {
                removeCommitted(entry, admission);
                reportRouteSubmitted(entry.context, publication.value());
                return Outcome.DONE;
            }
            if (publication.status() == PlacementResult.Status.LIMIT_REACHED) {
                remove(entry);
                completeAcceptanceLimit(entry);
                return Outcome.DONE;
            }
            if (publication.status() == PlacementResult.Status.REJECTED
                    || publication.status() == PlacementResult.Status.CLOSED) {
                remove(entry);
                return Outcome.DONE;
            }
            entry.blockedKey = publication.blocker();
            plan.rememberBlockedEndpoint(admission.blockedEndpoint());
            boolean staleSelection = admission.blockedSelectionBecameStale();
            if (staleSelection) {
                return Outcome.REPLAN;
            }
            if (tryPriorityRescue(plan)) {
                remove(entry);
                return Outcome.DONE;
            }
            return Outcome.BLOCKED;
        } finally {
            plan.close();
        }
    }

    private boolean tryPriorityRescue(Plan plan) {
        GlobalQueueEntry entry = plan.entry;
        if (!priorityOrdering || entry.future.isDone()) {
            return false;
        }
        WorkerEndpoint blockedEndpoint = plan.blockedEndpoint();
        plan.closeMutation();
        QueueRouteAdmission admission = plan.admission();
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

    /** Revalidate PRIORITY at the commit linearization point. */
    private boolean hasHigherPriorityEntry(GlobalQueueEntry entry) {
        lock.lock();
        try {
            return orderedQueue.hasHigherPriorityEntry(
                    entry, this::isEligible);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Preserve queue order per endpoint while allowing independent endpoints
     * to progress. Conflict discovery and park publication share the ordering
     * lock, so an endpoint release cannot linearize between those operations.
     */
    private boolean parkIfConflicting(Plan plan) {
        QueueRouteAdmission admission = plan.admission;
        if (admission == null) {
            return false;
        }
        lock.lock();
        try {
            BlockedRequestIndex.Conflict conflict =
                    blockedRequests.conflict(plan.entry, admission);
            if (conflict == null) {
                return false;
            }
            parkEndpointUnderLock(
                    plan.entry,
                    conflict.blocker(),
                    conflict.endpoint());
            return true;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Atomically close the plan-snapshot/availability-edge race and publish
     * either an exact endpoint blocker or a strict selector frontier.
     *
     * @return false when a newer capacity edge requires immediate replanning
     */
    private boolean parkIfCapacityUnchanged(
            Plan plan,
            PlacementKey blocker,
            WorkerEndpoint endpoint) {
        lock.lock();
        try {
            if (!isQueued(plan.entry)) {
                return true;
            }
            if (availability.lastChangedSequence(blocker)
                    > plan.availabilitySequence) {
                return false;
            }
            if (endpoint == null) {
                blockedRequests.parkFrontier(plan.entry, blocker);
                changed.signal();
            } else {
                parkEndpointUnderLock(plan.entry, blocker, endpoint);
            }
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** Caller holds {@link #lock}. */
    private void parkEndpointUnderLock(
            GlobalQueueEntry entry,
            PlacementKey blocker,
            WorkerEndpoint endpoint) {
        if (!isQueued(entry)) {
            return;
        }
        blockedRequests.parkExact(entry, blocker, endpoint);
        changed.signal();
    }

    private void remove(GlobalQueueEntry entry) {
        lock.lock();
        try {
            if (!orderedQueue.remove(entry)) {
                return;
            }
            blockedRequests.clearEntry(entry);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void removeCommitted(
            GlobalQueueEntry entry,
            QueueRouteAdmission admission) {
        lock.lock();
        try {
            if (!orderedQueue.remove(entry)) {
                return;
            }
            blockedRequests.routeCommitted(entry, admission);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    /** Unlink a completed request from its ordering bucket in O(1). */
    private void markCompleted(GlobalQueueEntry entry) {
        if (entry.removed) {
            return;
        }
        lock.lock();
        try {
            if (!orderedQueue.markCompleted(entry)) {
                return;
            }
            blockedRequests.clearEntry(entry);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private static boolean resolvePriorityOrdering(ConfigService configService) {
        return configService.loadBalanceConfig().isPriorityOrdering();
    }

    /**
     * Resolve the independent requests admitted to one planning pass. The
     * endpoint runtime owns dispatcher-specific credit accounting; this
     * global pump neither groups requests nor interprets decision policy.
     */
    private int planningFrontierSize() {
        var activeConfig = configService.loadBalanceConfig();
        RoleType admissionRole = router.queueAdmissionRole();
        long globalLimit = Math.max(
                MIN_PLANNING_FRONTIER_SIZE,
                activeConfig.queueScheduler().getCapacity()
                        .getMaxOutstandingRequestsGlobal());
        long available = endpointRegistry
                .availablePrefillDeliveryCredits(admissionRole);
        if (available <= 0L) {
            // With a live fleet, zero aggregate credit is authoritative:
            // wait for an endpoint edge instead of selecting requests
            // which cannot be committed. With no endpoint yet, route one
            // request so the selector can establish its role/group wait.
            return endpointRegistry.getEndpointCount(admissionRole) == 0
                    ? MIN_PLANNING_FRONTIER_SIZE : 0;
        }
        // Capture the complete release budget, but PlanningPipeline keeps only
        // plannerCount computations in flight and refills one as each ordered
        // plan is consumed. This removes a barrier every plannerCount requests
        // without allowing speculative work to exceed the planner pool.
        long frontier = Math.min(available, globalLimit);
        return (int) Math.max(MIN_PLANNING_FRONTIER_SIZE, frontier);
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
            if (event.kind() == PlacementAvailability.ChangeKind.TOPOLOGY) {
                blockedRequests.topologyChanged(event.key());
            } else {
                blockedRequests.capacityChanged(event.key());
            }
            // Aggregate planning credits can become available even when no
            // parked entry matches this exact edge.
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void completeAcceptanceLimit(GlobalQueueEntry entry) {
        BalanceContext context = entry.context;
        int limit = context.getConfig().queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        String detail = "admission capacity is temporarily exhausted"
                + "; active_admissions=" + lifecycle.decodeAcceptanceCount()
                + " limit=" + limit;
        Response failure = Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        failure.setErrorMessage(
                StrategyErrorType.RESOURCE_EXHAUSTED.buildErrorMessage(detail));
        completeDecisionResponse(entry, failure);
    }

    private void completeDecisionResponse(GlobalQueueEntry entry, Response response) {
        try {
            lifecycle.publishQueueDecisionResponseAsync(
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
                    RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (Throwable failure) {
            Logger.warn("Failed to record route-submit telemetry", failure);
        }
    }

    private void drainOnClose() {
        List<GlobalQueueEntry> abandoned;
        lock.lock();
        try {
            abandoned = orderedQueue.drain();
            blockedRequests.clear();
        } finally {
            lock.unlock();
        }
        for (GlobalQueueEntry entry : abandoned) {
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
            completeDecisionResponse(entry, error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
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
        // Do not discard queued CompletableFuture tasks: the decision thread
        // may be joining one while it drains the current planning pipeline.
        planners.shutdown();
        if (Thread.currentThread() != decisionThread) {
            try {
                decisionThread.join(decisionThreadJoinTimeoutMs());
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        }
        if (!decisionThread.isAlive()) {
            planners.shutdownNow();
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

    /** Ordered, bounded submission view over one captured planning frontier. */
    private final class PlanningPipeline {
        private final List<GlobalQueueEntry> entries;
        private final ArrayDeque<SubmittedPlan> submitted;
        private final int maxInFlight;
        private int nextToSubmit;

        private PlanningPipeline(
                List<GlobalQueueEntry> entries,
                int maxInFlight) {
            this.entries = List.copyOf(entries);
            this.maxInFlight = Math.max(MIN_PLANNER_THREADS, maxInFlight);
            this.submitted = new ArrayDeque<>(Math.min(
                    this.maxInFlight, entries.size()));
        }

        private int size() {
            return entries.size();
        }

        private Plan awaitNext() {
            fill();
            SubmittedPlan next = submitted.removeFirst();
            return awaitPlan(next.future(), next.entry());
        }

        private void fill() {
            while (submitted.size() < maxInFlight
                    && nextToSubmit < entries.size()) {
                GlobalQueueEntry entry = entries.get(nextToSubmit++);
                submitted.addLast(new SubmittedPlan(entry, submit(entry)));
            }
        }

        private void closeSubmitted() {
            // A submitted plan owns its entry's sole AdmissionMutation.
            // Re-entering the queue before it closes can misread temporary
            // ownership as terminal state. Abandonment is rare and bounded by
            // plannerCount, so retire the submitted suffix synchronously.
            while (!submitted.isEmpty()) {
                SubmittedPlan pending = submitted.removeFirst();
                closePlan(awaitPlan(pending.future(), pending.entry()));
            }
        }

        private CompletableFuture<Plan> submit(GlobalQueueEntry entry) {
            try {
                return CompletableFuture.supplyAsync(
                        () -> plan(entry), planners);
            } catch (Throwable failure) {
                Logger.error("Global queue planner submission failed", failure);
                return CompletableFuture.completedFuture(
                        Plan.failure(entry, failure, availability.sequence()));
            }
        }

        private record SubmittedPlan(
                GlobalQueueEntry entry,
                CompletableFuture<Plan> future) {}
    }

    private static final class Plan implements AutoCloseable {
        private final GlobalQueueEntry entry;
        private AdmissionMutation mutation;
        private QueueRouteAdmission admission;
        private final PlacementResult<QueueRouteAdmission, PlacementKey> result;
        private final Throwable failure;
        private final long availabilitySequence;
        private WorkerEndpoint blockedEndpoint;

        private Plan(
                GlobalQueueEntry entry,
                AdmissionMutation mutation,
                QueueRouteAdmission admission,
                PlacementResult<QueueRouteAdmission, PlacementKey> result,
                Throwable failure,
                long availabilitySequence) {
            this.entry = entry;
            this.mutation = mutation;
            this.admission = admission;
            this.result = result;
            this.failure = failure;
            this.availabilitySequence = availabilitySequence;
        }

        static Plan success(GlobalQueueEntry entry, AdmissionMutation mutation,
                            QueueRouteAdmission admission,
                            long availabilitySequence) {
            return new Plan(entry, mutation, admission,
                    PlacementResult.success(admission), null,
                    availabilitySequence);
        }

        static Plan result(GlobalQueueEntry entry,
                           PlacementResult<QueueRouteAdmission, PlacementKey> result,
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

        PlacementKey blocker() {
            if (result != null && result.status() == PlacementResult.Status.BLOCKED) {
                return result.blocker();
            }
            return entry.blockedKey;
        }

        WorkerEndpoint blockedEndpoint() {
            if (blockedEndpoint != null) {
                return blockedEndpoint;
            }
            if (admission != null) {
                WorkerEndpoint endpoint = admission.blockedEndpoint();
                if (endpoint != null) {
                    return endpoint;
                }
            }
            return entry.blockedEndpoint;
        }

        void rememberBlockedEndpoint(WorkerEndpoint endpoint) {
            blockedEndpoint = endpoint;
        }

        QueueRouteAdmission takeAdmission() {
            QueueRouteAdmission owned = admission;
            admission = null;
            return owned;
        }

        QueueRouteAdmission admission() {
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
            QueueRouteAdmission ownedAdmission = admission;
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

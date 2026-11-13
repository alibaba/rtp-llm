package org.flexlb.balance.scheduler;

import io.opentelemetry.api.common.Attributes;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillQueueSheddingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

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
import java.util.concurrent.ConcurrentLinkedQueue;
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
    /** Decision-loop wake period while shedding is enabled; otherwise unbounded. */
    private static final long SHED_WAKEUP_MS = 100L;

    private final DefaultRouter router;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestRegistry lifecycle;
    private final PlacementAvailability availability;
    private final boolean priorityOrdering;
    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final ConfigService configService;
    /** Optional age-based shedding; null keeps the queue purely deadline-driven. */
    private final PrefillQueueSheddingConfig sheddingConfig;
    /** Optional rescue of queued routes from congested Engines; null disables it. */
    private final RoutingConfig.HotspotRescueConfig hotspotRescue;
    /** Last rescue round start; rounds never accumulate while idle. */
    private long lastRescueRoundNanos = Long.MIN_VALUE;
    // Decision-thread confined feedback state, keyed by routing group.
    private final Map<String, PrefillQueueShedding.Controller> sheddingControllers =
            new LinkedHashMap<>();
    private final Map<String, PrefillTrafficTrend> trafficTrends = new LinkedHashMap<>();
    private final Map<String, PrefillDeadlineBudget> deadlineBudgets = new LinkedHashMap<>();
    /** Arrival classification funnel from ingress threads to the decision thread. */
    private final ConcurrentLinkedQueue<GlobalQueueEntry> arrivals = new ConcurrentLinkedQueue<>();
    /** Hard-timeout evidence funnel from planner threads to the decision thread. */
    private final ConcurrentLinkedQueue<HardTimeout> hardTimeouts = new ConcurrentLinkedQueue<>();
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final OrderedRequestQueue orderedQueue;
    private final PlacementWaitQueue waitingRequests;
    /** Retained until response completion so withdrawal reuses the original FIFO identity. */
    private final Map<CompletableFuture<Response>, GlobalQueueEntry> registered = new IdentityHashMap<>();
    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ArrayDeque<Plan> completedPlans = new ArrayDeque<>();
    /** Published by the decision thread; timeout readers never acquire the queue lock. */
    private volatile Map<String, Object> waitDiagnostics = Map.of("cause", "waiting for placement");
    /** Per-pool trend snapshots published by the decision thread; lock-free read-only view. */
    private volatile Map<String, PrefillTrafficTrend.Snapshot> publishedTrafficTrends = Map.of();
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
        this.sheddingConfig = checkedConfig.loadBalanceConfig()
                .queueScheduler().getPrefillQueueShedding();
        this.hotspotRescue = hotspotRescueConfig(checkedConfig);
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
        lifecycle.attachGlobalQueue(this);
        decisionThread.start();
    }

    Map<String, Object> waitDiagnostics() {
        return waitDiagnostics;
    }

    /** Read-only per-pool prefill traffic snapshot; readers never take the queue lock. */
    Map<String, PrefillTrafficTrend.Snapshot> prefillTrafficSnapshot() {
        return publishedTrafficTrends;
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
            registered.put(future, entry);
            FlexlbTrace.event(context.getTraceContext(), "flexlb.queue.enqueued",
                    Attributes.builder()
                            .put("flexlb.queue.depth", orderedQueue.size())
                            .put("flexlb.request.priority", priority).build());
            if (sheddingConfig != null) {
                // The decision thread classifies arrivals; retries and
                // migrations never re-enter this funnel.
                arrivals.add(entry);
            }
            // Completion removes this exact request from the ordering and wait
            // indexes without scanning the backlog.
            future.whenComplete((ignored, failure) -> completeRequest(entry));
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

    /** A withdrawal is not a new request: keep its sequence, context, Future and absolute deadline. */
    boolean requeue(ScheduledRequest previous) {
        lock.lock();
        try {
            if (previous.future().isDone()) { return true; }
            GlobalQueueEntry entry = registered.get(previous.future());
            if (closed.get() || entry == null) { return false; }
            if (!entry.removed) { throw new IllegalStateException("withdrawn route still has a global queue entry"); }
            entry.context.setPlanType("");
            entry.context.setPlanCost(0L);
            entry.context.setVictimCount(0);
            orderedQueue.restore(entry);
            FlexlbTrace.event(entry.context.getTraceContext(), "flexlb.queue.requeued",
                    Attributes.builder()
                            .put("flexlb.queue.depth", orderedQueue.size()).build());
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    private void completeRequest(GlobalQueueEntry entry) {
        lock.lock();
        try {
            registered.remove(entry.future, entry);
            removeRequestUnderLock(entry);
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
                shedOverloadedQueue();
                rescueHotspotQueues();
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
                // A cadence wake leaves the park so the loop re-runs its control
                // rounds: rescue acts on worker queues while this queue is idle.
                if (!awaitChanged()) {
                    break;
                }
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
            recordQueueHardTimeout(entry);
            lifecycle.cancelRequest(
                    entry.context.getRequestId(),
                    0L,
                    CancelReason.DEADLINE_EXCEEDED);
            return Plan.done(entry, availabilitySequence);
        }
        AdmissionHandle handle = lifecycle.claimAdmissionHandle(
                entry.context.getRequestId(), entry.future);
        if (handle == null) {
            return Plan.done(entry, availabilitySequence);
        }
        long decisionStartedNanos = System.nanoTime();
        FlexlbTrace.event(entry.context.getTraceContext(), "flexlb.decision.started");
        try {
            // A wakeup grants a fresh fleet-wide decision, not an obligation
            // to return to the endpoint which published the capacity event.
            PlacementResult<RouteAdmission, PlacementKey> result =
                    FlexlbTrace.stage(entry.context.getTraceContext(), "rtp_llm.flexlb.pd_decision",
                            () -> router.select(entry.context, entry.routingGroup));
            handle.recordDiagnostics(result.diagnostics());
            FlexlbTrace.recordPlacement(entry.context.getTraceContext(), decisionStartedNanos,
                    result.status().name(), result.diagnostics(), null);
            if (sheddingConfig != null
                    && result.status() == PlacementResult.Status.BLOCKED) {
                // Advisory fleet estimate captured while the request keeps
                // waiting; eviction decisions read it later.
                entry.sheddingEstimate = router.estimateQueuedWork(
                        entry.context, entry.routingGroup);
            }
            if (result.status() == PlacementResult.Status.SUCCESS) {
                return Plan.success(
                        entry, handle, result, availabilitySequence);
            }
            handle.close();
            return Plan.result(entry, result, availabilitySequence);
        } catch (Throwable failure) {
            handle.close();
            FlexlbTrace.recordPlacement(entry.context.getTraceContext(), decisionStartedNanos,
                    "ERROR", null, failure);
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
            completeDecisionResponse(entry, result.failure());
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.BLOCKED) {
            return Outcome.BLOCKED;
        }
        if (result.status() == PlacementResult.Status.CLOSED) {
            removeRequest(entry);
            return Outcome.DONE;
        }
        RouteAdmission admission = plan.admission();
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
        plan.closeAdmissionHandle();
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
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (!isQueued(plan.entry)) { return true; }
            parked = waitingRequests.park(plan.entry, plan.waitKey(), plan.availabilitySequence);
            PrefillQueueShedding.Estimate estimate = plan.entry.sheddingEstimate;
            if (estimate != null && estimate.hitTokens() == 0L) {
                plan.entry.knownColdRequest = true;
            }
            plan.entry.lastBlocker = plan.waitKey();
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
        details.put("cause", plan.waitKey().role().name() + " placement unavailable");
        details.put("role", plan.waitKey().role().name());
        if (plan.waitKey().group() != null) { details.put("group", plan.waitKey().group()); }
        if (plan.waitKey().endpoint() != null) { details.put("endpoint", plan.waitKey().endpoint()); }
        details.put("capturedAtMs", System.currentTimeMillis());
        details.put("queueDepth", depth);
        details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
        if (plan.result != null && plan.result.diagnostics() != null) {
            details.put("decision", plan.result.diagnostics());
        }
        waitDiagnostics = Collections.unmodifiableMap(details);
        return parked;
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

    // ==================== Prefill queue shedding ====================

    /** Hard-deadline evidence enqueued by planner threads; atMs is wall clock. */
    private record HardTimeout(String pool, long atMs) { }

    /** Planner threads only enqueue the evidence; controllers stay decision-thread confined. */
    private void recordQueueHardTimeout(GlobalQueueEntry entry) {
        if (sheddingConfig == null || router.prefillRole() == null) {
            return;
        }
        hardTimeouts.add(new HardTimeout(entry.routingGroup, System.currentTimeMillis()));
    }

    private void drainArrivals() {
        GlobalQueueEntry entry;
        while ((entry = arrivals.poll()) != null) {
            if (!entry.classifyArrival || entry.removed || entry.future.isDone()) {
                continue;
            }
            entry.classifyArrival = false;
            PrefillQueueShedding.Controller controller = controllerFor(entry.routingGroup);
            PrefillTrafficTrend trend = trendFor(entry.routingGroup);
            if (trend != null) {
                trend.arrive(Math.max(0L, entry.context.getRequest().getSeqLen()));
            }
            PrefillQueueShedding.Estimate estimate = entry.sheddingEstimate;
            if (estimate != null && estimate.hitTokens() == 0L) {
                controller.coldArrival();
            } else {
                controller.arrive();
            }
        }
    }

    private void drainHardTimeouts() {
        HardTimeout timeout;
        while ((timeout = hardTimeouts.poll()) != null) {
            controllerFor(timeout.pool()).hardTimeout(timeout.atMs());
        }
    }

    /**
     * Decision-thread eviction round: expired queued requests get their hard
     * deadline enforced, then each routing-group pool applies bounded soft
     * eviction. Controllers throttle themselves to one active round per second.
     */
    private void shedOverloadedQueue() {
        if (sheddingConfig == null) {
            return;
        }
        long nowMs = System.currentTimeMillis();
        long nowNanos = System.nanoTime();
        drainArrivals();
        drainHardTimeouts();
        List<GlobalQueueEntry> expired = new ArrayList<>();
        Map<String, List<PrefillQueueShedding.Candidate<GlobalQueueEntry>>> pools =
                new LinkedHashMap<>();
        lock.lock();
        try {
            if (closed.get()) {
                return;
            }
            for (GlobalQueueEntry entry : orderedQueue.entries()) {
                if (entry.removed || entry.future.isDone()
                        || inFlight.contains(entry)) {
                    continue;
                }
                if (entry.context.requestExpired(nowMs)) {
                    expired.add(entry);
                    continue;
                }
                PrefillQueueShedding.Estimate estimate = entry.sheddingEstimate;
                if (estimate == null) {
                    // Never blocked yet: only the hard deadline applies.
                    continue;
                }
                long waitedMs = Math.max(0L, (nowNanos - entry.queuedAtNanos) / 1_000_000L);
                pools.computeIfAbsent(entry.routingGroup, ignored -> new ArrayList<>())
                        .add(new PrefillQueueShedding.Candidate<>(
                                entry, entry.priority, entry.sequence, estimate, waitedMs));
            }
        } finally {
            lock.unlock();
        }
        for (GlobalQueueEntry entry : expired) {
            expireQueuedRequest(entry, nowMs);
        }
        for (Map.Entry<String, List<PrefillQueueShedding.Candidate<GlobalQueueEntry>>> pool
                : pools.entrySet()) {
            shedPool(pool.getKey(), pool.getValue(), nowMs);
        }
        publishTrafficTrends();
    }

    /** Decision-thread publication of the per-pool trend snapshots for diagnostics. */
    private void publishTrafficTrends() {
        if (trafficTrends.isEmpty()) {
            return;
        }
        Map<String, PrefillTrafficTrend.Snapshot> snapshot = new LinkedHashMap<>();
        for (Map.Entry<String, PrefillTrafficTrend> entry : trafficTrends.entrySet()) {
            PrefillTrafficTrend.Snapshot latest = entry.getValue().latestSnapshot();
            if (latest != null) {
                snapshot.put(entry.getKey(), latest);
            }
        }
        publishedTrafficTrends = Collections.unmodifiableMap(snapshot);
    }

    private void expireQueuedRequest(GlobalQueueEntry entry, long nowMs) {
        controllerFor(entry.routingGroup).hardTimeout(nowMs);
        lifecycle.cancelRequest(
                entry.context.getRequestId(),
                0L,
                CancelReason.DEADLINE_EXCEEDED);
        removeRequest(entry);
    }

    private void shedPool(
            String pool,
            List<PrefillQueueShedding.Candidate<GlobalQueueEntry>> candidates,
            long nowMs) {
        CostBasedPrefillStrategy.PrefillPoolSnapshot snapshot = router.prefillPoolSnapshot(pool);
        if (snapshot == null || snapshot.workers() <= 0) {
            return;
        }
        PrefillQueueShedding.Controller controller = controllerFor(pool);
        PrefillTrafficTrend trend = trendFor(pool);
        double queuedWorkMs = 0.0;
        for (PrefillQueueShedding.Candidate<GlobalQueueEntry> candidate : candidates) {
            queuedWorkMs += candidate.estimate().fleetWorkMs();
        }
        PrefillTrafficTrend.Snapshot observed = trend == null ? null
                : trend.observe(nowMs, queuedWorkMs,
                        sheddingConfig.getExpectedWaitThresholdMs(),
                        sheddingConfig.getMaxWaitMs());
        List<GlobalQueueEntry> victims;
        if (sheddingConfig.getDeadlineBudget() != null) {
            victims = deadlineBudgetVictims(pool, candidates, snapshot, nowMs);
        } else {
            // Soft eviction requires sustained pool-wide Engine queue pressure;
            // recovery headroom only reduces, never triggers, eviction.
            victims = PrefillQueueShedding.adaptiveVictims(
                    candidates, sheddingConfig, observed, controller, nowMs,
                    router.isPrefillClusterOverloaded(pool, nowMs));
        }
        double acceptingPercent = snapshot.accepting() * 100.0 / snapshot.workers();
        int selected = victims.size();
        int accepted = 0;
        for (GlobalQueueEntry victim : victims) {
            if (evictPoolVictim(victim, acceptingPercent, trend)) {
                accepted++;
            }
        }
        controller.recordEvictions(nowMs, selected, accepted);
    }

    private List<GlobalQueueEntry> deadlineBudgetVictims(
            String pool,
            List<PrefillQueueShedding.Candidate<GlobalQueueEntry>> candidates,
            CostBasedPrefillStrategy.PrefillPoolSnapshot snapshot,
            long nowMs) {
        PrefillDeadlineBudget budget = deadlineBudgets.computeIfAbsent(
                pool, ignored -> new PrefillDeadlineBudget());
        PrefillDeadlineBudget.Fleet fleet = new PrefillDeadlineBudget.Fleet(
                snapshot.workers(),
                snapshot.workers() - snapshot.accepting(),
                snapshot.pendingTokens(),
                true);
        List<PrefillDeadlineBudget.Candidate<GlobalQueueEntry>> budgeted =
                new ArrayList<>(candidates.size());
        for (PrefillQueueShedding.Candidate<GlobalQueueEntry> candidate : candidates) {
            GlobalQueueEntry entry = candidate.value();
            PrefillQueueShedding.Estimate estimate = candidate.estimate();
            long computeTokens = Math.max(
                    0L, estimate.inputTokens() - estimate.hitTokens());
            boolean cold = entry.knownColdRequest || estimate.hitTokens() == 0L;
            long remainingMs = Math.max(
                    0L, entry.context.getRequestExpiresAtMs() - nowMs);
            budgeted.add(new PrefillDeadlineBudget.Candidate<>(
                    entry, candidate.priority(), candidate.sequence(),
                    computeTokens, cold, remainingMs));
        }
        return budget.victims(budgeted, fleet,
                sheddingConfig.getDeadlineBudget(), priorityOrdering, nowMs);
    }

    /** Evict one victim, or veto a cold request while placement capacity exists. */
    private boolean evictPoolVictim(
            GlobalQueueEntry victim,
            double acceptingPercent,
            PrefillTrafficTrend trend) {
        PrefillQueueShedding.Estimate estimate = victim.sheddingEstimate;
        boolean cold = victim.knownColdRequest
                || (estimate != null && estimate.hitTokens() == 0L);
        lock.lock();
        try {
            if (closed.get() || victim.removed || victim.future.isDone()
                    || inFlight.contains(victim)) {
                return false;
            }
            if (cold && acceptingPercent >= sheddingConfig.getMinAcceptingWorkerPercent()) {
                // Cold-request veto: workers still admit, so retry placement
                // instead of dropping work that has not started anywhere.
                orderedQueue.markRequestReadyForRetry(victim);
                return false;
            }
            removeRequestUnderLock(victim);
        } finally {
            lock.unlock();
        }
        completeDecisionResponse(victim, error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                "prefill queue overloaded: request evicted while waiting for placement"));
        if (trend != null && estimate != null) {
            // Own evictions must not masquerade as natural queue drainage.
            trend.discard(estimate.fleetWorkMs());
        }
        Logger.info(
                "Prefill queue shedding evicted request: request_id={}, pool={}",
                victim.context.getRequestId(), victim.routingGroup);
        return true;
    }

    private PrefillQueueShedding.Controller controllerFor(String pool) {
        return sheddingControllers.computeIfAbsent(
                pool, ignored -> new PrefillQueueShedding.Controller());
    }

    private PrefillTrafficTrend trendFor(String pool) {
        PrefillQueueSheddingConfig.RecoveryConfig recovery = sheddingConfig.getRecovery();
        if (recovery == null) {
            return null;
        }
        return trafficTrends.computeIfAbsent(
                pool, ignored -> new PrefillTrafficTrend(recovery));
    }

    // ==================== Hotspot queue rescue ====================

    /**
     * Decision-thread rescue round: withdraw a bounded set of small, aged
     * queued routes from congested Engines and return them to this queue for
     * re-selection on an idle, cache-eligible worker. One round per migration
     * interval; unused rounds never accumulate. A rescue keeps the request
     * alive, so it is not an eviction and never completes the Future.
     */
    void rescueHotspotQueues() {
        RoutingConfig.HotspotRescueConfig rescue = hotspotRescue;
        if (rescue == null || closed.get()) {
            return;
        }
        long nowNanos = System.nanoTime();
        if (lastRescueRoundNanos != Long.MIN_VALUE && nowNanos - lastRescueRoundNanos
                < TimeUnit.MILLISECONDS.toNanos(rescue.getMigrationIntervalMs())) {
            return;
        }
        lastRescueRoundNanos = nowNanos;
        int limit = Math.min(plannerCount, rescue.getMaxMigrationsPerRound());
        List<CostBasedPrefillStrategy.HotspotRescuePool> pools =
                router.hotspotRescuePools(rescue);
        int migrated = 0;
        for (CostBasedPrefillStrategy.HotspotRescuePool pool : pools) {
            if (migrated >= limit) {
                break;
            }
            if (!pool.idleTarget()) {
                continue;
            }
            for (PrefillEndpoint source : pool.sources()) {
                if (migrated >= limit) {
                    break;
                }
                migrated += rescueSourceQueue(source, pool.group(), rescue, limit - migrated);
            }
        }
    }

    /** Withdraw the best rescue candidates from one congested source queue. */
    private int rescueSourceQueue(
            PrefillEndpoint source,
            String group,
            RoutingConfig.HotspotRescueConfig rescue,
            int remaining) {
        List<ScheduledRequest> items = source.captureQueueSnapshot().items();
        // Bounded scan in scheduling order; the shortlist is then ranked by total
        // input and original worker-queue sequence, like ordinary queue ordering.
        int budget = (int) Math.min(items.size(), Math.max(64L, 64L * remaining));
        long nowMs = System.currentTimeMillis();
        List<ScheduledRequest> candidates = new ArrayList<>();
        for (int index = 0; index < budget; index++) {
            ScheduledRequest item = items.get(index);
            if (item.future().isDone() || item.requestExpired(nowMs)
                    || item.seqLen() > rescue.getMaxInputTokens()
                    || item.ctx().getPrefillMigrationSource() != null
                    || nowMs - item.ctx().getStartTime() < rescue.getMinWaitMs()) {
                continue;
            }
            candidates.add(item);
        }
        candidates.sort(java.util.Comparator
                .comparingLong(ScheduledRequest::seqLen)
                .thenComparingLong(item -> item.ctx().getWorkerEnqueueSequence()));
        // Validate only the bounded shortlist, ranked first; final placement
        // rechecks the current cache fleet.
        List<ScheduledRequest> shortlist = candidates.subList(
                0, Math.min(candidates.size(), remaining));
        int migrated = 0;
        for (ScheduledRequest candidate : shortlist) {
            if (migrated >= remaining) {
                break;
            }
            if (!router.hasCacheEligibleRescueTarget(candidate.ctx(), group, source)) {
                continue;
            }
            if (lifecycle.migrateQueuedPrefillRoute(source, candidate)) {
                migrated++;
                Logger.debug(
                        "Hotspot rescue migrated queued prefill request: request_id={}, source={}",
                        candidate.requestId(), source.ipPort());
            }
        }
        return migrated;
    }

    private int calculateScanBudget(int candidates) {
        return (int) Math.min(Integer.MAX_VALUE, Math.ceil(candidates * scanBudgetMultiplier));
    }

    private static RoutingConfig.HotspotRescueConfig hotspotRescueConfig(
            ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        RoutingConfig.TokenBalanceConfig balance = config == null ? null
                : config.getRouter().getRoles().getPrefill().getTokenBalance();
        return balance == null ? null : balance.getHotspotRescue();
    }

    private static boolean resolvePriorityOrdering(ConfigService configService) {
        return configService.loadBalanceConfig().isPriorityOrdering();
    }

    /** True when parked without a cadence deadline; false after a cadence wake. */
    private boolean awaitChanged() {
        try {
            if (sheddingConfig == null && hotspotRescue == null) {
                changed.await();
                return true;
            }
            // Shedding and rescue cadence needs periodic control rounds even
            // without arrivals, capacity events or completed plans.
            changed.await(SHED_WAKEUP_MS, TimeUnit.MILLISECONDS);
            return false;
        } catch (InterruptedException interruption) {
            if (closed.get()) {
                Thread.currentThread().interrupt();
            }
            return true;
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
            registered.clear();
            arrivals.clear();
            hardTimeouts.clear();
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
        return Response.buildErrorResponse(type, detail);
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
        private AdmissionHandle handle;
        // A successful result owns its admission until transfer or close.
        private PlacementResult<RouteAdmission, PlacementKey> result;
        private final Throwable failure;
        private final long availabilitySequence;
        // Retained after admission closes so the request can be parked.
        private PlacementKey waitKey;

        private Plan(
                GlobalQueueEntry entry,
                AdmissionHandle handle,
                PlacementResult<RouteAdmission, PlacementKey> result,
                Throwable failure,
                long availabilitySequence) {
            this.entry = entry;
            this.handle = handle;
            this.result = result;
            this.failure = failure;
            this.availabilitySequence = availabilitySequence;
            if (result != null && result.status() == PlacementResult.Status.BLOCKED) {
                waitKey = result.blocker();
            }
        }

        static Plan success(GlobalQueueEntry entry, AdmissionHandle handle,
                            PlacementResult<RouteAdmission, PlacementKey> result,
                            long availabilitySequence) {
            return new Plan(entry, handle, result, null,
                    availabilitySequence);
        }

        static Plan result(GlobalQueueEntry entry,
                           PlacementResult<RouteAdmission, PlacementKey> result,
                           long availabilitySequence) {
            return new Plan(entry, null, result, null,
                    availabilitySequence);
        }

        static Plan done(GlobalQueueEntry entry, long availabilitySequence) {
            return new Plan(entry, null,
                    PlacementResult.closed(), null, availabilitySequence);
        }

        static Plan failure(GlobalQueueEntry entry, Throwable failure,
                            long availabilitySequence) {
            return new Plan(entry, null, null, failure,
                    availabilitySequence);
        }

        PlacementKey waitKey() {
            return waitKey;
        }

        void blockedOn(PlacementKey key) {
            waitKey = Objects.requireNonNull(key, "waitKey");
        }

        RouteAdmission takeAdmission() {
            RouteAdmission owned = admission();
            result = null;
            return owned;
        }

        RouteAdmission admission() {
            return result == null ? null : result.value();
        }

        void closeAdmissionHandle() {
            AdmissionHandle owned = handle;
            handle = null;
            if (owned != null) {
                owned.close();
            }
        }

        @Override
        public void close() {
            RouteAdmission ownedAdmission = takeAdmission();
            if (ownedAdmission != null) {
                try {
                    ownedAdmission.close();
                } catch (Throwable failure) {
                    Logger.warn("Failed to close abandoned route plan", failure);
                }
            }
            closeAdmissionHandle();
        }
    }
}

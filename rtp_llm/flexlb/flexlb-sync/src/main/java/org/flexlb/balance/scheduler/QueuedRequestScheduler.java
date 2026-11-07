package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.ProjectionQueueSnapshot.AdmissionBlock;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter.QueueWaitSnapshot;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/**
 * Schedules global placement and worker delivery. Queues provide ordered candidates;
 * GroupPlanner chooses each delivery prefix and its next check time. This scheduler
 * owns the bounded planning tasks, capacity subscriptions and timer wakeups.
 */
public final class QueuedRequestScheduler extends AbstractRequestScheduler {

    /**
     * One finite scheduling turn, submitted by schedule(); no endpoint owns a decision loop.
     */
    static PrefillQueueScheduler.Decision decideQueueScheduling(
            PrefillQueueScheduler queueScheduler) {

        int maxRequests = queueScheduler.settings().maxRequests();
        PrefillEndpoint.QueueSnapshot snapshot = queueScheduler.captureQueue(maxRequests);
        RouteExecution head = snapshot.head();
        if (head == null) return PrefillQueueScheduler.Decision.NO_ACTION;
        long nowMs = System.currentTimeMillis();
        var requestState = head.ctx().snapshot();
        if (requestState.placement() != null && !requestState.placement().published()) {
            return PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                    head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    Long.MAX_VALUE,
                    "Route commit in progress");
        }
        if (head.requestExpired(nowMs)) {
            queueScheduler.removeAndNotify(head, null, false);
            return PrefillQueueScheduler.Decision.NO_ACTION;
        }
        GroupPlanner.Constraints constraints = queueScheduler.schedulingConstraints();
        if (Math.max(0L, head.seqLen()) > constraints.batchKvCapacity()) {
            return PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                    head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    head.expiresAtMs(),
                    "Prefill KV capacity exhausted");
        }
        for (int index = 1; index < Math.min(maxRequests, snapshot.items().size()); index++) {
            RouteExecution member = snapshot.items().get(index);
            if (member.requestExpired(nowMs)) {
                queueScheduler.removeAndNotify(member, null, false);
                return PrefillQueueScheduler.Decision.NO_ACTION;
            }
        }
        PrefillTimePredictor.Evaluator evaluator =
                queueScheduler.prefillEndpoint().predictor.evaluator();
        var decision = decideGroup(queueScheduler, snapshot, null, evaluator, constraints);
        if (decision.unavailable() != null
                && decision.unavailable().resource() == CapacityBoundary.Resource.DECODE_ENGINE) {
            queueScheduler.bypass = head;
            queueScheduler.blockedHead = decision;
            try {
                var alternative =
                        decideGroup(
                                queueScheduler,
                                queueScheduler
                                        .prefillEndpoint()
                                        .captureQueue(
                                                Math.max(2, maxRequests),
                                                head,
                                                queueScheduler.settings().priorityOrdering()),
                                head,
                                evaluator,
                                queueScheduler.schedulingConstraints());
                if (alternative.unavailable() == null && alternative.wakeAtMs() > 0L) {
                    decision =
                            new PrefillQueueScheduler.Decision(
                                    head,
                                    decision.unavailable(),
                                    decision.queueVersion(),
                                    decision.schedulingInputVersion(),
                                    alternative.wakeAtMs(),
                                    decision.waitReason());
                }
            } finally {
                queueScheduler.bypass = null;
                queueScheduler.blockedHead = null;
            }
        }
        return decision;
    }

    private static PrefillQueueScheduler.Decision decideGroup(
            PrefillQueueScheduler queueScheduler,
            PrefillEndpoint.QueueSnapshot snapshot,
            RouteExecution bypassedHead,
            PrefillTimePredictor.Evaluator evaluator,
            GroupPlanner.Constraints constraints) {

        var candidates = snapshot.items();
        if (candidates.isEmpty()) {
            return PrefillQueueScheduler.Decision.NO_ACTION;
        }
        RouteExecution first = candidates.getFirst();
        try {
            var predictor =
                    queueScheduler.settings().executionBudgetMs() > 0L
                            ? RequestTimeEstimation.newGroupPredictor(
                                    queueScheduler.settings().dispatcherType(), evaluator)
                            : null;
            long planningAtMs = System.currentTimeMillis();
            var selection =
                    queueScheduler.settings().grouping().select(candidates, constraints, predictor);
            if (selection.items().isEmpty()) {
                return PrefillQueueScheduler.Decision.NO_ACTION;
            }
            String reason =
                    queueScheduler
                            .settings()
                            .grouping()
                            .dispatchReason(selection, constraints, planningAtMs);
            if (reason == null) {
                return PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                        first,
                        snapshot.queueVersion(),
                        snapshot.schedulingInputVersion(),
                        Math.min(
                                queueScheduler
                                        .settings()
                                        .grouping()
                                        .nextCheckAtMs(selection, constraints),
                                first.expiresAtMs()),
                        "Prefill collection window");
            }
            // Recheck advisory capacity after prediction, before acquiring hard-capacity leases.
            constraints = queueScheduler.schedulingConstraints();
            if ((selection.items().size() > 1
                            && !selection.fitsCompute(constraints.batchTokenCapacity()))
                    || !selection.fitsKv(constraints.batchKvCapacity())) {
                return PrefillQueueScheduler.Decision.NO_ACTION;
            }
            OptionalLong prediction =
                    queueScheduler.settings().dispatcherType() != DispatcherConfig.Type.BATCH
                                    || selection.selectedPredictionMs().isEmpty()
                            ? OptionalLong.empty()
                            : OptionalLong.of(
                                    PrefillTimePredictor.committedDecisionGroupMs(
                                            selection.selectedPredictionMs().getAsDouble()));
            CapacityBoundary boundary;
            try {
                boundary =
                        queueScheduler.processSelection(
                                selection.items(),
                                reason,
                                evaluator,
                                prediction,
                                bypassedHead,
                                Delivery.PrefixPolicy.ORDERED_PREFIX,
                                queueScheduler);
            } catch (InvalidPrefillPredictionException failure) {
                first = selection.items().get(failure.memberIndex());
                boundary = CapacityBoundary.failed(failure);
            }
            long retryVersion =
                    boundary != null
                                    && boundary.resource()
                                            == CapacityBoundary.Resource.DECODE_ENGINE
                            ? snapshot.queueVersion()
                            : 0L;
            return queueScheduler.commitBoundary(first, boundary, retryVersion);
        } catch (InvalidPrefillPredictionException failure) {
            return queueScheduler.commitBoundary(
                    candidates.get(failure.memberIndex()),
                    CapacityBoundary.failed(failure),
                    snapshot.queueVersion());
        }
    }

    private static final int MIN_PLANNER_THREADS = 1;

    private final RequestRouteSelector router;
    private final DecodePreemption decodePreemption;
    private final PlacementAvailability availability;
    private final QueueExecutionSettings queueSettings;
    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final ReentrantLock lock = new ReentrantLock();
    private final java.util.concurrent.locks.Condition schedulingStopped = lock.newCondition();
    private final PlacementQueue orderedQueue;
    private final PlacementCapacityWaitQueue waitingRequests;

    /**
     * Entries exist only while the scheduler owns global placement.
     */
    private final Map<RequestContext, GlobalQueueEntry> queuedEntries = new IdentityHashMap<>();

    private final Set<WorkerEndpoint> unavailableEndpoints =
            Collections.newSetFromMap(new IdentityHashMap<>());

    private final Set<PlacementPlan> pendingPreemptions =
            Collections.newSetFromMap(new IdentityHashMap<>());

    /**
     * Waiting on local preemption slots, independent of physical Worker capacity edges.
     */
    private final Set<GlobalQueueEntry> preemptionQuotaWaiters =
            Collections.newSetFromMap(new IdentityHashMap<>());

    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ArrayDeque<PlacementPlan> completedPlans = new ArrayDeque<>();

    private final ExecutorService planners;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            new PlacementAvailability.Listener() {
                @Override
                public void onAvailabilityChanged(PlacementKey key) {
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

    public QueuedRequestScheduler(
            FlexlbConfig config,
            RequestRouteSelector router,
            DecodePreemption decodePreemption,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            PlacementAvailability availability,
            PlacementQueue queue,
            GroupPlanner grouping) {
        super(
                requests,
                tasks,
                deadlines,
                dispatcher,
                responseCompletions,
                deliveryReporter,
                cacheTrace);
        this.queueSettings = QueueExecutionSettings.capture(config, grouping);
        this.router = Objects.requireNonNull(router, "router");
        this.decodePreemption = Objects.requireNonNull(decodePreemption, "decodePreemption");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.orderedQueue = Objects.requireNonNull(queue, "queue");
        this.waitingRequests = new PlacementCapacityWaitQueue(queue.order(), availability);
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
        availability.addListener(availabilityListener);
    }

    private boolean started;
    private Thread schedulingThread;
    private volatile long nextCheck = Long.MAX_VALUE;
    private final java.util.concurrent.atomic.AtomicLong notifications =
            new java.util.concurrent.atomic.AtomicLong();
    private final Thread schedulerThread =
            Thread.ofPlatform().daemon().name("flexlb-scheduling").unstarted(this::run);

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

    private void requestRun() {
        notifications.incrementAndGet();
        java.util.concurrent.locks.LockSupport.unpark(schedulerThread);
    }

    private void run() {
        try {
            while (!closed.get()) {
                long observed = notifications.get();
                schedule();
                if (closed.get()) break;
                // A blocking operation inside schedule() may consume an unpark permit.
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
            // The driver may close its planner pool without waiting for its own thread.
            Throwable cleanup = Failures.run(null, this::close);
            if (cleanup != null) tasks.recordFailure(cleanup);
        }
    }

    @Override
    public void schedule() {
        lock.lock();
        try {
            if (closed.get()) return;
            if (schedulingThread != null) {
                if (Thread.currentThread() != schedulerThread) requestRun();
                return;
            }
            schedulingThread = Thread.currentThread();
        } finally {
            lock.unlock();
        }
        try {
            nextCheck = scheduleAt(System.nanoTime());
        } catch (RuntimeException | Error failure) {
            tasks.recordFailure(failure);
            stop();
            throw failure;
        } finally {
            lock.lock();
            try {
                schedulingThread = null;
                schedulingStopped.signalAll();
                if (Thread.currentThread() != schedulerThread) requestRun();
            } finally {
                lock.unlock();
            }
        }
    }

    @Override
    public void stop() {
        closed.set(true);
        requestRun();
        availability.removeListener(availabilityListener);
        planners.shutdown();
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (Boolean.TRUE.equals(planningThread.get())) return false;
        lock.lock();
        try {
            if (schedulerThread == Thread.currentThread()
                    || schedulingThread == Thread.currentThread()) return false;
        } finally {
            lock.unlock();
        }
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
        interrupted = Thread.interrupted();
        lock.lock();
        try {
            if (schedulingThread == Thread.currentThread()) return false;
            while (schedulingThread != null) {
                long remaining = remainingNanos(deadlineNanos);
                if (remaining <= 0) return false;
                try {
                    schedulingStopped.awaitNanos(remaining);
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        } finally {
            lock.unlock();
            if (interrupted) Thread.currentThread().interrupt();
        }
        return awaitPlanners(deadlineNanos);
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
    public boolean shouldRetryRouteFailure(ContextResult.RouteFailure failure) {
        return !closed.get()
                && failure.reason() == ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE;
    }

    @Override
    public void onRequest(RequestContext context) {
        if (!checkRequestMode(context, true)) return;
        if (!enqueue(context)) shutdownRequest(context);
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
                context.data().setVictimCount(0);
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

    private long scheduleAt(long nowNanos) {
        if (closed.get()) return Long.MAX_VALUE;
        withdrawUnavailableRoutes();
        long queueSchedulingCheck = schedulePrefillQueues(nowNanos);
        PlacementPlan completed;
        List<GlobalQueueEntry> candidates;
        lock.lock();
        try {
            completed = completedPlans.pollFirst();
        } finally {
            lock.unlock();
        }
        if (completed != null) {
            processCompletedPlan(completed);
        }
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
                    !completedPlans.isEmpty()
                            || (!closed.get()
                                    && inFlight.size() < plannerCount
                                    && (orderedQueue.hasUnscannedRequests()
                                            || waitingRequests.hasReadyRequests()));
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
            endpoints = Set.copyOf(unavailableEndpoints);
            unavailableEndpoints.clear();
        } finally {
            lock.unlock();
        }
        for (var request : requests.snapshotActive()) {
            var route = request.snapshot().route();
            if (route != null && endpoints.stream().anyMatch(route::usesEndpoint))
                onRouteFailure(
                        route,
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        "Selected Endpoint no longer accepts requests");
        }
    }

    @Override
    public void onEndpointChanged(WorkerEndpoint endpoint) {
        var status = Objects.requireNonNull(endpoint, "endpoint").getStatus();
        PrefillQueueScheduler newQueueScheduler = null;
        lock.lock();
        try {
            if (closed.get()) return;
            if (!endpoint.acceptsNewRequests()) unavailableEndpoints.add(endpoint);
            if (endpoint instanceof PrefillEndpoint prefill) {
                var queueScheduler = prefillQueueSchedulers.get(prefill);
                if (queueScheduler == null && prefill.queueEnabled() && !prefill.queueStopped()) {
                    queueScheduler =
                            new PrefillQueueScheduler(
                                    prefill.ipPort(), prefill, queueSettings, dispatcher);
                    prefillQueueSchedulers.put(prefill, queueScheduler);
                    newQueueScheduler = queueScheduler;
                }
                if (queueScheduler != null && !queueScheduler.ready) {
                    queueScheduler.ready = true;
                    if (!queueScheduler.running) {
                        timedPrefillQueues.remove(queueScheduler);
                        readyPrefillQueues.addLast(queueScheduler);
                    }
                }
            }
            waitingRequests.capacityChanged(
                    PlacementKey.exact(
                            status.getRole(),
                            status.topologySnapshot().group(),
                            endpoint.ipPort()));
        } finally {
            lock.unlock();
        }
        // Never wait for an Endpoint lock while holding the Scheduler lock.
        if (newQueueScheduler != null)
            newQueueScheduler.prefillEndpoint.observeQueueWait(
                    newQueueScheduler::admissionBlock,
                    newQueueScheduler::getLatestQueueWaitSnapshot);
        requestRun();
    }

    /**
     * Schedules one Prefill queue, including delivery attempts, capacity waits and retries.
     */
    public static final class PrefillQueueScheduler implements java.util.function.BooleanSupplier {

        /**
         * Exact predicate captured by one scheduling cycle.
         */
        public record Decision(
                RouteExecution request,
                CapacityBoundary unavailable,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs,
                String waitReason) {

            public static final Decision NO_ACTION = new Decision(null, null, 0L, 0L, 0L, null);

            public static Decision capacityBlocked(
                    RouteExecution item, CapacityBoundary unavailable, long queueVersion) {
                return new Decision(
                        Objects.requireNonNull(item),
                        Objects.requireNonNull(unavailable),
                        queueVersion,
                        0L,
                        Long.MAX_VALUE,
                        unavailable.resource().getDescription());
            }

            public static Decision awaitingSchedulingChange(
                    RouteExecution head,
                    long queueVersion,
                    long schedulingInputVersion,
                    long wakeAtMs,
                    String waitReason) {
                return new Decision(
                        Objects.requireNonNull(head),
                        null,
                        queueVersion,
                        schedulingInputVersion,
                        wakeAtMs,
                        waitReason);
            }
        }

        private static final Map<String, Object> INITIAL_QUEUE_WAIT_SNAPSHOT =
                Map.of("cause", "waiting for Prefill decision");

        private static final RequestTimeEstimation.AdmissionBlockSemantics BATCH_DELIVERY_BLOCK =
                new RequestTimeEstimation.AdmissionBlockSemantics(
                        "DELIVERY_CAPACITY_BATCH_ADMISSION",
                        RequestTimeEstimation.State.BLOCKED,
                        "DELIVERY_CAPACITY_BATCH_ADMISSION");
        private static final RequestTimeEstimation.AdmissionBlockSemantics DECODE_DELIVERY_BLOCK =
                new RequestTimeEstimation.AdmissionBlockSemantics(
                        "DELIVERY_CAPACITY_DECODE_ENGINE",
                        RequestTimeEstimation.State.UNAVAILABLE,
                        "DECODE_CAPACITY_SCOPE_UNKNOWN");

        private final String key;
        private final PrefillEndpoint prefillEndpoint;
        private final QueueExecutionSettings settings;
        private final DeliveryDispatcher dispatcher;
        // Protects this scheduler's waiting decision, listener replacement and stop ownership.
        private final ReentrantLock waitLock = new ReentrantLock();
        private Runnable removeCapacityListener;

        /**
         * Endpoint owns queue membership; this scheduler owns waiting decisions and subscriptions.
         */
        public PrefillEndpoint prefillEndpoint() {
            return prefillEndpoint;
        }

        public PrefillEndpoint.QueueSnapshot captureQueue(int limit) {
            return prefillEndpoint.captureQueue(limit);
        }

        public QueueExecutionSettings settings() {
            return settings;
        }

        private final CompletableFuture<Throwable> stopped = new CompletableFuture<>();
        private volatile Thread stopOwner;
        public final Runnable capacityAvailableSignal;

        /**
         * Last decision; capacity subscriptions and scheduling retry use the same evidence.
         */
        private volatile Decision waiting;

        /**
         * Last waiting decision; timeout readers use the published immutable snapshot.
         */
        private volatile QueueWaitSnapshot latestQueueWaitSnapshot;

        public PrefillQueueScheduler(
                String key,
                PrefillEndpoint prefillEp,
                QueueExecutionSettings settings,
                DeliveryDispatcher dispatcher) {
            this.key = key;
            this.prefillEndpoint = Objects.requireNonNull(prefillEp, "prefillEndpoint");
            this.settings = Objects.requireNonNull(settings);
            this.capacityAvailableSignal =
                    () -> {
                        signal(true);
                        prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
                    };
            this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        }

        public boolean stopping() {
            return stopOwner != null || prefillEndpoint.queueStopped();
        }

        /**
         * Read the published decision without taking the scheduler wait lock.
         * Projection may call this while the Endpoint holds its own state lock.
         */
        public AdmissionBlock admissionBlock() {
            Decision blocked = waiting;
            return stopping()
                            || blocked == null
                            || blocked.unavailable() == null
                            || !prefillEndpoint.isQueueWaitCurrent(
                                    blocked.request(), 0L, 0L, true, System.currentTimeMillis())
                            || capacityAvailable(blocked.request(), blocked.unavailable())
                    ? null
                    : new AdmissionBlock(
                            blocked.request().requestId(),
                            blocked.request().enqueueSeq(),
                            projectionSemantics(blocked.unavailable().resource()));
        }

        private static RequestTimeEstimation.AdmissionBlockSemantics projectionSemantics(
                CapacityBoundary.Resource resource) {
            return switch (resource) {
                case DISPATCH_EXECUTOR -> BATCH_DELIVERY_BLOCK;
                case DECODE_ENGINE -> DECODE_DELIVERY_BLOCK;
                case PREFILL_BATCH, PREFILL_REQUEST -> null;
            };
        }

        public Throwable stopAndAwait() {
            return stopAndDrain(
                    new CancellationException("FlexLB worker scheduling stopped: " + key));
        }

        Throwable stopAndDrain(Throwable cause) {
            Thread caller = Thread.currentThread();
            boolean first;
            waitLock.lock();
            try {
                boolean startsDrain = stopOwner == null;
                checkState(
                        startsDrain || stopped.isDone() || stopOwner != caller,
                        "cannot wait for own queue drain");
                if (startsDrain) stopOwner = caller;
                first = startsDrain;
            } finally {
                waitLock.unlock();
            }
            if (!first) return stopped.join();
            Throwable failure = null;
            try {
                failure = Failures.run(failure, () -> observeWait(null));
                List<RouteExecution> removed = prefillEndpoint.closeQueue();
                for (var route : removed) {
                    failure =
                            Failures.run(
                                    failure,
                                    () -> prefillEndpoint.signal(PrefillEndpoint.Signal.CAPACITY));
                    failure =
                            Failures.run(
                                    failure,
                                    () ->
                                            route.ctx()
                                                    .data()
                                                    .getRequestScheduler()
                                                    .rejectRoute(route, cause));
                }
            } catch (Throwable problem) {
                failure = Failures.append(failure, problem);
            } finally {
                stopped.complete(failure);
            }
            return failure;
        }

        /**
         * Read the last captured queue wait state without waiting for an Endpoint lock.
         */
        public Map<String, Object> getLatestQueueWaitSnapshot() {
            QueueWaitSnapshot snapshot = latestQueueWaitSnapshot;
            return snapshot == null ? INITIAL_QUEUE_WAIT_SNAPSHOT : snapshot.asMap();
        }

        /**
         * Capture fixed-size counters without formatting the PV record on the scheduling loop.
         */
        public void recordQueueWait(RouteExecution head, String reason) {
            EndpointStats.QueueCounts counters = prefillEndpoint.queueCounters();
            DecodeRoutingView decode =
                    head.decodeEndpoint() == null ? null : head.decodeEndpoint().routingView();
            latestQueueWaitSnapshot = new QueueWaitSnapshot(key, reason, counters, decode);
        }

        // ==================== Queue ownership and projection ====================

        public GroupPlanner.Constraints schedulingConstraints() {
            WorkerStatus status = prefillEndpoint.getStatus();
            var engine = status == null ? null : status.committedEngineObservation();
            return settings.constraints(engine);
        }

        // ==================== Delivery ownership ====================

        public CapacityBoundary processSelection(
                List<RouteExecution> candidates,
                String reason,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong prediction,
                RouteExecution bypassedHead,
                Delivery.PrefixPolicy prefixPolicy,
                java.util.function.BooleanSupplier selectionStillAllowed) {
            Delivery delivery =
                    new Delivery(
                            prefillEndpoint,
                            settings.dispatcherType() == DispatcherConfig.Type.BATCH,
                            candidates.size(),
                            evaluator,
                            reason);
            return dispatcher.submit(
                    delivery,
                    candidates,
                    prediction,
                    bypassedHead,
                    prefixPolicy,
                    () -> !stopping() && selectionStillAllowed.getAsBoolean());
        }

        boolean capacityAvailable(RouteExecution item, CapacityBoundary boundary) {
            return dispatcher.capacityAvailable(item, boundary);
        }

        Runnable watchCapacity(RouteExecution item, CapacityBoundary boundary, Runnable wake) {
            return dispatcher.watchCapacity(item, boundary, wake);
        }

        public Decision commitBoundary(
                RouteExecution blockedItem, CapacityBoundary blockedResult, long queueVersion) {
            if (blockedItem != null && !blockedItem.selectedEndpointsAvailable()) {
                blockedItem
                        .ctx()
                        .data()
                        .getRequestScheduler()
                        .onRouteFailure(
                                blockedItem,
                                ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                                "Selected Endpoint unavailable during delivery preparation");
                return Decision.NO_ACTION;
            }
            if (blockedItem == null || blockedResult == null || stopping())
                return Decision.NO_ACTION;
            if (blockedResult.unavailable()) {
                return prefillEndpoint.isQueueWaitCurrent(
                                blockedItem, 0L, 0L, true, System.currentTimeMillis())
                        ? Decision.capacityBlocked(blockedItem, blockedResult, queueVersion)
                        : Decision.NO_ACTION;
            }
            if (blockedResult.status() != CapacityBoundary.Status.FAILED
                    || !prefillEndpoint.removeFailedQueuedRequest(blockedItem))
                return Decision.NO_ACTION;
            removeAndNotify(blockedItem, blockedResult, true);
            return Decision.NO_ACTION;
        }

        /**
         * Once removed, the request must be notified even if a capacity listener fails.
         */
        void removeAndNotify(
                RouteExecution item, CapacityBoundary failure, boolean alreadyRemoved) {
            if (!alreadyRemoved && !prefillEndpoint.removeQueuedMember(item)) return;
            Throwable problem = Failures.run(null, capacityAvailableSignal);
            try {
                if (failure == null) {
                    item.ctx()
                            .data()
                            .getRequestScheduler()
                            .cancel(
                                    item.ctx(),
                                    0L,
                                    org.flexlb.balance.scheduler.CancelReason.DEADLINE_EXCEEDED);
                } else {
                    AbstractRequestScheduler.preparationFailed(
                            item, String.valueOf(failure.cause().getMessage()));
                }
            } catch (Throwable callbackFailure) {
                item.ctx().data().getRequestScheduler().tasks.recordFailure(callbackFailure);
                problem =
                        Failures.run(
                                problem,
                                () ->
                                        Logger.error(
                                                "PrefillQueueScheduler[{}] removed request callback failed request_id={}",
                                                key,
                                                item.requestId(),
                                                callbackFailure));
            }
            Failures.rethrow(problem, "queue removal notification failed");
        }

        /**
         * Subscribe before checking the resource so capacity changes cannot be missed.
         */
        public void observeWait(Decision decision) {
            boolean invalidateProjection = false;
            waitLock.lock();
            try {
                Decision previous = waiting;
                waiting = stopping() ? null : decision;
                if (previous != waiting
                        && ((previous != null && previous.unavailable() != null)
                                || (waiting != null && waiting.unavailable() != null))) {
                    invalidateProjection = true;
                    updateCapacityListener(
                            waiting == null || waiting.unavailable() == null ? null : waiting);
                }
            } finally {
                waitLock.unlock();
                if (invalidateProjection) prefillEndpoint.schedulingInputsChanged();
            }
        }

        /**
         * Whether the observed queue/resource facts still hold; the scheduler chooses when to retry.
         */
        public boolean waitCurrent(Decision decision, long nowMs) {
            boolean capacity = decision != null && decision.unavailable() != null;
            return !stopping()
                    && prefillEndpoint.isQueueWaitCurrent(
                            decision == null ? null : decision.request(),
                            decision == null ? 0L : decision.queueVersion(),
                            decision == null ? 0L : decision.schedulingInputVersion(),
                            capacity,
                            nowMs)
                    && (!capacity
                            || !capacityAvailable(decision.request(), decision.unavailable()));
        }

        /**
         * Replace the capacity subscription while holding only the scheduler wait lock.
         */
        private void updateCapacityListener(Decision blocked) {
            checkState(
                    waitLock.isHeldByCurrentThread(),
                    "capacity subscriptions require the scheduler wait lock");
            Runnable previous = removeCapacityListener;
            removeCapacityListener = null;
            try {
                if (previous != null) {
                    previous.run();
                }
            } finally {
                if (blocked != null) {
                    removeCapacityListener =
                            watchCapacity(
                                    blocked.request(),
                                    blocked.unavailable(),
                                    capacityAvailableSignal);
                }
            }
        }

        /**
         * Wake decisions whose advisory worker-status or predictor input changed.
         */
        public void signal(boolean capacityOnly) {
            Decision blocked = waiting;
            if (!capacityOnly || (blocked != null && blocked.unavailable() != null)) {
                prefillEndpoint.schedulingInputsChanged();
            }
        }

        Decision blockedHead;
        RouteExecution bypass;
        long nextCheck = Long.MAX_VALUE;
        boolean ready, running;

        @Override
        public boolean getAsBoolean() {
            return bypass == null || !capacityAvailable(bypass, blockedHead.unavailable());
        }
    }

    private final Map<PrefillEndpoint, PrefillQueueScheduler> prefillQueueSchedulers =
            new IdentityHashMap<>();
    private final ArrayDeque<PrefillQueueScheduler> readyPrefillQueues = new ArrayDeque<>();
    private final java.util.PriorityQueue<PrefillQueueScheduler> timedPrefillQueues =
            new java.util.PriorityQueue<>(
                    java.util.Comparator.comparingLong(queueScheduler -> queueScheduler.nextCheck));
    private int queueSchedulingInFlight;

    /**
     * One task owns decision, wait subscription and next check; no completed-result queue.
     */
    private void runQueueScheduling(PrefillQueueScheduler queueScheduler) {

        long next = Long.MAX_VALUE;
        Throwable failure = null;
        try {
            if (queueScheduler.stopping()) queueScheduler.observeWait(null);
            if (!closed.get() && !queueScheduler.stopping()) {
                long nowNanos = System.nanoTime();
                long nowMs = System.currentTimeMillis();
                if (queueScheduler.waiting == null
                        || !queueScheduler.waitCurrent(queueScheduler.waiting, nowMs)
                        || queueScheduler.nextCheck <= nowNanos) {
                    queueScheduler.observeWait(null);
                    var decision = decideQueueScheduling(queueScheduler);
                    if (decision.waitReason() != null)
                        queueScheduler.recordQueueWait(decision.request(), decision.waitReason());
                    queueScheduler.observeWait(decision);
                }
                // Earlier notifications are covered by the checks below; later ones remain pending.
                lock.lock();
                try {
                    queueScheduler.ready = false;
                } finally {
                    lock.unlock();
                }
                // Subscribe before checking again. Events racing this check are retained by ready.
                nowNanos = System.nanoTime();
                nowMs = System.currentTimeMillis();
                var wait = queueScheduler.waiting;
                if (!queueScheduler.waitCurrent(
                        wait == null || wait.request() == null ? null : wait, nowMs)) {
                    next = nowNanos;
                } else if (wait != null && wait.request() != null) {
                    long at = Math.min(wait.wakeAtMs(), wait.request().expiresAtMs());
                    if (at != Long.MAX_VALUE)
                        next = nowNanos + TimeUnit.MILLISECONDS.toNanos(Math.max(0L, at - nowMs));
                }
            }
        } catch (Throwable problem) {
            failure = problem;
        } finally {
            finishQueueScheduling(queueScheduler, next, failure);
        }
    }

    private void finishQueueScheduling(
            PrefillQueueScheduler queueScheduler, long next, Throwable failure) {
        try {
            if (failure != null) {
                tasks.recordFailure(failure);
                tasks.recordFailure(queueScheduler.stopAndDrain(failure));
            }
        } finally {
            lock.lock();
            try {
                queueScheduler.running = false;
                queueSchedulingInFlight--;
                if (queueScheduler.stopping()) {
                    prefillQueueSchedulers.remove(queueScheduler.prefillEndpoint());
                } else if (!closed.get()) {
                    queueScheduler.nextCheck = next;
                    if (queueScheduler.ready || next <= System.nanoTime()) {
                        queueScheduler.ready = true;
                        readyPrefillQueues.addLast(queueScheduler);
                    } else if (next != Long.MAX_VALUE) {
                        timedPrefillQueues.add(queueScheduler);
                    }
                }
            } finally {
                lock.unlock();
                requestRun();
            }
        }
    }

    private long schedulePrefillQueues(long nowNanos) {
        PrefillQueueScheduler queueScheduler;
        lock.lock();
        try {
            while (!timedPrefillQueues.isEmpty()
                    && timedPrefillQueues.peek().nextCheck <= nowNanos) {
                var dueQueueScheduler = timedPrefillQueues.remove();
                dueQueueScheduler.ready = true;
                readyPrefillQueues.addLast(dueQueueScheduler);
            }
            queueScheduler =
                    queueSchedulingInFlight < plannerCount ? readyPrefillQueues.pollFirst() : null;
            if (queueScheduler != null) {
                // Claim before releasing the lock: later notifications belong to this task.
                queueScheduler.ready = false;
                queueScheduler.running = true;
                queueSchedulingInFlight++;
            }
        } finally {
            lock.unlock();
        }
        if (queueScheduler != null) {
            try {
                planners.execute(() -> runQueueScheduling(queueScheduler));
            } catch (java.util.concurrent.RejectedExecutionException rejected) {
                // Stop may close the pool after this cycle claims its slot. No task was submitted.
                finishQueueScheduling(
                        queueScheduler,
                        Long.MAX_VALUE,
                        closed.get() && planners.isShutdown() ? null : rejected);
            } catch (Throwable failure) {
                finishQueueScheduling(queueScheduler, Long.MAX_VALUE, failure);
            }
        }
        lock.lock();
        try {
            return !readyPrefillQueues.isEmpty() && queueSchedulingInFlight < plannerCount
                    ? nowNanos
                    : timedPrefillQueues.isEmpty()
                            ? Long.MAX_VALUE
                            : timedPrefillQueues.peek().nextCheck;
        } finally {
            lock.unlock();
        }
    }

    private void submitPlan(GlobalQueueEntry entry) {
        try {
            planners.execute(
                    () -> {
                        PlacementPlan plan = new PlacementPlan(entry, availability.sequence());
                        try {
                            if (!entry.removed && !entry.context.data().getFuture().isDone()) {
                                if (entry.context
                                        .data()
                                        .getConfig()
                                        .getDispatcher()
                                        .requiresGenerateInput()) {
                                    entry.context.data().prepareGenerateInput();
                                }
                                if (entry.context.data().getRequestExpiresAtMs() <= 0L
                                        || System.currentTimeMillis()
                                                >= entry.context.data().getRequestExpiresAtMs()) {
                                    cancel(entry.context, 0L, CancelReason.DEADLINE_EXCEEDED);
                                } else {
                                    plan.handle = claimPlacement(entry.context);
                                    if (plan.handle != null) {
                                        plan.result =
                                                prepareSelectedRoute(
                                                        entry.context,
                                                        Objects.requireNonNull(
                                                                router.select(
                                                                        entry.context,
                                                                        entry.routingGroup),
                                                                "placement result"));
                                        plan.handle
                                                .owner()
                                                .recordPlacementDiagnostics(
                                                        plan.handle, plan.result.diagnostics());
                                        plan.waitKey = plan.result.blocker();
                                    }
                                }
                            }
                        } catch (Throwable failure) {
                            plan.planningFailure = failure;
                        } finally {
                            publishPlan(plan);
                        }
                    });
        } catch (Throwable failure) {
            var plan = new PlacementPlan(entry, availability.sequence());
            plan.planningFailure = failure;
            publishPlan(plan);
        }
    }

    private void publishPlan(PlacementPlan plan) {
        boolean closing;
        lock.lock();
        try {
            closing = closed.get();
            if (!closing) completedPlans.addLast(plan);
        } finally {
            lock.unlock();
        }
        if (closing) processCompletedPlan(plan);
        else requestRun();
    }

    private void processCompletedPlan(PlacementPlan plan) {
        boolean retry = false;
        boolean completingPreemption = plan.preemptionCompleted;
        Map<String, Object> diagnostics = plan.result.diagnostics();
        try {
            boolean abandoned =
                    closed.get()
                            || completingPreemption
                                    && (plan.entry.removed
                                            || plan.entry.context.data().getFuture().isDone());
            Outcome outcome = Outcome.DONE;
            Throwable commitFailure = null;
            try {
                if (!abandoned) {
                    Failures.rethrow(plan.planningFailure, "placement planning failed");
                    outcome = commit(plan);
                }
            } catch (Throwable failure) {
                commitFailure = failure;
            } finally {
                if (abandoned || completingPreemption || !pendingPreemptions.contains(plan)) {
                    commitFailure = Failures.run(commitFailure, plan::close);
                }
            }
            Failures.rethrow(commitFailure, "placement commit failed");
            retry =
                    outcome == Outcome.REPLAN
                            || (outcome == Outcome.BLOCKED && !park(plan, diagnostics));
        } catch (Throwable failure) {
            if (closed.get()) tasks.recordFailure(failure);
            removeRequest(plan.entry);
            completeDecisionResponse(
                    plan.entry,
                    buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Placement failed: " + failure.getMessage()));
            Logger.error(
                    "Global queue commit failed: request_id={}",
                    plan.entry.context.data().getRequestId(),
                    failure);
        } finally {
            // PlacementPlan ownership has settled before this slot becomes eligible again.
            lock.lock();
            try {
                inFlight.remove(plan.entry);
                boolean closing = closed.get();
                if (closing || completingPreemption) pendingPreemptions.remove(plan);
                if (completingPreemption && !closing) {
                    for (GlobalQueueEntry waiter : preemptionQuotaWaiters) {
                        waitingRequests.remove(waiter);
                        if (!waiter.removed && !waiter.context.data().getFuture().isDone()) {
                            orderedQueue.markRequestReadyForRetry(waiter);
                        }
                    }
                    preemptionQuotaWaiters.clear();
                }
                if (!closing
                        && retry
                        && (!plan.entry.removed
                                && !plan.entry.context.data().getFuture().isDone())) {
                    orderedQueue.markRequestReadyForRetry(plan.entry);
                }
                requestRun();
            } finally {
                lock.unlock();
            }
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
                            && pendingPreemptions.stream().noneMatch(plan -> plan.entry == entry)
                            && !waitingRequests.isWaiting(entry);
                });
    }

    private Outcome commit(PlacementPlan plan) {
        GlobalQueueEntry entry = plan.entry;
        boolean reclaimed = plan.preemptionCompleted;
        String rejection = null;
        try {
            if (reclaimed) {
                if (plan.preemptionFailure != null || plan.preemptionResult == null) {
                    rejection = "Decode eviction control failed before commit";
                } else if (!plan.preemptionResult.committed()) {
                    rejection = plan.preemptionResult.detail();
                }
            }
            if (rejection == null
                    && (reclaimed
                            || (!entry.removed && !entry.context.data().getFuture().isDone()))) {
                PlacementResult<RouteExecution, PlacementKey> result = plan.result;
                switch (result.status()) {
                    case REJECTED -> {
                        if (plan.handle != null)
                            AbstractRequestScheduler.finishPlacement(plan.handle);
                        completeDecisionResponse(entry, result.failure());
                    }
                    case CLOSED -> {}
                    case ENDPOINT_CHANGED -> {
                        return Outcome.REPLAN;
                    }
                    case BLOCKED -> {
                        return Outcome.BLOCKED;
                    }
                    case SUCCESS -> {
                        RouteExecution routing = result.value();
                        PlacementResult<RouteExecution, PlacementKey> publication;
                        boolean waitingForCapacity = false;
                        try {
                            publication = publishPlacement(routing, plan.handle, queueSettings);
                            waitingForCapacity =
                                    publication.status() == PlacementResult.Status.BLOCKED;
                        } finally {
                            // publishPlacement finishes every result except a retained capacity
                            // wait.
                            if (!waitingForCapacity) plan.handle = null;
                        }
                        if (publication.status() == PlacementResult.Status.REJECTED) {
                            completeDecisionResponse(entry, publication.failure());
                        }
                        if (publication.status() == PlacementResult.Status.ENDPOINT_CHANGED) {
                            return Outcome.REPLAN;
                        } else if (reclaimed
                                && publication.status() != PlacementResult.Status.SUCCESS) {
                            rejection =
                                    "selected Prefill capacity changed before canonical placement";
                        } else if (publication.status() == PlacementResult.Status.BLOCKED) {
                            plan.waitKey = publication.blocker();
                            if (routing.encoderEndpoint() != null) return Outcome.BLOCKED;
                            WorkerEndpoint blocked = routing.blockedEndpointIfCurrent(plan.waitKey);
                            if (blocked == null) {
                                return Outcome.REPLAN;
                            }
                            if (!tryPriorityRescue(plan, blocked)) {
                                return Outcome.BLOCKED;
                            }
                            return Outcome.PENDING;
                        }
                    }
                }
            }
        } catch (RuntimeException | Error failure) {
            if (!reclaimed) {
                throw failure;
            }
            rejection = "Decode eviction placement failed: " + failure.getMessage();
        }
        if (rejection != null) {
            var response =
                    Response.error(
                            StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            rejection);
            if (plan.handle != null) AbstractRequestScheduler.failPlacement(plan.handle, response);
            else completeDecisionResponse(entry, response);
        }
        removeRequest(entry);
        return Outcome.DONE;
    }

    private boolean tryPriorityRescue(PlacementPlan plan, WorkerEndpoint blockedEndpoint) {
        GlobalQueueEntry entry = plan.entry;
        if (!queueSettings.priorityOrdering() || entry.context.data().getFuture().isDone()) {
            return false;
        }
        if (pendingPreemptions.size() >= plannerCount) {
            preemptionQuotaWaiters.add(entry);
            return false;
        }
        var preemptionResult = decodePreemption.tryReclaim(plan.context(), blockedEndpoint);
        if (preemptionResult == null) {
            return false;
        }
        pendingPreemptions.add(plan);
        plan.preemption =
                preemptionResult
                        .thenApply(result -> acceptPreemption(plan, blockedEndpoint, result))
                        .whenComplete(
                                (result, failure) -> {
                                    plan.preemptionResult = result;
                                    plan.preemptionFailure = failure;
                                    if (failure != null) {
                                        tasks.recordFailure(
                                                failure
                                                                        instanceof
                                                                        java.util.concurrent
                                                                                .CompletionException
                                                                && failure.getCause() != null
                                                        ? failure.getCause()
                                                        : failure);
                                    }
                                    plan.preemptionCompleted = true;
                                    publishPlan(plan);
                                });
        return true;
    }

    private DecodePreemption.PreemptionResult acceptPreemption(
            PlacementPlan plan,
            WorkerEndpoint blockedEndpoint,
            DecodePreemption.PreemptionResult result) {
        if (result == null || !result.committed()) {
            return result;
        }
        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)) {
            return result;
        }
        var reservation = result.reservation();
        if (plan.accept(decodeEndpoint, reservation)) {
            return result;
        }
        var status =
                decodeEndpoint
                        .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                        .status();
        checkState(
                status == DecodeResources.ReservationReleaseResult.RELEASED
                        || status == DecodeResources.ReservationReleaseResult.STALE,
                "unaccepted preemption retained Decode resources");
        return new DecodePreemption.PreemptionResult(null, false, "plan_no_longer_current");
    }

    @Override
    public int blockedRequestCount() {
        lock.lock();
        try {
            return waitingRequests.size() + preemptionQuotaWaiters.size();
        } finally {
            lock.unlock();
        }
    }

    private boolean park(PlacementPlan plan, Map<String, Object> diagnostics) {
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (plan.entry.removed || plan.entry.context.data().getFuture().isDone()) {
                return true;
            }
            parked = waitingRequests.park(plan.entry, plan.waitKey, plan.availabilitySequence);
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        plan.entry
                .context
                .data()
                .setSchedulingDiagnostics(
                        RequestSchedulerReporter.placementWait(
                                plan.waitKey.role().name(),
                                plan.waitKey.group(),
                                plan.waitKey.endpoint(),
                                depth,
                                counts,
                                diagnostics));
        return parked;
    }

    private void removeRequest(GlobalQueueEntry entry) {
        boolean acquire = !lock.isHeldByCurrentThread();
        if (acquire) {
            lock.lock();
        }
        try {
            orderedQueue.remove(entry);
            waitingRequests.remove(entry);
            preemptionQuotaWaiters.remove(entry);
            queuedEntries.remove(entry.context, entry);
            requestRun();
        } finally {
            if (acquire) {
                lock.unlock();
            }
        }
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
        if (context.data().getRequestScheduler() != this || !requests.isCurrent(context)) {
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

    @Override
    public void close() {
        Throwable failure = Failures.run(null, decodePreemption::stopAdmission);
        stop();
        drainOnClose();
        checkState(
                Thread.currentThread() == schedulerThread
                        ? awaitPlanners(Long.MAX_VALUE)
                        : awaitStopped(Long.MAX_VALUE),
                "scheduler cannot wait for its own scheduling tasks");
        PlacementPlan plan;
        while (true) {
            lock.lock();
            try {
                plan = completedPlans.pollFirst();
            } finally {
                lock.unlock();
            }
            if (plan == null) break;
            processCompletedPlan(plan);
        }
        List<PlacementPlan> preemptions;
        lock.lock();
        try {
            preemptions = List.copyOf(pendingPreemptions);
        } finally {
            lock.unlock();
        }
        for (var pending : preemptions) {
            try {
                pending.preemption.toCompletableFuture().join();
            } catch (java.util.concurrent.CompletionException ignored) {
                // The completion callback records failure and closes the exact plan before this
                // join.
            }
        }
        for (var queueScheduler : prefillQueueSchedulers.values()) queueScheduler.observeWait(null);
        prefillQueueSchedulers.clear();
        readyPrefillQueues.clear();
        timedPrefillQueues.clear();
        Failures.rethrow(failure, "scheduler failed to stop preemption");
    }

    private enum Outcome {
        DONE,
        BLOCKED,
        REPLAN,
        PENDING
    }

    public final class PlacementPlan implements AutoCloseable {
        private boolean closed;
        private final GlobalQueueEntry entry;
        private PlacementOperation handle;
        private Throwable planningFailure;
        private java.util.concurrent.CompletionStage<?> preemption;
        private volatile boolean preemptionCompleted;
        private DecodePreemption.PreemptionResult preemptionResult;
        private Throwable preemptionFailure;
        // The plan retains the route and its uncommitted Decode reservation.
        private PlacementResult<RouteExecution, PlacementKey> result;
        private final long availabilitySequence;
        // Keep the blocking Endpoint identity while the request waits for capacity.
        private PlacementKey waitKey;

        private PlacementPlan(GlobalQueueEntry entry, long availabilitySequence) {
            this.entry = entry;
            this.availabilitySequence = availabilitySequence;
            this.result = PlacementResult.closed();
        }

        public RequestContext context() {
            return entry.context;
        }

        /**
         * A false result leaves the reservation with the caller for rollback.
         */
        public synchronized boolean accept(
                DecodeEndpoint endpoint, DecodeResources.RequestReservation reservation) {
            var route = result == null ? null : result.value();
            var ctx = context();
            synchronized (ctx) {
                var state = ctx.snapshot();
                if (closed
                        || QueuedRequestScheduler.this.closed.get()
                        || route == null
                        || route.decodeEndpoint() != endpoint
                        || route.decodeReservation() != null
                        || !java.util.Objects.equals(reservation.requestId(), route.requestId())
                        || ctx.data().getFuture().isDone()
                        || state.phase() != Phase.OPEN
                        || state.cancellation() != null
                        || handle == null
                        || state.placement() == null
                        || state.placement().operation() != handle) {
                    return false;
                }
            }
            // markQueued publishes capacity notifications; never invoke them under Context.
            if (!endpoint.markQueued(reservation).status().applied()) {
                return false;
            }
            route.attachReservation(reservation);
            return true;
        }

        @Override
        public void close() {
            RouteExecution routing;
            synchronized (this) {
                if (closed) {
                    return;
                }
                closed = true;
                routing = result == null ? null : result.value();
                result = null;
            }
            var failure =
                    handle == null
                            ? null
                            : AbstractRequestScheduler.finishPlacement(routing, handle);
            if (failure != null) tasks.recordFailure(failure);
        }
    }
}

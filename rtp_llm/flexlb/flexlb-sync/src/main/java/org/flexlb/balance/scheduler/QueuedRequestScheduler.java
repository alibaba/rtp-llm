package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.RequestLifecycle.Publication;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
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
import static org.flexlb.balance.scheduler.ContextResult.PlacementToken;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/**
 * The single QUEUE admission owner for one FlexLB model.
 *
 * <p>This queue orders the start of placement attempts for the model. A request is
 * selected from the complete live candidate fleet before the endpoint runtime
 * receives it. A bounded set of in-flight requests controls how many
 * independent routes can be prepared together; request collection and grouping
 * are decided by this scheduler for each endpoint. The queue lock protects only
 * index operations; route projection and RPCs are never performed while it is
 * held.</p>
 *
 * <p>Ready requests receive planning slots in FIFO/priority order. A request
 * that cannot be admitted waits for a relevant capacity event; it does not fence
 * later requests from trying the same worker with different resource demands.
 * Planning is bounded and parallel; completed plans are published without waiting
 * for earlier planners. Arrivals and wakeups do not invalidate in-flight work.
 * This scheduler owns SINGLE/FIXED_WINDOW grouping.</p>
 */
public final class QueuedRequestScheduler extends AbstractRequestScheduler {

    public static void runPrefill(RequestExecution execution) {
        runPrefill(execution, QueuedRequestScheduler::schedulePrefill);
    }

    public static void runPrefill(
            RequestExecution execution,
            java.util.function.Function<RequestExecution, RequestExecution.Decision> decisions) {
        while (!execution.stopping() && !Thread.currentThread().isInterrupted()) {
            try {
                execution.awaitWork(null);
                if (execution.stopping()) {
                    return;
                }
                RequestExecution.Decision result = decisions.apply(execution);
                if (result.waitReason() != null) {
                    execution.recordQueueWait(result.request(), result.waitReason());
                }
                if (result.request() != null) {
                    execution.awaitWork(result);
                }
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    public static RequestExecution.Decision schedulePrefill(RequestExecution execution) {
        int maxRequests = execution.settings().maxRequests();
        PrefillState.QueueSnapshot snapshot = execution.captureQueue(maxRequests);
        RequestRoute head = snapshot.head();
        if (head == null) {
            return RequestExecution.Decision.NO_ACTION;
        }
        long nowMs = System.currentTimeMillis();
        var requestState = head.ctx().snapshot();
        if (requestState.placement() != null && !requestState.placement().published()) {
            return RequestExecution.Decision.awaitingSchedulingChange(
                    head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    Long.MAX_VALUE,
                    "Route commit in progress");
        }
        if (head.requestExpired(nowMs)) {
            execution.removeAndNotify(head, null, false);
            return RequestExecution.Decision.NO_ACTION;
        }
        GroupPlanner.Constraints constraints = execution.schedulingConstraints();
        if (Math.max(0L, head.seqLen()) > constraints.batchKvCapacity()) {
            return RequestExecution.Decision.awaitingSchedulingChange(
                    head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    head.expiresAtMs(),
                    "Prefill KV capacity exhausted");
        }
        for (int index = 1; index < Math.min(maxRequests, snapshot.items().size()); index++) {
            RequestRoute member = snapshot.items().get(index);
            if (member.requestExpired(nowMs)) {
                execution.removeAndNotify(member, null, false);
                return RequestExecution.Decision.NO_ACTION;
            }
        }
        PrefillTimePredictor.Evaluator evaluator =
                execution.prefillEndpoint().predictor.evaluator();
        // First try dispatching from the queue head in normal queue order.
        var result = decideGroup(execution, snapshot, null, evaluator, constraints);
        if (execution.settings().priorityOrdering()
                && result.unavailable() != null
                && result.unavailable().resource() == CapacityBoundary.Resource.DECODE_ENGINE) {
            // The head's Decode endpoint is full. Try one queue-permitted bypass batch on
            // other Decode endpoints, using fresh queue/capacity snapshots after the first attempt.
            // Dispatch happens inside decideGroup; keep the original result so the run loop
            // continues waiting for the blocked head rather than a bypass candidate.
            decideGroup(
                    execution,
                    execution.captureQueue(Math.max(2, maxRequests)),
                    head,
                    evaluator,
                    execution.schedulingConstraints());
        }
        return result;
    }

    private static RequestExecution.Decision decideGroup(
            RequestExecution execution,
            PrefillState.QueueSnapshot snapshot,
            RequestRoute bypassedHead,
            PrefillTimePredictor.Evaluator evaluator,
            GroupPlanner.Constraints constraints) {
        var candidates =
                bypassedHead == null
                        ? snapshot.items()
                        : bypassCandidates(execution, snapshot, bypassedHead);
        if (candidates.isEmpty()) {
            return RequestExecution.Decision.NO_ACTION;
        }
        RequestRoute first = candidates.getFirst();
        try {
            var predictor =
                    execution.settings().executionBudgetMs() > 0L
                            ? RequestTimeEstimation.newGroupPredictor(
                                    execution.settings().dispatcherType(), evaluator)
                            : null;
            long planningAtMs = System.currentTimeMillis();
            var selection = execution.grouping().select(candidates, constraints, predictor);
            if (selection.items().isEmpty()) {
                return RequestExecution.Decision.NO_ACTION;
            }
            String reason =
                    bypassedHead == null
                            ? execution
                                    .grouping()
                                    .dispatchReason(selection, constraints, planningAtMs)
                            : "decode_head_bypass";
            if (reason == null) {
                return RequestExecution.Decision.awaitingSchedulingChange(
                        first,
                        snapshot.queueVersion(),
                        snapshot.schedulingInputVersion(),
                        Math.min(
                                execution.grouping().nextCheckAtMs(selection, constraints),
                                first.expiresAtMs()),
                        "Prefill collection window");
            }
            // Recheck advisory capacity after prediction, before acquiring hard-capacity leases.
            constraints = execution.schedulingConstraints();
            if ((selection.items().size() > 1
                            && !selection.fitsCompute(constraints.batchTokenCapacity()))
                    || !selection.fitsKv(constraints.batchKvCapacity())) {
                return RequestExecution.Decision.NO_ACTION;
            }
            OptionalLong prediction =
                    execution.settings().dispatcherType() != DispatcherConfig.Type.BATCH
                                    || selection.selectedPredictionMs().isEmpty()
                            ? OptionalLong.empty()
                            : OptionalLong.of(
                                    PrefillPredictionBoundary.committedDecisionGroupMs(
                                            selection.selectedPredictionMs().getAsDouble()));
            var boundary =
                    execution.processSelection(
                            selection.items(), reason, evaluator, prediction, bypassedHead);
            long retryOnQueueChange =
                    execution.settings().priorityOrdering()
                                    && boundary != null
                                    && boundary.resource()
                                            == CapacityBoundary.Resource.DECODE_ENGINE
                            ? snapshot.queueVersion()
                            : 0L;
            return execution.commitBoundary(first, boundary, retryOnQueueChange);
        } catch (InvalidPrefillPredictionException failure) {
            return execution.commitBoundary(
                    candidates.get(failure.memberIndex()),
                    CapacityBoundary.failed(failure),
                    snapshot.queueVersion());
        }
    }

    public static boolean canBypass(RequestExecution execution,
            RequestRoute head, RequestRoute candidate) {
        return execution.settings().priorityOrdering()
                && head.priority() == candidate.priority()
                && execution.queueAllowsBypass(head, candidate);
    }

    /** Only the immediate blocked head may be skipped; a second blocked member stops this batch. */
    private static List<RequestRoute> bypassCandidates(
            RequestExecution execution, PrefillState.QueueSnapshot snapshot, RequestRoute head) {
        if (snapshot.head() != head) {
            return List.of();
        }
        int end = 1;
        while (end < snapshot.items().size()
                && canBypass(execution, head, snapshot.items().get(end))) {
            end++;
        }
        return snapshot.items().subList(1, end);
    }

    private static final int MIN_PLANNER_THREADS = 1;

    private final RequestWorkerSelector router;
    private final DeliveryMetricsReporter reporter;
    private final DecodePreemption decodePreemption;
    private final PlacementAvailability availability;
    private final QueueExecutionSettings queueSettings;
    private final RequestExecution.Dependencies queueDependencies;
    private final int plannerCount;
    private final double scanBudgetMultiplier;
    private final ReentrantLock lock = new ReentrantLock();
    // Decision work and graceful drain wait for different predicates on this condition.
    private final PlacementQueue orderedQueue;
    private final PlacementCapacityWaitQueue waitingRequests;

    /** Entries exist only while the scheduler owns global placement. */
    private final Map<RequestContext, GlobalQueueEntry> queuedEntries = new IdentityHashMap<>();

    private final ArrayDeque<QueueEvent> events = new ArrayDeque<>();
    private final Set<PlacementPlan> pendingPreemptions =
            Collections.newSetFromMap(new IdentityHashMap<>());

    /** Waiting on local preemption slots, independent of physical Worker capacity edges. */
    private final Set<GlobalQueueEntry> preemptionQuotaWaiters =
            Collections.newSetFromMap(new IdentityHashMap<>());

    // Protected by lock. A slot stays occupied until its result has been handled,
    // including when the request is cancelled while planning.
    private final Set<GlobalQueueEntry> inFlight =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private final ArrayDeque<PlacementPlan> completedPlans = new ArrayDeque<>();

    /** Published by the decision thread; timeout readers never acquire the queue lock. */
    private volatile Map<String, Object> latestQueueWaitSnapshot =
            Map.of("cause", "waiting for placement");

    private final ExecutorService planners;
    private final SchedulerLoop schedulingLoop;
    private boolean closeDrained;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            new PlacementAvailability.Listener() {
                @Override
                public void onAvailabilityChanged(PlacementKey key) {
                    postEvent(new QueueEvent(EventKind.CAPACITY, null, null, key));
                }

                @Override
                public void onEndpointChanged(WorkerEndpoint endpoint) {
                    QueuedRequestScheduler.this.onEndpointChanged(endpoint);
                }
            };

    public QueuedRequestScheduler(
            FlexlbConfig config,
            RequestWorkerSelector router,
            DeliveryMetricsReporter reporter,
            DecodePreemption decodePreemption,
            RequestLifecycle lifecycle,
            PlacementAvailability availability,
            PlacementQueue queue,
            GroupPlanner grouping,
            RequestExecution.Dependencies queueDependencies) {
        super(lifecycle);
        Objects.requireNonNull(queueDependencies);
        this.queueDependencies =
                new RequestExecution.Dependencies(
                        queueDependencies.deliveries,
                        queueDependencies.reporter,
                        queueDependencies.failures,
                        QueuedRequestScheduler::runPrefill);
        this.queueSettings = QueueExecutionSettings.capture(config, grouping);
        this.router = Objects.requireNonNull(router, "router");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.decodePreemption = Objects.requireNonNull(decodePreemption, "decodePreemption");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.orderedQueue = Objects.requireNonNull(queue, "queue");
        this.waitingRequests = new PlacementCapacityWaitQueue(queue.order(), availability);
        this.scanBudgetMultiplier = config.queueScheduler().getScanBudgetMultiplier();

        this.plannerCount =
                Math.max(MIN_PLANNER_THREADS, config.getInternalRuntime().getQueuePlannerThreads());
        this.planners =
                Executors.newFixedThreadPool(
                        this.plannerCount,
                        Thread.ofPlatform().daemon().name("flexlb-global-planner-", 1).factory());
        this.schedulingLoop = new SchedulerLoop(
                Thread.ofPlatform().daemon().name("flexlb-global-decision")
                        .uncaughtExceptionHandler((thread, failure) -> {
                            Logger.error("Global queue decision thread failed", failure);
                            close();
                        }),
                this,
                () -> {
                    closed.set(true);
                    availability.removeListener(availabilityListener);
                    planners.shutdown();
                    drainOnClose();
                });
        availability.addListener(availabilityListener);
    }

    /** Start scheduler workers before accepting requests. */
    public void start() {
        schedulingLoop.start();
    }

    @Override
    public void onRequestChanged(RequestContext request) {
        postEvent(new QueueEvent(EventKind.INVALIDATE, null, request, null));
    }

    @Override
    public boolean reschedule(RequestRoute previousRoute) {
        return requeue(previousRoute);
    }

    @Override
    public Map<String, Object> describeWait(
            RequestContext ctx, CancelReason reason, String message) {
        if (reason != CancelReason.DEADLINE_EXCEEDED) {
            return null;
        }
        var state = ctx.snapshot();
        if (state.delivery() != null) {
            return Map.of("cause", message);
        }
        return state.route() != null
                ? state.route().prefillEp().getLatestQueueWaitSnapshot()
                : latestQueueWaitSnapshot;
    }

    @Override
    public void onRequest(RequestContext context) {
        if (!trySubmitRegistered(context)) {
            shutdownRequest(context);
        }
    }

    boolean trySubmitRegistered(RequestContext context) {
        try {
            Objects.requireNonNull(context, "context");
            CompletableFuture<Response> future =
                    Objects.requireNonNull(context.data().getFuture(), "registered future");
            var state = context.snapshot();
            if (future.isDone() || state.phase() != Phase.OPEN || state.cancellation() != null) {
                return true;
            }
            GlobalQueueEntry entry =
                    new GlobalQueueEntry(context, router.resolvePolicyGroup(context));
            if (!postEvent(new QueueEvent(EventKind.SUBMIT, entry, null, null))) {
                return false;
            }
            return true;
        } catch (RuntimeException failure) {
            // Once registration has been handed here, this mode owns execution failures.
            if (!lifecycle.publish(
                    context,
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Queue submission failed: " + failure.getMessage()),
                    false)) {
                lifecycle.closeRequest(context, null);
            }
            return true;
        }
    }

    /** A withdrawal is not a new request: keep its sequence, context, Future and absolute deadline. */
    public boolean requeue(RequestRoute previous) {
        if (previous.future().isDone()) {
            return true;
        }
        RequestContext context = previous.ctx();
        GlobalQueueEntry entry = new GlobalQueueEntry(context, router.resolvePolicyGroup(context));
        entry.sequence = context.data().getPlacementSequence();
        return postEvent(new QueueEvent(EventKind.REQUEUE, entry, null, null));
    }

    private boolean postEvent(QueueEvent event) {
        lock.lock();
        try {
            if (closed.get()) {
                return false;
            }
            events.addLast(event);
            schedulingLoop.wake();
            return true;
        } finally {
            lock.unlock();
        }
    }

    private void consumeEventsLocked() {
        QueueEvent event;
        while ((event = events.pollFirst()) != null) {
            switch (event.kind) {
                case CAPACITY -> waitingRequests.capacityChanged(event.key);
                case INVALIDATE -> {
                    GlobalQueueEntry entry = queuedEntries.get(event.context);
                    if (entry != null) {
                        removeRequest(entry);
                    }
                }
                case SUBMIT, REQUEUE -> {
                    GlobalQueueEntry entry = event.entry;
                    var state = entry.context.snapshot();
                    // Invalidation may arrive before SUBMIT. The recorded state also fences
                    // an archived request whose asynchronous response has not completed yet.
                    if (entry.context.data().getFuture().isDone()
                            || state.phase() != Phase.OPEN
                            || state.cancellation() != null) {
                        continue;
                    }
                    checkState(
                            queuedEntries.putIfAbsent(entry.context, entry) == null,
                            "duplicate global queue identity");
                    if (event.kind == EventKind.REQUEUE) {
                        entry.context.data().setPlanType("");
                        entry.context.data().setPlanCost(0L);
                        entry.context.data().setVictimCount(0);
                        orderedQueue.restore(entry);
                    } else {
                        orderedQueue.add(entry);
                        if (entry.context.data().getPlacementSequence() == 0L) {
                            entry.context.data().setPlacementSequence(entry.sequence);
                        }
                    }
                }
            }
        }
    }

    @Override
    public long schedule(long nowNanos) {
        if (closed.get() && !closeDrained) {
            drainOnClose();
            closeDrained = true;
        }
        PlacementPlan completed;
        List<GlobalQueueEntry> candidates;
        lock.lock();
        try {
            consumeEventsLocked();
            completed = completedPlans.pollFirst();
        } finally {
            lock.unlock();
        }
        if (completed != null) {
            processCompletedPlan(completed);
        }
        lock.lock();
        try {
            // A completed route can publish capacity or invalidate queued work.
            // Apply those index changes before selecting the next planning slots.
            consumeEventsLocked();
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
                schedulingLoop.stop();
                return Long.MAX_VALUE;
            }
            boolean ready = !events.isEmpty() || !completedPlans.isEmpty()
                    || (!closed.get() && inFlight.size() < plannerCount
                            && (orderedQueue.hasUnscannedRequests() || waitingRequests.hasReadyRequests()));
            return ready ? nowNanos : Long.MAX_VALUE;
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void onEndpointChanged(WorkerEndpoint endpoint) {
        var status = Objects.requireNonNull(endpoint, "endpoint").getStatus();
        postEvent(
                new QueueEvent(
                        EventKind.CAPACITY,
                        null,
                        null,
                        PlacementKey.exact(
                                status.getRole(),
                                status.topologySnapshot().group(),
                                endpoint.ipPort())));
    }

    private void submitPlan(GlobalQueueEntry entry) {
        try {
            planners.execute(
                    () -> {
                        PlacementPlan plan =
                                new PlacementPlan(entry, availability.sequence(), closed);
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
                                    lifecycle.cancel(
                                            entry.context, 0L, CancelReason.DEADLINE_EXCEEDED);
                                } else {
                                    plan.handle =
                                            lifecycle.claimAdmissionHandle(
                                                    entry.context.data().getRequestId(),
                                                    entry.context.data().getFuture());
                                    if (plan.handle != null) {
                                        plan.result =
                                                Objects.requireNonNull(
                                                        router.select(
                                                                entry.context, entry.routingGroup),
                                                        "placement result");
                                        plan.handle.recordDiagnostics(plan.result.diagnostics());
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
            publishPlan(PlacementPlan.failure(entry, failure, availability.sequence(), closed));
        }
    }

    private void publishPlan(PlacementPlan plan) {
        lock.lock();
        try {
            completedPlans.addLast(plan);
            schedulingLoop.wake();
        } finally {
            lock.unlock();
        }
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
                    commitFailure = Failures.run(commitFailure, () -> plan.close(lifecycle.tasks));
                }
            }
            Failures.rethrow(commitFailure, "placement commit failed");
            retry =
                    outcome == Outcome.REPLAN
                            || (outcome == Outcome.BLOCKED && !park(plan, diagnostics));
        } catch (Throwable failure) {
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
                if (completingPreemption) {
                    pendingPreemptions.remove(plan);
                    for (GlobalQueueEntry waiter : preemptionQuotaWaiters) {
                        waitingRequests.remove(waiter);
                        if (!waiter.removed && !waiter.context.data().getFuture().isDone()) {
                            orderedQueue.markRequestReadyForRetry(waiter);
                        }
                    }
                    preemptionQuotaWaiters.clear();
                }
                if (retry
                        && (!plan.entry.removed
                                && !plan.entry.context.data().getFuture().isDone())) {
                    orderedQueue.markRequestReadyForRetry(plan.entry);
                }
                schedulingLoop.wake();
            } finally {
                lock.unlock();
            }
        }
    }

    /** Caller holds lock; every examined entry counts against the scan budget. */
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
                PlacementResult<RequestRoute, PlacementKey> result = plan.result;
                switch (result.status()) {
                    case REJECTED -> {
                        plan.finishAdmission(false);
                        completeDecisionResponse(entry, result.failure());
                    }
                    case CLOSED -> {}
                    case BLOCKED -> {
                        return Outcome.BLOCKED;
                    }
                    case SUCCESS -> {
                        RequestRoute routing = result.value();
                        var publication = enqueueRoute(plan);
                        if (reclaimed && publication.status() != PlacementResult.Status.SUCCESS) {
                            rejection =
                                    "selected Prefill capacity changed before canonical placement";
                        } else if (publication.status() == PlacementResult.Status.BLOCKED) {
                            plan.waitKey = publication.blocker();
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
            plan.handle.terminate(
                    Response.error(
                            StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            rejection));
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
        var execution = decodePreemption.tryReclaim(plan.context(), blockedEndpoint);
        if (execution == null) {
            return false;
        }
        pendingPreemptions.add(plan);
        execution
                .thenApply(result -> acceptPreemption(plan, blockedEndpoint, result))
                .whenComplete(
                        (result, failure) -> {
                            plan.preemptionResult = result;
                            plan.preemptionFailure = failure;
                            if (failure != null) {
                                lifecycle.tasks.recordFailure(
                                        failure instanceof java.util.concurrent.CompletionException
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

    PlacementResult<RequestRoute, PlacementKey> enqueueRoute(PlacementPlan plan) {
        RequestContext context = plan.entry.context;
        RequestRoute routing = plan.result.value();
        if (plan.pendingDecodeRollback == null) {
            var request = context.data().getRequest();
            plan.pendingDecodeRollback =
                    routing.decodeEp() == null
                            ? null
                            : routing.decodeEp()
                                    .tryReserveQueuedRequest(
                                            routing.decodePin(),
                                            routing.requestId(),
                                            request.getHardKvTokens(),
                                            request.getExpectedKvTokens(),
                                            routing.priority(),
                                            decodeAdmissionCapacity)
                                    .handle();
            if (routing.decodeEp() != null && plan.pendingDecodeRollback == null) {
                return PlacementResult.blocked(routing.placementKey(RoleType.DECODE));
            }
        }
        RequestRoute route = RequestRoute.create(context, routing, plan.pendingDecodeRollback);
        plan.acquired = route;
        var placement = context.snapshot().placement();
        plan.placementToken = placement == null ? null : placement.operation();
        context.data().setRouteSubmittedNanos(System.nanoTime());
        PlacementResult<RequestRoute, PlacementKey> result =
                switch (lifecycle.commitRoute(route, new QueuePublication(plan, route))) {
                    case SUCCESS -> PlacementResult.success(route);
                    case BLOCKED -> PlacementResult.blocked(routing.placementKey(RoleType.PREFILL));
                    case CLOSED -> PlacementResult.closed();
                    case REJECTED ->
                            throw new IllegalStateException(
                                    "route commit cannot reject with a response");
                };
        if (result.status() == PlacementResult.Status.SUCCESS) {
            try {
                reporter.reportLatency(
                        DeliveryMetricsReporter.Latency.ROUTE_SUBMIT,
                        route.prefill().getRole().name(),
                        route.prefillEp().getIp(),
                        System.currentTimeMillis() - context.data().getStartTime());
            } catch (Throwable failure) {
                Logger.warn("Failed to record route-submit telemetry", failure);
            }
        }
        return result;
    }

    private final class QueuePublication implements Publication {
        private final PlacementPlan plan;
        private final RequestRoute route;
        private boolean published;

        private QueuePublication(PlacementPlan plan, RequestRoute route) {
            this.plan = plan;
            this.route = route;
        }

        @Override
        public void publish() {
            RequestRoute routing = plan.result.value();
            if (routing.prefillEp()
                    .offerPinned(routing.prefillPin(), route, queueSettings, queueDependencies)) {
                published = true;
                plan.published = true;
                plan.pendingDecodeRollback = null;
            }
        }

        @Override
        public boolean published() {
            return published;
        }
    }

    private boolean park(PlacementPlan plan, Map<String, Object> diagnostics) {
        int depth;
        int[] counts;
        boolean parked;
        lock.lock();
        try {
            if (!(!plan.entry.removed && !plan.entry.context.data().getFuture().isDone())) {
                return true;
            }
            parked = waitingRequests.park(plan.entry, plan.waitKey, plan.availabilitySequence);
            depth = orderedQueue.size();
            counts = orderedQueue.priorityCounts();
        } finally {
            lock.unlock();
        }
        Map<Integer, Integer> priorityCounts = new LinkedHashMap<>();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) {
                priorityCounts.put(priority, counts[priority]);
            }
        }
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("cause", plan.waitKey.role().name() + " placement unavailable");
        details.put("role", plan.waitKey.role().name());
        if (plan.waitKey.group() != null) {
            details.put("group", plan.waitKey.group());
        }
        if (plan.waitKey.endpoint() != null) {
            details.put("endpoint", plan.waitKey.endpoint());
        }
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
        if (acquire) {
            lock.lock();
        }
        try {
            orderedQueue.remove(entry);
            waitingRequests.remove(entry);
            preemptionQuotaWaiters.remove(entry);
            queuedEntries.remove(entry.context, entry);
            schedulingLoop.wake();
        } finally {
            if (acquire) {
                lock.unlock();
            }
        }
    }

    private void completeDecisionResponse(GlobalQueueEntry entry, Response response) {
        try {
            lifecycle.publish(entry.context, response, false);
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
            for (QueueEvent event : events) {
                if (event.kind == EventKind.SUBMIT || event.kind == EventKind.REQUEUE) {
                    all.add(event.entry);
                }
            }
            abandoned = List.copyOf(all);
            events.clear();
            queuedEntries.clear();
        } finally {
            schedulingLoop.wake();
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
        if (context.data().getScheduler() != lifecycle || !requests.isCurrent(context)) {
            return false;
        }
        lifecycle.closeRequest(
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
        Throwable failure = Failures.run(null, decodePreemption::freezeAdmission);
        if (closed.compareAndSet(false, true)) {
            availability.removeListener(availabilityListener);
            lock.lock();
            try {
                schedulingLoop.wake();
            } finally {
                lock.unlock();
            }
            // Late planner results release their own route ownership.
            planners.shutdown();
        }
        // Every caller observes the same drain, including a second shutdown owner.
        boolean interrupted = false;
        schedulingLoop.awaitStopped();
        while (!planners.isTerminated()) {
            try {
                planners.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        Failures.rethrow(failure, "scheduler failed to stop preemption");
    }

    private enum Outcome {
        DONE,
        BLOCKED,
        REPLAN,
        PENDING
    }

    private enum EventKind {
        SUBMIT,
        REQUEUE,
        INVALIDATE,
        CAPACITY
    }

    private record QueueEvent(
            EventKind kind, GlobalQueueEntry entry, RequestContext context, PlacementKey key) {}

    public static final class PlacementPlan implements AutoCloseable {
        private boolean closed;
        private final AtomicBoolean schedulerClosed;
        private final GlobalQueueEntry entry;
        private AdmissionHandle handle;
        private DecodeResources.ReservationHandle pendingDecodeRollback;
        private RequestRoute acquired;
        private PlacementToken placementToken;
        private boolean published;
        private Throwable planningFailure;
        private volatile boolean preemptionCompleted;
        private org.flexlb.balance.eviction.DecodePreemption.PreemptionResult preemptionResult;
        private Throwable preemptionFailure;
        // The plan owns selection pins and uncommitted Decode capacity.
        private PlacementResult<RequestRoute, PlacementKey> result;
        private final long availabilitySequence;
        // Retained after selection pins close so the request can be parked.
        private PlacementKey waitKey;

        private PlacementPlan(
                GlobalQueueEntry entry, long availabilitySequence, AtomicBoolean schedulerClosed) {
            this.entry = entry;
            this.schedulerClosed = schedulerClosed;
            this.availabilitySequence = availabilitySequence;
            this.result = PlacementResult.closed();
        }

        static PlacementPlan failure(
                GlobalQueueEntry entry,
                Throwable failure,
                long availabilitySequence,
                AtomicBoolean schedulerClosed) {
            PlacementPlan plan = new PlacementPlan(entry, availabilitySequence, schedulerClosed);
            plan.planningFailure = failure;
            return plan;
        }

        public RequestContext context() {
            return entry.context;
        }

        /** A false result leaves the reservation with the caller for rollback. */
        public synchronized boolean accept(
                DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
            var route = result == null ? null : result.value();
            var ctx = context();
            synchronized (ctx) {
                var state = ctx.snapshot();
                if (closed
                        || schedulerClosed.get()
                        || route == null
                        || route.decodeEp() != endpoint
                        || pendingDecodeRollback != null
                        || reservation.requestId() != route.requestId()
                        || ctx.data().getFuture().isDone()
                        || state.phase() != Phase.OPEN
                        || state.cancellation() != null
                        || handle == null
                        || state.placement() == null
                        || state.placement().operation() != handle.token) {
                    return false;
                }
            }
            // markQueued publishes capacity notifications; never invoke them under Context.
            if (!endpoint.markQueued(route.decodePin(), reservation).status().applied()) {
                return false;
            }
            pendingDecodeRollback = reservation;
            return true;
        }

        private void finishAdmission(boolean rollbackFailed) {
            AdmissionHandle owned = handle;
            handle = null;
            if (owned != null) {
                if (!rollbackFailed) {
                    owned.finish();
                } else {
                    owned.terminate(
                            buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "placement rollback failed"));
                }
            }
        }

        @Override
        public void close() {
            close(null);
        }

        void close(RequestTaskPool tasks) {
            RequestRoute routing;
            DecodeResources.ReservationHandle rollback;
            synchronized (this) {
                if (closed) {
                    return;
                }
                closed = true;
                routing = result == null ? null : result.value();
                result = null;
                rollback = pendingDecodeRollback;
                pendingDecodeRollback = null;
            }
            boolean rollbackFailed = false;
            Throwable cleanupFailure = null;
            try (routing) {
                try {
                    if (rollback != null) {
                        var released =
                                routing.decodeEp()
                                        .release(
                                                rollback,
                                                DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                                        .status();
                        checkState(
                                released == DecodeResources.ReservationReleaseResult.RELEASED
                                        || released
                                                == DecodeResources.ReservationReleaseResult.STALE,
                                "placement rollback retained Decode resources");
                    }
                    if (!published && acquired != null && placementToken != null) {
                        acquired.ctx()
                                .bindPlacement(
                                        placementToken,
                                        acquired,
                                        PublicationResult.ROLLBACK_COMPLETED);
                    }
                } catch (RuntimeException | Error failure) {
                    rollbackFailed = true;
                    throw failure;
                }
            } catch (Throwable failure) {
                cleanupFailure = failure;
            }
            if (tasks == null) {
                Failures.rethrow(cleanupFailure, "placement cleanup failed");
                return;
            }
            try {
                if (cleanupFailure != null) {
                    tasks.recordFailure(cleanupFailure);
                }
            } finally {
                finishAdmission(rollbackFailed);
            }
        }
    }
}

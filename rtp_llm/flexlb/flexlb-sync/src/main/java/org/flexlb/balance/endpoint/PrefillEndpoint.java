package org.flexlb.balance.endpoint;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.EstimationSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.CapacityNotifier;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.scheduler.RoutingContext;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.PriorityNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.function.LongSupplier;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Maintains request records, waiting queue membership and delivery capacity for one worker generation.
 * Reconciles worker reports under the state lock, then publishes scheduler notifications outside it.
 */
public class PrefillEndpoint extends WorkerEndpoint implements EndpointStats.Prefill {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final GroupPlanner.Constraints DIRECT_CONSTRAINTS =
            new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);

    /**
     * Non-owning index containing only ACTIVE RouteResult identities.
     */
    private PrefillRequestQueue queuedRequests = PrefillRequestQueue.disabled();

    /**
     * Canonical request records, changed only in this Endpoint state lock.
     */
    private final Map<String, RequestRecord> requests = new HashMap<>();

    private final LongSupplier clock = System::currentTimeMillis;

    /**
     * Occupied delivery slots, updated with their request records in the same event.
     */
    private int inflightRequests;

    // Updated and read under stateLock; captured versions travel with selection inputs.
    private long requestStateVersion;
    private long schedulingInputVersion;

    /**
     * Derived immutable work; ACTIVE queue mutations leave committed work unchanged.
     */
    private WorkSnapshot committedWorkSnapshot;

    private long unknownEngineRequestCount;

    private final Map<Long, BatchRecord> batches = new HashMap<>();
    public final PrefillTimePredictor predictor;

    public final long requestLimit;

    private volatile boolean queueEnabled;
    private volatile boolean queueStopped;

    // Changed in Endpoint lock; settings are fixed before execution is enabled.
    private final QueueExecutionSettings queueSettings;
    public final DispatcherConfig.Type deliveryMode;

    private RequestTimeEstimation.Inputs cachedEstimationInputs;

    private final DeliveryMetricsReporter reporter;
    private final CapacityNotifier capacityNotifier;

    PrefillEndpoint(
            WorkerStatus status,
            FlexlbConfig config,
            DispatcherConfig.Type deliveryMode,
            DeliveryMetricsReporter reporter,
            CapacityNotifier capacityNotifier,
            QueueExecutionSettings queueSettings) {
        super(status);
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.capacityNotifier =
                java.util.Objects.requireNonNull(capacityNotifier, "capacityNotifier");
        var estimator = config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator();
        this.predictor =
                estimator.getType() == RoutingConfig.EstimatorType.LEARNING
                        ? new LearningPredictor()
                        : new FormulaPredictor(estimator.getExpression());
        this.queueSettings = queueSettings;
        this.requestLimit =
                config.deliveryType(org.flexlb.dao.route.RequestPhase.GENERATION)
                                == DispatcherConfig.Type.NON_BATCH
                        ? config.getDispatcher().getMaxInflightPerPrefillWorker()
                        : 0L;
        this.deliveryMode = deliveryMode;
    }

    /**
     * The selected Endpoint checks the request record and retirement state before committing.
     */
    public WorkSnapshot commitDirect(RouteResult route) {
        stateLock().lock();
        try {
            var entry = route.prefillReservation();
            if (!acceptsNewRequests()
                    || entry == null
                    || requests.get(route.requestId()) != entry
                    || entry.route != route
                    || entry.ownership != OwnershipStage.RESERVED) return null;
            long now = clock.getAsLong();
            var preceding = captureWorkLocked(now, false, entry);
            entry.commit(null, entry.remainingWorkMs, now);
            committedWorkSnapshot = null;
            requestStateVersion++;
            return preceding;
        } finally {
            stateLock().unlock();
        }
    }

    public RequestTimeEstimation.Inputs captureRouteProjectionInputs() {
        RequestTimeEstimation.Inputs captured;
        stateLock().lock();
        try {
            captured = cachedEstimationInputs;
            if (captured == null
                    || captured.selectionVersion().ownershipVersion() != requestStateVersion
                    || captured.selectionVersion().schedulingInputVersion()
                            != schedulingInputVersion) {
                long nowMs = clock.getAsLong();
                List<GroupPlanner.Item> selected =
                        queueSettings == null ? List.of() : new ArrayList<>();
                var work =
                        captureWorkAndSelectedLocked(
                                nowMs, true, null, queueSettings == null ? null : selected);
                var queue =
                        new EstimationSnapshot(
                                nowMs,
                                queueSettings != null,
                                queueSettings == null ? null : queueSettings.grouping(),
                                QueueExecutionSettings.queueOrder(),
                                queueSettings == null
                                        ? DIRECT_CONSTRAINTS
                                        : queueSettings.constraints(
                                                getStatus().committedEngineObservation()),
                                queuedRequests.capture().items(),
                                selected);
                captured =
                        new RequestTimeEstimation.Inputs(
                                queue,
                                work,
                                new RequestTimeEstimation.SelectionVersion(
                                        getStatus().getGenerationId(),
                                        requestStateVersion,
                                        schedulingInputVersion),
                                null,
                                deliveryMode);
                cachedEstimationInputs = captured;
            }
        } finally {
            stateLock().unlock();
        }

        var evaluator = predictor == null ? null : predictor.evaluator();
        captured.queue().activeItems();
        captured.work().calculate();
        if (captured.evaluator() == evaluator) return captured;

        var updated =
                new RequestTimeEstimation.Inputs(
                        captured.queue(),
                        captured.work(),
                        captured.selectionVersion(),
                        evaluator,
                        deliveryMode);
        stateLock().lock();
        try {
            // A delayed reader must not replace a newer state snapshot or model capture.
            if (cachedEstimationInputs == captured) {
                cachedEstimationInputs = updated;
            } else if (cachedEstimationInputs.queue() == captured.queue()
                    && cachedEstimationInputs.evaluator() == evaluator) {
                return cachedEstimationInputs;
            }
            return updated;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Publishes a route before queue readers can consume it. Request cancellation uses the
     * same request lock; notifications and resource rollback run after both locks are released.
     */
    public boolean acceptQueuedRoute(
            RouteResult route, QueueExecutionSettings settings, RoutingContext routingContext) {
        checkArgument(
                route.prefillEndpoint() == this, "request belongs to another Prefill generation");
        checkArgument(
                settings != null && settings.equals(queueSettings),
                "queue rules must match endpoint configuration");
        var context = route.ctx();
        synchronized (context) {
            stateLock().lock();
            try {
                if (!acceptsNewRequests() || queueStopped) return false;
                var entry = requests.get(route.requestId());
                if (entry == null
                        || entry != route.prefillReservation()
                        || entry.route != route
                        || entry.ownership != OwnershipStage.WAITING
                                && entry.ownership != OwnershipStage.SELECTED) return false;
                if (!queueEnabled) {
                    queuedRequests = PrefillRequestQueue.fifo(16);
                    queueEnabled = true;
                    schedulingInputVersion++;
                }
                if (!queuedRequests.add(route)) return false;
                entry.ownership = OwnershipStage.WAITING;
                requestStateVersion++;
                boolean accepted = false;
                try {
                    accepted = context.publishRoute(routingContext, route);
                    return accepted;
                } finally {
                    if (!accepted) {
                        queuedRequests.remove(route);
                        requestStateVersion++;
                    }
                }
            } finally {
                stateLock().unlock();
            }
        }
    }

    public boolean queueEnabled() {
        return queueEnabled;
    }

    public boolean rollback(RequestRecord request) {
        checkArgument(request.owner == this, "request belongs to another endpoint");
        stateLock().lock();
        try {
            if (requests.get(request.planningInput.requestId()) != request
                    || request.ownership != OwnershipStage.RESERVED
                            && request.ownership != OwnershipStage.SELECTED) return false;
            removeRequestRecordLocked(request);
        } finally {
            stateLock().unlock();
        }
        signal(Signal.CAPACITY);
        return true;
    }

    /**
     * INPUT changes planning facts; CAPACITY also wakes placement; PLACEMENT only publishes that edge.
     */
    public void signal(Signal kind) {
        try {
            if (kind != Signal.PLACEMENT) {
                stateLock().lock();
                try {
                    schedulingInputVersion++;
                } finally {
                    stateLock().unlock();
                }
            }
            capacityNotifier.notifyChanged(this);
        } catch (Throwable failed) {
            if (kind != Signal.CAPACITY) {
                Failures.rethrow(failed, "Prefill scheduling notification failed");
            }
            try {
                logger.error("Prefill capacity notification failed", failed);
            } catch (Throwable ignored) {
            }
        }
    }

    /**
     * Releases a request only while it is still waiting in this worker queue.
     * Removes its local record and publishes capacity after unlocking; committed delivery is left untouched.
     */
    public boolean releaseQueuedRequest(RouteResult route) {
        stateLock().lock();
        try {
            var entry = requests.get(route.requestId());
            if (entry == null
                    || entry.route != route
                    || entry.ownership != OwnershipStage.WAITING
                    || !queuedRequests.contains(route)) return false;
            removeRequestRecordLocked(entry);
        } finally {
            stateLock().unlock();
        }
        signal(Signal.CAPACITY);
        return true;
    }

    /**
     * Remove only this exact request's resources, then publish capacity outside the Endpoint state lock.
     */
    public boolean releaseRequest(RouteResult route) {
        Objects.requireNonNull(route, "route");
        stateLock().lock();
        try {
            RequestRecord entry = requests.get(route.requestId());
            if (entry == null || entry.route != route) return false;
            if (entry.batch != null) {
                entry.batch.lastObservedAtMs =
                        Math.max(entry.batch.lastObservedAtMs, clock.getAsLong());
                entry.batch.outcome.learningEligible = false;
            }
            removeRequestRecordLocked(entry);
        } finally {
            stateLock().unlock();
        }
        signal(Signal.CAPACITY);
        return true;
    }

    /**
     * Updates request and capacity facts and publishes the Worker snapshot in one state-lock interval.
     * Remaining batch time is calculated later by WorkSnapshot readers outside this lock.
     */
    @Override
    public StatusChange applyWorkerStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        var observation = prepared.observation();
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        stateLock().lock();
        try {
            checkState(!isRetired(), "Prefill endpoint is retired");
            try {
                var changes = updateRequestAndBatchStatusLocked(observation);
                if (!observation.alive()) beginRetirement();
                ws.publishReducedStatus(prepared);
                if (changes.schedulingInputsChanged()) schedulingInputVersion++;
                return changes;
            } catch (Throwable failure) {
                beginRetirement();
                throw Failures.propagate(failure, "Prefill status update/publication failed");
            }
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) {
            return;
        }
        var reconciliation = (StatusChange.Prefill) change;
        Throwable reporting = reportBatchCompletions(reconciliation.batchCompletions(), null);
        if (reporting != null) {
            try {
                logger.warn(
                        "Prefill status committed but reporting failed: engine={}",
                        getIp(),
                        reporting);
            } catch (Throwable ignored) {
            }
        }
        if (reconciliation.capacityReleased()) {
            signal(Signal.CAPACITY);
        } else if (reconciliation.schedulingInputsChanged()) {
            signal(Signal.PLACEMENT);
        }
        for (var status : reconciliation.requestStatuses()) {
            var route = status.route();
            if (route.prefillEndpoint() != this) {
                continue;
            }
            route.ctx()
                    .data()
                    .getRequestScheduler()
                    .onWorkerRequestEvent(
                            route.ctx(),
                            new ContextResult.WorkerRequestEvent(
                                    route,
                                    null,
                                    getStatus().getRole(),
                                    switch (status.kind()) {
                                        case ACTIVE -> ContextResult.WorkerRequestEventType.ACTIVE;
                                        case COMPLETED ->
                                                ContextResult.WorkerRequestEventType.COMPLETED;
                                        case FAILED -> ContextResult.WorkerRequestEventType.FAILED;
                                        case PRIORITY_CANCELED ->
                                                ContextResult.WorkerRequestEventType
                                                        .PRIORITY_CANCELLED;
                                    },
                                    status.errorCode(),
                                    "",
                                    System.currentTimeMillis()));
        }
    }

    /**
     * Evict endpoint orphans while retaining IDs still registered by the scheduler.
     */
    public int evictExpiredInflight(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        var orphanCandidates = new java.util.HashSet<RouteResult>();
        for (RouteResult route : cleanupCandidates()) {
            if (!retainForSchedulerCleanup.test(route.requestId())) {
                orphanCandidates.add(route);
            }
        }
        int released = evictExpiredInflight(ttlMs, orphanCandidates);
        if (released > 0) {
            signal(Signal.CAPACITY);
        }
        return released;
    }

    /**
     * Capture exact committed identities before the workflow checks request ownership outside this event.
     */
    public List<RouteResult> cleanupCandidates() {
        stateLock().lock();
        try {
            return requests.values().stream()
                    .filter(entry -> entry.ownership == OwnershipStage.COMMITTED)
                    .map(entry -> entry.route)
                    .toList();
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Captures cleanup candidates, then checks and removes each batch or individual under its own lock interval.
     * A retained member protects its whole batch; the result counts batches and individuals.
     */
    public int evictExpiredInflight(long ttlMs, Set<RouteResult> orphanCandidates) {
        List<BatchRecord> batchCandidates;
        List<RequestRecord> individualCandidates = new ArrayList<>();
        stateLock().lock();
        try {
            batchCandidates = List.copyOf(batches.values());
            for (RequestRecord entry : requests.values()) {
                if (entry.batch == null && entry.ownership == OwnershipStage.COMMITTED) {
                    individualCandidates.add(entry);
                }
            }
        } finally {
            stateLock().unlock();
        }
        int evicted = 0;
        long ttl = Math.max(0L, ttlMs);
        for (BatchRecord batch : batchCandidates) {
            stateLock().lock();
            try {
                if (batches.get(batch.batchId) != batch) continue;
                Set<RequestRecord> members = batch.members;
                if (clock.getAsLong() - batch.lastObservedAtMs < ttl) continue;
                boolean retained = false;
                for (RequestRecord entry : members) {
                    if (!orphanCandidates.contains(entry.route)) {
                        retained = true;
                        break;
                    }
                }
                if (retained) {
                    continue;
                }
                for (RequestRecord entry : List.copyOf(members)) {
                    removeRequestRecordLocked(entry);
                }
                evicted++;
            } finally {
                stateLock().unlock();
            }
        }
        for (RequestRecord entry : individualCandidates) {
            stateLock().lock();
            try {
                if (requests.get(entry.planningInput.requestId()) == entry
                        && entry.batch == null
                        && entry.ownership == OwnershipStage.COMMITTED
                        && clock.getAsLong() - entry.phaseBaseMs >= ttl
                        && orphanCandidates.contains(entry.route)) {
                    removeRequestRecordLocked(entry);
                    evicted++;
                }
            } finally {
                stateLock().unlock();
            }
        }
        return evicted;
    }

    /**
     * Current number of requests waiting for this worker.
     */
    public int queueDepth() {
        stateLock().lock();
        try {
            return queuedRequests.size();
        } finally {
            stateLock().unlock();
        }
    }

    public EndpointStats.QueueCounts queueCounters() {
        stateLock().lock();
        try {
            int[] byPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < byPriority.length; priority++) {
                byPriority[priority] = queuedRequests.size(priority);
            }
            return new EndpointStats.QueueCounts(
                    queuedRequests.version(),
                    byPriority,
                    saturatedAdd(requests.size(), unknownEngineRequestCount),
                    batches.size());
        } finally {
            stateLock().unlock();
        }
    }

    public boolean queueStopped() {
        return queueStopped;
    }

    public QueueSnapshot captureQueue(int limit) {
        checkArgument(limit > 0, "queue capture limit must be positive");
        stateLock().lock();
        try {
            List<RouteResult> items = new ArrayList<>(Math.min(limit, queuedRequests.size()));
            var ordered = queuedRequests.iterator();
            while (items.size() < limit && ordered.hasNext()) {
                items.add(ordered.next());
            }
            var head = items.isEmpty() ? null : items.getFirst();
            return new QueueSnapshot(
                    queuedRequests.version(), schedulingInputVersion, List.copyOf(items), head);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Stop queue admission and detach membership; the caller owns request settlement.
     */
    public List<RouteResult> closeQueue() {
        stateLock().lock();
        try {
            queueStopped = true;
            var removed = List.copyOf(queuedRequests.capture().items());
            queuedRequests.clear();
            if (!removed.isEmpty()) requestStateVersion++;
            return removed;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Explains a capacity rejection using current request priorities; does not admit a request.
     */
    public AdmissionRejectReason explainAdmissionRejection(int priority) {
        stateLock().lock();
        try {
            long lower = 0, same = 0, higher = 0, unknown = unknownEngineRequestCount;
            for (RequestRecord entry : requests.values()) {
                int occupant = entry.planningInput.priority();
                if (!PriorityNormalizer.isValid(occupant)) unknown++;
                else if (occupant < priority) lower++;
                else if (occupant == priority) same++;
                else higher++;
            }
            long occupied = saturatedAdd(requests.size(), unknownEngineRequestCount);
            long requestsToRelease =
                    requestLimit > 0 ? Math.max(0L, occupied - requestLimit + 1L) : 0L;
            long residual = Math.max(0L, requestsToRelease - lower);
            long protectedRequests = higher + same;
            if (residual > protectedRequests && unknown > 0L) {
                return AdmissionRejectReason.UNSPECIFIED;
            }
            if (residual > 0L && protectedRequests >= residual) {
                return higher > 0L
                        ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                        : AdmissionRejectReason.SAME_PRIORITY_AHEAD;
            }
            return AdmissionRejectReason.RESOURCE_EXHAUSTED;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Validate the queue prefix and capacity, then dequeue and register its work under one lock.
     * Returns the preceding work or a rejection reason; the caller sends the accepted requests.
     */
    public CapacityBoundary.Attempt<WorkSnapshot> tryStartQueuedDelivery(
            Delivery.Selection selection) {
        stateLock().lock();
        try {
            List<RouteResult> items = selection.items();
            long now = clock.getAsLong();
            if (!acceptsNewRequests())
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            if (selection.batch() && batches.size() >= selection.maxInflightBatches())
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH));
            if (!selection.batch() && !requestCapacityAvailableLocked(requestLimit, items.size()))
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_REQUEST));
            var members = new java.util.LinkedHashSet<RequestRecord>();
            var queue = queuedRequests.iterator();
            for (var route : items) {
                var entry = requests.get(route.requestId());
                if (!queue.hasNext()
                        || queue.next() != route
                        || route.requestExpired(now)
                        || entry == null
                        || entry.route != route
                        || entry.ownership != OwnershipStage.WAITING)
                    return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
                members.add(entry);
            }
            if (selection.batch())
                checkState(!batches.containsKey(selection.batchId()), "batch id already committed");
            var precedingWork = captureWorkLocked(now, true, null);
            var batch = selection.batch() ? new BatchRecord(selection, members, now) : null;
            if (batch != null) batches.put(batch.batchId, batch);
            int index = 0;
            for (var member : members) {
                queuedRequests.remove(member.route);
                member.commit(
                        batch,
                        batch == null ? selection.predictionAt(index++) : selection.predictedMs(),
                        now);
            }
            inflightRequests += members.size();
            committedWorkSnapshot = null;
            requestStateVersion++;
            return CapacityBoundary.Attempt.accepted(precedingWork);
        } finally {
            stateLock().unlock();
        }
    }

    public boolean batchCapacityAvailable(int maximum) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
        stateLock().lock();
        try {
            return batches.size() < maximum;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Queued routes consume no delivery slots; reserved and committed work do.
     */
    public boolean requestCapacityAvailable(long maximum, int requested) {
        checkArgument(maximum >= 0L && requested > 0, "invalid request capacity");
        stateLock().lock();
        try {
            return requestCapacityAvailableLocked(maximum, requested);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Register selected work before it can be dispatched. The same record is published later.
     */
    public Registration tryRegisterSelection(
            GroupPlanner.Item input,
            RequestTimeEstimation.Inputs expected,
            long workMs,
            boolean queued) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return new Registration(RoutingResult.Registration.UNAVAILABLE, null);
            var version = expected.selectionVersion();
            if (getStatus().getGenerationId() != version.generationId()
                    || requestStateVersion != version.ownershipVersion()
                    || schedulingInputVersion != version.schedulingInputVersion()
                    || predictor != null && predictor.evaluator() != expected.evaluator())
                return new Registration(RoutingResult.Registration.CONFLICT, null);
            if (requests.containsKey(input.requestId()))
                return new Registration(RoutingResult.Registration.CONFLICT, null);
            if (!queued && !requestCapacityAvailableLocked(requestLimit, 1))
                return new Registration(RoutingResult.Registration.BLOCKED, null);
            var record =
                    new RequestRecord(
                            this,
                            input,
                            queued ? OwnershipStage.SELECTED : OwnershipStage.RESERVED);
            record.remainingWorkMs = Math.max(0L, workMs);
            requests.put(record.planningInput.requestId(), record);
            if (!queued) inflightRequests++;
            if (!queued || queueSettings == null) committedWorkSnapshot = null;
            requestStateVersion++;
            return new Registration(RoutingResult.Registration.REGISTERED, record);
        } finally {
            stateLock().unlock();
        }
    }

    public EndpointStats.PrefillCounts stats() {
        int locallyOwned = 0;
        int individual = 0;
        long maxAgeMs = 0L;
        long nowMs = clock.getAsLong();
        stateLock().lock();
        try {
            for (RequestRecord entry : requests.values()) {
                if (entry.ownership != OwnershipStage.COMMITTED) {
                    continue;
                }
                locallyOwned++;
                if (entry.batch == null) {
                    individual++;
                    maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - entry.phaseBaseMs));
                }
            }
            for (BatchRecord batch : batches.values()) {
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs));
            }
            return new EndpointStats.PrefillCounts(
                    locallyOwned, individual, batches.size(), maxAgeMs);
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public OptionalLong tryGetLoadMetric() {
        if (!stateLock().tryLock()) return OptionalLong.empty();
        WorkSnapshot capture;
        try {
            capture = captureWorkLocked(clock.getAsLong(), false, null);
        } finally {
            stateLock().unlock();
        }
        return capture.calculate().totalRemainingWorkMs();
    }

    /**
     * Clears request records once and publishes retirement notifications outside the state lock.
     */
    @Override
    protected void closeEndpoint() {
        Throwable retirementFailure = null;
        try {
            Retirement retirement = retireResources();
            for (RouteResult route : retirement.ownedItems()) {
                try {
                    var context = route.ctx();
                    var event =
                            new ContextResult.EndpointRetirement(
                                    route,
                                    null,
                                    getStatus().getRole(),
                                    this,
                                    "Prefill generation retired: " + ipPort(),
                                    System.currentTimeMillis());
                    context.data().getRequestScheduler().onEndpointRetired(context, event);
                } catch (Throwable notificationFailure) {
                    retirementFailure = Failures.append(retirementFailure, notificationFailure);
                }
            }
            retirementFailure =
                    reportBatchCompletions(retirement.batchCompletions(), retirementFailure);
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = Failures.append(retirementFailure, committedRetirementFailure);
        }
        signal(Signal.CAPACITY);
        Failures.rethrow(retirementFailure, "Prefill endpoint retirement failed");
    }

    /**
     * Captures cleanup notifications and clears all records in one Endpoint lock.
     */
    Retirement retireResources() {
        List<RouteResult> ownedItems = new ArrayList<>();
        List<BatchCompletion> completions = new ArrayList<>();
        stateLock().lock();
        try {
            for (var entry : requests.values()) {
                if (entry.route != null) ownedItems.add(entry.route);
            }
            for (var batch : batches.values()) {
                completions.add(
                        new BatchCompletion(
                                batch.batchId,
                                batch.originalFeatures,
                                batch.originalPredictionMs,
                                batch.outcome.maxExecutionTimeMs,
                                batch.outcome.successfulCompletion,
                                false));
            }
            for (BatchRecord batch : batches.values()) {
                if (!batch.members.isEmpty()) batch.members.clear();
            }
            queueStopped = true;
            requests.clear();
            batches.clear();
            queuedRequests.clear();
            unknownEngineRequestCount = 0L;
            committedWorkSnapshot = null;
            inflightRequests = 0;
            requestStateVersion++;
        } finally {
            stateLock().unlock();
        }
        ownedItems.sort(
                Comparator.comparingLong(RouteResult::enqueueSeq)
                        .thenComparing(RouteResult::requestId));
        completions.sort(Comparator.comparingLong(BatchCompletion::batchId));
        return new Retirement(ownedItems, completions);
    }

    /**
     * Updates requests first, then aggregates only the members reported for each affected batch.
     * Finished records remain available until batch feedback has consumed their results.
     */
    private StatusChange.Prefill updateRequestAndBatchStatusLocked(
            WorkerStatus.StatusObservation observation) {
        long nowMs = clock.getAsLong();
        var finishedRequests = matchFinishedRequests(observation.finishedTasks());
        var engine = observation.engine();
        var statuses =
                new ArrayList<PrefillRequestStatus>(
                        finishedRequests.size() + engine.runningTaskList().size());
        Map<BatchRecord, List<RequestRecord>> reportedBatchMembers = new IdentityHashMap<>();
        for (var finished : finishedRequests.values()) {
            PrefillRequestStatus.Kind kind;
            if (finished.errorCode() == 0L) {
                kind = PrefillRequestStatus.Kind.COMPLETED;
            } else if (finished.preemptionProgress() == PriorityPreemptionProgress.CANCELED
                    && finished.errorCode()
                            == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()) {
                kind = PrefillRequestStatus.Kind.PRIORITY_CANCELED;
            } else {
                kind = PrefillRequestStatus.Kind.FAILED;
            }
            var entry = finished.owner();
            statuses.add(new PrefillRequestStatus(entry.route, kind, finished.errorCode()));
            if (entry.batch != null) {
                reportedBatchMembers
                        .computeIfAbsent(entry.batch, ignored -> new ArrayList<>())
                        .add(entry);
            }
        }

        boolean changed = !finishedRequests.isEmpty();
        Set<String> unknown = new HashSet<>(), known = new HashSet<>();
        for (var task : engine.runningTaskList().values()) {
            if (finishedRequests.containsKey(task.requestId())) continue;
            var entry = requests.get(task.requestId());
            if (entry == null
                    || entry.batch != null
                            && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) {
                if (!task.isPriorityCancelOverlayOnly()) unknown.add(task.requestId());
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                known.add(task.requestId());
                if (entry.ownership == OwnershipStage.COMMITTED) {
                    statuses.add(
                            new PrefillRequestStatus(
                                    entry.route, PrefillRequestStatus.Kind.ACTIVE, 0L));
                }
            }
            if (entry.ownership != OwnershipStage.COMMITTED) continue;
            Phase phase =
                    task.phase() == TaskPhase.RUNNING ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
            if (entry.batch == null) {
                changed |= entry.executionPhase != phase;
                entry.remainingWorkMs =
                        remainingWorkAt(
                                entry.executionPhase,
                                entry.remainingWorkMs,
                                entry.phaseBaseMs,
                                nowMs);
            } else {
                reportedBatchMembers
                        .computeIfAbsent(entry.batch, ignored -> new ArrayList<>())
                        .add(entry);
            }
            entry.phaseBaseMs = Math.max(entry.phaseBaseMs, nowMs);
            entry.executionPhase = phase;
        }

        var completions = new ArrayList<BatchCompletion>();
        for (var batchReport : reportedBatchMembers.entrySet()) {
            changed |=
                    updateBatchStatusLocked(
                            batchReport.getKey(),
                            batchReport.getValue(),
                            finishedRequests,
                            nowMs,
                            completions);
        }
        for (var finished : finishedRequests.values()) {
            removeRequestRecordLocked(finished.owner());
        }
        long total =
                saturatedAdd(
                        Math.max(0L, engine.waitingQueryLen()),
                        Math.max(0L, engine.runningQueryLen()));
        long unknownCount = Math.max(unknown.size(), Math.max(0L, total - known.size()));
        boolean released = !finishedRequests.isEmpty() || unknownCount < unknownEngineRequestCount;
        changed |= unknownCount != unknownEngineRequestCount;
        unknownEngineRequestCount = unknownCount;
        if (changed) {
            committedWorkSnapshot = null;
            requestStateVersion++;
        }
        boolean inputsChanged =
                changed
                        || QueueExecutionSettings.capacityChanged(
                                getStatus().committedEngineObservation(), engine);
        return new StatusChange.Prefill(statuses, completions, inputsChanged, released);
    }

    private Map<String, TerminalObservation> matchFinishedRequests(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        // Both maps are keyed by request ID; worker history may outnumber local requests.
        var candidateIds =
                requests.size() < finishedTasks.size() ? requests.keySet() : finishedTasks.keySet();
        Map<String, TerminalObservation> matched = new HashMap<>();
        for (String requestId : candidateIds) {
            var entry = requests.get(requestId);
            var task = finishedTasks.get(requestId);
            if (entry == null || task == null || entry.ownership != OwnershipStage.COMMITTED) {
                continue;
            }
            if (entry.batch != null
                    && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) {
                continue;
            }
            var next =
                    new TerminalObservation(
                            entry,
                            task.executionTimeMs(),
                            task.errorCode(),
                            task.priorityPreemptionProgress());
            matched.put(requestId, next);
        }
        return matched;
    }

    private static long remainingWorkAt(
            Phase phase, long remainingWorkMs, long phaseBaseMs, long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        long elapsedMs = Math.max(0L, nowMs - phaseBaseMs);
        return Math.max(0L, remainingWorkMs - elapsedMs);
    }

    /**
     * Uses this report's member phases and finished results to update one batch.
     * Historical outcomes survive member removal; prediction remains deferred to WorkSnapshot.
     */
    private boolean updateBatchStatusLocked(
            BatchRecord batch,
            List<RequestRecord> reportedMembers,
            Map<String, TerminalObservation> finishedRequests,
            long nowMs,
            List<BatchCompletion> completions) {
        Phase reportedPhase = null;
        int finishedCount = 0;
        var outcome = batch.outcome;
        for (var member : reportedMembers) {
            var finished = finishedRequests.get(member.planningInput.requestId());
            if (finished != null) {
                finishedCount++;
                boolean success = finished.errorCode() == 0L;
                outcome.executionStarted |= success || finished.executionTimeMs() > 0L;
                outcome.maxExecutionTimeMs =
                        Math.max(outcome.maxExecutionTimeMs, finished.executionTimeMs());
                outcome.successfulCompletion |= success;
                outcome.learningEligible &= success;
            } else if (reportedPhase != Phase.ENGINE_RUNNING) {
                reportedPhase = member.executionPhase;
            }
        }
        batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, nowMs);
        boolean changed = reportedPhase != null && batch.servicePhase != reportedPhase;
        if (reportedPhase != null) {
            if (batch.servicePhase == Phase.ENGINE_RUNNING) {
                batch.elapsedRunningMs =
                        saturatedAdd(
                                batch.elapsedRunningMs, Math.max(0L, nowMs - batch.phaseBaseMs));
            }
            batch.phaseBaseMs = Math.max(batch.phaseBaseMs, nowMs);
            batch.servicePhase = reportedPhase;
            outcome.executionStarted |= reportedPhase == Phase.ENGINE_RUNNING;
        }
        if (finishedCount == batch.members.size()) {
            completions.add(
                    new BatchCompletion(
                            batch.batchId,
                            batch.originalFeatures,
                            batch.originalPredictionMs,
                            outcome.maxExecutionTimeMs,
                            outcome.successfulCompletion,
                            outcome.learningEligible));
        } else if (finishedCount > 0 && !outcome.executionStarted) {
            batch.estimate =
                    new WorkSnapshot.BatchEstimate(
                            batch.members.stream()
                                    .filter(
                                            member ->
                                                    !finishedRequests.containsKey(
                                                            member.planningInput.requestId()))
                                    .map(
                                            member -> {
                                                var route = member.route;
                                                var input = route.planningInput();
                                                long seqLen = Math.max(0L, route.seqLen());
                                                return new GroupPlanner.Item(
                                                        input.requestId(),
                                                        input.priority(),
                                                        input.enqueueSeq(),
                                                        input.enqueuedAtMs(),
                                                        input.expiresAtMs(),
                                                        seqLen,
                                                        Math.clamp(route.hitCache(), 0L, seqLen));
                                            })
                                    .toList(),
                            predictor);
        }
        return changed;
    }

    /**
     * Removes the exact request, releases its queue or batch capacity, and advances the request version.
     */
    private void removeRequestRecordLocked(RequestRecord entry) {
        if (requests.get(entry.planningInput.requestId()) != entry) {
            return;
        }
        if (entry.route != null && queuedRequests.contains(entry.route)) {
            checkState(
                    queuedRequests.remove(entry.route), "exact queue member could not be removed");
        }
        if (entry.batch != null) {
            var batch = entry.batch;
            checkState(
                    batches.get(batch.batchId) == batch && batch.members.remove(entry),
                    "terminal member lost its batch");
            if (batch.members.isEmpty()) {
                batches.remove(batch.batchId, batch);
            }
        }
        if (entry.ownership != OwnershipStage.WAITING
                && entry.ownership != OwnershipStage.SELECTED) {
            checkState(inflightRequests > 0, "Prefill delivery slot underflow");
            inflightRequests--;
            committedWorkSnapshot = null;
        }
        if (entry.ownership == OwnershipStage.SELECTED && queueSettings == null)
            committedWorkSnapshot = null;
        requests.remove(entry.planningInput.requestId(), entry);
        requestStateVersion++;
    }

    private WorkSnapshot captureWorkLocked(long nowMs, boolean cached, RequestRecord excluded) {
        return captureWorkAndSelectedLocked(nowMs, cached, excluded, null);
    }

    /**
     * Captures selected inputs and uncached work in one traversal while their version is locked.
     */
    private WorkSnapshot captureWorkAndSelectedLocked(
            long nowMs, boolean cached, RequestRecord excluded, List<GroupPlanner.Item> selected) {
        boolean reuseWork =
                cached
                        && committedWorkSnapshot != null
                        && committedWorkSnapshot.capturedAtMs() <= nowMs;
        if (reuseWork && selected == null) {
            return committedWorkSnapshot;
        }
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RequestRecord entry : requests.values()) {
            if (selected != null && entry.ownership == OwnershipStage.SELECTED) {
                selected.add(entry.planningInput);
            }
            if (reuseWork
                    || entry == excluded
                    || entry.batch != null
                    || entry.ownership == OwnershipStage.WAITING
                    || entry.ownership == OwnershipStage.SELECTED && queueSettings != null) {
                continue;
            }
            Phase phase =
                    entry.ownership == OwnershipStage.COMMITTED
                            ? entry.executionPhase
                            : Phase.COMMITTED;
            individual.add(
                    new WorkSnapshot.RequestWork(
                            entry.planningInput.requestId(),
                            phase,
                            remainingWorkAt(
                                    phase, entry.remainingWorkMs, entry.phaseBaseMs, nowMs)));
        }
        if (reuseWork) return committedWorkSnapshot;
        List<WorkSnapshot.BatchWork> capturedBatches = new ArrayList<>(batches.size());
        for (BatchRecord batch : batches.values()) {
            List<String> members = new ArrayList<>(batch.members.size());
            for (RequestRecord member : batch.members) {
                members.add(member.route.requestId());
            }
            if (!members.isEmpty()) {
                capturedBatches.add(
                        new WorkSnapshot.BatchWork(
                                members,
                                batch.servicePhase,
                                batch.estimate,
                                saturatedAdd(
                                        batch.elapsedRunningMs,
                                        batch.servicePhase == Phase.ENGINE_RUNNING
                                                ? Math.max(0L, nowMs - batch.phaseBaseMs)
                                                : 0L)));
            }
        }
        var capture =
                new WorkSnapshot(nowMs, individual, capturedBatches, unknownEngineRequestCount);
        if (cached) {
            committedWorkSnapshot = capture;
        }
        return capture;
    }

    private Throwable reportBatchCompletions(List<BatchCompletion> completions, Throwable failure) {
        for (var completion : completions) {
            try {
                long batchId = completion.batchId(), actualMs = completion.actualWorkMs();
                if (!completion.successfulCompletion() || actualMs <= 0) {
                    logger.debug(
                            "batch completion not reportable: batchId={} success={} actualMs={}",
                            batchId,
                            completion.successfulCompletion(),
                            actualMs);
                    continue;
                }
                long predictedMs = completion.predictedWorkMs();
                org.flexlb.util.Logger.debug(
                        "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                        batchId,
                        predictedMs,
                        actualMs,
                        actualMs - predictedMs,
                        completion.originalFeatures().batchSize(),
                        getIp());
                // A failed/removed member invalidates the original batch as a learning sample.
                if (completion.learningEligible()) {
                    try {
                        if (predictor.learn(completion.originalFeatures(), predictedMs, actualMs)
                                == PrefillTimePredictor.LearningResult.MODEL_UPDATED) {
                            signal(Signal.INPUT);
                        }
                    } catch (RuntimeException learningFailure) {
                        logger.warn(
                                "batch predictor learning failed after settlement: batchId={} engine={}",
                                batchId,
                                getIp(),
                                learningFailure);
                    }
                }
                try {
                    reporter.reportBatchCompletion(getIp(), predictedMs, actualMs);
                } catch (RuntimeException telemetryFailure) {
                    logger.warn(
                            "batch completion metrics failed: batchId={} engine={}",
                            batchId,
                            getIp(),
                            telemetryFailure);
                }
            } catch (Throwable reportingFailure) {
                failure = Failures.append(failure, reportingFailure);
            }
        }
        return failure;
    }

    private boolean requestCapacityAvailableLocked(long maximum, int requested) {
        long inFlight = saturatedAdd(inflightRequests, unknownEngineRequestCount);
        return maximum == 0L || requested <= maximum && inFlight <= maximum - requested;
    }

    public enum Signal {
        CAPACITY,
        INPUT,
        PLACEMENT
    }

    /**
     * Request status matched to its exact Prefill route, published after ledger reconciliation.
     */
    public record PrefillRequestStatus(RouteResult route, Kind kind, long errorCode) {
        public PrefillRequestStatus {
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(kind, "kind");
            checkArgument(
                    kind != Kind.ACTIVE || errorCode == 0L,
                    "an active Prefill request status cannot carry an error code");
        }

        public enum Kind {
            ACTIVE,
            COMPLETED,
            FAILED,
            PRIORITY_CANCELED
        }
    }

    public record BatchCompletion(
            long batchId,
            PrefillBatchFeatures originalFeatures,
            long predictedWorkMs,
            long actualWorkMs,
            boolean successfulCompletion,
            boolean learningEligible) {
        public BatchCompletion {
            Objects.requireNonNull(originalFeatures, "originalFeatures");
        }
    }

    /**
     * Freezes the routes and batch completions collected while retiring a Prefill endpoint.
     */
    public record Retirement(List<RouteResult> ownedItems, List<BatchCompletion> batchCompletions) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    /**
     * One committed batch and its members; created atomically when queue work is submitted.
     */
    private static final class BatchRecord {
        final long batchId;
        final Set<RequestRecord> members;
        final long originalPredictionMs;
        final PrefillBatchFeatures originalFeatures;
        WorkSnapshot.BatchEstimate estimate;
        long elapsedRunningMs;
        Phase servicePhase = Phase.COMMITTED;
        BatchOutcome outcome = new BatchOutcome();
        long phaseBaseMs;
        long lastObservedAtMs;

        private BatchRecord(Delivery.Selection selection, Set<RequestRecord> members, long nowMs) {
            checkArgument(selection.batchId() >= 0L, "batchId must be non-negative");
            this.batchId = selection.batchId();
            this.members = members;
            originalPredictionMs = selection.predictedMs();
            originalFeatures = selection.features();
            estimate = new WorkSnapshot.BatchEstimate(originalPredictionMs);
            phaseBaseMs = nowMs;
            lastObservedAtMs = nowMs;
        }
    }

    /**
     * Resource ownership within this Prefill generation, independent of request protocol.
     */
    enum OwnershipStage {
        SELECTED,
        WAITING,
        RESERVED,
        COMMITTED
    }

    /**
     * One selection registration. The complete route is linked before publication.
     * Mutable resource facts change only in the Endpoint lock.
     */
    public static final class RequestRecord {
        final PrefillEndpoint owner;

        OwnershipStage ownership;

        final GroupPlanner.Item planningInput;

        RouteResult route;

        /**
         * Starts at COMMITTED on delivery and follows this request's Worker reports, including batch members.
         * Finished requests are removed; this phase does not determine capacity ownership.
         */
        Phase executionPhase;

        long remainingWorkMs;
        long phaseBaseMs;

        BatchRecord batch;

        private RequestRecord(
                PrefillEndpoint owner, GroupPlanner.Item planningInput, OwnershipStage ownership) {
            this.owner = owner;
            this.ownership = ownership;
            this.planningInput = Objects.requireNonNull(planningInput);
        }

        /**
         * The immutable request input used for registration, queueing and projection.
         */
        public GroupPlanner.Item planningInput() {
            return planningInput;
        }

        /**
         * Associate the complete route with its exact selection record before publication.
         */
        public void bind(RouteResult completed) {
            owner.stateLock().lock();
            try {
                checkArgument(
                        planningInput.requestId().equals(completed.requestId()),
                        "Prefill registration belongs to another selection");
                checkState(route == null || route == completed, "registration already bound");
                route = completed;
            } finally {
                owner.stateLock().unlock();
            }
        }

        private void commit(BatchRecord committedBatch, long predictedMs, long nowMs) {
            if (committedBatch == null) {
                remainingWorkMs = Math.clamp(predictedMs, 0L, (long) Integer.MAX_VALUE);
            } else {
                batch = committedBatch;
            }
            phaseBaseMs = nowMs;
            executionPhase = Phase.COMMITTED;
            ownership = OwnershipStage.COMMITTED;
        }
    }

    record TerminalObservation(
            RequestRecord owner,
            long executionTimeMs,
            long errorCode,
            PriorityPreemptionProgress preemptionProgress) {}

    /**
     * One bounded, ordered queue view; all metadata belongs to the same ownership revision.
     */
    public record QueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<RouteResult> items,
            RouteResult head) {}

    public record Registration(RoutingResult.Registration status, RequestRecord record) {}

    /**
     * Accumulates member results until the batch's last request is removed.
     */
    static final class BatchOutcome {
        long maxExecutionTimeMs;
        boolean successfulCompletion;
        boolean learningEligible = true;

        /**
         * Historical execution evidence survives a later queued observation.
         */
        boolean executionStarted;
    }
}

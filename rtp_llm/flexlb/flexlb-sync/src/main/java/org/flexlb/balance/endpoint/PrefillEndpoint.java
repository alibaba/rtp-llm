package org.flexlb.balance.endpoint;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.CapacityNotifier;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.scheduler.RoutingContext;
import org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
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
    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR =
            new FormulaPredictor(new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());
    private static final GroupPlanner.Constraints DIRECT_CONSTRAINTS =
            new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);

    /**
     * Non-owning index containing only ACTIVE RouteResult identities.
     */
    private PrefillRequestQueue queuedRequests = OrderedPrefillRequestQueue.disabled();

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
    private final PrefillRequestQueue.Factory workerQueues;
    public final DispatcherConfig.Type deliveryMode;

    private ProjectionSource projectionSource;

    private final DeliveryMetricsReporter reporter;
    private final CapacityNotifier capacityNotifier;

    PrefillEndpoint(
            WorkerStatus status,
            FlexlbConfig config,
            DispatcherConfig.Type deliveryMode,
            DeliveryMetricsReporter reporter,
            CapacityNotifier capacityNotifier,
            QueueExecutionSettings queueSettings,
            PrefillRequestQueue.Factory workerQueues) {
        super(status);
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.capacityNotifier =
                java.util.Objects.requireNonNull(capacityNotifier, "capacityNotifier");
        var estimator = config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator();
        this.predictor =
                estimator.getType() == RoutingConfig.EstimatorType.LEARNING
                        ? new LearningPredictor()
                        : new FormulaPredictor(estimator.getExpression());
        this.workerQueues = java.util.Objects.requireNonNull(workerQueues);
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
        ProjectionSource source;
        RequestTimeEstimation.Inputs observedInputs;
        stateLock().lock();
        try {
            source = projectionSource;
            if (source == null
                    || source.requestStateVersion != requestStateVersion
                    || source.schedulingInputVersion != schedulingInputVersion) {
                source = new ProjectionSource();
                projectionSource = source;
            }
            observedInputs = source.materialized;
        } finally {
            stateLock().unlock();
        }
        // Materialization only publishes to this captured source, never to a newer one.
        var evaluator = predictor == null ? null : predictor.evaluator();
        return source.materialize(evaluator, observedInputs);
    }

    /**
     * Publishes a route before queue readers can consume it. Request cancellation uses the
     * same request lock; notifications and resource rollback run after both locks are released.
     */
    public boolean acceptQueuedRoute(
            RouteResult route, QueueExecutionSettings settings, RoutingContext routingContext) {
        var context = route.ctx();
        synchronized (context) {
            stateLock().lock();
            try {
                if (!enqueueRoute(route, settings)) return false;
                boolean accepted = false;
                try {
                    accepted = context.publishRoute(routingContext, route);
                    return accepted;
                } finally {
                    if (!accepted) undoEnqueueLocked(route);
                }
            } finally {
                stateLock().unlock();
            }
        }
    }

    /**
     * Commit queue membership without notifying consumers. False means no member was added;
     * queue validation failures leave membership unchanged.
     */
    public boolean enqueueRoute(RouteResult route, QueueExecutionSettings settings) {
        enableQueue(settings);
        return tryEnqueueRoute(route);
    }

    /**
     * Enable exact queue membership; only the Scheduler owns decisions and execution tasks.
     */
    public void enableQueue(QueueExecutionSettings settings) {
        stateLock().lock();
        try {
            checkArgument(
                    settings != null && settings.equals(queueSettings),
                    "queue rules must match endpoint configuration");
            if (queueEnabled) return;
            var queue = workerQueues.create();
            checkArgument(queue.enabled(), "worker queue must support admission");
            queuedRequests = queue;
            schedulingInputVersion++;
            requestStateVersion++;
            queueEnabled = true;
        } finally {
            stateLock().unlock();
        }
    }

    public boolean tryEnqueueRoute(RouteResult route) {
        checkArgument(
                route.prefillEndpoint() == this, "request belongs to another Prefill generation");
        stateLock().lock();
        try {
            return addQueuedLocked(route) != null;
        } finally {
            stateLock().unlock();
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

    @Override
    public StatusChange applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        return reduceStatus(ws, prepared.observation(), prepared);
    }

    @Override
    public void initializeFromPreparedStatus(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        StatusChange.Prefill reconciliation = reduceStatus(ws, observation, null);
        checkState(
                reconciliation.requestStatuses().isEmpty()
                        && reconciliation.batchCompletions().isEmpty(),
                "Private Prefill candidate produced locally-owned request statuses");
    }

    @Override
    public StatusChange applyStatusHeartbeat(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        stateLock().lock();
        try {
            if (isRetired()) return NO_STATUS_CHANGE;
            var reconciliation =
                    commitStatusLocked(prepareStatusLocked(observation, false), Map.of());
            if (reconciliation.schedulingInputsChanged()) {
                schedulingInputVersion++;
            }
            return reconciliation;
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
        }
        if (reconciliation.schedulingInputsChanged()) {
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
     * One orphan pass: a retained member protects its whole batch. Counts batches and individuals.
     */
    public int evictExpiredInflight(long ttlMs, Set<RouteResult> orphanCandidates) {
        stateLock().lock();
        try {
            int evicted = 0;
            long nowMs = clock.getAsLong();
            long ttl = Math.max(0L, ttlMs);
            for (BatchRecord batch : List.copyOf(batches.values())) {
                Set<RequestRecord> members = batch.members;
                boolean retained = nowMs - batch.lastObservedAtMs < ttl;
                for (RequestRecord entry : members) {
                    retained |= !orphanCandidates.contains(entry.route);
                }
                if (retained) {
                    continue;
                }
                for (RequestRecord entry : List.copyOf(members)) {
                    removeRequestRecordLocked(entry);
                }
                evicted++;
            }
            List<RequestRecord> individuals = new ArrayList<>();
            for (RequestRecord entry : requests.values()) {
                if (entry.batch == null
                        && entry.ownership == OwnershipStage.COMMITTED
                        && nowMs - entry.phaseBaseMs >= ttl
                        && orphanCandidates.contains(entry.route)) {
                    individuals.add(entry);
                }
            }
            for (RequestRecord entry : individuals) {
                removeRequestRecordLocked(entry);
            }
            evicted += individuals.size();
            return evicted;
        } finally {
            stateLock().unlock();
        }
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
     * Exact failure-path capacity accounting; queue priority ordering stays in the queue.
     */
    public AdmissionSummary admissionSummary(int priority, long requestLimit) {
        stateLock().lock();
        try {
            return admissionSummaryLocked(priority, requestLimit);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Validate the queue prefix and capacity, then dequeue and register its work under one lock.
     * The result records either local capacity or the rejection reason; the caller starts delivery.
     */
    public CapacityBoundary.Attempt<WorkSnapshot> dequeueForDelivery(DeliverySelection selection) {
        stateLock().lock();
        try {
            List<RouteResult> items = selection.items();
            long now = clock.getAsLong();
            boolean batched = selection.batch();
            if (!acceptsNewRequests())
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            if (batched && batches.size() >= selection.maxInflightBatches())
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH));
            if (!batched && !requestCapacityAvailableLocked(requestLimit, items.size()))
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
            if (batched)
                checkState(!batches.containsKey(selection.batchId()), "batch id already committed");
            var work = captureWorkLocked(now, true, null);
            var batch = batched ? new BatchRecord(selection.batchId()) : null;
            if (batch != null) {
                batch.members = members;
                batch.originalPredictionMs = selection.predictedMs();
                batch.remainingWorkMs = selection.predictedMs();
                batch.originalFeatures = selection.features();
                batch.phaseBaseMs = now;
                batch.lastObservedAtMs = now;
                batches.put(batch.batchId, batch);
            }
            int index = 0;
            for (var member : members) {
                queuedRequests.remove(member.route);
                member.commit(
                        batch,
                        batch == null ? selection.predictions[index++] : selection.predictedMs(),
                        now);
                inflightRequests++;
            }
            committedWorkSnapshot = null;
            requestStateVersion++;
            return CapacityBoundary.Attempt.accepted(work);
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
     * Checks worker availability, queue admission and the exact request record before inserting it.
     */
    RouteResult addQueuedLocked(RouteResult route) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        Objects.requireNonNull(route, "route");
        if (!acceptsNewRequests() || queueStopped()) return null;
        var entry = requests.get(route.requestId());
        if (entry == null
                || entry != route.prefillReservation()
                || entry.route != route
                || entry.ownership != OwnershipStage.WAITING
                        && entry.ownership != OwnershipStage.SELECTED
                || queuedRequests.contains(route)) return null;
        if (!queuedRequests.add(route)) return null;
        if (entry.ownership == OwnershipStage.SELECTED && queueSettings == null)
            committedWorkSnapshot = null;
        entry.ownership = OwnershipStage.WAITING;
        requestStateVersion++;
        return route;
    }

    /**
     * Capture exact activity into the same transaction that will commit it.
     */
    /**
     * Null means prediction inputs changed between events; no record has changed.
     */
    StatusChange.Prefill commitStatusLocked(
            StatusReduction reduction, Map<Long, Long> predictions) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        checkArgument(reduction.owner == this, "Status reduction belongs to another endpoint");
        if (reduction.version != requestStateVersion) {
            return null;
        }
        checkArgument(
                predictions.keySet().equals(reduction.predictionInputs.keySet()),
                "Predictions do not match the prepared batches");
        for (long prediction : predictions.values()) {
            checkArgument(prediction >= 0L, "Negative batch prediction");
        }
        // The version check also validates the capacity facts captured in the prepared result.
        for (var change : reduction.batchOutcomes.entrySet()) {
            BatchRecord batch = change.getKey();
            batch.outcome = change.getValue();
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
        }
        for (TerminalObservation terminal : reduction.terminals.values()) {
            removeRequestRecordLocked(terminal.owner);
        }
        for (var phase : reduction.individualPhases.entrySet()) {
            var entry = phase.getKey();
            entry.remainingWorkMs =
                    remainingWorkAt(
                            entry.individualPhase,
                            entry.remainingWorkMs,
                            entry.phaseBaseMs,
                            reduction.nowMs);
            entry.phaseBaseMs = Math.max(entry.phaseBaseMs, reduction.nowMs);
            entry.individualPhase = phase.getValue();
        }
        for (var phase : reduction.batchPhases.entrySet()) {
            var batch = phase.getKey();
            batch.remainingWorkMs =
                    remainingWorkAt(
                            batch.servicePhase,
                            batch.remainingWorkMs,
                            batch.phaseBaseMs,
                            reduction.nowMs);
            batch.phaseBaseMs = Math.max(batch.phaseBaseMs, reduction.nowMs);
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
            batch.outcome.executionStarted |= phase.getValue() == Phase.ENGINE_RUNNING;
            batch.servicePhase = phase.getValue();
        }
        unknownEngineRequestCount = reduction.unknownRequests;
        for (var prediction : predictions.entrySet()) {
            BatchRecord batch = batches.get(prediction.getKey());
            batch.remainingWorkMs = prediction.getValue();
            batch.phaseBaseMs = reduction.nowMs;
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
        }
        if (reduction.result.schedulingInputsChanged()) {
            committedWorkSnapshot = null;
            requestStateVersion++;
        }
        return reduction.result;
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

    StatusReduction prepareStatusLocked(WorkerStatus.StatusObservation observation, boolean full) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill observation requires an Endpoint lock");
        long nowMs = full ? clock.getAsLong() : 0L;
        var reduction =
                new StatusReduction(this, full ? terminals(observation.finishedTasks()) : Map.of());
        List<PrefillRequestStatus> statuses =
                new ArrayList<>(
                        reduction.terminals.size() + observation.engine().runningTaskList().size());
        for (var terminal : reduction.terminals.values()) {
            var kind =
                    terminal.errorCode() == 0L
                            ? PrefillRequestStatus.Kind.COMPLETED
                            : terminal.preemptionProgress() == PriorityPreemptionProgress.CANCELED
                                            && terminal.errorCode()
                                                    == StrategyErrorType.PRIORITY_PREEMPTED
                                                            .getErrorCode()
                                    ? PrefillRequestStatus.Kind.PRIORITY_CANCELED
                                    : PrefillRequestStatus.Kind.FAILED;
            statuses.add(
                    new PrefillRequestStatus(terminal.owner().route, kind, terminal.errorCode()));
            var batch = terminal.owner().batch;
            if (batch == null) {
                continue;
            }
            var outcome = reduction.batchOutcomes.get(batch);
            if (outcome == null) {
                outcome = new BatchOutcome();
                outcome.maxExecutionTimeMs = batch.outcome.maxExecutionTimeMs;
                outcome.successfulCompletion = batch.outcome.successfulCompletion;
                outcome.learningEligible = batch.outcome.learningEligible;
                outcome.executionStarted = batch.outcome.executionStarted;
                reduction.batchOutcomes.put(batch, outcome);
            }
            boolean success = terminal.errorCode() == 0L;
            outcome.executionStarted |= success || terminal.executionTimeMs() > 0L;
            outcome.maxExecutionTimeMs =
                    Math.max(outcome.maxExecutionTimeMs, terminal.executionTimeMs());
            outcome.successfulCompletion |= success;
            outcome.learningEligible &= success;
        }
        observeActive(observation.engine(), reduction, statuses);
        boolean released =
                !reduction.terminals.isEmpty()
                        || reduction.unknownRequests < unknownEngineRequestCount;
        reduction.nowMs = full ? nowMs : clock.getAsLong();
        Map<Long, List<RouteResult>> inputs =
                reduction.batchOutcomes.isEmpty() ? Map.of() : new HashMap<>();
        List<BatchCompletion> completions =
                reduction.batchOutcomes.isEmpty() ? List.of() : new ArrayList<>();
        for (var change : reduction.batchOutcomes.entrySet()) {
            var batch = change.getKey();
            var outcome = change.getValue();
            List<RouteResult> survivors = new ArrayList<>(batch.members.size());
            for (var member : batch.members) {
                if (!reduction.terminals.containsKey(member.route.requestId())) {
                    survivors.add(member.route);
                }
            }
            if (survivors.isEmpty()) {
                completions.add(
                        new BatchCompletion(
                                batch.batchId,
                                batch.originalFeatures,
                                batch.originalPredictionMs,
                                outcome.maxExecutionTimeMs,
                                outcome.successfulCompletion,
                                outcome.learningEligible));
            } else if (!outcome.executionStarted
                    && reduction.batchPhases.get(batch) != Phase.ENGINE_RUNNING) {
                survivors.sort(
                        Comparator.comparingLong(RouteResult::enqueueSeq)
                                .thenComparing(RouteResult::requestId));
                inputs.put(batch.batchId, List.copyOf(survivors));
            }
        }
        boolean changed = reduction.unknownRequests != unknownEngineRequestCount;
        for (var phase : reduction.individualPhases.entrySet()) {
            changed |= phase.getKey().individualPhase != phase.getValue();
        }
        for (var phase : reduction.batchPhases.entrySet()) {
            changed |= phase.getKey().servicePhase != phase.getValue();
        }
        reduction.version = requestStateVersion;
        reduction.predictionInputs = Map.copyOf(inputs);
        reduction.result =
                new StatusChange.Prefill(
                        statuses,
                        completions,
                        changed || !reduction.terminals.isEmpty() || !inputs.isEmpty(),
                        released);
        return reduction;
    }

    /**
     * Undo queue membership after rejected route publication; the route's owner releases its record.
     */
    private boolean undoEnqueueLocked(RouteResult route) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        Objects.requireNonNull(route, "route");
        var entry = requests.get(route.requestId());
        if (entry == null
                || entry.route != route
                || entry.ownership != OwnershipStage.WAITING
                || !queuedRequests.contains(route)) return false;
        checkState(queuedRequests.remove(route), "queue member disappeared during Endpoint lock");
        requestStateVersion++;
        return true;
    }

    private WorkSnapshot captureWorkLocked(long nowMs, boolean cached, RequestRecord excluded) {
        if (cached
                && committedWorkSnapshot != null
                && committedWorkSnapshot.capturedAtMs() <= nowMs) {
            return committedWorkSnapshot;
        }
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RequestRecord entry : requests.values()) {
            if (entry == excluded
                    || entry.batch != null
                    || entry.ownership == OwnershipStage.WAITING
                    || entry.ownership == OwnershipStage.SELECTED && queueSettings != null) {
                continue;
            }
            Phase phase =
                    entry.ownership == OwnershipStage.COMMITTED
                            ? entry.individualPhase
                            : Phase.COMMITTED;
            individual.add(
                    new WorkSnapshot.RequestWork(
                            entry.planningInput.requestId(),
                            phase,
                            remainingWorkAt(
                                    phase, entry.remainingWorkMs, entry.phaseBaseMs, nowMs)));
        }
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
                                OptionalLong.of(
                                        remainingWorkAt(
                                                batch.servicePhase,
                                                batch.remainingWorkMs,
                                                batch.phaseBaseMs,
                                                nowMs))));
            }
        }
        var capture =
                new WorkSnapshot(nowMs, individual, capturedBatches, unknownEngineRequestCount);
        if (cached) {
            committedWorkSnapshot = capture;
        }
        return capture;
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
     * Reduce and publish status in one Endpoint lock; predict a shrunk batch outside it.
     */
    private StatusChange.Prefill reduceStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation,
            WorkerStatus.PreparedStatus prepared) {
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        Map<Long, List<RouteResult>> predictedInputs = Map.of();
        Map<Long, Long> predictions = Map.of();
        while (true) {
            var expectedInputs = predictedInputs;
            var expectedPredictions = predictions;
            Map<Long, List<RouteResult>> inputs;
            stateLock().lock();
            try {
                checkState(!isRetired(), "Prefill endpoint is retired");
                try {
                    var reduction = prepareStatusLocked(observation, true);
                    inputs = reduction.predictionInputs();
                    // Always capture current resource facts. Reuse
                    // prediction only for the same
                    // members.
                    if (inputs.isEmpty() || inputs.equals(expectedInputs)) {
                        var result =
                                commitStatusLocked(
                                        reduction,
                                        inputs.isEmpty() ? Map.of() : expectedPredictions);
                        if (prepared != null && !observation.alive()) beginRetirement();
                        boolean inputsChanged =
                                result.schedulingInputsChanged()
                                        || prepared != null
                                                && QueueExecutionSettings.capacityChanged(
                                                        ws.committedEngineObservation(),
                                                        observation.engine());
                        if (prepared != null) ws.publishReducedStatus(prepared);
                        if (inputsChanged) {
                            schedulingInputVersion++;
                            if (!result.schedulingInputsChanged()) {
                                result =
                                        new StatusChange.Prefill(
                                                result.requestStatuses(),
                                                result.batchCompletions(),
                                                true,
                                                result.capacityReleased());
                            }
                        }
                        return result;
                    }
                } catch (Throwable failure) {
                    // Reject later admissions if status reduction fails.
                    beginRetirement();
                    throw Failures.propagate(
                            failure, "Prefill status reduction/publication failed");
                }
            } finally {
                stateLock().unlock();
            }
            try {
                predictions = new HashMap<>();
                for (var batch : inputs.entrySet()) {
                    var features =
                            PrefillBatchFeatures.from(
                                    batch.getValue(),
                                    item -> Math.max(0L, item.seqLen()),
                                    item ->
                                            Math.clamp(
                                                    item.hitCache(),
                                                    0L,
                                                    Math.max(0L, item.seqLen())));
                    predictions.put(
                            batch.getKey(),
                            predictRemainingBatch(features, batch.getValue().size()));
                }
                predictedInputs = inputs;
            } catch (Throwable failure) {
                beginRetirement();
                throw Failures.propagate(failure, "Prefill status prediction failed");
            }
        }
    }

    /**
     * Model failures use the existing default formula; callers run prediction outside the Endpoint state lock.
     */
    private long predictRemainingBatch(PrefillBatchFeatures features, int survivingRequests) {
        try {
            return PrefillTimePredictor.predictCommittedBatchMs(predictor.evaluator(), features);
        } catch (RuntimeException predictionFailure) {
            try {
                logger.error(
                        "Prefill batch repack prediction failed; using default formula "
                                + "engine={} surviving_requests={}",
                        getIp(),
                        survivingRequests,
                        predictionFailure);
            } catch (RuntimeException ignoredLoggingFailure) {
                // Prediction logging cannot block membership settlement.
            }
            return PrefillTimePredictor.predictCommittedBatchMs(DEFAULT_BATCH_PREDICTOR, features);
        }
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

    private boolean requestCapacityAvailableLocked(long maximum, int requested) {
        long inFlight = saturatedAdd(inflightRequests, unknownEngineRequestCount);
        return maximum == 0L || requested <= maximum && inFlight <= maximum - requested;
    }

    private AdmissionSummary admissionSummaryLocked(int priority, long requestLimit) {
        long lower = 0, same = 0, higher = 0, unknown = unknownEngineRequestCount;
        if (!requests.isEmpty()) {
            for (RequestRecord entry : requests.values()) {
                int occupant = entry.planningInput.priority();
                if (!PriorityNormalizer.isValid(occupant)) unknown++;
                else if (occupant < priority) lower++;
                else if (occupant == priority) same++;
                else higher++;
            }
        }
        long outstandingRequestCount = saturatedAdd(requests.size(), unknownEngineRequestCount);
        return new AdmissionSummary(
                outstandingRequestCount,
                requestLimit > 0 ? Math.max(0L, outstandingRequestCount - requestLimit + 1L) : 0L,
                lower,
                same,
                higher,
                unknown);
    }

    private Map<String, TerminalObservation> terminals(
            Map<String, WorkerStatus.TaskObservation> tasks) {
        Map<String, TerminalObservation> facts = new HashMap<>();
        for (var task : tasks.values()) {
            var entry = requests.get(task.requestId());
            if (entry == null
                    || entry.ownership != OwnershipStage.COMMITTED
                    || entry.batch != null
                            && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) {
                continue;
            }
            var next =
                    new TerminalObservation(
                            entry,
                            task.executionTimeMs(),
                            task.errorCode(),
                            task.priorityPreemptionProgress());
            facts.merge(
                    task.requestId(),
                    next,
                    (first, other) ->
                            new TerminalObservation(
                                    first.owner(),
                                    Math.max(first.executionTimeMs(), other.executionTimeMs()),
                                    first.errorCode() != 0L ? first.errorCode() : other.errorCode(),
                                    PriorityPreemptionProgress.merge(
                                            first.preemptionProgress(),
                                            other.preemptionProgress())));
        }
        return facts;
    }

    private void observeActive(
            WorkerStatus.EngineObservation engine,
            StatusReduction reduction,
            List<PrefillRequestStatus> statuses) {
        Set<String> unknown = new HashSet<>(), known = new HashSet<>();
        for (var task : engine.runningTaskList().values()) {
            if (reduction.terminals.containsKey(task.requestId())) {
                continue;
            }
            var entry = requests.get(task.requestId());
            if (entry == null
                    || entry.batch != null
                            && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) {
                if (!task.isPriorityCancelOverlayOnly()) {
                    unknown.add(task.requestId());
                }
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
            if (entry.ownership != OwnershipStage.COMMITTED) {
                continue;
            }
            Phase phase =
                    task.phase() == TaskPhase.RUNNING ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
            if (entry.batch == null) {
                reduction.individualPhases.put(entry, phase);
            } else {
                reduction.batchPhases.merge(
                        entry.batch,
                        phase,
                        (left, right) ->
                                left == Phase.ENGINE_RUNNING || right == Phase.ENGINE_RUNNING
                                        ? Phase.ENGINE_RUNNING
                                        : Phase.ENGINE_QUEUED);
            }
        }
        long total =
                saturatedAdd(
                        Math.max(0L, engine.waitingQueryLen()),
                        Math.max(0L, engine.runningQueryLen()));
        reduction.unknownRequests = Math.max(unknown.size(), Math.max(0L, total - known.size()));
    }

    /**
     * Captured in an Endpoint lock; expensive projection work runs afterward.
     */
    private final class ProjectionSource {
        private final long requestStateVersion;
        private final long schedulingInputVersion;
        private final long capturedAtMs;
        private PrefillRequestQueue.Snapshot active;
        private WorkSnapshot work;
        private List<GroupPlanner.Item> selected;
        private final GroupPlanner.Constraints constraints;
        private final boolean queueEnabled;
        private final GroupPlanner grouping;
        private volatile RequestTimeEstimation.Inputs materialized;

        private ProjectionSource() {
            this.requestStateVersion = PrefillEndpoint.this.requestStateVersion;
            this.schedulingInputVersion = PrefillEndpoint.this.schedulingInputVersion;
            this.capturedAtMs = clock.getAsLong();
            this.active = queuedRequests.capture();
            this.selected =
                    queueSettings == null
                            ? List.of()
                            : requests.values().stream()
                                    .filter(r -> r.ownership == OwnershipStage.SELECTED)
                                    .map(r -> r.planningInput)
                                    .toList();
            this.work = captureWorkLocked(capturedAtMs, true, null);
            this.constraints =
                    queueSettings == null
                            ? DIRECT_CONSTRAINTS
                            : queueSettings.constraints(getStatus().committedEngineObservation());
            this.queueEnabled = queueSettings != null;
            this.grouping = queueSettings == null ? null : queueSettings.grouping();
        }

        private RequestTimeEstimation.Inputs materialize(
                PrefillTimePredictor.Evaluator evaluator, RequestTimeEstimation.Inputs observed) {
            RequestTimeEstimation.Inputs result = materialized;
            if (result != null && result.evaluator() == evaluator) {
                return result;
            }
            synchronized (this) {
                result = materialized;
                if (result != null && result.evaluator() == evaluator) {
                    return result;
                }
                var queueSnapshot =
                        result == null
                                ? new RequestTimeEstimation.ProjectionQueueSnapshot(
                                        capturedAtMs,
                                        queueEnabled,
                                        grouping,
                                        QueueExecutionSettings.queueOrder(),
                                        constraints,
                                        (selected.isEmpty()
                                                        ? active.items().stream()
                                                                .map(RouteResult::planningInput)
                                                        : java.util.stream.Stream.concat(
                                                                active.items().stream()
                                                                        .map(
                                                                                RouteResult
                                                                                        ::planningInput),
                                                                selected.stream()))
                                                .toList())
                                : result.queue();
                var updated =
                        new RequestTimeEstimation.Inputs(
                                queueSnapshot,
                                result == null ? work.calculate() : result.work(),
                                new RequestTimeEstimation.SelectionVersion(
                                        getStatus().getGenerationId(),
                                        requestStateVersion,
                                        schedulingInputVersion),
                                evaluator,
                                deliveryMode);
                // A delayed model capture must not replace concurrently published inputs.
                if (result == observed) {
                    materialized = updated;
                    // The cached projection must not retain completed request contexts.
                    active = null;
                    selected = null;
                    work = null;
                }
                return updated;
            }
        }
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
        Set<RequestRecord> members = Set.of();
        long originalPredictionMs;
        PrefillBatchFeatures originalFeatures;
        long remainingWorkMs;
        Phase servicePhase = Phase.COMMITTED;
        BatchOutcome outcome = new BatchOutcome();
        long phaseBaseMs;
        long lastObservedAtMs;

        private BatchRecord(long batchId) {
            checkArgument(batchId >= 0L, "batchId must be non-negative");
            this.batchId = batchId;
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
        Phase individualPhase;

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
                phaseBaseMs = nowMs;
                individualPhase = Phase.COMMITTED;
            } else {
                batch = committedBatch;
            }
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
     * Facts prepared in the Endpoint state lock, then read during prediction and exact-version commit.
     */
    public static final class StatusReduction {
        final PrefillEndpoint owner;
        long version;
        final Map<String, TerminalObservation> terminals;
        final Map<RequestRecord, Phase> individualPhases = new IdentityHashMap<>();
        final Map<BatchRecord, Phase> batchPhases = new IdentityHashMap<>();
        final Map<BatchRecord, BatchOutcome> batchOutcomes;
        // Assigned before publication; only this Endpoint applies the facts in its state lock.
        long nowMs;
        long unknownRequests;
        Map<Long, List<RouteResult>> predictionInputs;
        StatusChange.Prefill result;

        StatusReduction(PrefillEndpoint owner, Map<String, TerminalObservation> terminals) {
            this.owner = owner;
            this.terminals = terminals;
            this.batchOutcomes = terminals.isEmpty() ? Map.of() : new IdentityHashMap<>();
        }

        public Map<Long, List<RouteResult>> predictionInputs() {
            return predictionInputs;
        }
    }

    /**
     * Owned by one batch or by an uncommitted status reduction, never both while preparing.
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

    /**
     * Exact queue prefix and its precomputed work. Batch predictions apply to the whole batch;
     * route predictions correspond to each item. Inputs do not own resources or delivery state.
     */
    public record DeliverySelection(
            List<RouteResult> items,
            boolean batch,
            long batchId,
            int maxInflightBatches,
            long predictedMs,
            PrefillBatchFeatures features,
            long[] predictions) {
        public DeliverySelection {
            items = List.copyOf(items);
            predictions = predictions == null ? null : predictions.clone();
            checkArgument(!items.isEmpty(), "delivery selection must not be empty");
            if (batch) {
                checkArgument(maxInflightBatches > 0, "maximumInflightBatches must be positive");
                Objects.requireNonNull(features, "batch features");
                checkArgument(predictedMs >= 0L, "predicted batch work must be non-negative");
            } else {
                checkArgument(
                        predictions != null && predictions.length >= items.size(),
                        "missing predictions");
            }
        }

        @Override
        public long[] predictions() {
            return predictions == null ? null : predictions.clone();
        }
    }
}

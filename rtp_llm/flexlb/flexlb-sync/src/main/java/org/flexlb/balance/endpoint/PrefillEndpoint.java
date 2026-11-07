package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.ProjectionQueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RouteExecution;
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
 * Owns the exact Prefill resources and queue membership for one worker generation.
 */
public class PrefillEndpoint extends WorkerEndpoint implements EndpointStats.Prefill {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR =
            new FormulaPredictor(new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());
    private static final GroupPlanner.Constraints DIRECT_CONSTRAINTS =
            new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);

    /**
     * Non-owning index containing only ACTIVE RouteExecution identities.
     */
    private PrefillRequestQueue activeIndex = OrderedPrefillRequestQueue.disabled();

    /**
     * Canonical request records, changed only in this Endpoint state lock.
     */
    private final Map<String, RequestRecord> requests = new HashMap<>();

    private final LongSupplier clock = System::currentTimeMillis;

    /**
     * Occupied delivery slots, updated with their request records in the same event.
     */
    private int inflightRequests;

    private static final Progress EMPTY_PROGRESS = new Progress(0L, 0L, 0L, 0);
    private volatile Progress progress = EMPTY_PROGRESS;

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
    private volatile java.util.function.Supplier<AdmissionBlock> admissionBlock = () -> null;
    private volatile java.util.function.Supplier<Map<String, Object>> queueWait =
            () -> Map.of("cause", "waiting for Prefill decision");
    // Changed in Endpoint lock; settings are fixed before execution is enabled.
    private QueueExecutionSettings queueSettings;
    private final PrefillRequestQueue.Factory workerQueues;
    private final DispatcherConfig.Type dispatcherType;
    public final DispatcherConfig.Type deliveryMode;
    private volatile ProjectionSource projectionSource;

    private final DeliveryMetricsReporter reporter;

    private final PlacementAvailability placementAvailability;

    PrefillEndpoint(
            WorkerStatus status,
            FlexlbConfig config,
            DispatcherConfig.Type deliveryMode,
            DeliveryMetricsReporter reporter,
            PlacementAvailability placementAvailability,
            GroupPlanner grouping,
            PrefillRequestQueue.Factory workerQueues) {
        super(status);
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.placementAvailability =
                java.util.Objects.requireNonNull(placementAvailability, "placementAvailability");
        var estimator = config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator();
        this.predictor =
                estimator.getType() == RoutingConfig.EstimatorType.LEARNING
                        ? new LearningPredictor()
                        : new FormulaPredictor(estimator.getExpression());
        this.workerQueues = java.util.Objects.requireNonNull(workerQueues);
        this.queueSettings =
                config.isQueue() ? QueueExecutionSettings.capture(config, grouping) : null;
        this.requestLimit =
                config.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH
                        ? config.getDispatcher().getMaxInflightPerPrefillWorker()
                        : 0L;
        this.dispatcherType = config.getDispatcher().getType();
        this.deliveryMode = deliveryMode;
    }

    /**
     * The selected Endpoint checks the request record and retirement state before committing.
     */
    public WorkSnapshot commitDirect(RouteExecution route, RequestRecord reservation) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return null;
            var entry = requests.get(route.requestId());
            checkState(
                    entry != null
                            && entry.route == route
                            && entry.ownership == OwnershipStage.RESERVED,
                    "DIRECT member is not canonical");
            checkArgument(reservation.owner == this, "preparation belongs to another ledger");
            checkState(entry == reservation, "DIRECT commit does not own exact preparation");
            checkState(!activeIndex.contains(route), "DIRECT member has queue ownership");
            long now = clock.getAsLong();
            var preceding = captureWorkLocked(now, false, entry);
            entry.commit(null, entry.remainingWorkMs, now);
            committedWorkSnapshot = null;
            recordMutationLocked();
            return preceding;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Captured in an Endpoint lock; expensive projection work runs afterward.
     */
    private final class ProjectionSource {
        private final Progress version;
        private final long capturedAtMs;
        private PrefillRequestQueue.Snapshot active;
        private WorkSnapshot work;
        private final GroupPlanner.Constraints constraints;
        private final AdmissionBlock admissionBlock;
        private final boolean queueEnabled;
        private final GroupPlanner grouping;
        private final Comparator<PrefillRequestQueue.QueueItem> capturedOrder;
        private volatile RequestTimeEstimation.Inputs materialized;

        private ProjectionSource(AdmissionBlock admissionBlock) {
            checkState(
                    stateLock().isHeldByCurrentThread(),
                    "Prefill projection capture requires an Endpoint lock");
            this.version = progress;
            this.capturedAtMs = clock.getAsLong();
            this.active = activeIndex.capture();
            this.work = captureWorkLocked(capturedAtMs, true, null);
            this.constraints =
                    queueSettings == null
                            ? DIRECT_CONSTRAINTS
                            : queueSettings.constraints(getStatus().committedEngineObservation());
            this.admissionBlock = admissionBlock;
            this.queueEnabled = queueSettings != null;
            this.grouping = queueSettings == null ? null : queueSettings.grouping();
            this.capturedOrder = active.order();
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
                                        capturedOrder,
                                        constraints,
                                        active.items().stream()
                                                .map(RouteExecution::planningInput)
                                                .toList(),
                                        admissionBlock)
                                : result.queue();
                var updated =
                        new RequestTimeEstimation.Inputs(
                                queueSnapshot,
                                result == null ? work.calculate() : result.work(),
                                version.ownershipVersion(),
                                evaluator,
                                deliveryMode);
                // A delayed model capture must not replace concurrently published inputs.
                if (result == observed) {
                    materialized = updated;
                    // The cached projection must not retain completed request contexts.
                    active = null;
                    work = null;
                }
                return updated;
            }
        }
    }

    public RequestTimeEstimation.Inputs captureRouteProjectionInputs() {
        ProjectionSource observed = projectionSource;
        var observedInputs = observed == null ? null : observed.materialized;
        var evaluator = predictor == null ? null : predictor.evaluator();
        ProjectionSource source = observed;
        if (source == null || source.version != progress()) {
            for (; ; ) {
                var expected = progress();
                var block = admissionBlock.get();
                stateLock().lock();
                try {
                    var current = projectionSource;
                    if (current != null && current.version == progress()) {
                        source = current;
                        break;
                    }
                    // The probe ran outside the lock; only bind it to the version it observed.
                    if (progress() != expected) continue;
                    source = new ProjectionSource(block);
                    projectionSource = source;
                    break;
                } finally {
                    stateLock().unlock();
                }
            }
        } else if (observedInputs != null && observedInputs.evaluator() == evaluator) {
            return observedInputs;
        }
        // Concurrent callers share one source per ownership version. A late build only
        // fills its own source; it can never overwrite a newer capture.
        return source.materialize(evaluator, source == observed ? observedInputs : null);
    }

    /**
     * Enable exact queue membership; only the Scheduler owns decisions and execution tasks.
     */
    public void enableQueue(QueueExecutionSettings settings) {
        stateLock().lock();
        try {
            checkArgument(
                    settings != null && settings.dispatcherType() == dispatcherType,
                    "queued admission requires a compatible QUEUE configuration");
            checkArgument(
                    !queueEnabled || queueSettings.equals(settings), "queue rules already bound");
            if (queueEnabled) return;
            enableQueueLocked(
                    workerQueues.create(
                            QueueExecutionSettings.queueOrder(settings.priorityOrdering())));
            queueSettings = settings;
            projectionSource = null;
            queueEnabled = true;
        } finally {
            stateLock().unlock();
        }
    }

    public boolean queueEnabled() {
        return queueEnabled;
    }

    public boolean queueStopped() {
        return queueStopped;
    }

    /**
     * Read-only scheduling evidence used by prediction and timeout diagnostics.
     */
    public void observeQueueWait(
            java.util.function.Supplier<AdmissionBlock> block,
            java.util.function.Supplier<Map<String, Object>> diagnostics) {
        stateLock().lock();
        try {
            admissionBlock = java.util.Objects.requireNonNull(block);
            queueWait = java.util.Objects.requireNonNull(diagnostics);
            schedulingInputsChangedLocked();
        } finally {
            stateLock().unlock();
        }
    }

    public boolean enqueueRoute(RouteExecution route, QueueExecutionSettings settings) {
        enableQueue(settings);
        route.ctx().data().initializeExecutionOrder(System.currentTimeMillis());
        return tryEnqueueRoute(route);
    }

    public boolean tryEnqueueRoute(RouteExecution route) {
        checkArgument(
                route.prefillEndpoint() == this, "request belongs to another Prefill generation");
        boolean accepted;
        stateLock().lock();
        try {
            accepted = !queueStopped() && acceptsNewRequests() && addQueuedLocked(route) != null;
        } finally {
            stateLock().unlock();
        }
        if (accepted) signal(Signal.PLACEMENT);
        return accepted;
    }

    public enum Signal {
        CAPACITY,
        INPUT,
        PLACEMENT
    }

    /**
     * INPUT changes planning facts; CAPACITY also wakes placement; PLACEMENT only publishes that edge.
     */
    public void signal(Signal kind) {
        try {
            if (kind != Signal.PLACEMENT) {
                stateLock().lock();
                try {
                    this.schedulingInputsChangedLocked();
                } finally {
                    stateLock().unlock();
                }
            }
            placementAvailability.changed(this);
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

    public boolean removeQueued(RouteExecution route) {
        checkArgument(
                route.prefillEndpoint() == this, "request belongs to another Prefill generation");
        boolean removed;
        stateLock().lock();
        try {
            removed = !queueStopped() && removeQueuedLocked(route);
        } finally {
            stateLock().unlock();
        }
        if (removed) signal(Signal.CAPACITY);
        return removed;
    }

    public Map<String, Object> getLatestQueueWaitSnapshot() {
        return queueWait.get();
    }

    /**
     * Clears request records once and publishes retirement notifications outside the state lock.
     */
    @Override
    protected void closeEndpoint() {
        Throwable retirementFailure = null;
        try {
            Retirement retirement = retireResources();
            for (RouteExecution route : retirement.ownedItems()) {
                retirementFailure =
                        Failures.run(
                                retirementFailure,
                                () ->
                                        route.ctx()
                                                .data()
                                                .getRequestScheduler()
                                                .onEndpointRetired(
                                                        route.ctx(),
                                                        new ContextResult.EndpointRetirement(
                                                                route,
                                                                null,
                                                                getStatus().getRole(),
                                                                this,
                                                                "Prefill generation retired: "
                                                                        + ipPort(),
                                                                System.currentTimeMillis())));
            }
            retirementFailure =
                    reportBatchCompletions(retirement.batchCompletions(), retirementFailure);
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = Failures.append(retirementFailure, committedRetirementFailure);
        }
        signal(Signal.CAPACITY);
        Failures.rethrow(retirementFailure, "Prefill endpoint retirement failed");
    }

    public CapacityBoundary.Attempt<BatchRecord> reserveBatch(
            long batchId, int maximumInflightBatches) {
        checkArgument(maximumInflightBatches > 0, "maximumInflightBatches must be positive");
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            if (batches.containsKey(batchId)) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException(
                                        "Batch id already reserved: batch_id=" + batchId)));
            }
            if (batches.size() >= maximumInflightBatches) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH));
            }
            BatchRecord reservation = new BatchRecord(this, batchId);
            batches.put(batchId, reservation);
            recordMutationLocked();
            return CapacityBoundary.Attempt.accepted(reservation);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Admit on the selected generation using its current canonical occupancy
     * and bound dispatcher policy. The Endpoint reference identifies the generation;
     * its state lock checks retirement and occupies current capacity.
     */
    public CapacityBoundary.Attempt<RequestRecord> reserveUnqueuedRoute(
            RouteExecution route, long predictedMs) {
        var result = reserveUnqueuedRoute(route, predictedMs, requestLimit);
        if (result.accepted()) route.attachPrefillReservation(result.value());
        return result;
    }

    /**
     * Undo an uncommitted preparation; queued members remain eligible for another delivery.
     */
    public boolean rollback(BatchRecord batch) {
        checkArgument(batch.owner == this, "batch belongs to another endpoint");
        stateLock().lock();
        try {
            if (batch.committed || !batches.remove(batch.batchId, batch)) return false;
            recordMutationLocked();
        } finally {
            stateLock().unlock();
        }
        signal(Signal.CAPACITY);
        return true;
    }

    public boolean rollback(RequestRecord request) {
        checkArgument(request.owner == this, "request belongs to another endpoint");
        stateLock().lock();
        try {
            if (requests.get(request.route.requestId()) != request
                    || request.ownership != OwnershipStage.RESERVED) return false;
            checkState(removeEntry(request), "request record changed during rollback");
            committedWorkSnapshot = null;
            recordMutationLocked();
        } finally {
            stateLock().unlock();
        }
        signal(Signal.CAPACITY);
        return true;
    }

    /**
     * Remove only this exact request's resources, then publish capacity outside the Endpoint state lock.
     */
    public boolean releaseRequest(RouteExecution route) {
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
            checkState(removeEntry(entry), "exact release lost canonical ownership");
            recordMutationLocked();
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

    private record StatusStep(
            Map<Long, List<RouteExecution>> predictionInputs, StatusChange.Prefill result) {}

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
        Map<Long, List<RouteExecution>> predictedInputs = Map.of();
        Map<Long, Long> predictions = Map.of();
        while (true) {
            var expectedInputs = predictedInputs;
            var expectedPredictions = predictions;
            StatusStep step;
            stateLock().lock();
            try {
                checkState(!isRetired(), "Prefill endpoint is retired");
                try {
                    var reduction = prepareStatusLocked(observation, true);
                    var inputs = reduction.predictionInputs();
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
                            schedulingInputsChangedLocked();
                            if (!result.schedulingInputsChanged()) {
                                result =
                                        new StatusChange.Prefill(
                                                result.requestStatuses(),
                                                result.batchCompletions(),
                                                true,
                                                result.capacityReleased());
                            }
                        }
                        step = new StatusStep(Map.of(), result);
                    } else {
                        step = new StatusStep(inputs, null);
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
            if (step.result() != null) return step.result();
            var inputs = step.predictionInputs();
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
                schedulingInputsChangedLocked();
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
        var orphanCandidates = new java.util.HashSet<RouteExecution>();
        for (RouteExecution route : cleanupCandidates()) {
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
     * Current number of requests waiting for this worker.
     */
    public int queueDepth() {
        return progress().queueDepth();
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
     * Request status matched to its exact Prefill route, published after ledger reconciliation.
     */
    public record PrefillRequestStatus(RouteExecution route, Kind kind, long errorCode) {
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

    public record Retirement(
            List<RouteExecution> ownedItems, List<BatchCompletion> batchCompletions) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    /**
     * One batch reservation becomes its committed record; the delivery retains the exact identity.
     */
    public static final class BatchRecord {
        final PrefillEndpoint owner;
        final long batchId;
        boolean committed;
        Set<RequestRecord> members = Set.of();
        long originalPredictionMs;
        PrefillBatchFeatures originalFeatures;
        long remainingWorkMs;
        Phase servicePhase = Phase.COMMITTED;
        BatchOutcome outcome = new BatchOutcome();
        long phaseBaseMs;
        long lastObservedAtMs;

        private BatchRecord(PrefillEndpoint owner, long batchId) {
            this.owner = Objects.requireNonNull(owner);
            checkArgument(batchId >= 0L, "batchId must be non-negative");
            this.batchId = batchId;
        }
    }

    /**
     * Resource ownership within this Prefill generation, independent of request protocol.
     */
    enum OwnershipStage {
        WAITING,
        RESERVED,
        COMMITTED
    }

    /**
     * Canonical resource facts for one exact route. Changed only in Endpoint lock.
     */
    public static final class RequestRecord {
        final PrefillEndpoint owner;
        OwnershipStage ownership;
        final RouteExecution route;
        Phase individualPhase;
        long remainingWorkMs;
        long phaseBaseMs;
        BatchRecord batch;

        public boolean belongsTo(RouteExecution exact) {
            return route == exact;
        }

        private RequestRecord(
                PrefillEndpoint owner, RouteExecution route, OwnershipStage ownership) {
            this.owner = Objects.requireNonNull(owner);
            this.ownership = ownership;
            this.route = Objects.requireNonNull(route, "route");
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
     * Publish the capacity summary before readers observe a new ownership revision.
     */
    private void recordMutationLocked() {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        progress =
                new Progress(
                        activeIndex.version(),
                        progress.schedulingInputVersion(),
                        progress.ownershipVersion() + 1L,
                        activeIndex.size());
    }

    public record Progress(
            long queueVersion,
            long schedulingInputVersion,
            long ownershipVersion,
            int queueDepth) {}

    public Progress progress() {
        return progress;
    }

    /**
     * Enable waiting work without replacing the ledger of existing DIRECT reservations.
     */
    void enableQueueLocked(PrefillRequestQueue queue) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        if (!activeIndex.enabled()) {
            checkArgument(queue.enabled(), "worker queue must support admission");
            activeIndex = queue;
            schedulingInputsChangedLocked();
            recordMutationLocked();
        }
    }

    /**
     * One bounded, ordered queue view; all metadata belongs to the same ownership revision.
     */
    public record QueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<RouteExecution> items,
            RouteExecution head) {}

    public EndpointStats.QueueCounts queueCounters() {
        stateLock().lock();
        try {
            int[] byPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < byPriority.length; priority++) {
                byPriority[priority] = activeIndex.size(priority);
            }
            return new EndpointStats.QueueCounts(
                    activeIndex.version(),
                    byPriority,
                    saturatedAdd(requests.size(), unknownEngineRequestCount),
                    batches.size());
        } finally {
            stateLock().unlock();
        }
    }

    public QueueSnapshot captureQueue(int limit) {
        return captureQueue(limit, null, false);
    }

    /**
     * Queue policy supplies alternatives; the Endpoint captures them in one event.
     */
    public QueueSnapshot captureQueue(
            int limit, RouteExecution blockedHead, boolean priorityOrdering) {
        checkArgument(limit > 0, "queue capture limit must be positive");
        stateLock().lock();
        try {
            List<RouteExecution> items = new ArrayList<>(Math.min(limit, activeIndex.size()));
            var ordered = activeIndex.iterator();
            while (items.size() < limit && ordered.hasNext()) {
                items.add(ordered.next());
            }
            var head = items.isEmpty() ? null : items.getFirst();
            if (blockedHead != null) {
                items =
                        head == blockedHead
                                ? activeIndex.candidatesAfterBlockedHead(
                                        items, priorityOrdering, clock.getAsLong())
                                : List.of();
            }
            return new QueueSnapshot(
                    activeIndex.version(),
                    progress.schedulingInputVersion(),
                    List.copyOf(items),
                    head);
        } finally {
            stateLock().unlock();
        }
    }

    void schedulingInputsChangedLocked() {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        progress =
                new Progress(
                        activeIndex.version(),
                        progress.schedulingInputVersion() + 1L,
                        progress.ownershipVersion(),
                        activeIndex.size());
    }

    /**
     * Remove a known exact owner and its non-owning indexes; uncommitted batch records retain their reserved slots.
     */
    private boolean removeEntry(RequestRecord entry) {
        if (requests.get(entry.route.requestId()) != entry) {
            return false;
        }
        if (activeIndex.contains(entry.route)) {
            checkState(activeIndex.remove(entry.route), "exact queue member could not be removed");
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
        if (entry.ownership != OwnershipStage.WAITING) {
            checkState(inflightRequests > 0, "Prefill delivery slot underflow");
            inflightRequests--;
            committedWorkSnapshot = null;
        }
        return requests.remove(entry.route.requestId(), entry);
    }

    /**
     * Register exact queue membership; request settlement remains with its Lifecycle.
     */
    RouteExecution addQueuedLocked(RouteExecution route) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        Objects.requireNonNull(route, "route");
        if (!acceptsNewRequests() || queueStopped()) return null;
        var entry = requests.get(route.requestId());
        if (entry != null
                && (entry.route != route
                        || entry.ownership != OwnershipStage.WAITING
                        || activeIndex.contains(route))) return null;
        boolean created = entry == null;
        if (created) {
            entry = new RequestRecord(this, route, OwnershipStage.WAITING);
            requests.put(route.requestId(), entry);
        }
        try {
            if (!activeIndex.add(route)) {
                if (created) requests.remove(route.requestId(), entry);
                return null;
            }
            route.attachPrefillReservation(entry);
        } catch (RuntimeException | Error failure) {
            if (created) requests.remove(route.requestId(), entry);
            throw failure;
        }
        recordMutationLocked();
        return route;
    }

    /**
     * Exact deletion only; the scheduler decides expiry and notifies the request outside this event.
     */
    boolean removeQueuedLocked(RouteExecution route) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        Objects.requireNonNull(route, "route");
        var entry = requests.get(route.requestId());
        if (entry == null
                || entry.route != route
                || entry.ownership != OwnershipStage.WAITING
                || !activeIndex.contains(route)) return false;
        checkState(activeIndex.remove(route), "queue member disappeared during Endpoint lock");
        recordMutationLocked();
        return true;
    }

    /**
     * Remove exact queue membership; the scheduler publishes capacity and completes the request.
     */
    public boolean removeQueuedMember(RouteExecution route) {
        stateLock().lock();
        try {
            return removeQueuedLocked(route);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Remove a failed preparation only while its waiting request is still live.
     */
    public boolean removeFailedQueuedRequest(RouteExecution route) {
        stateLock().lock();
        try {
            return !queueStopped()
                    && !route.requestExpired(System.currentTimeMillis())
                    && removeQueuedLocked(route);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Check the observed head and queue versions in one critical section.
     * Unrelated arrivals do not invalidate a capacity wait unless the queue permits an alternative.
     */
    public boolean isQueueWaitCurrent(
            RouteExecution head,
            long queueVersion,
            long inputVersion,
            boolean capacityWait,
            long nowMs) {
        stateLock().lock();
        try {
            if (queueStopped()) return false;
            if (head == null) return activeIndex.peek() == null;
            if (activeIndex.peek() != head) return false;
            return capacityWait
                    ? !head.requestExpired(nowMs)
                            && (queueVersion == 0L
                                    || progress.queueVersion() == queueVersion
                                    || captureQueue(
                                                    Math.max(2, queueSettings.maxRequests()),
                                                    head,
                                                    queueSettings.priorityOrdering())
                                            .items()
                                            .isEmpty())
                    : progress.queueVersion() == queueVersion
                            && progress.schedulingInputVersion() == inputVersion;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Verify exact selected queue members before preparing delivery resources.
     */
    public boolean isSelectionCurrent(List<RouteExecution> items, long nowMs) {
        stateLock().lock();
        try {
            return !items.isEmpty() && ownsSelectionLocked(items, nowMs);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Invalidate captured scheduling inputs after a scheduler decision or capacity change.
     */
    public void schedulingInputsChanged() {
        stateLock().lock();
        try {
            schedulingInputsChangedLocked();
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Stop queue admission and detach membership; the caller owns request settlement.
     */
    public List<RouteExecution> closeQueue() {
        stateLock().lock();
        try {
            queueStopped = true;
            var removed = List.copyOf(activeIndex.capture().items());
            activeIndex.clear();
            if (!removed.isEmpty()) recordMutationLocked();
            return removed;
        } finally {
            stateLock().unlock();
        }
    }

    boolean ownsSelectionLocked(List<RouteExecution> items, long nowMs) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        for (RouteExecution item : items) {
            if (!activeIndex.contains(item) || item.requestExpired(nowMs)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Validate the exact observed head and selected member; ordering policy belongs to the queue.
     */
    boolean selectionCurrentLocked(RouteExecution head, RouteExecution candidate) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        var entry = requests.get(head.requestId());
        return activeIndex.peek() == head
                && entry != null
                && entry.route == head
                && activeIndex.contains(candidate)
                && !head.requestExpired(clock.getAsLong())
                && !candidate.requestExpired(clock.getAsLong());
    }

    /**
     * Read queue identity and request records in the same Endpoint lock.
     */
    RouteExecution queueHeadLocked() {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        return activeIndex.peek();
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

    private AdmissionSummary admissionSummaryLocked(int priority, long requestLimit) {
        long lower = 0, same = 0, higher = 0, unknown = unknownEngineRequestCount;
        if (!requests.isEmpty()) {
            for (RequestRecord entry : requests.values()) {
                int occupant = entry.route.priority();
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

    /**
     * Queued routes consume no delivery slots; reserved and committed work do.
     */
    public boolean requestCapacityAvailable(long maximum, int requested) {
        checkArgument(maximum >= 0L && requested > 0, "invalid request capacity");
        stateLock().lock();
        try {
            long inFlight = saturatedAdd(inflightRequests, unknownEngineRequestCount);
            return maximum == 0L || requested <= maximum && inFlight <= maximum - requested;
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
     * Recheck exact queued members, deadlines and capacity, then commit the delivery atomically.
     */
    public Delivery commitDelivery(Delivery delivery) {
        stateLock().lock();
        try {
            return commitLocked(delivery);
        } finally {
            stateLock().unlock();
        }
    }

    Delivery commitLocked(Delivery delivery) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        checkState(!delivery.committed(), "delivery already committed");
        List<RouteExecution> items = delivery.items();
        long selectedAt = delivery.selectionTimeMs();
        if (!ownsSelectionLocked(items, selectedAt)) return null;
        var batch = delivery.reservation();
        var predictions = delivery.predictions();
        if (batch == null && !requestCapacityAvailable(requestLimit, items.size())) return null;
        checkArgument(batch != null || predictions.length >= items.size(), "missing predictions");
        var members = new ArrayList<RequestRecord>(items.size());
        Set<RequestRecord> unique = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (var route : items) {
            var entry = requests.get(route.requestId());
            if (entry == null || entry.route != route || entry.ownership != OwnershipStage.WAITING)
                return null;
            checkState(unique.add(entry), "duplicate plan member");
            if (delivery.bypassedHead() != null
                    && !selectionCurrentLocked(delivery.bypassedHead(), route)) return null;
            members.add(entry);
        }
        if (batch != null) {
            checkState(
                    batch.owner == this && batches.get(batch.batchId) == batch,
                    "batch plan lost its exact reservation");
            checkState(!batch.committed, "batch has already been committed");
            Objects.requireNonNull(delivery.features(), "batch features");
            checkArgument(
                    delivery.predictedMs() >= 0L, "predicted batch work must be non-negative");
        }
        var failedRoute = delivery.failedMember();
        var failed = failedRoute == null ? null : requests.get(failedRoute.requestId());
        if (failed != null
                && (failed.route != failedRoute
                        || failed.ownership != OwnershipStage.WAITING
                        || failedRoute.requestExpired(selectedAt))) failed = null;
        if (failed != null) {
            checkState(!unique.contains(failed), "failed member belongs to committed selection");
            checkState(activeIndex.contains(failedRoute), "failed member lost queue index");
        }
        if (!acceptsNewRequests()) return null;
        {
            long now = clock.getAsLong();
            var work = captureWorkLocked(now, true, null);
            if (batch != null) {
                batch.members = unique;
                batch.originalPredictionMs = delivery.predictedMs();
                batch.remainingWorkMs = delivery.predictedMs();
                batch.originalFeatures = delivery.features();
                batch.phaseBaseMs = now;
                batch.lastObservedAtMs = now;
                batch.committed = true;
            }
            for (int i = 0; i < members.size(); i++) {
                var member = members.get(i);
                checkState(activeIndex.remove(member.route), "committed owner lost queue index");
                member.commit(batch, batch == null ? predictions[i] : delivery.predictedMs(), now);
                inflightRequests++;
            }
            if (failed != null)
                checkState(activeIndex.remove(failed.route), "failed member lost queue membership");
            activeIndex.recordDispatch(delivery.bypassedHead());
            committedWorkSnapshot = null;
            recordMutationLocked();
            delivery.committed(work, activeIndex.size(), failed != null);
            return delivery;
        }
    }

    /**
     * Reserve immediate route work against current ownership in one transaction.
     * Selection revisions are advisory; only exact identity and current capacity
     * decide admission. A zero request limit leaves count admission disabled.
     */
    public CapacityBoundary.Attempt<RequestRecord> reserveUnqueuedRoute(
            RouteExecution item, long predictedMs, long maxOutstandingRequests) {
        Objects.requireNonNull(item, "item");
        checkArgument(maxOutstandingRequests >= 0L, "request limit must be non-negative");
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            if (requests.containsKey(item.requestId())) {
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            }
            if (!requestCapacityAvailable(maxOutstandingRequests, 1)) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_REQUEST));
            }
            RequestRecord reservation = new RequestRecord(this, item, OwnershipStage.RESERVED);
            reservation.remainingWorkMs = Math.clamp(predictedMs, 0L, (long) Integer.MAX_VALUE);
            CapacityBoundary.Attempt<RequestRecord> result =
                    CapacityBoundary.Attempt.accepted(reservation);
            requests.put(item.requestId(), reservation);
            inflightRequests++;
            committedWorkSnapshot = null;
            recordMutationLocked();
            return result;
        } finally {
            stateLock().unlock();
        }
    }

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
        Map<Long, List<RouteExecution>> predictionInputs;
        StatusChange.Prefill result;

        StatusReduction(PrefillEndpoint owner, Map<String, TerminalObservation> terminals) {
            this.owner = owner;
            this.terminals = terminals;
            this.batchOutcomes = terminals.isEmpty() ? Map.of() : new IdentityHashMap<>();
        }

        public Map<Long, List<RouteExecution>> predictionInputs() {
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

    private static long remainingWorkAt(
            Phase phase, long remainingWorkMs, long phaseBaseMs, long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        long elapsedMs = Math.max(0L, nowMs - phaseBaseMs);
        return Math.max(0L, remainingWorkMs - elapsedMs);
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
        if (reduction.version != progress.ownershipVersion()) {
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
            removeEntry(terminal.owner);
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
            recordMutationLocked();
        }
        return reduction.result;
    }

    /**
     * Captures cleanup notifications and clears all records in one Endpoint lock.
     */
    Retirement retireResources() {
        List<RouteExecution> ownedItems = new ArrayList<>();
        List<BatchCompletion> completions = new ArrayList<>();
        stateLock().lock();
        try {
            for (var entry : requests.values()) ownedItems.add(entry.route);
            for (var batch : batches.values()) {
                if (batch.committed)
                    completions.add(
                            new BatchCompletion(
                                    batch.batchId,
                                    batch.originalFeatures,
                                    batch.originalPredictionMs,
                                    batch.outcome.maxExecutionTimeMs,
                                    batch.outcome.successfulCompletion,
                                    false));
            }
            var retiredProgress =
                    new Progress(
                            activeIndex.version() + (activeIndex.isEmpty() ? 0L : 1L),
                            progress.schedulingInputVersion(),
                            progress.ownershipVersion() + 1L,
                            0);
            for (BatchRecord batch : batches.values()) {
                if (!batch.members.isEmpty()) batch.members.clear();
            }
            queueStopped = true;
            requests.clear();
            batches.clear();
            activeIndex.clear();
            unknownEngineRequestCount = 0L;
            committedWorkSnapshot = null;
            inflightRequests = 0;
            progress = retiredProgress;
        } finally {
            stateLock().unlock();
        }
        ownedItems.sort(
                Comparator.comparingLong(RouteExecution::enqueueSeq)
                        .thenComparing(RouteExecution::requestId));
        completions.sort(Comparator.comparingLong(BatchCompletion::batchId));
        return new Retirement(ownedItems, completions);
    }

    /**
     * Capture exact committed identities before the workflow checks request ownership outside this event.
     */
    public List<RouteExecution> cleanupCandidates() {
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
    public int evictExpiredInflight(long ttlMs, Set<RouteExecution> orphanCandidates) {
        stateLock().lock();
        try {
            int evicted = 0;
            long nowMs = clock.getAsLong();
            long ttl = Math.max(0L, ttlMs);
            for (BatchRecord batch : List.copyOf(batches.values())) {
                if (!batch.committed) {
                    continue;
                }
                Set<RequestRecord> members = batch.members;
                boolean retained = nowMs - batch.lastObservedAtMs < ttl;
                for (RequestRecord entry : members) {
                    retained |= !orphanCandidates.contains(entry.route);
                }
                if (retained) {
                    continue;
                }
                for (RequestRecord entry : List.copyOf(members)) {
                    removeEntry(entry);
                    recordMutationLocked();
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
                removeEntry(entry);
                recordMutationLocked();
            }
            evicted += individuals.size();
            return evicted;
        } finally {
            stateLock().unlock();
        }
    }

    public EndpointStats.PrefillCounts stats() {
        stateLock().lock();
        try {
            int locallyOwned = 0;
            int individual = 0;
            int committedBatches = 0;
            long maxAgeMs = 0L;
            long nowMs = clock.getAsLong();
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
                if (!batch.committed) {
                    continue;
                }
                committedBatches++;
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs));
            }
            return new EndpointStats.PrefillCounts(
                    locallyOwned, individual, committedBatches, maxAgeMs);
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

    public WorkSnapshot committedSnapshot() {
        WorkSnapshot capture;
        stateLock().lock();
        try {
            capture = captureWorkLocked(clock.getAsLong(), false, null);
        } finally {
            stateLock().unlock();
        }
        return capture.calculate();
    }

    private WorkSnapshot captureWorkLocked(long nowMs, boolean cached, RequestRecord excluded) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Prefill request changes require an Endpoint lock");
        if (cached
                && committedWorkSnapshot != null
                && committedWorkSnapshot.capturedAtMs() <= nowMs) {
            return committedWorkSnapshot;
        }
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RequestRecord entry : requests.values()) {
            if (entry == excluded
                    || entry.batch != null
                    || entry.ownership == OwnershipStage.WAITING) {
                continue;
            }
            Phase phase =
                    entry.ownership == OwnershipStage.COMMITTED
                            ? entry.individualPhase
                            : Phase.COMMITTED;
            individual.add(
                    new WorkSnapshot.RequestWork(
                            entry.route.requestId(),
                            phase,
                            remainingWorkAt(
                                    phase, entry.remainingWorkMs, entry.phaseBaseMs, nowMs)));
        }
        List<WorkSnapshot.BatchWork> capturedBatches = new ArrayList<>(batches.size());
        for (BatchRecord batch : batches.values()) {
            if (!batch.committed) {
                continue;
            }
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
        Map<Long, List<RouteExecution>> inputs =
                reduction.batchOutcomes.isEmpty() ? Map.of() : new HashMap<>();
        List<BatchCompletion> completions =
                reduction.batchOutcomes.isEmpty() ? List.of() : new ArrayList<>();
        for (var change : reduction.batchOutcomes.entrySet()) {
            var batch = change.getKey();
            var outcome = change.getValue();
            List<RouteExecution> survivors = new ArrayList<>(batch.members.size());
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
                        Comparator.comparingLong(RouteExecution::enqueueSeq)
                                .thenComparing(RouteExecution::requestId));
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
        reduction.version = progress().ownershipVersion();
        reduction.predictionInputs = Map.copyOf(inputs);
        reduction.result =
                new StatusChange.Prefill(
                        statuses,
                        completions,
                        changed || !reduction.terminals.isEmpty() || !inputs.isEmpty(),
                        released);
        return reduction;
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
}

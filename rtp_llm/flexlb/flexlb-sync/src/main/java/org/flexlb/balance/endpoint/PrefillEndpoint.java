package org.flexlb.balance.endpoint;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestExecution;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

public class PrefillEndpoint extends WorkerEndpoint {

    /** Exact plan commit; only the accepted handoff transfers generation ownership to the sender. */
    public PrefillState.CommittedHandoff commit(PrefillState.CommitPlan plan) {
        var lock = ownership.queueGuard();
        lock.lock();
        try {
            return commitLocked(plan);
        } finally {
            lock.unlock();
        }
    }

    public PrefillState.CommittedHandoff commitLocked(PrefillState.CommitPlan plan) {
        ownership.queueGuard().requireHeld();
        if (plan.kind() == PrefillState.CommitKind.QUEUED
                && !ownership.ownsSelectionLocked(plan.items(), plan.selectionTimeMs())) {
            return null;
        }
        if (plan.kind() == PrefillState.CommitKind.QUEUED
                && !ownership.requestCapacityAvailable(requestLimit, plan.items().size())) {
            return null;
        }
        if (plan.batchReservation() != null) {
            return ownership.commitLocked(plan, null);
        }
        var handoff = tryAcquireGenerationHandoff();
        if (handoff == null) {
            return null;
        }
        PrefillState.CommittedHandoff committed = null;
        try {
            committed = ownership.commitLocked(plan, handoff);
            return committed;
        } finally {
            if (committed == null) {
                handoff.close();
            }
        }
    }

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR =
            new FormulaPredictor(new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());
    private static final GroupPlanner.Constraints DIRECT_CONSTRAINTS =
            new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);

    public final PrefillTimePredictor predictor;

    public final long requestLimit;

    private volatile RequestExecution requestExecution;
    // Guarded by ownership.queueGuard(); policy exists before the execution thread starts.
    private QueueExecutionSettings queueSettings;
    private RequestExecution.Dependencies queueDependencies;
    private final DeliveryDispatcher dispatcher;
    private final PrefillRequestQueue.Factory workerQueues;
    private final DispatcherConfig.Type dispatcherType;
    public final DispatcherConfig.Type deliveryMode;
    private volatile Comparator<GroupPlanner.Item> projectionOrder;
    private volatile ProjectionSource projectionSource;

    public final PrefillState ownership;

    private final DeliveryMetricsReporter reporter;

    private final PlacementAvailability placementAvailability;

    PrefillEndpoint(
            WorkerStatus status,
            FlexlbConfig config,
            DeliveryDispatcher dispatcher,
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
        this.dispatcher = dispatcher;
        this.workerQueues = java.util.Objects.requireNonNull(workerQueues);
        this.queueSettings =
                config.isQueue() ? QueueExecutionSettings.capture(config, grouping) : null;
        this.requestLimit =
                config.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH
                        ? config.getDispatcher().getMaxInflightPerPrefillWorker()
                        : 0L;
        this.dispatcherType = config.getDispatcher().getType();
        this.deliveryMode = dispatcher.mode();
        this.projectionOrder = QueueExecutionSettings.projectionOrder(config.isPriorityOrdering());
        this.ownership =
                new PrefillState(
                        new ReentrantLock(),
                        PrefillActiveIndex.disabled(),
                        System::currentTimeMillis);
    }

    /** Captured under ownership.queueGuard(); the shared result is built without that lock. */
    private final class ProjectionSource {
        private final PrefillState.Progress version;
        private PrefillState.Snapshot ownership;
        private final GroupPlanner.Constraints constraints;
        private final AdmissionBlock admissionBlock;
        private final boolean queueEnabled;
        private final org.flexlb.balance.planner.GroupPlanner grouping;
        private final Comparator<GroupPlanner.Item> capturedOrder;
        private volatile RequestTimeEstimation.Inputs materialized;

        private ProjectionSource(PrefillState.Snapshot ownership, AdmissionBlock admissionBlock) {
            this.version = ownership.version();
            this.ownership = ownership;
            this.constraints =
                    queueSettings == null
                            ? DIRECT_CONSTRAINTS
                            : queueSettings.constraints(getStatus().committedEngineObservation());
            this.admissionBlock = admissionBlock;
            this.queueEnabled = queueSettings != null;
            this.grouping = queueSettings == null ? null : queueSettings.grouping();
            this.capturedOrder = projectionOrder;
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
                                ? new org.flexlb.balance.projection.QueueSnapshot(
                                        ownership.capturedAtMs(),
                                        queueEnabled,
                                        grouping,
                                        capturedOrder,
                                        constraints,
                                        ownership.active().items().stream().map(RequestRoute::planningInput).toList(),
                                        admissionBlock)
                                : result.queue();
                var updated =
                        new RequestTimeEstimation.Inputs(
                                queueSnapshot,
                                result == null ? ownership.work().materialize() : result.work(),
                                version.ownershipVersion(),
                                evaluator,
                                deliveryMode);
                // A delayed model capture must not replace concurrently published inputs.
                if (result == observed) {
                    materialized = updated;
                    // The cached projection must not retain completed request contexts.
                    ownership = null;
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
        if (source == null || source.version != ownership.progress()) {
            ownership.queueGuard().lock();
            try {
                source = projectionSource;
                if (source == null || source.version != ownership.progress()) {
                    var block = requestExecution == null ? null : requestExecution.admissionBlockLocked();
                    source = new ProjectionSource(ownership.snapshotLocked(), block);
                    projectionSource = source;
                }
            } finally {
                ownership.queueGuard().unlock();
            }
        } else if (observedInputs != null && observedInputs.evaluator() == evaluator) {
            return observedInputs;
        }
        // Concurrent callers share one source per ownership version. A late build only
        // fills its own source; it can never overwrite a newer capture.
        return source.materialize(evaluator, source == observed ? observedInputs : null);
    }

    /**
     * Publish one exact route after validating its generation pin.
     */
    public boolean offerPinned(
            GenerationPin exactPin,
            RequestRoute exactRoute,
            QueueExecutionSettings settings,
            RequestExecution.Dependencies dependencies) {
        requirePinnedGeneration(exactPin);
        if (requestExecution == null) {
            enableQueueRuntime(settings, dependencies);
        }
        checkArgument(
                queueDependencies == dependencies && queueSettings.equals(settings),
                "queue generation is already bound to different runtime facilities or settings");
        exactRoute.ctx().data().initializeExecutionOrder(System.currentTimeMillis());
        return requestExecution.offer(exactRoute);
    }

    private void enableQueueRuntime(
            QueueExecutionSettings settings, RequestExecution.Dependencies dependencies) {
        ownership.queueGuard().lock();
        try {
            if (requestExecution != null) {
                return;
            }
            checkArgument(
                    settings != null && settings.dispatcherType() == dispatcherType,
                    "queued admission requires a compatible QUEUE configuration");
            ownership.enableQueueLocked(
                    workerQueues.create(
                            QueueExecutionSettings.queueOrder(settings.priorityOrdering())));
            RequestExecution newExecution =
                    new RequestExecution(
                            ipPort(), this, settings, dispatcher, ownership, dependencies);
            queueSettings = settings;
            queueDependencies = dependencies;
            projectionOrder = QueueExecutionSettings.projectionOrder(settings.priorityOrdering());
            projectionSource = null;
            newExecution.start();
            requestExecution = newExecution;
        } finally {
            ownership.queueGuard().unlock();
        }
    }

    /**
     * Read the last scheduling decision without traversing or locking the queue.
     */
    /** Latest wait state from this endpoint's queued requests; not a per-request failure cause. */
    public enum Signal {
        CAPACITY,
        INPUT,
        PLACEMENT
    }

    /** INPUT changes planning facts; CAPACITY also wakes placement; PLACEMENT only publishes that edge. */
    public void signal(Signal kind) {
        try {
            if (kind != Signal.PLACEMENT) {
                if (requestExecution != null) {
                    requestExecution.signal(kind == Signal.CAPACITY);
                } else if (kind == Signal.INPUT) {
                    ownership.queueGuard().lock();
                    try {
                        ownership.schedulingInputsChangedLocked();
                    } finally {
                        ownership.queueGuard().unlock();
                    }
                }
            }
            if (kind != Signal.INPUT) {
                placementAvailability.changed(this);
            }
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

    public boolean removeQueued(RequestRoute exact) {
        return requestExecution != null && requestExecution.removeQueued(exact);
    }

    public Map<String, Object> getLatestQueueWaitSnapshot() {
        return requestExecution == null
                ? Map.of("cause", "waiting for Prefill decision")
                : requestExecution.getLatestQueueWaitSnapshot();
    }

    /**
     * Runs once, after every accepted generation handoff has released its pin.
     */
    @Override
    protected void closeEndpoint() {
        // A self-await invariant escapes here before ledger mutation/event
        // publication. Ordinary stop cleanup failures are returned only after
        // the exact worker has exited, and are aggregated below.
        Throwable retirementFailure = requestExecution == null ? null : requestExecution.stopAndAwait();
        try {
            PrefillState.Retirement retirement = ownership.retireGenerationOwnership();
            for (var handoff : retirement.orphanedHandoffs()) {
                retirementFailure = Failures.append(retirementFailure, Failures.close(handoff));
            }
            for (RequestRoute route : retirement.ownedItems()) {
                retirementFailure =
                        Failures.run(
                                retirementFailure,
                                () ->
                                        route.ctx()
                                                .data()
                                                .getScheduler()
                                                .onEngineEvidence(
                                                        route.ctx(),
                                                        new ContextResult.EngineEvidence(
                                                                route,
                                                                null,
                                                                getStatus().getRole(),
                                                                ContextResult.EngineKind.RETIRED,
                                                                0L,
                                                                "Prefill generation retired: "
                                                                        + ipPort(),
                                                                System.currentTimeMillis())));
            }
            retirementFailure = Failures.append(retirementFailure, retirement.invariantFailure());
            retirementFailure =
                    reportBatchCompletions(retirement.batchCompletions(), retirementFailure);
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = Failures.append(retirementFailure, committedRetirementFailure);
        }
        signal(Signal.CAPACITY);
        Failures.rethrow(retirementFailure, "Prefill endpoint retirement failed");
    }

    public PrefillState.ReservationResult<PrefillState.BatchReservation> reserveBatch(
            RequestRoute exactHead, long batchId, int maximumInflightBatches) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit = tryAcquireGenerationHandoff();
        if (handoffPermit == null) {
            return new PrefillState.ReservationResult<>(
                    PrefillState.CapacityStatus.ENDPOINT_RETIRED, null);
        }
        PrefillState.ReservationResult<PrefillState.BatchReservation> result = null;
        try {
            result =
                    ownership.reserveBatch(
                            exactHead, batchId, maximumInflightBatches, handoffPermit);
            return result;
        } finally {
            if (result == null || result.reservation() == null) {
                handoffPermit.close();
            }
        }
    }

    /**
     * Admit on the selected generation using its current canonical occupancy
     * and bound dispatcher policy. The pin preserves generation identity while
     * PrefillState checks and occupies capacity under its ownership lock.
     */
    public PrefillState.ReservationResult<PrefillState.RouteReservation> reserveUnqueuedRoute(
            GenerationPin pin, RequestRoute route, long predictedMs) {
        requirePinnedGeneration(pin);
        return ownership.reserveUnqueuedRoute(route, predictedMs, requestLimit);
    }

    /** The ledger supplies facts; generation cleanup and notifications run after unlocking. */
    public boolean release(PrefillState.Reservation preparation, RequestRoute exact) {
        var released = ownership.release(preparation, exact);
        if (!released.released()) {
            return false;
        }
        try {
            if (released.generationHandoff() != null) {
                Failures.rethrow(
                        Failures.close(released.generationHandoff()),
                        "Prefill preparation cleanup failed");
            }
            if (released.queued()) {
                signal(Signal.INPUT);
            }
        } finally {
            signal(Signal.CAPACITY);
        }
        return true;
    }

    @Override
    public StatusChange applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        PrefillState.StatusReconciliation reconciliation = reduceStatus(ws, observation, prepared);
        return new StatusChange.Prefill(reconciliation);
    }

    @Override
    public void initializeFromPreparedStatus(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        PrefillState.StatusReconciliation reconciliation = reduceStatus(ws, observation, null);
        checkState(
                reconciliation.requestStatuses().isEmpty()
                        && reconciliation.batchCompletions().isEmpty(),
                "Private Prefill candidate produced locally-owned request statuses");
    }

    /** Keep ordinary reduction/publication in one lock; predict a shrunk batch outside it. */
    private PrefillState.StatusReconciliation reduceStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation,
            WorkerStatus.PreparedStatus prepared) {
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        var lock = ownership.queueGuard();
        PrefillState.StatusReduction reduction = null;
        PrefillState.StatusReconciliation result = null;
        Map<Long, Long> predictions = Map.of();
        try {
            while (result == null) {
                lock.lock();
                try {
                    if (reduction == null) {
                        reduction = PrefillObservation.prepareLocked(ownership, observation, true);
                    } else if (!predictions.isEmpty()) {
                        // Rebase resource facts after prediction. Unrelated queue mutations do not
                        // invalidate a prediction for the same exact surviving batch members.
                        var current =
                                PrefillObservation.prepareLocked(ownership, observation, true);
                        if (!current.predictionInputs().equals(reduction.predictionInputs())) {
                            predictions = Map.of();
                        }
                        reduction = current;
                    }
                    if (reduction.predictionInputs().isEmpty() || !predictions.isEmpty()) {
                        result = ownership.commitStatusLocked(reduction, predictions);
                        checkState(
                                result != null, "Locked Prefill reduction changed during commit");
                        if (prepared != null && !observation.alive()) {
                            beginRetirement();
                        }
                        boolean inputsChanged =
                                result.schedulingInputsChanged()
                                        || prepared != null
                                                && QueueExecutionSettings.capacityChanged(
                                                        ws.committedEngineObservation(),
                                                        observation.engine());
                        if (prepared != null) {
                            ws.publishPreparedStatus(prepared);
                        }
                        if (inputsChanged) {
                            ownership.schedulingInputsChangedLocked();
                            if (!result.schedulingInputsChanged()) {
                                result =
                                        new PrefillState.StatusReconciliation(
                                                result.requestStatuses(),
                                                result.batchCompletions(),
                                                true,
                                                result.capacityReleased());
                            }
                        }
                    }
                } catch (Throwable failure) {
                    beginRetirement();
                    throw failure;
                } finally {
                    lock.unlock();
                }
                if (result == null) {
                    predictions = new java.util.HashMap<>();
                    for (var batch : reduction.predictionInputs().entrySet()) {
                        PrefillBatchFeatures features =
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
                }
            }
            return result;
        } catch (Throwable failure) {
            beginRetirement();
            throw Failures.propagate(failure, "Prefill status reduction/publication failed");
        }
    }

    /** Model failures use the existing default formula; callers run prediction outside queueGuard. */
    private long predictRemainingBatch(PrefillBatchFeatures features, int survivingRequests) {
        try {
            return PrefillPredictionBoundary.predictCommittedBatchMs(
                    predictor.evaluator(), features);
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
            return PrefillPredictionBoundary.predictCommittedBatchMs(
                    DEFAULT_BATCH_PREDICTOR, features);
        }
    }

    @Override
    public StatusChange applyStatusHeartbeat(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        PrefillState.StatusReconciliation reconciliation;
        ownership.queueGuard().lock();
        try {
            reconciliation =
                    ownership.commitStatusLocked(
                            PrefillObservation.prepareLocked(ownership, observation, false),
                            Map.of());
            if (reconciliation.schedulingInputsChanged()) {
                ownership.schedulingInputsChangedLocked();
            }
        } finally {
            ownership.queueGuard().unlock();
        }
        return new StatusChange.Prefill(reconciliation);
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) {
            return;
        }
        var reconciliation = ((StatusChange.Prefill) change).reconciliation();
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
        if (reconciliation.schedulingInputsChanged() && requestExecution != null) {
            requestExecution.wake();
        }
        for (var status : reconciliation.requestStatuses()) {
            var route = status.route();
            if (route.prefillEp() != this) {
                continue;
            }
            route.ctx()
                    .data()
                    .getScheduler()
                    .onEngineEvidence(
                            route.ctx(),
                            new ContextResult.EngineEvidence(
                                    route,
                                    null,
                                    getStatus().getRole(),
                                    switch (status.kind()) {
                                        case ACTIVE -> ContextResult.EngineKind.ACTIVE;
                                        case COMPLETED -> ContextResult.EngineKind.COMPLETED;
                                        case FAILED -> ContextResult.EngineKind.FAILED;
                                        case PRIORITY_CANCELED ->
                                                ContextResult.EngineKind.PRIORITY_CANCELLED;
                                    },
                                    status.errorCode(),
                                    "",
                                    System.currentTimeMillis()));
        }
    }

    /**
     * Evict endpoint orphans while retaining IDs still registered by the scheduler.
     */
    public int evictExpiredInflight(long ttlMs, LongPredicate retainForSchedulerCleanup) {
        var orphanCandidates = new java.util.HashSet<RequestRoute>();
        for (RequestRoute route : ownership.cleanupCandidates()) {
            if (!retainForSchedulerCleanup.test(route.requestId())) {
                orphanCandidates.add(route);
            }
        }
        int released = ownership.evictExpiredInflight(ttlMs, orphanCandidates);
        if (released > 0) {
            signal(Signal.CAPACITY);
        }
        return released;
    }

    @Override
    public OptionalLong tryGetLoadMetric() {
        return ownership.tryLoadMetric();
    }

    @Override
    public OptionalLong getLoadMetric() {
        return ownership.committedSnapshot().totalRemainingWorkMs();
    }

    /** Current number of requests waiting for this worker. */
    public int queueDepth() {
        return ownership.progress().queueDepth();
    }

    // ==================== Metrics ====================
    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.service.monitor.DeliveryMetricsReporter}.
     */
    public void reportBatchMetrics(DeliveryMetricsReporter reporter) {
        int queueSize = ownership.progress().queueDepth();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority =
                requestExecution == null ? Map.of() : requestExecution.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), 0, 0);
        } else {
            sizeByPriority.forEach(
                    (priority, size) ->
                            reporter.reportBatcherQueueDepthByPriority(
                                    RoleType.PREFILL.name(), getIp(), priority, size));
        }
        var stats = ownership.stats();
        reporter.reportPrefillInflight(
                getIp(),
                stats.batchCount(),
                stats.locallyOwnedRequests(),
                stats.maxObservedAgeMs());
    }

    private Throwable reportBatchCompletions(
            List<PrefillState.BatchCompletion> completions, Throwable failure) {
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
}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RequestRoute;
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
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;
import java.util.function.LongSupplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/** Owns the exact Prefill resources and queue membership for one worker generation. */
public class PrefillEndpoint extends WorkerEndpoint implements EndpointStats.Prefill {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR =
            new FormulaPredictor(new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());
    private static final GroupPlanner.Constraints DIRECT_CONSTRAINTS =
            new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);

    private final ReentrantLock lock = new ReentrantLock();

    /** Non-owning index containing only ACTIVE RequestRoute identities. */
    private PrefillRequestQueue activeIndex = PrefillActiveIndex.disabled();

    /** Canonical request ownership, changed only under the endpoint lock. */
    private final Map<Long, RouteReservation> requests = new HashMap<>();

    private final LongSupplier clock = System::currentTimeMillis;

    /** Exact occupied delivery slots, updated with their ledger rows under lock. */
    private int inflightRequests;

    private static final Progress EMPTY_PROGRESS = new Progress(0L, 0L, 0L, 0);
    private volatile Progress progress = EMPTY_PROGRESS;

    /** Derived immutable work; ACTIVE queue mutations leave committed work unchanged. */
    private WorkSnapshot committedWorkSnapshot;

    private long unknownEngineRequestCount;
    private final Map<Long, BatchReservation> batches = new HashMap<>();

    public final PrefillTimePredictor predictor;

    public final long requestLimit;

    private volatile boolean queueEnabled;
    private volatile boolean queueStopped;
    private java.util.function.Supplier<AdmissionBlock> admissionBlock = () -> null;
    private volatile java.util.function.Supplier<Map<String, Object>> queueWait =
            () -> Map.of("cause", "waiting for Prefill decision");
    // Guarded by resourceLock(); settings are fixed before execution is enabled.
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

    /** The caller retains its selection pin through route publication; no second handoff is needed. */
    public WorkSnapshot commitDirect(
            GenerationPin pin, RequestRoute route, RouteReservation reservation) {
        requirePinnedGeneration(pin);
        lock.lock();
        try {
            if (isRetired() || !getStatus().isAvailableForScheduling()) return null;
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
            lock.unlock();
        }
    }

    /** Captured under resourceLock(); the shared result is built without that lock. */
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
                    lock.isHeldByCurrentThread(),
                    "Prefill projection capture requires ownership lock");
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
                                ? new RequestTimeEstimation.QueueSnapshot(
                                        capturedAtMs,
                                        queueEnabled,
                                        grouping,
                                        capturedOrder,
                                        constraints,
                                        active.items().stream()
                                                .map(RequestRoute::planningInput)
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
            resourceLock().lock();
            try {
                source = projectionSource;
                if (source == null || source.version != progress()) {
                    var block = admissionBlock.get();
                    source = new ProjectionSource(block);
                    projectionSource = source;
                }
            } finally {
                resourceLock().unlock();
            }
        } else if (observedInputs != null && observedInputs.evaluator() == evaluator) {
            return observedInputs;
        }
        // Concurrent callers share one source per ownership version. A late build only
        // fills its own source; it can never overwrite a newer capture.
        return source.materialize(evaluator, source == observed ? observedInputs : null);
    }

    /** Enable exact queue membership; only the Scheduler owns decisions and execution tasks. */
    public void enableQueue(QueueExecutionSettings settings) {
        var guard = resourceLock();
        guard.lock();
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
            guard.unlock();
        }
    }

    public boolean queueEnabled() {
        return queueEnabled;
    }

    public boolean queueStopped() {
        return queueStopped;
    }

    /** Read-only scheduling evidence used by prediction and timeout diagnostics. */
    public void observeQueueWait(
            java.util.function.Supplier<AdmissionBlock> block,
            java.util.function.Supplier<Map<String, Object>> diagnostics) {
        var guard = resourceLock();
        guard.lock();
        try {
            admissionBlock = java.util.Objects.requireNonNull(block);
            queueWait = java.util.Objects.requireNonNull(diagnostics);
            schedulingInputsChangedLocked();
        } finally {
            guard.unlock();
        }
    }

    public boolean offerPinned(
            GenerationPin pin, RequestRoute route, QueueExecutionSettings settings) {
        requirePinnedGeneration(pin);
        enableQueue(settings);
        route.ctx().data().initializeExecutionOrder(System.currentTimeMillis());
        return offerQueued(route);
    }

    public boolean offerQueued(RequestRoute route) {
        checkArgument(route.prefillEp() == this, "request belongs to another Prefill generation");
        var guard = resourceLock();
        boolean accepted;
        guard.lock();
        try {
            accepted = !queueStopped() && !isRetired() && addQueuedLocked(route) != null;
        } finally {
            guard.unlock();
        }
        if (accepted) signal(Signal.PLACEMENT);
        return accepted;
    }

    public enum Signal {
        CAPACITY,
        INPUT,
        PLACEMENT
    }

    /** INPUT changes planning facts; CAPACITY also wakes placement; PLACEMENT only publishes that edge. */
    public void signal(Signal kind) {
        try {
            if (kind != Signal.PLACEMENT) {
                resourceLock().lock();
                try {
                    schedulingInputsChangedLocked();
                } finally {
                    resourceLock().unlock();
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

    public boolean removeQueued(RequestRoute route) {
        checkArgument(route.prefillEp() == this, "request belongs to another Prefill generation");
        boolean removed;
        var guard = resourceLock();
        guard.lock();
        try {
            removed = !queueStopped() && removeQueuedLocked(route);
        } finally {
            guard.unlock();
        }
        if (removed) signal(Signal.CAPACITY);
        return removed;
    }

    public Map<String, Object> getLatestQueueWaitSnapshot() {
        return queueWait.get();
    }

    /**
     * Runs once, after every accepted generation handoff has released its pin.
     */
    @Override
    protected void closeEndpoint() {
        Throwable retirementFailure = null;
        try {
            Retirement retirement = retireResources();
            for (RequestRoute route : retirement.ownedItems()) {
                retirementFailure =
                        Failures.run(
                                retirementFailure,
                                () ->
                                        route.ctx()
                                                .data()
                                                .getLifecycle()
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
            retirementFailure =
                    reportBatchCompletions(retirement.batchCompletions(), retirementFailure);
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = Failures.append(retirementFailure, committedRetirementFailure);
        }
        signal(Signal.CAPACITY);
        Failures.rethrow(retirementFailure, "Prefill endpoint retirement failed");
    }

    public CapacityBoundary.Attempt<BatchReservation> reserveBatch(
            long batchId, int maximumInflightBatches) {
        WorkerEndpoint.GenerationPin handoffPermit = tryPinGeneration();
        if (handoffPermit == null) {
            return CapacityBoundary.Attempt.rejected(
                    CapacityBoundary.failed(
                            new IllegalStateException("Prefill endpoint retired: " + ipPort())));
        }
        CapacityBoundary.Attempt<BatchReservation> result = null;
        try {
            result = reserveBatch(batchId, maximumInflightBatches, handoffPermit);
            return result;
        } finally {
            if (result == null || result.value() == null) {
                handoffPermit.close();
            }
        }
    }

    /**
     * Admit on the selected generation using its current canonical occupancy
     * and bound dispatcher policy. The pin preserves generation identity while
     * PrefillEndpoint checks and occupies capacity under its ownership lock.
     */
    public CapacityBoundary.Attempt<RouteReservation> reserveUnqueuedRoute(
            GenerationPin pin, RequestRoute route, long predictedMs) {
        requirePinnedGeneration(pin);
        var result = reserveUnqueuedRoute(route, predictedMs, requestLimit);
        if (result.accepted()) route.attachPrefillReservation(result.value());
        return result;
    }

    /** Undo an uncommitted preparation; queued members remain eligible for another delivery. */
    public boolean rollback(Reservation preparation) {
        checkArgument(preparation.owner == this, "preparation belongs to another endpoint");
        checkState(!lock.isHeldByCurrentThread(), "rollback cannot run under resource lock");
        GenerationPin handoff = null;
        lock.lock();
        try {
            if (preparation instanceof BatchReservation batch) {
                if (batch.generationHandoff == null) return false;
                checkState(
                        batches.get(batch.batchId) == batch, "batch reservation is not canonical");
                batches.remove(batch.batchId);
                handoff = batch.generationHandoff;
                batch.generationHandoff = null;
            } else {
                var route = (RouteReservation) preparation;
                if (requests.get(route.route.requestId()) != route
                        || route.ownership != OwnershipStage.RESERVED) return false;
                checkState(removeEntry(route), "route reservation is not canonical");
                committedWorkSnapshot = null;
            }
            recordMutationLocked();
        } finally {
            lock.unlock();
        }
        try {
            if (handoff != null)
                Failures.rethrow(Failures.close(handoff), "Prefill preparation cleanup failed");
        } finally {
            signal(Signal.CAPACITY);
        }
        return true;
    }

    /** Remove only this exact request's resources, then publish capacity outside the lock. */
    public boolean releaseRequest(RequestRoute route) {
        Objects.requireNonNull(route, "route");
        lock.lock();
        try {
            RouteReservation entry = requests.get(route.requestId());
            if (entry == null || entry.route != route) return false;
            if (entry.batch != null) {
                entry.batch.lastObservedAtMs =
                        Math.max(entry.batch.lastObservedAtMs, clock.getAsLong());
                entry.batch.outcome.learningEligible = false;
            }
            checkState(removeEntry(entry), "exact release lost canonical ownership");
            recordMutationLocked();
        } finally {
            lock.unlock();
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

    /** Keep ordinary reduction/publication in one lock; predict a shrunk batch outside it. */
    private StatusChange.Prefill reduceStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation,
            WorkerStatus.PreparedStatus prepared) {
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another Prefill generation");
        var lock = resourceLock();
        Map<Long, List<RequestRoute>> predictedInputs = Map.of();
        Map<Long, Long> predictions = Map.of();
        while (true) {
            Map<Long, List<RequestRoute>> inputs;
            lock.lock();
            try {
                var reduction = prepareStatusLocked(observation, true);
                inputs = reduction.predictionInputs();
                // Always capture current resource facts. Reuse prediction only for the same
                // members.
                if (inputs.isEmpty() || inputs.equals(predictedInputs)) {
                    var result =
                            commitStatusLocked(
                                    reduction, inputs.isEmpty() ? Map.of() : predictions);
                    if (prepared != null && !observation.alive()) beginRetirement();
                    boolean inputsChanged =
                            result.schedulingInputsChanged()
                                    || prepared != null
                                            && QueueExecutionSettings.capacityChanged(
                                                    ws.committedEngineObservation(),
                                                    observation.engine());
                    if (prepared != null) ws.publishPreparedStatus(prepared);
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
                    return result;
                }
            } catch (Throwable failure) {
                // Retire before unlocking so no new admission can observe a partial status commit.
                beginRetirement();
                throw Failures.propagate(failure, "Prefill status reduction/publication failed");
            } finally {
                lock.unlock();
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

    /** Model failures use the existing default formula; callers run prediction outside queueGuard. */
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
        StatusChange.Prefill reconciliation;
        resourceLock().lock();
        try {
            reconciliation = commitStatusLocked(prepareStatusLocked(observation, false), Map.of());
            if (reconciliation.schedulingInputsChanged()) {
                schedulingInputsChangedLocked();
            }
        } finally {
            resourceLock().unlock();
        }
        return reconciliation;
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
            if (route.prefillEp() != this) {
                continue;
            }
            route.ctx()
                    .data()
                    .getLifecycle()
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
        for (RequestRoute route : cleanupCandidates()) {
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

    /** Current number of requests waiting for this worker. */
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

    /** Request status matched to its exact Prefill route, published after ledger reconciliation. */
    public record PrefillRequestStatus(RequestRoute route, Kind kind, long errorCode) {
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
            List<RequestRoute> ownedItems, List<BatchCompletion> batchCompletions) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    /** Exact preparation owned by this ledger; consumed by commit or rollback. */
    public abstract static class Reservation {
        final PrefillEndpoint owner;

        private Reservation(PrefillEndpoint owner) {
            this.owner = owner;
        }
    }

    /** One batch reservation becomes its committed record; the delivery retains the exact identity. */
    public static final class BatchReservation extends Reservation {
        final long batchId;
        private WorkerEndpoint.GenerationPin generationHandoff;
        Set<RouteReservation> members = Set.of();
        long originalPredictionMs;
        PrefillBatchFeatures originalFeatures;
        long remainingWorkMs;
        Phase servicePhase = Phase.COMMITTED;
        BatchOutcome outcome = new BatchOutcome();
        long phaseBaseMs;
        long lastObservedAtMs;

        private BatchReservation(
                PrefillEndpoint owner,
                long batchId,
                WorkerEndpoint.GenerationPin generationHandoff) {
            super(owner);
            checkArgument(batchId >= 0L, "batchId must be non-negative");
            this.batchId = batchId;
            this.generationHandoff = Objects.requireNonNull(generationHandoff);
        }
    }

    /** Resource ownership within this Prefill generation, independent of request protocol. */
    enum OwnershipStage {
        WAITING,
        RESERVED,
        COMMITTED
    }

    /** Canonical resource facts for one exact route. Guarded by {@link #lock}. */
    public static final class RouteReservation extends Reservation {
        OwnershipStage ownership;
        final RequestRoute route;
        Phase individualPhase;
        long remainingWorkMs;
        long phaseBaseMs;
        BatchReservation batch;

        public boolean belongsTo(RequestRoute exact) {
            return route == exact;
        }

        private RouteReservation(
                PrefillEndpoint owner, RequestRoute route, OwnershipStage ownership) {
            super(owner);
            this.ownership = ownership;
            this.route = Objects.requireNonNull(route, "route");
        }

        private void commit(BatchReservation committedBatch, long predictedMs, long nowMs) {
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
            RouteReservation owner,
            long executionTimeMs,
            long errorCode,
            PriorityPreemptionProgress preemptionProgress) {}

    /** The resource lock also protects exact queue membership and commits. */
    public ReentrantLock resourceLock() {
        return lock;
    }

    /** Publish the capacity summary before readers observe a new ownership revision. */
    private void recordMutationLocked() {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
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

    /** Enable waiting work without replacing the ledger of existing DIRECT reservations. */
    void enableQueueLocked(PrefillRequestQueue queue) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        if (!activeIndex.enabled()) {
            checkArgument(queue.enabled(), "worker queue must support admission");
            activeIndex = queue;
            schedulingInputsChangedLocked();
            recordMutationLocked();
        }
    }

    /** One bounded, ordered queue view; all metadata belongs to the same ownership revision. */
    public record QueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<RequestRoute> items,
            RequestRoute head) {}

    public EndpointStats.QueueCounts queueCounters() {
        lock.lock();
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
            lock.unlock();
        }
    }

    public QueueSnapshot captureQueue(int limit) {
        return captureQueue(limit, null, false);
    }

    /** Queue policy supplies alternatives; the endpoint only captures them under its existing lock. */
    public QueueSnapshot captureQueue(
            int limit, RequestRoute blockedHead, boolean priorityOrdering) {
        checkArgument(limit > 0, "queue capture limit must be positive");
        lock.lock();
        try {
            List<RequestRoute> items = new ArrayList<>(Math.min(limit, activeIndex.size()));
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
            lock.unlock();
        }
    }

    public void schedulingInputsChangedLocked() {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        progress =
                new Progress(
                        activeIndex.version(),
                        progress.schedulingInputVersion() + 1L,
                        progress.ownershipVersion(),
                        activeIndex.size());
    }

    /** Remove a known exact owner and its non-owning indexes; batch leases keep their own preparation. */
    private boolean removeEntry(RouteReservation entry) {
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

    /** Register exact queue membership; request settlement remains with its Lifecycle. */
    RequestRoute addQueuedLocked(RequestRoute route) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        Objects.requireNonNull(route, "route");
        var entry = requests.get(route.requestId());
        if (entry != null
                && (entry.route != route
                        || entry.ownership != OwnershipStage.WAITING
                        || activeIndex.contains(route))) return null;
        boolean created = entry == null;
        if (created) {
            entry = new RouteReservation(this, route, OwnershipStage.WAITING);
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

    /** Exact deletion only; the scheduler decides expiry and notifies the request outside this lock. */
    public boolean removeQueuedLocked(RequestRoute route) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        Objects.requireNonNull(route, "route");
        var entry = requests.get(route.requestId());
        if (entry == null
                || entry.route != route
                || entry.ownership != OwnershipStage.WAITING
                || !activeIndex.contains(route)) return false;
        checkState(activeIndex.remove(route), "queue member disappeared under ledger lock");
        recordMutationLocked();
        return true;
    }

    /** Stop queue admission and detach membership; the caller owns request settlement. */
    public List<RequestRoute> closeQueue() {
        lock.lock();
        try {
            queueStopped = true;
            var removed = List.copyOf(activeIndex.capture().items());
            activeIndex.clear();
            if (!removed.isEmpty()) recordMutationLocked();
            return removed;
        } finally {
            lock.unlock();
        }
    }

    public boolean ownsSelectionLocked(List<RequestRoute> items, long nowMs) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        for (RequestRoute item : items) {
            if (!activeIndex.contains(item) || item.requestExpired(nowMs)) {
                return false;
            }
        }
        return true;
    }

    /** Validate the exact observed head and selected member; ordering policy belongs to the queue. */
    public boolean selectionCurrentLocked(RequestRoute head, RequestRoute candidate) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        var entry = requests.get(head.requestId());
        return activeIndex.peek() == head
                && entry != null
                && entry.route == head
                && activeIndex.contains(candidate)
                && !head.requestExpired(clock.getAsLong())
                && !candidate.requestExpired(clock.getAsLong());
    }

    /** Read queue identity under the same lock as resource ownership. */
    public RequestRoute queueHeadLocked() {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        return activeIndex.peek();
    }

    /** Exact failure-path capacity accounting; queue priority ordering stays in the queue. */
    public AdmissionSummary admissionSummary(int priority, long requestLimit) {
        lock.lock();
        try {
            long lower = 0, same = 0, higher = 0, unknown = unknownEngineRequestCount;
            if (!requests.isEmpty()) {
                for (RouteReservation entry : requests.values()) {
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
                    requestLimit > 0
                            ? Math.max(0L, outstandingRequestCount - requestLimit + 1L)
                            : 0L,
                    lower,
                    same,
                    higher,
                    unknown);
        } finally {
            lock.unlock();
        }
    }

    CapacityBoundary.Attempt<BatchReservation> reserveBatch(
            long batchId, int maximum, WorkerEndpoint.GenerationPin generationHandoff) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
        lock.lock();
        try {
            if (batches.containsKey(batchId)) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException(
                                        "Batch id already reserved: batch_id=" + batchId)));
            }
            if (batches.size() >= maximum) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH));
            }
            BatchReservation reservation = new BatchReservation(this, batchId, generationHandoff);
            batches.put(batchId, reservation);
            recordMutationLocked();
            return CapacityBoundary.Attempt.accepted(reservation);
        } finally {
            lock.unlock();
        }
    }

    /** Queued routes consume no delivery slots; reserved and committed work do. */
    public boolean requestCapacityAvailable(long maximum, int requested) {
        checkArgument(maximum >= 0L && requested > 0, "invalid request capacity");
        lock.lock();
        try {
            long inFlight = saturatedAdd(inflightRequests, unknownEngineRequestCount);
            return maximum == 0L || requested <= maximum && inFlight <= maximum - requested;
        } finally {
            lock.unlock();
        }
    }

    public boolean batchCapacityAvailable(int maximum) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
        lock.lock();
        try {
            return batches.size() < maximum;
        } finally {
            lock.unlock();
        }
    }

    /** Validate all members before mutating ownership; a failed commit returns its unused pin. */
    public Delivery commitLocked(Delivery delivery) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        checkState(!delivery.committed(), "delivery already committed");
        List<RequestRoute> items = delivery.items();
        long selectedAt = delivery.selectionTimeMs();
        if (!ownsSelectionLocked(items, selectedAt)) return null;
        var batch = delivery.reservation();
        var predictions = delivery.predictions();
        if (batch == null && !requestCapacityAvailable(requestLimit, items.size())) return null;
        checkArgument(batch != null || predictions.length >= items.size(), "missing predictions");
        var members = new ArrayList<RouteReservation>(items.size());
        Set<RouteReservation> unique = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
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
            checkState(batch.generationHandoff != null, "batch preparation lost its handoff");
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
        var pin = batch == null ? tryPinGeneration() : batch.generationHandoff;
        if (pin == null) return null;
        try {
            long now = clock.getAsLong();
            var work = captureWorkLocked(now, true, null);
            if (batch != null) {
                batch.members = unique;
                batch.originalPredictionMs = delivery.predictedMs();
                batch.remainingWorkMs = delivery.predictedMs();
                batch.originalFeatures = delivery.features();
                batch.phaseBaseMs = now;
                batch.lastObservedAtMs = now;
                batch.generationHandoff = null;
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
            delivery.committed(pin, work, activeIndex.size(), failed != null);
            return delivery;
        } finally {
            if (!delivery.committed() && batch == null) pin.close();
        }
    }

    /**
     * Reserve immediate route work against current ownership in one transaction.
     * Selection revisions are advisory; only exact identity and current capacity
     * decide admission. A zero request limit leaves count admission disabled.
     */
    public CapacityBoundary.Attempt<RouteReservation> reserveUnqueuedRoute(
            RequestRoute item, long predictedMs, long maxOutstandingRequests) {
        Objects.requireNonNull(item, "item");
        checkArgument(maxOutstandingRequests >= 0L, "request limit must be non-negative");
        lock.lock();
        try {
            if (requests.containsKey(item.requestId())) {
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
            }
            if (!requestCapacityAvailable(maxOutstandingRequests, 1)) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_REQUEST));
            }
            RouteReservation reservation =
                    new RouteReservation(this, item, OwnershipStage.RESERVED);
            reservation.remainingWorkMs = Math.clamp(predictedMs, 0L, (long) Integer.MAX_VALUE);
            CapacityBoundary.Attempt<RouteReservation> result =
                    CapacityBoundary.Attempt.accepted(reservation);
            requests.put(item.requestId(), reservation);
            inflightRequests++;
            committedWorkSnapshot = null;
            recordMutationLocked();
            return result;
        } finally {
            lock.unlock();
        }
    }

    /** Facts prepared under the lock, then read during prediction and exact-version commit. */
    public static final class StatusReduction {
        final PrefillEndpoint owner;
        long version;
        final Map<Long, TerminalObservation> terminals;
        final Map<RouteReservation, Phase> individualPhases = new IdentityHashMap<>();
        final Map<BatchReservation, Phase> batchPhases = new IdentityHashMap<>();
        final Map<BatchReservation, BatchOutcome> batchOutcomes;
        // Assigned before publication; only this endpoint applies the facts under its lock.
        long nowMs;
        long unknownRequests;
        Map<Long, List<RequestRoute>> predictionInputs;
        StatusChange.Prefill result;

        StatusReduction(PrefillEndpoint owner, Map<Long, TerminalObservation> terminals) {
            this.owner = owner;
            this.terminals = terminals;
            this.batchOutcomes = terminals.isEmpty() ? Map.of() : new IdentityHashMap<>();
        }

        public Map<Long, List<RequestRoute>> predictionInputs() {
            return predictionInputs;
        }
    }

    /** Owned by one batch or by an uncommitted status reduction, never both while preparing. */
    static final class BatchOutcome {
        long maxExecutionTimeMs;
        boolean successfulCompletion;
        boolean learningEligible = true;

        /** Historical execution evidence survives a later queued observation. */
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

    /** Capture exact activity into the same transaction that will commit it. */
    /** Null means the out-of-lock prediction was invalidated; no fact has changed. */
    public StatusChange.Prefill commitStatusLocked(
            StatusReduction reduction, Map<Long, Long> predictions) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
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
            BatchReservation batch = change.getKey();
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
            BatchReservation batch = batches.get(prediction.getKey());
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

    /** WorkerEndpoint invokes this only after all generation handoffs have finished. */
    Retirement retireResources() {
        Retirement retirement;
        lock.lock();
        try {
            checkState(
                    batches.values().stream().allMatch(batch -> batch.generationHandoff == null),
                    "generation still owns an uncommitted batch");
            List<RequestRoute> ownedItems =
                    requests.values().stream()
                            .map(entry -> entry.route)
                            .sorted(
                                    Comparator.comparingLong(RequestRoute::enqueueSeq)
                                            .thenComparingLong(RequestRoute::requestId))
                            .toList();
            List<BatchCompletion> completions =
                    batches.values().stream()
                            .map(
                                    batch ->
                                            new BatchCompletion(
                                                    batch.batchId,
                                                    batch.originalFeatures,
                                                    batch.originalPredictionMs,
                                                    batch.outcome.maxExecutionTimeMs,
                                                    batch.outcome.successfulCompletion,
                                                    false))
                            .sorted(Comparator.comparingLong(BatchCompletion::batchId))
                            .toList();
            retirement = new Retirement(ownedItems, completions);
            var retiredProgress =
                    new Progress(
                            activeIndex.version() + (activeIndex.isEmpty() ? 0L : 1L),
                            progress.schedulingInputVersion(),
                            progress.ownershipVersion() + 1L,
                            0);
            for (BatchReservation batch : batches.values()) {
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
            lock.unlock();
        }
        return retirement;
    }

    /** Capture exact committed identities before the workflow checks request ownership outside this lock. */
    public List<RequestRoute> cleanupCandidates() {
        lock.lock();
        try {
            return requests.values().stream()
                    .filter(entry -> entry.ownership == OwnershipStage.COMMITTED)
                    .map(entry -> entry.route)
                    .toList();
        } finally {
            lock.unlock();
        }
    }

    /** One orphan pass: a retained member protects its whole batch. Counts batches and individuals. */
    public int evictExpiredInflight(long ttlMs, Set<RequestRoute> orphanCandidates) {
        int evicted = 0;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            long ttl = Math.max(0L, ttlMs);
            for (BatchReservation batch : List.copyOf(batches.values())) {
                if (batch.generationHandoff != null) {
                    continue;
                }
                Set<RouteReservation> members = batch.members;
                boolean retained = nowMs - batch.lastObservedAtMs < ttl;
                for (RouteReservation entry : members) {
                    retained |= !orphanCandidates.contains(entry.route);
                }
                if (retained) {
                    continue;
                }
                for (RouteReservation entry : List.copyOf(members)) {
                    removeEntry(entry);
                    recordMutationLocked();
                }
                evicted++;
            }
            List<RouteReservation> individuals = new ArrayList<>();
            for (RouteReservation entry : requests.values()) {
                if (entry.batch == null
                        && entry.ownership == OwnershipStage.COMMITTED
                        && nowMs - entry.phaseBaseMs >= ttl
                        && orphanCandidates.contains(entry.route)) {
                    individuals.add(entry);
                }
            }
            for (RouteReservation entry : individuals) {
                removeEntry(entry);
                recordMutationLocked();
            }
            evicted += individuals.size();
        } finally {
            lock.unlock();
        }
        return evicted;
    }

    public EndpointStats.PrefillCounts stats() {
        lock.lock();
        try {
            int locallyOwned = 0;
            int individual = 0;
            int committedBatches = 0;
            long maxAgeMs = 0L;
            long nowMs = clock.getAsLong();
            for (RouteReservation entry : requests.values()) {
                if (entry.ownership != OwnershipStage.COMMITTED) {
                    continue;
                }
                locallyOwned++;
                if (entry.batch == null) {
                    individual++;
                    maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - entry.phaseBaseMs));
                }
            }
            for (BatchReservation batch : batches.values()) {
                if (batch.generationHandoff != null) {
                    continue;
                }
                committedBatches++;
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs));
            }
            return new EndpointStats.PrefillCounts(
                    locallyOwned, individual, committedBatches, maxAgeMs);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public OptionalLong tryGetLoadMetric() {
        if (!lock.tryLock()) {
            return OptionalLong.empty();
        }
        WorkSnapshot capture;
        try {
            capture = captureWorkLocked(clock.getAsLong(), false, null);
        } finally {
            lock.unlock();
        }
        return capture.calculate().totalRemainingWorkMs();
    }

    public WorkSnapshot committedSnapshot() {
        WorkSnapshot capture;
        lock.lock();
        try {
            capture = captureWorkLocked(clock.getAsLong(), false, null);
        } finally {
            lock.unlock();
        }
        return capture.calculate();
    }

    private WorkSnapshot captureWorkLocked(long nowMs, boolean cached, RouteReservation excluded) {
        checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        if (cached
                && committedWorkSnapshot != null
                && committedWorkSnapshot.capturedAtMs() <= nowMs) {
            return committedWorkSnapshot;
        }
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RouteReservation entry : requests.values()) {
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
        for (BatchReservation batch : batches.values()) {
            if (batch.generationHandoff != null) {
                continue;
            }
            List<Long> members = new ArrayList<>(batch.members.size());
            for (RouteReservation member : batch.members) {
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
        checkState(lock.isHeldByCurrentThread(), "Prefill observation requires ownership lock");
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
        Map<Long, List<RequestRoute>> inputs =
                reduction.batchOutcomes.isEmpty() ? Map.of() : new HashMap<>();
        List<BatchCompletion> completions =
                reduction.batchOutcomes.isEmpty() ? List.of() : new ArrayList<>();
        for (var change : reduction.batchOutcomes.entrySet()) {
            var batch = change.getKey();
            var outcome = change.getValue();
            List<RequestRoute> survivors = new ArrayList<>(batch.members.size());
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
                        Comparator.comparingLong(RequestRoute::enqueueSeq)
                                .thenComparingLong(RequestRoute::requestId));
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

    private Map<Long, TerminalObservation> terminals(
            Map<String, WorkerStatus.TaskObservation> tasks) {
        Map<Long, TerminalObservation> facts = new HashMap<>();
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
        Set<Long> unknown = new HashSet<>(), known = new HashSet<>();
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

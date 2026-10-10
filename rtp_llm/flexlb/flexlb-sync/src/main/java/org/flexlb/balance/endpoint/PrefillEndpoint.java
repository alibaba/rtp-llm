package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.eviction.EvictionPlanner;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.PriorityOrdering;
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

    /**
     * Short-lived generation capability for committing one NON_BATCH route
     * group. Queued requests keep their canonical identity without keeping an
     * endpoint generation alive while waiting for delivery.
     */
    public final class RouteCommitAdmission implements AutoCloseable {

        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;

        private RouteCommitAdmission(
                EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            this.generationHandoff = generationHandoff;
        }

        /** Commit exact DIRECT preparation while acquiring the resource ownership lock. */
        public PrefillState.CommittedHandoff commit(
                List<RequestRoute> exactItems,
                List<PrefillState.RouteReservation> exactReservations) {
            prefillState.ownershipLock().lock();
            try {
                EndpointGenerationLifecycle.HandoffPermit exact = generationHandoff;
                checkState(exact != null, "route commit no longer owns its generation handoff");
                PrefillState.CommittedHandoff committed =
                        prefillState.commitRouteGroupLocked(exactItems, exactReservations, exact);
                generationHandoff = null;
                return committed;
            } finally {
                prefillState.ownershipLock().unlock();
            }
        }

        @Override
        public void close() {
            EndpointGenerationLifecycle.HandoffPermit exact = generationHandoff;
            generationHandoff = null;
            if (exact != null) {
                exact.close();
            }
        }
    }

    /** Revalidate before pinning the generation; a stale selection acquires no new capability. */
    public PrefillState.CommittedHandoff commitQueuedRoutesLocked(List<RequestRoute> items, long[] predictions,
            RequestRoute failedMember, long nowMs) {
        if (!prefillState.ownsSelectionLocked(items, nowMs)) { return null; }
        try (var admission = tryBeginRouteCommitAdmission()) {
            if (admission == null) {
                throw new IllegalStateException("Prefill endpoint generation retired: request_id=" + items.getFirst().requestId());
            }
            var committed = prefillState.commitQueuedRoutesLocked(items, predictions,
                    admission.generationHandoff, failedMember, nowMs);
            admission.generationHandoff = null;
            return committed;
        }
    }

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR = new FormulaPredictor(
            new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());

    private final PrefillTimePredictor predictor;

    private final long inflightRequestLimit;

    private volatile WorkerBatcher batcher;
    private final DeliveryStrategy deliveryStrategy;
    private final DispatcherConfig.Type dispatcherType;
    private final RouteProjection.DeliveryProjection deliveryProjection;
    private volatile Comparator<GroupPlanner.Item> projectionOrder;
    private volatile ProjectionSource projectionSource;

    private static final Comparator<GroupPlanner.Item> PRIORITY_PROJECTION_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_PROJECTION_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparingLong(GroupPlanner.Item::requestId);

    private final PrefillState prefillState;

    private final DeliveryMetricsReporter reporter;

    private final PlacementAvailability placementAvailability;

    PrefillEndpoint(WorkerStatus status,
                    FlexlbConfig config,
                    DeliveryStrategy deliveryStrategy,
                    DeliveryMetricsReporter reporter,
                    PlacementAvailability placementAvailability) {
        super(status);
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
        this.predictor = createPredictor(config);
        this.deliveryStrategy = deliveryStrategy;
        this.dispatcherType = config.getDispatcher().getType();
        this.inflightRequestLimit = config.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH
                ? config.getDispatcher().getMaxInflightPerPrefillWorker() : 0L;
        this.deliveryProjection = deliveryStrategy.projectionPolicy();
        this.projectionOrder = config.isPriorityOrdering() ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
        this.prefillState = new PrefillState(new ReentrantLock(), PrefillActiveIndex.disabled());

    }

    /** Captured under prefillState.ownershipLock(); the shared result is built without that lock. */
    private final class ProjectionSource {
        private final PrefillState.ProjectionVersion version;
        private PrefillState.Snapshot ownership;
        private final GroupPlanner.Constraints constraints;
        private final AdmissionBlock admissionBlock;
        private final WorkerBatcher capturedBatcher;
        private final Comparator<GroupPlanner.Item> capturedOrder;
        private volatile RouteProjection.Inputs materialized;

        private ProjectionSource(PrefillState.Snapshot ownership,
                                 GroupPlanner.Constraints constraints, AdmissionBlock admissionBlock) {
            this.version = ownership.version();
            this.ownership = ownership;
            this.constraints = constraints;
            this.admissionBlock = admissionBlock;
            this.capturedBatcher = batcher;
            this.capturedOrder = projectionOrder;
        }

        private RouteProjection.Inputs materialize() {
            RouteProjection.Inputs result = materialized;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (materialized == null) {
                    var queueSnapshot = new org.flexlb.balance.projection.QueueSnapshot(
                            ownership.capturedAtMs(), capturedBatcher != null, capturedBatcher == null ? null : capturedBatcher.groupingPolicy(), capturedOrder,
                            constraints, ownership.active().projectedItems(), admissionBlock);
                    materialized = new RouteProjection.Inputs(
                            queueSnapshot, ownership.work().materialize(), version.ownership());
                    // The cached projection must not retain completed request contexts.
                    ownership = null;
                }
                return materialized;
            }
        }
    }

    public RouteProjection.Inputs captureRouteProjectionInputs() {
        ProjectionSource source = projectionSource;
        if (source == null || !prefillState.isCurrentProjection(source.version)) {
            prefillState.ownershipLock().lock();
            try {
                source = projectionSource;
                if (source == null || !prefillState.isCurrentProjection(source.version)) {
                    source = captureProjectionSourceLocked();
                    projectionSource = source;
                }
            } finally {
                prefillState.ownershipLock().unlock();
            }
        }
        // Concurrent callers share one source per version. A late build only
        // fills its own source; it can never overwrite a newer capture.
        return source.materialize();
    }

    /** Caller holds the endpoint ownership lock. */
    private ProjectionSource captureProjectionSourceLocked() {
        PrefillState.Snapshot ownership = prefillState.snapshotLocked();
        return new ProjectionSource(ownership,
                batcher == null ? new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L)
                        : batcher.projectionConstraintsLocked(),
                batcher == null || ownership.active().isEmpty() ? null : batcher.admissionBlockLocked());
    }

    /**
     * Stable delivery semantics selected once for this endpoint generation.
     */
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return deliveryProjection;
    }

    /**
     * Publish one exact route after validating its generation pin.
     */
    public boolean offerPinned(
            GenerationPin exactPin,
            RequestRoute exactItem, QueueExecutionSettings settings) {
        requirePinnedGeneration(exactPin);
        if (batcher == null) {
            enableQueueRuntime(settings);
        }
        return batcher.offer(exactItem);
    }

    public void enableQueueRuntime(QueueExecutionSettings settings) {
        prefillState.ownershipLock().lock();
        try {
            if (batcher != null) { return; }
            checkArgument(settings != null && settings.dispatcherType() == dispatcherType,
                    "queued admission requires a compatible QUEUE configuration");
            prefillState.enableQueueLocked(settings.priorityOrdering()
                    ? WorkerBatcher.PRIORITY_QUEUE_ORDER : WorkerBatcher.FIFO_QUEUE_ORDER);
            WorkerBatcher newBatcher = new WorkerBatcher(ipPort(), this, settings, deliveryStrategy, prefillState);
            projectionOrder = settings.priorityOrdering() ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
            projectionSource = null;
            batcher = newBatcher;
            newBatcher.start();
        } finally {
            prefillState.ownershipLock().unlock();
        }
    }

    /**
     * Publish a role/group-scoped edge after real queue or status progress.
     */
    public void signalPlacementCapacityChanged() {
        WorkerStatus.TopologySnapshot topology =
                getStatus().topologySnapshot();
        placementAvailability.changed(
                getStatus().getRole(), topology.group(), ipPort());
    }

    /**
     * Remove only the supplied canonical ACTIVE queue identity.
     */
    public boolean removeQueued(
            RequestRoute exactItem,
            String reason) {
        return batcher != null && batcher.removeQueued(exactItem, reason);
    }

    public void signalRouteReady() {
        signalSchedulingInputsChanged();
    }

    private void notifyCapacityAvailable() {
        try {
            if (batcher != null) { batcher.signalDeliveryCapacityAvailable(); }
            signalPlacementCapacityChanged();
        }
        catch (Throwable failure) {
            try { logger.error("Prefill capacity notification failed", failure); }
            catch (Throwable ignored) { }
        }
    }

    private void signalSchedulingInputsChanged() {
        if (batcher != null) {
            batcher.signalSchedulingInputsChanged();
        } else {
            prefillState.ownershipLock().lock();
            try {
                prefillState.schedulingInputsChangedLocked();
            } finally {
                prefillState.ownershipLock().unlock();
            }
        }
    }

    public boolean signalQueuedControl(RequestRoute exactItem) {
        return batcher != null && batcher.signalControl(exactItem);
    }

    /**
     * Read the last scheduling decision without traversing or locking the queue.
     */
    /** Latest wait state from this endpoint's worker batcher; not a per-request failure cause. */
    public Map<String, Object> getLatestQueueWaitSnapshot() {
        return batcher == null ? Map.of("cause", "waiting for Prefill decision") : batcher.getLatestQueueWaitSnapshot();
    }

    public int queuedRequestCount() {
        return prefillState.queueDepth();
    }

    /**
     * Runs once, after every accepted generation handoff has released its pin.
     */
    @Override
    protected void closeEndpoint() {
        // A self-await invariant escapes here before ledger mutation/event
        // publication. Ordinary stop cleanup failures are returned only after
        // the exact worker has exited, and are aggregated below.
        Throwable retirementFailure = batcher == null ? null : batcher.stopAndAwait();
        try {
            PrefillState.Retirement retirement =
                    prefillState.retireGenerationOwnership();
            for (var handoff : retirement.orphanedHandoffs()) {
                retirementFailure = Failures.append(retirementFailure, Failures.close(handoff));
            }
            if (!retirement.ownedItems().isEmpty()) {
                try {
                    for (RequestRoute route : retirement.ownedItems()) {
                        route.ctx().scheduler().onPrefillGenerationRetired(this, route);
                    }
                } catch (Throwable callbackFailure) {
                    retirementFailure = Failures.append(
                            retirementFailure, callbackFailure);
                }
            }
            retirementFailure = Failures.append(
                    retirementFailure, retirement.invariantFailure());
            List<PrefillState.BatchCompletion> completions =
                    retirement.batchCompletions();
            for (int index = 0; index < completions.size(); index++) {
                PrefillState.BatchCompletion completion =
                        completions.get(index);
                try {
                    reportBatchCompletion(completion);
                } catch (Throwable reportingFailure) {
                    retirementFailure = Failures.append(
                            retirementFailure, reportingFailure);
                }
            }
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = Failures.append(
                    retirementFailure, committedRetirementFailure);
        }
        notifyCapacityAvailable();
        Failures.rethrow(retirementFailure, "Prefill endpoint retirement failed");
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig config) {
        RoutingConfig.ExecutionTimeEstimatorConfig estimator = config.getRouter()
                .getRoles().getPrefill().getExecutionTimeEstimator();
        if (estimator.getType() == RoutingConfig.EstimatorType.LEARNING) {
            return new LearningPredictor();
        }
        return new FormulaPredictor(estimator.getExpression());
    }

    public PrefillState.ReservationResult<PrefillState.BatchReservation> reserveBatch(
            RequestRoute exactHead,
            long batchId,
            int maximumInflightBatches) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        if (handoffPermit == null) {
            return new PrefillState.ReservationResult<>(
                    PrefillState.CapacityStatus.ENDPOINT_RETIRED, null);
        }
        PrefillState.ReservationResult<PrefillState.BatchReservation> result = null;
        try {
            result = prefillState.reserveBatch(
                    exactHead,
                    batchId,
                    maximumInflightBatches,
                    handoffPermit);
            return result;
        } finally {
            if (result == null || result.reservation() == null) {
                handoffPermit.close();
            }
        }
    }

    /**
     * Acquire the generation capability only for the final route commit.
     */
    public RouteCommitAdmission tryBeginRouteCommitAdmission() {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        return handoffPermit == null
                ? null : new RouteCommitAdmission(handoffPermit);
    }

    /**
     * Exact wake source for this generation's batch admission capacity.
     */
    public CapacityBoundary.Availability batchAdmissionAvailability(
            int maximumInflightBatches) {
        checkArgument(maximumInflightBatches > 0, "maximumInflightBatches must be positive");
        return new CapacityBoundary.Availability() {
            @Override public boolean isAvailable() { return prefillState.batchCapacityAvailable(maximumInflightBatches); }
            @Override public void addListener(Runnable listener) {
                checkArgument(batcher != null && listener == batcher.capacityAvailableSignal(),
                        "Expected this generation's worker wake signal");
            }
            @Override public void removeListener(Runnable listener) { }
        };
    }

    public PrefillState.AdmissionSummary admissionSummary(int priority) {
        return prefillState.admissionSummary(priority, inflightRequestLimit);
    }

    /**
     * Advisory capacity; publication repeats the count check under the ownership lock.
     */
    public boolean canAcceptRequest() {
        return inflightRequestLimit == 0L || prefillState.canAcceptRequest(inflightRequestLimit);
    }

    public boolean canPreemptQueuedRequest(int priority) {
        if (!PriorityNormalizer.hasPriority(priority)) { return false; }
        prefillState.ownershipLock().lock();
        try {
            return prefillState.hasUncommittedQueuedRequestsBelowPriorityLocked(priority,
                    prefillState.requestSlotsToReleaseLocked(inflightRequestLimit));
        } finally {
            prefillState.ownershipLock().unlock();
        }
    }

    /** Plan and commit local queue replacement while the batcher holds the ownership lock. */
    public List<RequestRoute> replaceQueuedRoutesLocked(RequestRoute incoming, long requestLimit) {
        long required = prefillState.requestSlotsToReleaseLocked(requestLimit);
        if (required == 0L || !PriorityNormalizer.hasPriority(incoming.priority())) { return List.of(); }
        List<RequestRoute> victims = EvictionPlanner.selectPrefillVictims(
                prefillState.uncommittedQueuedRoutesLocked(), incoming.priority(), required);
        return prefillState.replaceQueuedRoutesLocked(incoming, requestLimit, victims) ? victims : List.of();
    }

    /**
     * Advisory endpoint ownership revision captured by queue placement.
     */
    public long placementVersion() {
        return prefillState.mutationVersion();
    }

    /**
     * Admit on the selected generation using its current canonical occupancy
     * and bound dispatcher policy. The pin preserves generation identity while
     * PrefillState checks and occupies capacity under its ownership lock.
     */
    public PrefillState.ReservationResult<PrefillState.RouteReservation> reserveUnqueuedRoute(
            GenerationPin pin, RequestRoute item, long predictedMs) {
        requirePinnedGeneration(pin);
        return prefillState.reserveUnqueuedRoute(item, predictedMs, inflightRequestLimit);
    }

    /** Preparation rollback belongs to its workflow owner; wake only after cleanup and outside State's lock. */
    public void rollbackReservation(PrefillState.Reservation reservation) {
        if (reservation == null) { return; }
        var rollback = prefillState.rollbackPreparation(reservation);
        try {
            if (rollback.generationHandoff() != null) {
                Failures.rethrow(Failures.close(rollback.generationHandoff()), "Prefill preparation cleanup failed");
            }
        } finally {
            if (rollback.released()) { notifyCapacityAvailable(); }
        }
    }

    /** Release the exact queued, prepared or committed request, then publish capacity outside State's lock. */
    public boolean releaseRequest(RequestRoute exactItem) {
        PrefillState.RequestRelease released = prefillState.releaseRequest(exactItem);
        if (released == PrefillState.RequestRelease.NONE) { return false; }
        try {
            if (released == PrefillState.RequestRelease.QUEUED) { signalSchedulingInputsChanged(); }
        } finally { notifyCapacityAvailable(); }
        return true;
    }

    /** One consistent snapshot of batch, individual and total local ownership. */
    public PrefillState.Stats ownershipStats() {
        return prefillState.stats();
    }

    @Override
    public Runnable applyPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        PrefillState.StatusReconciliation reconciliation = reduceStatus(ws, observation, prepared);
        List<PrefillState.PrefillRequestStatus> requestStatuses =
                reconciliation.requestStatuses();
        return () -> requestStatuses.forEach(requestStatus -> requestStatus.route().ctx().scheduler().onPrefillStatus(
                requestStatus.route().ctx(), this, observation.role(), requestStatus));
    }

    @Override
    public void initializeFromPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        PrefillState.StatusReconciliation reconciliation = reduceStatus(ws, observation, null);
        checkState(reconciliation.requestStatuses().isEmpty() && reconciliation.batchCompletions().isEmpty(),
                "Private Prefill candidate produced locally-owned request statuses");
    }

    /** Keep ordinary reduction/publication in one lock; predict a shrunk batch outside it. */
    private PrefillState.StatusReconciliation reduceStatus(WorkerStatus ws,
            WorkerStatus.StatusObservation observation, WorkerStatus.PreparedStatus prepared) {
        checkArgument(observation.owner() == ws, "Status observation belongs to another Prefill generation");
        var lock = prefillState.ownershipLock();
        PrefillState.StatusReduction reduction = null;
        PrefillState.StatusReconciliation result = null;
        Map<Long, Long> predictions = Map.of();
        try {
            while (result == null) {
                lock.lock();
                try {
                    if (reduction == null) { reduction = prefillState.prepareStatusLocked(observation); }
                    else if (!predictions.isEmpty()) {
                        // Rebase resource facts after prediction. Unrelated queue mutations do not
                        // invalidate a prediction for the same exact surviving batch members.
                        var current = prefillState.prepareStatusLocked(observation);
                        if (!current.predictionInputs().equals(reduction.predictionInputs())) { predictions = Map.of(); }
                        reduction = current;
                    }
                    if (reduction.predictionInputs().isEmpty() || !predictions.isEmpty()) {
                        result = prefillState.commitStatusLocked(reduction, predictions);
                        checkState(result != null, "Locked Prefill reduction changed during commit");
                        if (prepared != null && !observation.alive()) { beginRetirement(); }
                        if (result.schedulingInputsChanged()) { signalSchedulingInputsChanged(); }
                        if (prepared != null) { ws.publishPreparedStatus(prepared); }
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
                        predictions.put(batch.getKey(), predictRepackedBatchMs(batch.getValue()));
                    }
                }
            }
            return result;
        } catch (Throwable failure) {
            beginRetirement();
            throw Failures.propagate(failure, "Prefill status reduction/publication failed");
        } finally {
            if (result != null) {
                reportBatchCompletionsNoFail(result.batchCompletions());
                if (result.capacityReleased()) { notifyCapacityAvailable(); }
            }
        }
    }

    @Override
    public Runnable applyStatusHeartbeat(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        checkArgument(observation.owner() == ws, "Status observation belongs to another Prefill generation");
        PrefillState.StatusReconciliation reconciliation =
                prefillState.reconcileHeartbeat(observation);
        if (reconciliation.capacityReleased()) { notifyCapacityAvailable(); }
        if (reconciliation.schedulingInputsChanged()) {
            signalSchedulingInputsChanged();
        }
        return () -> reconciliation.requestStatuses().forEach(requestStatus -> requestStatus.route().ctx().scheduler().onPrefillStatus(
                requestStatus.route().ctx(), this, observation.role(), requestStatus));
    }

    private void reportBatchCompletionsNoFail(
            List<PrefillState.BatchCompletion> completions) {
        try {
            completions.forEach(this::reportBatchCompletion);
        } catch (Throwable reportingFailure) {
            try {
                logger.warn("Prefill status committed but completion reporting failed: engine={}",
                        getIp(), reportingFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Status facts must still reach the exact scheduler projection.
            }
        }
    }

    /**
     * Re-estimate surviving members, using the default formula if the configured predictor fails.
     */
    private long predictRepackedBatchMs(List<RequestRoute> survivingRequests) {
        PrefillBatchFeatures features = PrefillBatchFeatures.from(
                survivingRequests,
                item -> Math.max(0L, item.seqLen()),
                item -> Math.clamp(item.hitCache(), 0L, Math.max(0L, item.seqLen())));
        try {
            return PrefillPredictionBoundary.predictCommittedBatchMs(predictor.evaluator(), features);
        } catch (RuntimeException predictionFailure) {
            try {
                logger.error("Prefill batch repack prediction failed; using default formula "
                                + "engine={} surviving_requests={}",
                        getIp(), survivingRequests.size(), predictionFailure);
            } catch (RuntimeException ignoredLoggingFailure) {
                // Prediction logging cannot block membership settlement.
            }
            return PrefillPredictionBoundary.predictCommittedBatchMs(DEFAULT_BATCH_PREDICTOR, features);
        }
    }

    /**
     * Evict endpoint orphans while retaining IDs still registered by the scheduler.
     */
    public int evictExpiredInflight(long ttlMs,
                                    LongPredicate retainForSchedulerCleanup) {
        var orphanCandidates = new java.util.HashSet<RequestRoute>();
        for (RequestRoute route : prefillState.cleanupCandidates()) {
            if (!retainForSchedulerCleanup.test(route.requestId())) { orphanCandidates.add(route); }
        }
        int released = prefillState.evictExpiredInflight(ttlMs, orphanCandidates);
        if (released > 0) { notifyCapacityAvailable(); }
        return released;
    }

    @Override
    public OptionalLong getLoadMetric() {
        return prefillState.committedSnapshot()
                .totalRemainingWorkMs();
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    // ==================== Metrics ====================
    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.SchedulerRuntime}.
     */
    public void reportBatchMetrics(DeliveryMetricsReporter reporter) {
        int queueSize = queuedRequestCount();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority =
                batcher == null ? Map.of() : batcher.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), 0, 0);
        } else {
            sizeByPriority.forEach((priority, size) ->
                    reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), priority, size));
        }
        reporter.reportPrefillInflight(getIp(), ownershipStats());
    }

    private void reportBatchCompletion(
            PrefillState.BatchCompletion completion) {
        long batchId = completion.batchId();
        long actualMs = completion.actualWorkMs();
        if (!completion.successfulCompletion() || actualMs <= 0) {
            logger.debug("batch completion not reportable: batchId={} success={} actualMs={}",
                    batchId, completion.successfulCompletion(), actualMs);
            return;
        }

        long predictedMs = completion.predictedWorkMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.debug(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs,
                completion.originalFeatures().batchSize(), getIp());

        // A failed/removed member makes the original batch an invalid learning
        // sample even if another member completed successfully.
        if (completion.learningEligible()) {
            try {
                PrefillTimePredictor.LearningResult learningResult = predictor.learn(
                        completion.originalFeatures(), predictedMs, actualMs);
                if (learningResult
                        == PrefillTimePredictor.LearningResult.MODEL_UPDATED) {
                    signalSchedulingInputsChanged();
                }
            } catch (RuntimeException learningFailure) {
                logger.warn("batch predictor learning failed after settlement: batchId={} engine={}",
                        batchId, getIp(), learningFailure);
            }
        }

        try {
            reporter.reportBatchCompletion(getIp(), batchId, predictedMs, actualMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch completion metrics failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
    }
}

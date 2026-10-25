package org.flexlb.balance.endpoint;

import lombok.Getter;
import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.prefill.CommitResult;
import org.flexlb.balance.endpoint.prefill.HeartbeatResult;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.PrefillStats;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.endpoint.prefill.ShutdownResult;
import org.flexlb.balance.endpoint.prefill.WorkerStatusFact;
import org.flexlb.balance.endpoint.prefill.WorkerStatusResult;
import org.flexlb.balance.endpoint.prefill.batch.BatchReservation;
import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.PrefillQueue;
import org.flexlb.balance.queue.QueuedRequestsSnapshot;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.RejectReason;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PrefillQueueEnqueueResult;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.Predicate;
import java.util.function.Supplier;

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

        public CommitResult commit(List<ScheduledRequest> exactItems, List<SingleReservation> exactReservations) {
            EndpointGenerationLifecycle.HandoffPermit exact = generationHandoff;
            if (exact == null) {
                throw new IllegalStateException(
                        "route commit no longer owns its generation handoff");
            }
            CommitResult committed = prefillState.commitSingleGroup(exactItems, exactReservations, exact);
            generationHandoff = null;
            return committed;
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

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final PrefillTimePredictor.Evaluator DEFAULT_BATCH_PREDICTOR = new FormulaPredictor(
            new RoutingConfig.ExecutionTimeEstimatorConfig().getExpression());
    @Getter
    private final PrefillTimePredictor predictor;
    private final Supplier<FlexlbConfig> configSupplier;
    @Getter
    private final boolean engineSubQueueMode;
    private final PrefillQueue prefillQueue;
    private final PrefillState prefillState;
    private final EndpointEventHandler endpointEvents;
    private final BatchSchedulerReporter reporter;
    private final PlacementAvailability placementAvailability;
    private final EndpointRegistry endpointRegistry;
    private final QueueCapacity queueCapacity;

    PrefillEndpoint(WorkerStatus status,
                    Supplier<FlexlbConfig> configSupplier,
                    DeliveryStrategy deliveryStrategy,
                    PrefillTtftEstimator ttftEstimator,
                    EndpointEventHandler endpointEvents,
                    BatchSchedulerReporter reporter,
                    PlacementAvailability placementAvailability,
                    EndpointRegistry endpointRegistry,
                    QueueCapacity queueCapacity) {
        super(status);
        this.configSupplier = configSupplier;
        this.reporter = reporter;
        this.endpointEvents = endpointEvents;
        this.placementAvailability = placementAvailability;
        this.endpointRegistry = endpointRegistry;
        this.queueCapacity = queueCapacity;
        FlexlbConfig startupConfig = configSupplier.get();
        this.engineSubQueueMode = QueueCapacity.isEngineSubQueueMode(startupConfig);
        this.predictor = createPredictor(startupConfig);
        this.prefillQueue = new PrefillQueue(
                status.getLogicalIpPort(), this, configSupplier.get(),
                deliveryStrategy, ttftEstimator, endpointEvents, queueCapacity);
        this.prefillState = prefillQueue.ownedState();
    }

    /** Start the attached generation exactly once before routing publication. */
    void startGeneration() {
        prefillQueue.start();
    }

    /** Capture the canonical queue/work inputs for one pure route estimate. */
    public PrefillTtftEstimate.LoadView captureLoadView() {
        return prefillQueue.captureLoadView();
    }

    /** Stable delivery semantics selected once for this endpoint generation. */
    public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
        return prefillQueue.ttftEstimationRule();
    }

    /** Publish one exact route after validating its generation pin. */
    public PrefillQueueEnqueueResult offerPinnedResult(GenerationPin exactPin, ScheduledRequest exactItem) {
        requirePinnedGeneration(exactPin);
        return prefillQueue.offerWithResult(exactItem);
    }

    /** Publish a role/group-scoped edge after real queue or status progress. */
    public void signalPlacementCapacityChanged() {
        WorkerStatus.TopologySnapshot topology =
                getStatus().topologySnapshot();
        placementAvailability.capacityChanged(
                getStatus().getRole(), topology.group(), ipPort());
    }

    /** Remove only the supplied canonical ACTIVE queue identity. */
    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        return prefillQueue.removeQueued(exactItem, reason);
    }

    /** Read the last scheduling decision without traversing or locking the queue. */
    public Map<String, Object> queueWaitDiagnostics() {
        return prefillQueue.waitDiagnostics();
    }

    /** Capture exact queued identities for eviction planning. */
    public QueuedRequestsSnapshot captureQueueSnapshot() {
        return prefillQueue.captureQueueSnapshot();
    }

    public int queuedRequestCount() {
        return prefillQueue.queueSize();
    }

    /** Runs once, after every accepted generation handoff has released its pin. */
    @Override
    protected void closeEndpoint() {
        // A self-await invariant escapes here before ledger mutation/event
        // publication. Ordinary stop cleanup failures are returned only after
        // the exact worker has exited, and are aggregated below.
        Throwable retirementFailure = prefillQueue.stopAndAwait();
        try {
            ShutdownResult shutdownResult = prefillState.shutdown();
            if (!shutdownResult.removedItems().isEmpty()) {
                try {
                    endpointEvents.onPrefillGenerationRetired(this, shutdownResult.removedItems());
                } catch (Throwable callbackFailure) {
                    retirementFailure = appendRetirementFailure(
                            retirementFailure, callbackFailure);
                }
            }
            retirementFailure = appendRetirementFailure(retirementFailure, shutdownResult.checkFailure());
            List<FinishedBatch> finishedBatches = shutdownResult.finishedBatches();
            for (int index = 0; index < finishedBatches.size(); index++) {
                FinishedBatch finishedBatch = finishedBatches.get(index);
                try {
                    reportFinishedBatch(finishedBatch);
                } catch (Throwable reportingFailure) {
                    retirementFailure = appendRetirementFailure(
                            retirementFailure, reportingFailure);
                }
            }
        } catch (Throwable committedRetirementFailure) {
            retirementFailure = appendRetirementFailure(
                    retirementFailure, committedRetirementFailure);
        }
        rethrowEndpointRetirementFailure(retirementFailure);
    }

    private static Throwable appendRetirementFailure(
            Throwable first,
            Throwable next) {
        if (next == null) {
            return first;
        }
        if (first == null) {
            return next;
        }
        // Cleanup is total even under allocation failure. Preserve the first
        // causal failure; later leaf failures must not escape aggregation.
        return first;
    }

    private static void rethrowEndpointRetirementFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Prefill endpoint retirement failed", failure);
        }
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig config) {
        RoutingConfig.ExecutionTimeEstimatorConfig estimator = config.getRouter()
                .getRoles().getPrefill().getExecutionTimeEstimator();
        if (estimator.getType() == RoutingConfig.EstimatorType.LEARNING) {
            return new LearningPredictor();
        }
        return new FormulaPredictor(estimator.getExpression());
    }

    public ReservationResult<BatchReservation> reserveBatch(ScheduledRequest exactHead,
                                                            long batchId,
                                                            int maximumInflightBatches) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        if (handoffPermit == null) {
            return ReservationResult.fail(ReservationResult.Status.ENDPOINT_RETIRED);
        }
        ReservationResult<BatchReservation> result;
        try {
            result = prefillState.reserveBatch(
                    exactHead,
                    batchId,
                    maximumInflightBatches,
                    handoffPermit);
        } catch (Throwable failure) {
            handoffPermit.close();
            throw failure;
        }
        if (result.reservation() == null) {
            handoffPermit.close();
        }
        return result;
    }

    /**
     * Bind exact NON_BATCH ownership inside the caller's pinned placement
     * transaction.
     */
    public ReservationResult<SingleReservation> reserveSingle(ScheduledRequest exactItem,
                                                                                  long predictedMs) {
        if (isGenerationRetiringOrRetired()) {
            return ReservationResult.fail(ReservationResult.Status.ENDPOINT_RETIRED);
        }
        return prefillState.reserveSingle(
                exactItem, predictedMs);
    }

    public boolean isRetired() {
        return isGenerationRetiringOrRetired();
    }

    /** Acquire the generation capability only for the final route commit. */
    public RouteCommitAdmission tryBeginRouteCommitAdmission() {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        return handoffPermit == null
                ? null : new RouteCommitAdmission(handoffPermit);
    }

    /** Exact wake source for this generation's batch admission capacity. */
    public DeliveryBlockReason.Availability batchAdmissionAvailability(
            int maximumInflightBatches) {
        return prefillState.batchAvailability(maximumInflightBatches);
    }

    /** Explain a failed admission using the current owner-priority summary. */
    public RejectReason rejectReason(int priority, long inflightRequestLimit) {
        return prefillState.rejectReason(priority, inflightRequestLimit);
    }

    /** Advisory capacity; publication repeats the count check under the ownership lock. */
    public boolean canAcceptRequest(long inflightRequestLimit) {
        return inflightRequestLimit == 0L || prefillState.canAcceptRequest(inflightRequestLimit);
    }

    public boolean tryReserveEngineSubQueueSlot(FlexlbConfig config) {
        RoleType role = getStatus().getRole();
        int logicalEndpointCount = endpointRegistry.getEndpointCount(role);
        return queueCapacity.tryReserveEngineSubQueueSlot(config, role, logicalEndpointCount);
    }

    public void increaseEngineSubQueueCount() {
        queueCapacity.increaseEngineSubQueueCount(getStatus().getRole());
    }

    public boolean canDeliver(long maxDispatchLimit, int newReqNum) {
        return prefillState.canDeliver(maxDispatchLimit, newReqNum);
    }

    public DeliveryBlockReason.Availability routeAdmissionAvailability(long limit) {
        return prefillState.routeAvailability(limit);
    }

    public boolean canPreemptQueuedRequest(int priority, long inflightRequestLimit) {
        return prefillState.canPreemptQueuedRequest(priority, inflightRequestLimit);
    }

    /** Advisory endpoint ownership revision captured by queue placement. */
    public long placementVersion() {
        return prefillState.mutationVersion();
    }

    /** Exact counterpart cleanup; stale item generations are a no-op. */
    public boolean releaseCommittedItem(ScheduledRequest exactItem) {
        return prefillState.removeCommittedRequest(exactItem);
    }

    /** End the exact failed member, whether still queued or committed to a batch. */
    public void settleFailedRequest(ScheduledRequest exactItem) {
        removeQueued(exactItem, "REQUEST_FAILED");
        releaseCommittedItem(exactItem);
    }

    /** Expire an exact local committed lease without claiming Engine completion. */
    public boolean expireCommittedItem(ScheduledRequest exactItem) {
        return prefillState.removeCommittedRequest(exactItem);
    }

    /**
     * 已提交、还没结束的请求数，包括批次成员和单个请求。
     */
    public int getCommittedRequestCount() {
        PrefillStats stats = prefillState.stats();
        return stats.committedRequests();
    }

    /**
     * 已提交、还没结束的单个请求数（NON_BATCH）。
     */
    public int getSingleRequestCount() {
        PrefillStats stats = prefillState.stats();
        return stats.singleRequests();
    }

    @Override
    public Runnable applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        WorkerStatusResult result =
                prefillState.updateFromWorkerStatus(observation, this::predictRepackedBatchMs, () -> {
                            if (!observation.alive()) {
                                beginRetirement();
                            }
                            prefillQueue.signalSchedulingInputsChanged();
                            ws.publishPreparedStatus(prepared);
                        },
                        this::beginRetirement);
        reportFinishedBatchesNoFail(result.finishedBatches());
        rethrowPublicationFailure(result.publicationFailure());
        List<WorkerStatusFact> facts = result.schedulerFacts();
        return () -> endpointEvents.onPrefillStatus(
                this, observation.role(), facts);
    }

    @Override
    public Runnable initializeFromPreparedStatus(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        WorkerStatusResult result = prefillState.updateFromWorkerStatus(observation, this::predictRepackedBatchMs,
                prefillQueue::signalSchedulingInputsChanged, this::beginRetirement);
        if (!result.schedulerFacts().isEmpty() || !result.finishedBatches().isEmpty()) {
            throw new IllegalStateException(
                    "Private Prefill candidate produced locally-owned status facts");
        }
        rethrowPublicationFailure(result.publicationFailure());
        return () -> { };
    }

    @Override
    public Runnable observeStatusHeartbeat(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        if (observation.owner() != ws) {
            throw new IllegalArgumentException(
                    "Status observation belongs to another Prefill generation");
        }
        HeartbeatResult result = prefillState.updateFromHeartbeat(observation);
        if (result.schedulingInputsChanged()) {
            prefillQueue.signalSchedulingInputsChanged();
        }
        return () -> endpointEvents.onPrefillStatus(this, observation.role(), result.schedulerFacts());
    }

    private static void rethrowPublicationFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Prefill status publication failed", failure);
        }
    }

    private void reportFinishedBatchesNoFail(List<FinishedBatch> finishedBatches) {
        try {
            reportFinishedBatches(finishedBatches);
        } catch (Throwable reportingFailure) {
            try {
                logger.warn("Prefill status committed but finished batch reporting failed: engine={}", getIp(),
                        reportingFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Status facts must still reach the exact scheduler estimate.
            }
        }
    }

    /** Re-estimate surviving members, using the default formula if the configured predictor fails. */
    private long predictRepackedBatchMs(List<ScheduledRequest> survivingRequests) {
        PrefillBatchFeatures features = PrefillBatchFeatures.from(
                survivingRequests,
                item -> Math.max(0L, item.seqLen()),
                item -> Math.max(0L, Math.min(item.hitCache(), item.seqLen())));
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

    // ==================== Ownership diagnostics ====================

    /** Diagnostic snapshot of canonical local plus worker-reported ownership. */
    public long observedRequestCount() {
        return prefillState.observedRequestCount();
    }

    public int getInflightBatchCount() {
        PrefillStats stats = prefillState.stats();
        return stats.batchCount();
    }

    /**
     * Evict inflight batches not observed for longer than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale prefill entries.
     *
     * @return number of batches evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        return evictExpiredBatches(ttlMs, ignored -> false);
    }

    /** Evict only batches whose member IDs are absent from the scheduler directory. */
    public int evictExpiredBatches(long ttlMs,
                                   Predicate<String> retainForSchedulerCleanup) {
        return prefillState.evictExpiredBatches(
                ttlMs, retainForSchedulerCleanup);
    }

    /** Evict stale individual requests whose IDs are absent from the scheduler directory. */
    public int evictExpiredRequests(long ttlMs,
                                    Predicate<String> retainForSchedulerCleanup) {
        return prefillState.evictExpiredSingles(
                ttlMs, retainForSchedulerCleanup);
    }

    /** Evict endpoint orphans while retaining IDs still registered by the scheduler. */
    public int evictExpiredInflight(long ttlMs,
                                    Predicate<String> retainForSchedulerCleanup) {
        return evictExpiredBatches(ttlMs, retainForSchedulerCleanup)
                + evictExpiredRequests(ttlMs, retainForSchedulerCleanup);
    }

    @Override
    public OptionalLong getLoadMetric() {
        return prefillState.committedSnapshot()
                .totalRemainingWorkMs();
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.RequestScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        String engineIp = getStatus().getMetricIpPort();
        String role = getStatus().getRole().name();
        int queueSize = prefillQueue.queueSize();
        reporter.reportBatcherQueueSize(role, engineIp, queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority =
                prefillQueue.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(role, engineIp, 0, 0);
        } else {
            sizeByPriority.forEach((priority, size) ->
                    reporter.reportBatcherQueueDepthByPriority(role, engineIp, priority, size));
        }
        PrefillStats stats = prefillState.stats();
        reporter.reportInflightBatchCount(role, engineIp, stats.batchCount());
        if (configSupplier.get().getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH) {
            reporter.reportInflightRequestCount(role, engineIp, engineSubQueueMode
                    ? prefillState.dispatchedRequestCount() : stats.inflightRequests());
            reporter.reportWorkerStatusUnconfirmedRequestCount(role, engineIp, stats.waitingForWorkerStatusRequests());
        } else {
            reporter.reportInflightRequestCount(role, engineIp, stats.waitingForWorkerStatusRequests());
        }
        reporter.reportInflightMaxAgeMs(role, engineIp, stats.maxStatusAgeMs());
        if (engineSubQueueMode) {
            reportQueueMetricSafely("oldest endpoint queue wait", () ->
                    reporter.reportEngineSubQueueOldestWaitMs(role, engineIp, prefillQueue.oldestSubQueueWaitMs()));
        }
    }

    public void reportSubQueueResidence(ScheduledRequest item) {
        if (!engineSubQueueMode) {
            return;
        }
        long durationMs = item.finishSubQueueVisitMs();
        if (durationMs < 0L) {
            return;
        }
        item.ctx().setSubQueueMs(item.ctx().getSubQueueMs() + durationMs);
        reportQueueMetricSafely("endpoint queue residence time", () ->
            reporter.reportEngineSubQueueResidenceMs(
                    getStatus().getRole().name(), getStatus().getMetricIpPort(), durationMs));
    }

    public void reportSharedSubQueueBlocked() {
        reportQueueMetricSafely("shared endpoint queue admission block", () ->
                reporter.reportSharedSubQueueBlocked(
                        getStatus().getRole().name(), getStatus().getMetricIpPort()));
    }

    private void reportQueueMetricSafely(String metricName, Runnable report) {
        try {
            report.run();
        } catch (Throwable failure) {
            try {
                logger.warn("Failed to report {}", metricName, failure);
            } catch (Throwable ignoredLoggingFailure) {
                // Metrics must not affect queue admission or delivery.
            }
        }
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportFinishedBatches(List<FinishedBatch> finishedBatches) {
        finishedBatches.forEach(this::reportFinishedBatch);
    }

    private void reportFinishedBatch(FinishedBatch finishedBatch) {
        long batchId = finishedBatch.batchId();
        long actualMs = finishedBatch.actualWorkMs();
        if (!finishedBatch.anyMemberSucceeded() || actualMs <= 0) {
            logger.debug("finished batch not reportable: batchId={} success={} actualMs={}", batchId,
                    finishedBatch.anyMemberSucceeded(), actualMs);
            return;
        }

        long predictedMs = finishedBatch.predictedWorkMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.debug(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, finishedBatch.featuresAtCommit().batchSize(), getIp());

        // A failed/removed member makes the original batch an invalid learning
        // sample even if another member completed successfully.
        if (finishedBatch.usableForLearning()) {
            try {
                PrefillTimePredictor.LearningResult learningResult =
                        predictor.learn(finishedBatch.featuresAtCommit(), predictedMs, actualMs);
                if (learningResult
                        == PrefillTimePredictor.LearningResult.MODEL_UPDATED) {
                    prefillQueue.signalSchedulingInputsChanged();
                }
            } catch (RuntimeException learningFailure) {
                logger.warn("batch predictor learning failed after settlement: batchId={} engine={}",
                        batchId, getIp(), learningFailure);
            }
        }

        // These are post-settlement observers. Isolate them individually so
        // a metrics outage cannot suppress the scheduler's WorkerStatus
        // reducer or prevent the remaining observations.
        try {
            reporter.reportBatchPredictedTimeMs(
                    getStatus().getRole().name(), getStatus().getMetricIpPort(), predictedMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch predicted-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchActualTimeMs(
                    getStatus().getRole().name(), getStatus().getMetricIpPort(), actualMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch actual-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchPredictGapMs(
                    getStatus().getRole().name(), getStatus().getMetricIpPort(), gapMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch prediction-gap metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
    }

}

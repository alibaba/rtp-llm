package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.LongPredicate;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final PrefillTimePredictor predictor;
    private final WorkerBatcher runtime;
    private final PrefillState prefillState;
    private final EndpointEventSink endpointEventSink;
    private final BatchSchedulerReporter reporter;
    private final PlacementAvailability placementAvailability;

    public enum StatusSemantics {
        PREFILL_STAGE,
        FUSION_TERMINAL
    }

    /** Exact RequestSlot facts produced by this Prefill-backed generation. */
    public record StatusReduction(
            PrefillEndpoint source,
            StatusSemantics semantics,
            List<PrefillState.WorkerStatusFact> facts,
            Throwable publicationFailure)
            implements EndpointStatusReduction {
        public StatusReduction {
            java.util.Objects.requireNonNull(source, "source");
            java.util.Objects.requireNonNull(semantics, "semantics");
            facts = List.copyOf(facts);
        }
    }

    PrefillEndpoint(WorkerStatus status,
                    FlexlbConfig config,
                    DeliveryStrategy deliveryStrategy,
                    EndpointEventSink endpointEventSink,
                    EndpointEventSink deliveryLifecycle,
                    BatchSchedulerReporter reporter) {
        this(status, config, deliveryStrategy,
                endpointEventSink, deliveryLifecycle, reporter,
                new PlacementAvailability());
    }

    PrefillEndpoint(WorkerStatus status,
                    FlexlbConfig config,
                    DeliveryStrategy deliveryStrategy,
                    EndpointEventSink endpointEventSink,
                    EndpointEventSink deliveryLifecycle,
                    BatchSchedulerReporter reporter,
                    PlacementAvailability placementAvailability) {
        super(status);
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.endpointEventSink = java.util.Objects.requireNonNull(
                endpointEventSink, "endpointEventSink");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
        this.predictor = createPredictor(config);
        this.runtime = new WorkerBatcher(
                status.getIpPort(), this, config,
                deliveryStrategy, deliveryLifecycle);
        this.prefillState = runtime.ownedState();
    }

    /** Start the attached generation exactly once before routing publication. */
    void startGeneration() {
        runtime.start();
    }

    /** Capture the canonical queue/work inputs for one pure route projection. */
    public RouteProjection.Inputs captureRouteProjectionInputs() {
        return runtime.captureRouteProjectionInputs();
    }

    /** Stable delivery semantics selected once for this endpoint generation. */
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return runtime.deliveryProjection();
    }

    /** Publish one exact route after validating its generation pin. */
    public boolean offerPinned(
            GenerationPin exactPin,
            ScheduledRequest exactItem) {
        requirePinnedGeneration(exactPin);
        return runtime.offer(exactItem);
    }

    /** Exact queue-capacity commit for a fresh placement attempt. */
    public boolean offerPinnedForPlacement(
            GenerationPin exactPin,
            ScheduledRequest exactItem) {
        requirePinnedGeneration(exactPin);
        return runtime.offerForPlacement(exactItem);
    }

    /** Publish a role/group-scoped edge after real queue or status progress. */
    public void signalPlacementCapacityChanged() {
        WorkerStatus.TopologySnapshot topology =
                getStatus().topologySnapshot();
        placementAvailability.capacityChanged(
                getStatus().getRole(), topology.group());
    }

    /** Remove only the supplied canonical ACTIVE queue identity. */
    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        return runtime.removeQueued(exactItem, reason);
    }

    /** Capture immutable queue facts for timeout and eviction planning. */
    public WorkerBatcher.QueueSnapshot captureQueueSnapshot() {
        return runtime.captureQueueSnapshot();
    }

    /** Replace exact queued victims after validating this generation's pin. */
    public WorkerBatcher.QueueReplacement replaceQueued(
            GenerationPin exactPin,
            List<ScheduledRequest> exactVictims,
            ScheduledRequest incoming) {
        requirePinnedGeneration(exactPin);
        return runtime.replaceQueued(exactVictims, incoming);
    }

    public int queuedRequestCount() {
        return runtime.queueSize();
    }

    /** Runs once, after every accepted generation handoff has released its pin. */
    @Override
    protected void closeEndpoint() {
        // A self-await invariant escapes here before ledger mutation/event
        // publication. Ordinary stop cleanup failures are returned only after
        // the exact worker has exited, and are aggregated below.
        Throwable retirementFailure = runtime.stopAndAwait();
        try {
            PrefillState.Retirement retirement =
                    prefillState.retireGenerationOwnership();
            if (!retirement.ownedItems().isEmpty()) {
                try {
                    endpointEventSink.onPrefillGenerationRetired(
                            this, retirement.ownedItems());
                } catch (Throwable callbackFailure) {
                    retirementFailure = appendRetirementFailure(
                            retirementFailure, callbackFailure);
                }
            }
            retirementFailure = appendRetirementFailure(
                    retirementFailure, retirement.invariantFailure());
            List<PrefillState.BatchCompletion> completions =
                    retirement.batchCompletions();
            for (int index = 0; index < completions.size(); index++) {
                PrefillState.BatchCompletion completion =
                        completions.get(index);
                try {
                    reportBatchCompletion(completion);
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
        if (estimator instanceof RoutingConfig.LearningEstimatorConfig) {
            return new LearningPredictor();
        }
        RoutingConfig.FormulaEstimatorConfig formula =
                (RoutingConfig.FormulaEstimatorConfig) estimator;
        return new FormulaPredictor(formula.getExpression());
    }

    public PrefillState.BatchReservationResult reserveBatch(
            ScheduledRequest exactHead,
            long batchId,
            int maximumInflightBatches) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        if (handoffPermit == null) {
            return new PrefillState.BatchReservationResult(
                    PrefillState.CapacityStatus.ENDPOINT_RETIRED, null);
        }
        PrefillState.BatchReservationResult result;
        try {
            result = prefillState.reserveBatch(
                    exactHead,
                    batchId,
                    maximumInflightBatches,
                    handoffPermit::close);
        } catch (Throwable failure) {
            handoffPermit.close();
            throw failure;
        }
        if (result.reservation() == null) {
            handoffPermit.close();
        }
        return result;
    }

    public PrefillState.RouteReservationResult reserveRoute(
            ScheduledRequest exactItem,
            long predictedMs,
            int maximumRequests) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                tryAcquireGenerationHandoff();
        if (handoffPermit == null) {
            return new PrefillState.RouteReservationResult(
                    PrefillState.CapacityStatus.ENDPOINT_RETIRED, null);
        }
        PrefillState.RouteReservationResult result;
        try {
            result = prefillState.reserveRoute(
                    exactItem,
                    predictedMs,
                    maximumRequests,
                    handoffPermit::close);
        } catch (Throwable failure) {
            handoffPermit.close();
            throw failure;
        }
        if (result.reservation() == null) {
            handoffPermit.close();
        }
        return result;
    }

    /** Exact wake source for this generation's batch admission capacity. */
    public CapacityBoundary.Availability batchAdmissionAvailability(
            int maximumInflightBatches) {
        return prefillState.batchAvailability(maximumInflightBatches);
    }

    /** Exact wake source for this generation's route admission capacity. */
    public CapacityBoundary.Availability routeAdmissionAvailability(
            int maximumRequests) {
        return prefillState.routeAvailability(maximumRequests);
    }

    /**
     * Register through the exact route pin and return the sole provisional
     * rollback capability. The caller commits it only after every DIRECT role
     * has registered successfully.
     */
    public PrefillState.DirectRegistration registerDirectRequest(
            GenerationPin pin,
            long requestId,
            long predictedMs) {
        requirePinnedGeneration(pin);
        PrefillState.DirectRegistration registration =
                prefillState.tryRegisterDirect(requestId, predictedMs);
        if (registration == null) {
            throw new IllegalStateException(
                    "DIRECT request already has a live Prefill owner request_id="
                            + requestId);
        }
        return registration;
    }

    /** Exact counterpart cleanup; stale item generations are a no-op. */
    public boolean releaseCommittedItem(ScheduledRequest exactItem) {
        return prefillState.terminalizeCommittedItem(exactItem);
    }

    /**
     * Protect one route request while an EngineFence reconciles
     * ambiguous delivery ownership.
     *
     * <p>The flag lives on the request entry and is mutated under the same fixed
     * lock as progress, terminal settlement, and TTL eviction. There is no
     * auxiliary set to leak after an authoritative release/status terminal. This
     * method never acquires the batcher queue lock or calls back into the scheduler.
     *
     * @return an opaque guard bound to the exact committed item, or {@code null}
     *         when that exact generation is no longer protectable
     */
    public PrefillState.Protection acquireEngineFenceProtection(
            ScheduledRequest exactItem) {
        return prefillState.tryAcquireProtection(exactItem);
    }

    /**
     * Acquire an exact batch-member guard. A stale batch id cannot protect a
     * newer generation which reused the same request id.
     */
    public PrefillState.Protection acquireBatchMemberProtection(
            long batchId,
            ScheduledRequest exactItem) {
        return prefillState.tryAcquireBatchProtection(batchId, exactItem);
    }

    /** Release one exact Engine-fence guard and apply any deferred terminal. */
    public void releaseEngineFenceProtection(
            PrefillState.Protection protection) {
        List<PrefillState.BatchCompletion> completions =
                prefillState.releaseProtection(
                        protection,
                        this::predictRepackedBatchMs);
        try {
            reportBatchCompletions(completions);
        } catch (Throwable reportingFailure) {
            logger.warn("Engine-fence protection released but batch completion"
                    + " reporting failed: engine={}", getIp(), reportingFailure);
        }
    }

    /**
     * Requests owned by a local Prefill lifecycle: admitted QUEUE batch members
     * plus individually tracked DIRECT and QUEUE_ROUTE requests.
     */
    public int getLocallyOwnedRequestCount() {
        PrefillState.Stats stats = prefillState.stats();
        return stats.locallyOwnedRequests();
    }

    /** Individually-accounted DIRECT and QUEUE_ROUTE requests. */
    public int getIndividuallyTrackedRequestCount() {
        PrefillState.Stats stats = prefillState.stats();
        return stats.individuallyOwnedRequests();
    }

    @Override
    public EndpointStatusReduction applyPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        long pendingBefore = prefillState.pendingRequestCount();
        PrefillState.StatusReconciliation reconciliation =
                prefillState.reconcileWorkerStatus(
                        observation,
                        this::predictRepackedBatchMs,
                        () -> {
                            if (!observation.alive()) {
                                beginRetirement();
                            }
                            runtime.signalSchedulingInputsChanged();
                            ws.publishPreparedStatus(prepared);
                        },
                        this::beginRetirement);
        reportBatchCompletionsNoFail(reconciliation.batchCompletions());
        if (prefillState.pendingRequestCount() < pendingBefore) {
            signalPlacementCapacityChanged();
        }
        return new StatusReduction(
                this,
                statusSemantics(observation.role()),
                reconciliation.schedulerFacts(),
                reconciliation.publicationFailure());
    }

    @Override
    public EndpointStatusReduction initializeFromPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        PrefillState.StatusReconciliation reconciliation =
                prefillState.reconcileWorkerStatus(
                        observation,
                        this::predictRepackedBatchMs,
                        runtime::signalSchedulingInputsChanged,
                        this::beginRetirement);
        if (!reconciliation.schedulerFacts().isEmpty()
                || !reconciliation.batchCompletions().isEmpty()) {
            throw new IllegalStateException(
                    "Private Prefill candidate produced locally-owned status facts");
        }
        return new StatusReduction(
                this,
                statusSemantics(observation.role()),
                reconciliation.schedulerFacts(),
                reconciliation.publicationFailure());
    }

    @Override
    public EndpointStatusReduction observeStatusHeartbeat(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        if (observation.owner() != ws) {
            throw new IllegalArgumentException(
                    "Status observation belongs to another Prefill generation");
        }
        return new StatusReduction(
                this,
                statusSemantics(observation.role()),
                prefillState.heartbeatFacts(observation),
                null);
    }

    private static StatusSemantics statusSemantics(RoleType role) {
        return switch (role) {
            case PREFILL -> StatusSemantics.PREFILL_STAGE;
            case PDFUSION -> StatusSemantics.FUSION_TERMINAL;
            case DECODE, VIT, FRONTEND -> throw new IllegalArgumentException(
                    "PrefillEndpoint cannot reduce role " + role);
        };
    }

    private void reportBatchCompletionsNoFail(
            List<PrefillState.BatchCompletion> completions) {
        try {
            reportBatchCompletions(completions);
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
     * Membership settlement must not depend on the optional cost estimator.
     * If prediction fails, the batch still loses the finished members while
     * its remaining-work estimate becomes explicitly unavailable.
     */
    private OptionalLong predictRepackedBatchMs(
            List<ScheduledRequest> survivingRequests) {
        try {
            PrefillTimePredictor.Evaluator evaluator = predictor.evaluator();
            return OptionalLong.of(
                    PrefillPredictionBoundary.predictCommittedBatchMs(
                            evaluator,
                            PrefillBatchFeatures.from(
                                    survivingRequests,
                                    ScheduledRequest::seqLen,
                                    ScheduledRequest::hitCache)));
        } catch (Throwable predictionFailure) {
            try {
                logger.error("Prefill batch repack prediction failed; marking work unavailable "
                                + "engine={} surviving_requests={}",
                        getIp(), survivingRequests.size(), predictionFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Optional prediction and its telemetry cannot block settlement.
            }
            return OptionalLong.empty();
        }
    }

    // ==================== Pending Count ====================

    /** Canonical pending ownership used by the hard admission threshold. */
    public long admissionPendingRequestCount() {
        return prefillState.pendingRequestCount();
    }

    public int getInflightBatchCount() {
        PrefillState.Stats stats = prefillState.stats();
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

    /** Evict only batches with no request generation still owned by the scheduler. */
    public int evictExpiredBatches(long ttlMs,
                                   LongPredicate schedulerOwnsRequest) {
        return prefillState.evictExpiredBatches(
                ttlMs, schedulerOwnsRequest);
    }

    /**
     * Evict individually-accounted requests that have not appeared in WorkerStatus
     * for longer than {@code ttlMs}.
     *
     * <p>The stale check is repeated while holding the request's stripe. Progress
     * observation, explicit release, and TTL removal are therefore linearizable and
     * an observation racing the first optimistic check cannot be evicted as stale.
     */
    /** Evict route-request entries which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return prefillState.evictExpiredIndividuals(
                ttlMs, schedulerOwnsRequest);
    }

    /** Evict endpoint orphans without racing scheduler-owned generations. */
    public int evictExpiredInflight(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return evictExpiredBatches(ttlMs, schedulerOwnsRequest)
                + evictExpiredRequests(ttlMs, schedulerOwnsRequest);
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
     * Called periodically by {@link org.flexlb.balance.scheduler.RequestScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        int queueSize = runtime.queueSize();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority =
                runtime.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), 0, 0);
        } else {
            sizeByPriority.forEach((priority, size) ->
                    reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), priority, size));
        }
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), getInflightBatchCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), getLocallyOwnedRequestCount());
        reporter.reportInflightMaxAgeMs(
                RoleType.PREFILL.name(),
                getIp(),
                prefillState.stats().maxObservedAgeMs());
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportBatchCompletions(
            List<PrefillState.BatchCompletion> completions) {
        completions.forEach(this::reportBatchCompletion);
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
                    runtime.signalSchedulingInputsChanged();
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
            reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch predicted-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch actual-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch prediction-gap metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
    }

}

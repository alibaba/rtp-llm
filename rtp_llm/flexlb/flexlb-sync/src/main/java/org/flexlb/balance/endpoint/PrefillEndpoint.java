package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import java.util.stream.Collectors;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final int COMBINED_WAIT_SNAPSHOT_MAX_ATTEMPTS = 4;

    enum WaitSnapshotStage {
        AFTER_REQUEST_QUEUED_READ,
        BEFORE_CACHE_PUBLISH
    }

    @FunctionalInterface
    interface WaitSnapshotHook {
        void onStage(WaitSnapshotStage stage);
    }

    private record FinishedObservation(long requestId,
                                       long executionTimeMs,
                                       long errorCode,
                                       String errorMessage) {
        static FinishedObservation from(TaskInfo task) {
            return new FinishedObservation(task.getRequestId(), task.getExecutionTimeMs(),
                    task.getErrorCode(), task.getErrorMessage());
        }

        FinishedObservation merge(FinishedObservation other) {
            long mergedErrorCode = errorCode != 0 ? errorCode : other.errorCode;
            String mergedErrorMessage = errorCode != 0 ? errorMessage : other.errorMessage;
            return new FinishedObservation(requestId,
                    Math.max(executionTimeMs, other.executionTimeMs),
                    mergedErrorCode, mergedErrorMessage);
        }
    }

    private record BatchMemberProtection(FinishedObservation deferredTerminal) {}

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, ConcurrentHashMap<Long, BatchMemberProtection>>
            batchMemberProtections = new ConcurrentHashMap<>();
    private final AtomicInteger inflightBatchRequestCount = new AtomicInteger();
    private final WorkerBatcher batcher;
    private final PrefillRequestLedger requestLedger;
    private final BatchSchedulerReporter reporter;

    /** Active Engine tasks not already represented in the local batch ledger. */
    private volatile long engineUntrackedRequestCount = 0;

    /**
     * Independent batch-ledger epoch used with the request-ledger epoch to form a
     * coherent combined wait snapshot without acquiring locks across ledgers.
     */
    private final AtomicLong batchWaitMutationsStarted = new AtomicLong();
    private final AtomicLong batchWaitMutationsCompleted = new AtomicLong();
    private final AtomicReference<WaitSnapshotHook> waitSnapshotHook = new AtomicReference<>();

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           DecisionGroupHandler decisionHandler,
                           BatchSchedulerReporter reporter) {
        super(status);
        this.reporter = reporter;
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, decisionHandler, reporter);
        AtomicReference<WaitSnapshotHook> snapshotHook = waitSnapshotHook;
        this.requestLedger = new PrefillRequestLedger(
                batcher::signalDeliveryCapacityAvailable,
                System::currentTimeMillis,
                stage -> notifyWaitSnapshotHook(snapshotHook, stage));
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config, DecisionGroupHandler decisionHandler,
                                        BatchSchedulerReporter reporter) {
        return new WorkerBatcher(status.getIpPort(), this, config, decisionHandler, reporter);
    }

    public WorkerBatcher getBatcher() {
        return batcher;
    }

    @Override
    public void close() {
        try {
            batcher.shutdown();
        } finally {
            super.close();
        }
    }

    public long batcherWaitMs() {
        return batcher.queueWaitMs();
    }

    /**
     * Auto-TPM priority-aware queue wait estimate (design doc 8.4):
     * counts only items ordered ahead of the incoming request.
     */
    public long batcherEstimatedWaitMs(int priority, long requestId) {
        return batcher.queueManager().estimateWaitMs(priority, requestId);
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        RoutingConfig.ExecutionTimeEstimatorConfig estimator = cfg.getRouter()
                .getRoles().getPrefill().getExecutionTimeEstimator();
        if (estimator instanceof RoutingConfig.LearningEstimatorConfig) {
            return new LearningPredictor();
        }
        RoutingConfig.FormulaEstimatorConfig formula =
                (RoutingConfig.FormulaEstimatorConfig) estimator;
        return new FormulaPredictor(formula.getExpression());
    }

    public void commitBatch(long batchId, long predictMs, List<BatchItem> requests) {
        BatchInflight newBatch = new BatchInflight(predictMs, requests);
        beginBatchWaitMutation();
        try {
            inflightBatches.compute(batchId, (id, previous) -> {
                int previousSize = previous != null ? previous.requests().size() : 0;
                inflightBatchRequestCount.addAndGet(
                        newBatch.requests().size() - previousSize);
                return newBatch;
            });
        } finally {
            endBatchWaitMutation();
        }
    }

    public void releaseBatch(long batchId) {
        long statusMs = System.currentTimeMillis();
        beginBatchWaitMutation();
        try {
            inflightBatches.compute(batchId, (id, batch) -> {
                // Keep the lock order consistent with protection/calibration:
                // inflight batch key first, batch-member protection key second.
                ConcurrentHashMap<Long, BatchMemberProtection> protectedRequests =
                        batchMemberProtections.get(id);
                if (batch == null || protectedRequests == null
                        || protectedRequests.isEmpty()) {
                    batchMemberProtections.remove(id);
                    if (batch != null) {
                        inflightBatchRequestCount.addAndGet(-batch.requests().size());
                    }
                    return null;
                }

                // A transport failure can race an Engine fence installation.
                // Retain exactly those members whose protection owner won the
                // batch-key linearization point; unprotected siblings are safe to
                // release immediately.
                List<BatchItem> survivors = batch.requests().stream()
                        .filter(item -> protectedRequests.containsKey(item.requestId()))
                        .toList();
                int removed = batch.requests().size() - survivors.size();
                if (survivors.isEmpty()) {
                    batchMemberProtections.remove(id, protectedRequests);
                    inflightBatchRequestCount.addAndGet(-batch.requests().size());
                    return null;
                }
                if (removed == 0) {
                    batch.touch(statusMs);
                    batch.observeFailure();
                    return batch;
                }
                batch.touch(statusMs);
                batch.observeFailure();
                long newPredMs = (long) predictor.predictBatchMs(survivors);
                BatchInflight repacked = batch.repack(newPredMs, survivors);
                inflightBatchRequestCount.addAndGet(-removed);
                return repacked;
            });
        } finally {
            endBatchWaitMutation();
        }
    }

    /**
     * Atomically account for one route-decision request.
     *
     * <p>A positive {@code maxPerWorker} is a hard admission cap over
     * individually delivered route requests. Real batch members retain their
     * independent batch-count controls and contribute only to the total request metric. A
     * non-positive limit disables the route cap. Repeating the same live request id
     * is idempotent and returns {@code true} without replacing its original prediction
     * or incrementing either counter.
     * Request ids are delivery identities and must not be reused while an old
     * WorkerStatus terminal for the id can still arrive; Engine status does not
     * carry a route-delivery generation token that could disambiguate such reuse.
     *
     * <p>Mutations for a request id are serialized by a fixed stripe. No request
     * stripe is acquired while holding a batch/protection map lock (or vice
     * versa), so request accounting cannot participate in a cross-ledger lock cycle.
     */
    public boolean tryCommitRequest(long requestId, long predictMs, int maxPerWorker) {
        return requestLedger.tryAcquire(requestId, predictMs, maxPerWorker);
    }

    /**
     * Idempotently release an individually-accounted request.
     *
     * @return {@code true} only when this call removed the live ledger entry
     */
    public boolean releaseRequest(long requestId) {
        return requestLedger.release(requestId);
    }

    /**
     * Protect one route-request ledger entry while an EngineFence reconciles
     * ambiguous delivery ownership.
     *
     * <p>The flag lives on the request entry and is mutated under the same fixed
     * stripe as progress, terminal settlement, and TTL eviction. There is no
     * auxiliary set to leak after an authoritative release/status terminal. This
     * method never acquires the batcher queue lock or calls back into the scheduler.
     *
     * @return {@code true} when the request is still locally accounted (including
     *         an already-protected request), otherwise {@code false}
     */
    public boolean beginEngineFenceProtection(long requestId) {
        return requestLedger.protect(requestId);
    }

    /**
     * End request-scoped EngineFence protection without refreshing its TTL age.
     * A request that was already released or authoritatively settled is a no-op.
     *
     * @return {@code true} only when a live protection flag was cleared
     */
    public boolean endEngineFenceProtection(long requestId) {
        return requestLedger.unprotect(requestId);
    }

    /** Advisory capacity snapshot; {@link #tryCommitRequest} is the hard gate. */
    public int availableRequestSlots(int maxPerWorker) {
        return requestLedger.available(maxPerWorker);
    }

    /** Total locally-accounted Prefill requests, including real batch members. */
    public int getInflightRequestCount() {
        return inflightBatchRequestCount.get() + requestLedger.count();
    }

    /** Individually-accounted route-decision requests only. */
    public int getInflightRouteRequestCount() {
        return requestLedger.count();
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     */
    public void repackBatch(long batchId, Set<Long> failedRequestIds) {
        long statusMs = System.currentTimeMillis();
        beginBatchWaitMutation();
        try {
            inflightBatches.computeIfPresent(batchId, (id, old) -> {
                List<BatchItem> survivors = old.requests().stream()
                        .filter(r -> !failedRequestIds.contains(r.requestId()))
                        .toList();
                int removed = old.requests().size() - survivors.size();
                if (removed == 0) {
                    return old;
                }
                old.touch(statusMs);
                old.observeFailure();
                if (survivors.isEmpty()) {
                    inflightBatchRequestCount.addAndGet(-old.requests().size());
                    return null; // removes entry from map
                }
                long newPredMs = (long) predictor.predictBatchMs(survivors);
                BatchInflight repacked = old.repack(newPredMs, survivors);
                inflightBatchRequestCount.addAndGet(-removed);
                return repacked;
            });
        } finally {
            endBatchWaitMutation();
        }
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        Set<Long> activeNonRouteRequestIds =
                calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
        updateEngineUntrackedRequestCount(resp, activeNonRouteRequestIds);
    }

    /**
     * Full calibration against worker status report.
     */
    private Set<Long> calibrate(Map<String, TaskInfo> finishedTaskInfo,
                                Map<String, TaskInfo> runningTaskInfo) {
        long statusMs = System.currentTimeMillis();

        int finishedSize = finishedTaskInfo != null ? finishedTaskInfo.size() : 0;
        int runningSize = runningTaskInfo != null ? runningTaskInfo.size() : 0;
        if (logger.isDebugEnabled()
                && (finishedSize > 0 || !inflightBatches.isEmpty() || requestLedger.count() > 0)) {
            logger.debug("Prefill calibrate: finishedTasks={}, runningTasks={}, "
                            + "inflightBatches={}, inflightRouteRequests={}",
                    finishedSize, runningSize, inflightBatches.size(), requestLedger.count());
        }

        // Phase 1: settle route-decision requests directly by request id, then collect
        // terminal observations owned by real batches. Checking the request ledger
        // first also tolerates Engine versions that attach a synthetic batch id to an
        // individually submitted request.
        Map<Long, List<FinishedObservation>> finishedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                if (settleRequest(task)) {
                    continue;
                }
                FinishedObservation observation = FinishedObservation.from(task);
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    reconcileFinishedWithoutBatchId(observation, statusMs);
                    continue;
                }
                finishedByBatch.computeIfAbsent(batchId, k -> new ArrayList<>()).add(observation);
            }
        }

        // Phase 2: settle only the locally-owned finished members. WorkerStatus
        // is request-granular and the Engine may partially admit a batch, so a
        // short member finishing must not release long-running siblings or
        // reopen the fixed-window inflight gate.
        for (Map.Entry<Long, List<FinishedObservation>> entry : finishedByBatch.entrySet()) {
            settleFinishedMembers(entry.getKey(), entry.getValue(), statusMs);
        }

        // Phase 3: update progress anchors. A queued batch cannot spend
        // predicted forward time until the worker reports it as RUNNING.
        Map<Long, List<TaskInfo>> activeByBatch = new HashMap<>();
        Set<Long> activeNonRouteRequestIds = new HashSet<>();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                if (observeRequestProgress(task, statusMs)) {
                    continue;
                }
                if (!isPriorityCancelOverlayOnly(task)) {
                    activeNonRouteRequestIds.add(task.getRequestId());
                }
                long batchId = task.getBatchId();
                if (batchId >= 0) {
                    activeByBatch.computeIfAbsent(batchId, ignored -> new ArrayList<>()).add(task);
                }
            }
        }
        for (Map.Entry<Long, List<TaskInfo>> entry : activeByBatch.entrySet()) {
            inflightBatches.computeIfPresent(entry.getKey(), (id, batch) -> {
                Set<Long> currentRequestIds = batch.requests().stream()
                        .map(BatchItem::requestId)
                        .collect(Collectors.toSet());
                boolean observedCurrentMember = false;
                boolean observedRunningMember = false;
                for (TaskInfo task : entry.getValue()) {
                    if (!currentRequestIds.contains(task.getRequestId())) {
                        continue;
                    }
                    observedCurrentMember = true;
                    observedRunningMember |= task.getPhase() == TaskPhase.RUNNING;
                }
                if (!observedCurrentMember) {
                    return batch;
                }
                beginBatchWaitMutation();
                try {
                    if (observedRunningMember) {
                        batch.markRunning(statusMs);
                    } else {
                        batch.markQueued(statusMs);
                    }
                } finally {
                    endBatchWaitMutation();
                }
                return batch;
            });
        }

        // Phase 4: check non-route running requests for anomalies.
        for (Map.Entry<Long, List<TaskInfo>> entry : activeByBatch.entrySet()) {
            if (!inflightBatches.containsKey(entry.getKey())) {
                for (TaskInfo task : entry.getValue()) {
                    logger.debug("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), entry.getKey());
                }
            }
        }
        return activeNonRouteRequestIds;
    }

    private boolean settleRequest(TaskInfo task) {
        long requestId = task.getRequestId();
        if (!requestLedger.settle(requestId)) {
            return false;
        }
        if (task.getErrorCode() != 0) {
            logger.debug("Prefill calibrate: request failure reqId={} error={}",
                    requestId, task.getErrorMessage());
        }
        return true;
    }

    private boolean observeRequestProgress(TaskInfo task, long statusMs) {
        return requestLedger.observe(task.getRequestId(),
                task.getPhase() == TaskPhase.RUNNING, statusMs);
    }

    private void settleFinishedMembers(long batchId,
                                       List<FinishedObservation> observations,
                                       long statusMs) {
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
            inflightBatches.computeIfPresent(batchId, (id, batch) ->
                    applyFinishedObservations(
                            id, batch, observations, statusMs, true, completed));
        } finally {
            endBatchWaitMutation();
        }

        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private BatchInflight applyFinishedObservations(long batchId,
                                                     BatchInflight batch,
                                                     List<FinishedObservation> observations,
                                                     long statusMs,
                                                     boolean deferProtectedMembers,
                                                     AtomicReference<BatchInflight> completed) {
        Set<Long> localRequestIds = batch.requests().stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toSet());
        Set<Long> finishedIds = new HashSet<>();
        int foreignCount = 0;

        for (FinishedObservation observation : observations) {
            long requestId = observation.requestId();
            if (!localRequestIds.contains(requestId)) {
                // Finished snapshots can repeat a member already settled
                // in a previous calibration pass. Warn only for a request
                // that never belonged to this batch generation.
                if (!batch.originalRequestIds().contains(requestId)) {
                    foreignCount++;
                }
                continue;
            }

            batch.touch(statusMs);
            batch.observeExecutionTime(observation.executionTimeMs());
            if (observation.errorCode() == 0) {
                batch.observeSuccessfulCompletion();
            } else {
                batch.observeFailure();
            }
            if (deferProtectedMembers
                    && deferIfBatchMemberProtected(batchId, observation)) {
                continue;
            }

            finishedIds.add(requestId);
            if (observation.errorCode() != 0) {
                logger.debug("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                        batchId, requestId, observation.errorMessage());
            }
        }

        if (foreignCount > 0) {
            logger.warn("Prefill calibrate: batchId={} has {} finished tasks with foreign requestIds; "
                            + "ignoring them",
                    batchId, foreignCount);
        }
        if (finishedIds.isEmpty()) {
            return batch;
        }

        List<BatchItem> survivors = batch.requests().stream()
                .filter(item -> !finishedIds.contains(item.requestId()))
                .toList();
        inflightBatchRequestCount.addAndGet(
                -(batch.requests().size() - survivors.size()));
        if (survivors.isEmpty()) {
            completed.set(batch);
            return null;
        }

        long newPredMs = (long) predictor.predictBatchMs(survivors);
        return batch.repack(newPredMs, survivors);
    }

    /**
     * Reconcile a finished task whose Engine status omitted the original batch id.
     *
     * <p>Legacy non-batch reservations are keyed by request id and carry an empty
     * member list. A real batch is keyed by its generated batch id and always
     * carries its request members. Checking the value shape before the direct
     * removal prevents an unrelated real batch from being erased when its batch
     * id happens to equal this request id.
     *
     * <p>Production priority-cancel terminals may currently report
     * {@code batch_id=-1} even though the Master committed the request as a member
     * of a real batch. In that case scan the live ledger for the unique owning
     * batch and remove only the matching member. The member is revalidated inside
     * the map compute, so a concurrent release/repack/TTL eviction is an idempotent
     * no-op rather than a counter double-decrement. No reverse index is retained,
     * keeping every existing ledger mutation path consistent automatically.
     */
    private void reconcileFinishedWithoutBatchId(FinishedObservation observation, long statusMs) {
        long requestId = observation.requestId();
        AtomicBoolean removedNonBatch = new AtomicBoolean(false);
        beginBatchWaitMutation();
        try {
            inflightBatches.computeIfPresent(requestId, (id, batch) -> {
                if (!batch.requests().isEmpty()) {
                    return batch;
                }
                removedNonBatch.set(true);
                batchMemberProtections.remove(id);
                inflightBatchRequestCount.addAndGet(-batch.requests().size());
                return null;
            });
        } finally {
            endBatchWaitMutation();
        }
        if (removedNonBatch.get()) {
            return;
        }

        List<Long> matchingBatchIds = new ArrayList<>();
        for (Map.Entry<Long, BatchInflight> entry : inflightBatches.entrySet()) {
            boolean containsRequest = entry.getValue().requests().stream()
                    .anyMatch(item -> item.requestId() == requestId);
            if (containsRequest) {
                matchingBatchIds.add(entry.getKey());
            }
        }
        if (matchingBatchIds.isEmpty()) {
            logger.debug("Prefill calibrate: finished task with no batch id reqId={} not in inflight",
                    requestId);
            return;
        }
        if (matchingBatchIds.size() != 1) {
            // A request is expected to belong to exactly one live batch. Do not
            // guess when that invariant is already broken: member-scoped cleanup
            // in multiple generations could erase a newer dispatch.
            logger.warn("Prefill calibrate: finished task with no batch id reqId={} matches batches={}; "
                            + "skipping ambiguous cleanup",
                    requestId, matchingBatchIds);
            return;
        }

        long resolvedBatchId = matchingBatchIds.get(0);
        settleFinishedMembers(resolvedBatchId, List.of(observation), statusMs);
    }

    private void updateEngineUntrackedRequestCount(
            WorkerStatusResponse response, Set<Long> activeNonRouteRequestIds) {
        // Real batches are few and need a membership set. Route membership was
        // classified while applying the same WorkerStatus observation, avoiding
        // a second request-ledger lookup or a copy of all live route ids.
        Set<Long> localBatchRequestIds = new HashSet<>();
        for (BatchInflight batch : inflightBatches.values()) {
            for (BatchItem request : batch.requests()) {
                localBatchRequestIds.add(request.requestId());
            }
        }

        activeNonRouteRequestIds.removeAll(localBatchRequestIds);

        long reportedActive = Math.max(0, response.getWaitingQueryLen())
                + Math.max(0, response.getRunningQueryLen());
        long scalarLowerBound = Math.max(
                0, reportedActive - Math.max(0, getInflightRequestCount()));
        // The protobuf converter represents an absent detail list as an empty map,
        // while older/newer Engine variants may still populate only the scalar
        // counts. Keep the request-id union when details exist and conservatively
        // retain the scalar lower bound when the detail list is empty or partial.
        engineUntrackedRequestCount = Math.max(
                activeNonRouteRequestIds.size(), scalarLowerBound);
    }

    private static boolean isPriorityCancelOverlayOnly(TaskInfo task) {
        PriorityPreemptionProgress progress = task.getPriorityPreemptionProgress();
        return (progress == PriorityPreemptionProgress.CANCELING
                || progress == PriorityPreemptionProgress.CANCELED)
                && task.getPhase() == TaskPhase.PENDING;
    }

    // ==================== Pending Count ====================

    /**
     * Real pending count: total requests the engine will face.
     * Includes master-tracked inflight + batcher queue + active Engine tasks
     * not already represented in the local ledger.
     */
    public long realPendingCount() {
        return getInflightRequestCount() + batcher.queueSize() + engineUntrackedRequestCount;
    }

    // ==================== Wait Time ====================

    /**
     * Real wait time: estimated time to drain current inflight batches.
     */
    public long realWaitTimeMs() {
        return estimateWaitingTimeMs(System.currentTimeMillis());
    }

    public int getInflightBatchCount() {
        return inflightBatches.size();
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
        long nowMs = System.currentTimeMillis();
        AtomicInteger evictedCount = new AtomicInteger();
        for (Long batchId : inflightBatches.keySet()) {
            BatchInflight candidate = inflightBatches.get(batchId);
            if (candidate == null
                    || hasProtectedBatchMember(batchId)
                    || batchHasOwnedRequest(candidate, schedulerOwnsRequest)
                    || nowMs - candidate.lastObservedAtMs() <= ttlMs) {
                continue;
            }
            AtomicReference<BatchInflight> evicted = new AtomicReference<>();
            beginBatchWaitMutation();
            try {
                inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                    if (hasProtectedBatchMember(id)
                            || batchHasOwnedRequest(batch, schedulerOwnsRequest)
                            || nowMs - batch.lastObservedAtMs() <= ttlMs) {
                        return batch;
                    }
                    evicted.set(batch);
                    return null;
                });
            } finally {
                endBatchWaitMutation();
            }
            BatchInflight removed = evicted.get();
            if (removed != null) {
                inflightBatchRequestCount.addAndGet(-removed.requests().size());
                evictedCount.incrementAndGet();
            }
        }
        return evictedCount.get();
    }

    private static boolean batchHasOwnedRequest(
            BatchInflight batch, LongPredicate schedulerOwnsRequest) {
        for (BatchItem item : batch.requests()) {
            if (schedulerOwnsRequest.test(item.requestId())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Evict individually-accounted requests that have not appeared in WorkerStatus
     * for longer than {@code ttlMs}.
     *
     * <p>The stale check is repeated while holding the request's stripe. Progress
     * observation, explicit release, and TTL removal are therefore linearizable and
     * an observation racing the first optimistic check cannot be evicted as stale.
     */
    public int evictExpiredRequests(long ttlMs) {
        return requestLedger.evict(ttlMs);
    }

    /** Evict route-request entries which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return requestLedger.evict(ttlMs, schedulerOwnsRequest);
    }

    /** Evict stale entries from both Prefill accounting ledgers. */
    public int evictExpiredInflight(long ttlMs) {
        return evictExpiredBatches(ttlMs) + evictExpiredRequests(ttlMs);
    }

    /** Evict endpoint orphans without racing scheduler-owned generations. */
    public int evictExpiredInflight(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return evictExpiredBatches(ttlMs, schedulerOwnsRequest)
                + evictExpiredRequests(ttlMs, schedulerOwnsRequest);
    }

    /**
     * Protect an ACK-ambiguous batch member from settlement and age-only eviction.
     *
     * <p>The ownership check and guard publication run inside the same
     * {@code inflightBatches} key computation used by WorkerStatus settlement,
     * explicit release, and TTL revalidation. A {@code true} result therefore
     * proves that every later destructive mutation for this batch observes the
     * guard; {@code false} proves that settlement won first or the request did not
     * belong to that batch, and no guard was installed.</p>
     */
    public boolean tryProtectBatchMember(long batchId, long requestId) {
        long nowMs = System.currentTimeMillis();
        AtomicBoolean protectedMember = new AtomicBoolean();
        inflightBatches.computeIfPresent(batchId, (id, batch) -> {
            boolean owned = false;
            for (BatchItem item : batch.requests()) {
                if (item.requestId() == requestId) {
                    owned = true;
                    break;
                }
            }
            if (!owned) {
                return batch;
            }
            batchMemberProtections.compute(id, (ignored, requests) -> {
                ConcurrentHashMap<Long, BatchMemberProtection> states = requests != null
                        ? requests : new ConcurrentHashMap<>();
                states.putIfAbsent(requestId, new BatchMemberProtection(null));
                return states;
            });
            batch.touch(nowMs);
            protectedMember.set(true);
            return batch;
        });
        return protectedMember.get();
    }

    /** Release one batch member's protection after authoritative settlement. */
    public void releaseBatchMemberProtection(long batchId, long requestId) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
            inflightBatches.compute(batchId, (id, batch) -> {
                AtomicReference<FinishedObservation> deferredTerminal = new AtomicReference<>();
                batchMemberProtections.computeIfPresent(id, (ignored, requests) -> {
                    BatchMemberProtection state = requests.remove(requestId);
                    if (state != null) {
                        deferredTerminal.set(state.deferredTerminal());
                    }
                    return requests.isEmpty() ? null : requests;
                });
                if (batch == null) {
                    return null;
                }
                FinishedObservation observation = deferredTerminal.get();
                if (observation == null) {
                    batch.touch(statusMs);
                    return batch;
                }
                // The protection was removed under the same inflight-key critical section,
                // so apply the cached terminal directly instead of trying to defer it again.
                return applyFinishedObservations(
                        id, batch, List.of(observation), statusMs, false, completed);
            });
        } finally {
            endBatchWaitMutation();
        }
        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private boolean deferIfBatchMemberProtected(
            long batchId, FinishedObservation observation) {
        AtomicBoolean deferred = new AtomicBoolean(false);
        batchMemberProtections.computeIfPresent(batchId, (ignored, requests) -> {
            requests.computeIfPresent(observation.requestId(), (requestId, state) -> {
                deferred.set(true);
                FinishedObservation existing = state.deferredTerminal();
                return new BatchMemberProtection(existing == null
                        ? observation : existing.merge(observation));
            });
            return requests.isEmpty() ? null : requests;
        });
        return deferred.get();
    }

    private boolean hasProtectedBatchMember(long batchId) {
        ConcurrentHashMap<Long, BatchMemberProtection> requests =
                batchMemberProtections.get(batchId);
        return requests != null && !requests.isEmpty();
    }

    @Override
    public long getLoadMetric() {
        return realWaitTimeMs();
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.PriorityScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        int queueSize = batcher.queueSize();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority = batcher.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), 0, 0);
        } else {
            sizeByPriority.forEach((priority, size) ->
                    reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), priority, size));
        }
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), getInflightBatchCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), getInflightRequestCount());
        long nowMs = System.currentTimeMillis();
        long maxAgeMs = Math.max(
                InflightEvictor.maxAgeMs(inflightBatches, nowMs),
                requestLedger.maxAge(nowMs));
        reporter.reportInflightMaxAgeMs(RoleType.PREFILL.name(), getIp(), maxAgeMs);
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportBatchCompletion(long batchId, BatchInflight batch) {
        long actualMs = batch.maxExecutionTimeMs();
        if (!batch.successfulCompletionObserved() || actualMs <= 0) {
            logger.debug("batch completion not reportable: batchId={} success={} actualMs={}",
                    batchId, batch.successfulCompletionObserved(), actualMs);
            return;
        }

        long predictedMs = batch.originalPredictTimeMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.debug(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, batch.originalFeatures().batchSize(), getIp());

        // A failed/removed member makes the original batch an invalid learning
        // sample even if another member completed successfully.
        if (batch.learningEligible()) {
            try {
                predictor.learn(batch.originalFeatures(), predictedMs, actualMs);
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

    private long estimateWaitingTimeMs(long nowMs) {
        for (int attempt = 0; attempt < COMBINED_WAIT_SNAPSHOT_MAX_ATTEMPTS; attempt++) {
            long batchVersionBefore = batchWaitMutationVersion();
            long requestVersionBefore = requestLedger.mutationVersion();
            if (batchVersionBefore < 0 || requestVersionBefore < 0) {
                Thread.onSpinWait();
                continue;
            }

            long batchWaitMs = computeBatchWaitingTimeMs(nowMs);
            long requestWaitMs = requestLedger.estimate(nowMs);
            if (requestWaitMs == Long.MAX_VALUE) {
                return Long.MAX_VALUE;
            }

            // Validate in reverse order so both component reads are bracketed by
            // their own monotonic epochs. This stays lock-free across ledgers while
            // preventing a BATCH/QUEUE transition from publishing a torn low sum.
            long requestVersionAfter = requestLedger.mutationVersion();
            long batchVersionAfter = batchWaitMutationVersion();
            if (batchVersionBefore == batchVersionAfter
                    && requestVersionBefore == requestVersionAfter) {
                return saturatedAdd(batchWaitMs, requestWaitMs);
            }
            Thread.onSpinWait();
        }

        // Continuous cross-ledger mutation cannot produce a coherent bounded
        // snapshot without blocking admission. Route away conservatively.
        return Long.MAX_VALUE;
    }

    private long computeBatchWaitingTimeMs(long nowMs) {
        if (inflightBatches.isEmpty()) {
            return 0;
        }
        long batchPredMs = 0;
        long earliestBatchProgressBaseMs = Long.MAX_VALUE;
        for (BatchInflight batch : inflightBatches.values()) {
            batchPredMs = saturatedAdd(batchPredMs, Math.max(0, batch.predictTimeMs()));
            // Preserve legacy batch semantics: an inflight batch begins aging at
            // commit and WorkerStatus may subsequently re-anchor it.
            earliestBatchProgressBaseMs = Math.min(
                    earliestBatchProgressBaseMs, batch.progressBaseMs());
        }
        long batchWaitMs = earliestBatchProgressBaseMs == Long.MAX_VALUE
                ? 0
                : Math.max(0, batchPredMs
                        - Math.max(0, nowMs - earliestBatchProgressBaseMs));
        return batchWaitMs;
    }

    private void beginBatchWaitMutation() {
        batchWaitMutationsStarted.incrementAndGet();
        VarHandle.storeStoreFence();
    }

    private void endBatchWaitMutation() {
        VarHandle.storeStoreFence();
        batchWaitMutationsCompleted.incrementAndGet();
    }

    private long batchWaitMutationVersion() {
        VarHandle.loadLoadFence();
        long started = batchWaitMutationsStarted.get();
        long completed = batchWaitMutationsCompleted.get();
        return started == completed ? completed : -1;
    }

    /** Package-private deterministic interleaving hook; always null in production. */
    void setWaitSnapshotHookForTest(WaitSnapshotHook hook) {
        waitSnapshotHook.set(hook);
    }

    private static void notifyWaitSnapshotHook(
            AtomicReference<WaitSnapshotHook> hookReference,
            PrefillRequestLedger.WaitSnapshotStage ledgerStage) {
        WaitSnapshotHook hook = hookReference.get();
        if (hook != null) {
            WaitSnapshotStage endpointStage = switch (ledgerStage) {
                case AFTER_QUEUED_READ -> WaitSnapshotStage.AFTER_REQUEST_QUEUED_READ;
                case BEFORE_CACHE_PUBLISH -> WaitSnapshotStage.BEFORE_CACHE_PUBLISH;
            };
            hook.onStage(endpointStage);
        }
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

}

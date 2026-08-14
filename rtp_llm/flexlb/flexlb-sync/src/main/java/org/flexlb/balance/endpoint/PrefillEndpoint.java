package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

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

    private record ReconciliationState(FinishedObservation deferredTerminal) {}

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, ConcurrentHashMap<Long, ReconciliationState>>
            reconciliationRequests = new ConcurrentHashMap<>();
    private final AtomicInteger inflightRequestCount = new AtomicInteger(0);
    private final WorkerBatcher batcher;
    private final BatchSchedulerReporter reporter;

    /** Active Engine tasks not already represented in the local batch ledger. */
    private volatile long engineUntrackedRequestCount = 0;

    private static final long WAIT_TIME_CACHE_TTL_MS = 2;
    private volatile long cachedWaitTimeMs = 0;
    private volatile long cachedWaitTimeExpireAtMs = 0;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           BatchDecisionHandler handler,
                           BatchSchedulerReporter reporter) {
        super(status);
        this.reporter = reporter;
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, handler, reporter);
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config, BatchDecisionHandler handler,
                                        BatchSchedulerReporter reporter) {
        return new WorkerBatcher(status.getIpPort(), this, config, handler, reporter);
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
    public long batcherEstimatedWaitMs(int priority, long deadlineMs, long requestId) {
        return batcher.queueManager().estimateWaitMs(priority, deadlineMs, requestId);
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        if ("learning".equalsIgnoreCase(cfg.getPrefillPredictorType())) {
            return new LearningPredictor();
        }
        return new FormulaPredictor(cfg.getCostFormula());
    }

    public void commitBatch(long batchId, long predictMs, List<BatchItem> requests) {
        BatchInflight newBatch = new BatchInflight(predictMs, requests);
        BatchInflight prev = inflightBatches.putIfAbsent(batchId, newBatch);
        if (prev != null) {
            // batchId already exists — subtract the old request count before overwriting,
            // otherwise the old value is silently lost and the counter stays inflated.
            inflightRequestCount.addAndGet(-prev.requests().size());
            inflightBatches.put(batchId, newBatch);
        }
        inflightRequestCount.addAndGet(requests.size());
        cachedWaitTimeExpireAtMs = 0;
    }

    public void releaseBatch(long batchId) {
        AtomicReference<BatchInflight> removedBatch = new AtomicReference<>();
        inflightBatches.compute(batchId, (id, batch) -> {
            // Keep the lock order consistent with begin/calibration:
            // inflight batch key first, reconciliation key second.
            reconciliationRequests.remove(id);
            removedBatch.set(batch);
            return null;
        });
        BatchInflight removed = removedBatch.get();
        if (removed != null) {
            inflightRequestCount.addAndGet(-removed.requests().size());
            cachedWaitTimeExpireAtMs = 0;
        }
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     */
    public void repackBatch(long batchId, Set<Long> failedRequestIds) {
        long statusMs = System.currentTimeMillis();
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
                inflightRequestCount.addAndGet(-old.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                return null; // removes entry from map
            }
            long newPredMs = (long) predictor.predictBatchMs(survivors);
            BatchInflight repacked = old.repack(newPredMs, survivors);
            inflightRequestCount.addAndGet(-removed);
            cachedWaitTimeExpireAtMs = 0;
            return repacked;
        });
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
        updateEngineUntrackedRequestCount(resp);
    }

    /**
     * Full calibration against worker status report.
     */
    private void calibrate(Map<String, TaskInfo> finishedTaskInfo, Map<String, TaskInfo> runningTaskInfo) {
        long statusMs = System.currentTimeMillis();

        int finishedSize = finishedTaskInfo != null ? finishedTaskInfo.size() : 0;
        int runningSize = runningTaskInfo != null ? runningTaskInfo.size() : 0;
        if (finishedSize > 0 || !inflightBatches.isEmpty()) {
            logger.debug("Prefill calibrate: finishedTasks={}, runningTasks={}, inflightBatches={}",
                    finishedSize, runningSize, inflightBatches.size());
        }

        // Phase 1: collect request-level terminal observations and reconcile tasks whose Engine
        // status omitted batch_id. Legacy non-batch requests use requestId as
        // the inflight key; real batch members are resolved by membership.
        Map<Long, List<FinishedObservation>> finishedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
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
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
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
                if (observedRunningMember) {
                    batch.markRunning(statusMs);
                } else {
                    batch.markQueued(statusMs);
                }
                return batch;
            });
        }

        // Phase 4: check running requests for anomalies
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    continue;
                }
                if (!inflightBatches.containsKey(batchId)) {
                    logger.debug("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), batchId);
                }
            }
        }
    }

    private void settleFinishedMembers(long batchId,
                                       List<FinishedObservation> observations,
                                       long statusMs) {
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        inflightBatches.computeIfPresent(batchId, (id, batch) ->
                applyFinishedObservations(id, batch, observations, statusMs, true, completed));

        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private BatchInflight applyFinishedObservations(long batchId,
                                                     BatchInflight batch,
                                                     List<FinishedObservation> observations,
                                                     long statusMs,
                                                     boolean deferReconciliation,
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
            if (deferReconciliation && deferIfReconciling(batchId, observation)) {
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
        inflightRequestCount.addAndGet(-(batch.requests().size() - survivors.size()));
        cachedWaitTimeExpireAtMs = 0;
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
        inflightBatches.computeIfPresent(requestId, (id, batch) -> {
            if (!batch.requests().isEmpty()) {
                return batch;
            }
            removedNonBatch.set(true);
            reconciliationRequests.remove(id);
            inflightRequestCount.addAndGet(-batch.requests().size());
            cachedWaitTimeExpireAtMs = 0;
            return null;
        });
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

    private void updateEngineUntrackedRequestCount(WorkerStatusResponse response) {
        Set<Long> localRequestIds = new HashSet<>();
        for (BatchInflight batch : inflightBatches.values()) {
            for (BatchItem request : batch.requests()) {
                localRequestIds.add(request.requestId());
            }
        }

        Set<Long> untracked = new HashSet<>();
        Map<String, TaskInfo> runningTasks = response.getRunningTaskInfo();
        if (runningTasks != null) {
            for (TaskInfo task : runningTasks.values()) {
                if (task == null || isPriorityCancelOverlayOnly(task)) {
                    continue;
                }
                if (!localRequestIds.contains(task.getRequestId())) {
                    untracked.add(task.getRequestId());
                }
            }
        }

        long reportedActive = Math.max(0, response.getWaitingQueryLen())
                + Math.max(0, response.getRunningQueryLen());
        long scalarLowerBound = Math.max(0, reportedActive - localRequestIds.size());
        // The protobuf converter represents an absent detail list as an empty map,
        // while older/newer Engine variants may still populate only the scalar
        // counts. Keep the request-id union when details exist and conservatively
        // retain the scalar lower bound when the detail list is empty or partial.
        engineUntrackedRequestCount = Math.max(untracked.size(), scalarLowerBound);
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
        return inflightRequestCount.get() + batcher.queueSize() + engineUntrackedRequestCount;
    }

    // ==================== Wait Time ====================

    /**
     * Real wait time: estimated time to drain current inflight batches.
     */
    public long realWaitTimeMs() {
        long waitMs = estimateWaitingTimeMs(System.currentTimeMillis());
        return waitMs;
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
        long nowMs = System.currentTimeMillis();
        AtomicInteger evictedCount = new AtomicInteger();
        for (Long batchId : inflightBatches.keySet()) {
            AtomicReference<BatchInflight> evicted = new AtomicReference<>();
            inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                if (hasDispatchReconciliation(id)
                        || nowMs - batch.lastObservedAtMs() <= ttlMs) {
                    return batch;
                }
                evicted.set(batch);
                return null;
            });
            BatchInflight removed = evicted.get();
            if (removed != null) {
                inflightRequestCount.addAndGet(-removed.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                evictedCount.incrementAndGet();
            }
        }
        return evictedCount.get();
    }

    /** Protect an ACK-ambiguous batch from age-only eviction. */
    public void beginDispatchReconciliation(long batchId, long requestId) {
        long nowMs = System.currentTimeMillis();
        inflightBatches.computeIfPresent(batchId, (id, batch) -> {
            boolean owned = batch.requests().stream()
                    .anyMatch(item -> item.requestId() == requestId);
            if (!owned) {
                return batch;
            }
            reconciliationRequests.compute(id, (ignored, requests) -> {
                ConcurrentHashMap<Long, ReconciliationState> states = requests != null
                        ? requests : new ConcurrentHashMap<>();
                states.putIfAbsent(requestId, new ReconciliationState(null));
                return states;
            });
            batch.touch(nowMs);
            return batch;
        });
    }

    /** Release one request's reconciliation fence after authoritative settlement. */
    public void endDispatchReconciliation(long batchId, long requestId) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        inflightBatches.compute(batchId, (id, batch) -> {
            AtomicReference<FinishedObservation> deferredTerminal = new AtomicReference<>();
            reconciliationRequests.computeIfPresent(id, (ignored, requests) -> {
                ReconciliationState state = requests.remove(requestId);
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
            // The fence was removed under the same inflight-key critical section,
            // so apply the cached terminal directly instead of trying to defer it again.
            return applyFinishedObservations(
                    id, batch, List.of(observation), statusMs, false, completed);
        });
        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private boolean deferIfReconciling(long batchId, FinishedObservation observation) {
        AtomicBoolean deferred = new AtomicBoolean(false);
        reconciliationRequests.computeIfPresent(batchId, (ignored, requests) -> {
            requests.computeIfPresent(observation.requestId(), (requestId, state) -> {
                deferred.set(true);
                FinishedObservation existing = state.deferredTerminal();
                return new ReconciliationState(existing == null
                        ? observation : existing.merge(observation));
            });
            return requests.isEmpty() ? null : requests;
        });
        return deferred.get();
    }

    private boolean hasDispatchReconciliation(long batchId) {
        ConcurrentHashMap<Long, ReconciliationState> requests = reconciliationRequests.get(batchId);
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
     * Called periodically by {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler}.
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
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), inflightRequestCount.get());
        reporter.reportInflightMaxAgeMs(RoleType.PREFILL.name(), getIp(),
                InflightEvictor.maxAgeMs(inflightBatches, System.currentTimeMillis()));
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
            predictor.learn(batch.originalFeatures(), predictedMs, actualMs);
        }

        reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
        reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
        reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
    }

    private long estimateWaitingTimeMs(long nowMs) {
        if (nowMs < cachedWaitTimeExpireAtMs) {
            return cachedWaitTimeMs;
        }
        if (inflightBatches.isEmpty()) {
            cachedWaitTimeMs = 0;
            cachedWaitTimeExpireAtMs = nowMs + WAIT_TIME_CACHE_TTL_MS;
            return 0;
        }
        long totalPredMs = 0;
        long earliestProgressBaseMs = Long.MAX_VALUE;
        for (BatchInflight batch : inflightBatches.values()) {
            totalPredMs += Math.max(0, batch.predictTimeMs());
            earliestProgressBaseMs = Math.min(earliestProgressBaseMs, batch.progressBaseMs());
        }
        long result;
        if (earliestProgressBaseMs == Long.MAX_VALUE) {
            result = 0;
        } else {
            long elapsedMs = Math.max(0, nowMs - earliestProgressBaseMs);
            result = Math.max(0, totalPredMs - elapsedMs);
        }
        cachedWaitTimeMs = result;
        cachedWaitTimeExpireAtMs = nowMs + WAIT_TIME_CACHE_TTL_MS;
        return result;
    }

}

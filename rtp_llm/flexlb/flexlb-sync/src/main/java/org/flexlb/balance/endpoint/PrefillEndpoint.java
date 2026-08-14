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
import java.util.stream.Collectors;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, Set<Long>> reconciliationRequests = new ConcurrentHashMap<>();
    private final AtomicInteger inflightRequestCount = new AtomicInteger(0);
    private final WorkerBatcher batcher;
    private final InflightEvictor<Long, BatchInflight> batchEvictor;
    private final BatchSchedulerReporter reporter;

    /**
     * Engine-reported waiting queue length from the latest WorkerStatus update.
     * Reflects requests queued on the engine side that the master hasn't
     * dispatched yet (e.g. traffic not tracked by the current master).
     */
    private volatile long engineWaitingQueryLen = 0;

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
        this.batchEvictor = new InflightEvictor<>(inflightBatches, batch -> {
            inflightRequestCount.addAndGet(-batch.requests().size());
            cachedWaitTimeExpireAtMs = 0;
        });
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
        reconciliationRequests.remove(batchId);
        BatchInflight removed = inflightBatches.remove(batchId);
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
        inflightBatches.computeIfPresent(batchId, (id, old) -> {
            List<BatchItem> survivors = old.requests().stream()
                    .filter(r -> !failedRequestIds.contains(r.requestId()))
                    .toList();
            if (survivors.isEmpty()) {
                inflightRequestCount.addAndGet(-old.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                return null; // removes entry from map
            }
            long newPredMs = (long) predictor.predictBatchMs(survivors);
            BatchInflight repacked = old.repack(newPredMs, survivors);
            inflightRequestCount.addAndGet(-(old.requests().size() - survivors.size()));
            cachedWaitTimeExpireAtMs = 0;
            return repacked;
        });
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        engineWaitingQueryLen = resp.getWaitingQueryLen();
        calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
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

        // Phase 1: classify finished requests and reconcile tasks whose Engine
        // status omitted batch_id. Legacy non-batch requests use requestId as
        // the inflight key; real batch members are resolved by membership.
        Set<Long> batchesWithSuccess = new HashSet<>();
        Map<Long, List<TaskInfo>> failedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    reconcileFinishedWithoutBatchId(task.getRequestId());
                    continue;
                }
                Set<Long> protectedRequests = reconciliationRequests.get(batchId);
                if (protectedRequests != null && !protectedRequests.isEmpty()) {
                    // A batch may contain both an ACK-ambiguous member and
                    // ordinary members. WorkerStatus is delivered to the
                    // endpoint before the scheduler consumes the typed cancel
                    // terminal. Therefore an ordinary sibling success must
                    // only retire that sibling, never the whole batch ledger
                    // (which would erase the reconciliation fence).
                    if (!protectedRequests.contains(task.getRequestId())) {
                        repackBatch(batchId, Set.of(task.getRequestId()));
                    }
                    continue;
                }
                if (task.getErrorCode() == 0) {
                    batchesWithSuccess.add(batchId);
                } else {
                    failedByBatch.computeIfAbsent(batchId, k -> new ArrayList<>()).add(task);
                }
            }
        }

        // Phase 2: any success request → remove entire batch, report predicted vs actual timing
        logger.debug("batchesWithSuccess size: {}", batchesWithSuccess.size());
        for (long batchId : batchesWithSuccess) {
            BatchInflight batch = inflightBatches.get(batchId);
            if (batch == null) {
                logger.debug("batch is null, batchId: {}", batchId);
                continue;
            }
            // Defense-in-depth: verify that at least one success task's requestId
            // belongs to this local batch. Mismatch indicates a stale engine report
            // from a stale or foreign status report with the same batchId.
            Set<Long> localRequestIds = batch.requests().stream()
                    .map(BatchItem::requestId)
                    .collect(Collectors.toSet());
            boolean owned = false;
            if (finishedTaskInfo != null) {
                for (TaskInfo task : finishedTaskInfo.values()) {
                    if (task.getBatchId() == batchId
                            && task.getErrorCode() == 0
                            && localRequestIds.contains(task.getRequestId())) {
                        owned = true;
                        break;
                    }
                }
            }
            if (!owned) {
                logger.warn("Prefill calibrate: batchId={} has success but no matching requestId in local batch. "
                        + "Likely stale or foreign status report. Skipping removal.", batchId);
                continue;
            }
            BatchInflight removed = inflightBatches.remove(batchId);
            if (removed != null) {
                reconciliationRequests.remove(batchId);
                inflightRequestCount.addAndGet(-removed.requests().size());
                cachedWaitTimeExpireAtMs = 0;
            }
            reportBatchCompletion(batchId, batch, finishedTaskInfo);
        }

        // Phase 3: fail-only batches → repack survivors
        for (Map.Entry<Long, List<TaskInfo>> entry : failedByBatch.entrySet()) {
            long batchId = entry.getKey();
            if (batchesWithSuccess.contains(batchId)) {
                continue;
            }
            BatchInflight batch = inflightBatches.get(batchId);
            if (batch == null) {
                continue;
            }
            // Defense-in-depth: verify failed tasks belong to this local batch
            Set<Long> localRequestIds = batch.requests().stream()
                    .map(BatchItem::requestId)
                    .collect(Collectors.toSet());
            List<TaskInfo> foreignTasks = new ArrayList<>();
            List<TaskInfo> localFailedTasks = new ArrayList<>();
            for (TaskInfo t : entry.getValue()) {
                if (localRequestIds.contains(t.getRequestId())) {
                    localFailedTasks.add(t);
                } else {
                    foreignTasks.add(t);
                }
            }
            if (!foreignTasks.isEmpty()) {
                logger.warn("Prefill calibrate: batchId={} has {} failed tasks with foreign requestIds. "
                        + "Skipping repack for foreign tasks.", batchId, foreignTasks.size());
            }
            if (localFailedTasks.isEmpty()) {
                continue;
            }
            Set<Long> failedIds = new HashSet<>();
            for (TaskInfo t : localFailedTasks) {
                logger.debug("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                        batchId, t.getRequestId(), t.getErrorMessage());
                failedIds.add(t.getRequestId());
            }
            repackBatch(batchId, failedIds);
        }

        // Phase 4: update progress anchors. A queued batch cannot spend
        // predicted forward time until the worker reports it as RUNNING.
        Map<Long, Boolean> activeBatchRunning = new HashMap<>();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0 || !inflightBatches.containsKey(batchId)) {
                    continue;
                }
                boolean running = task.getPhase() == TaskPhase.RUNNING;
                activeBatchRunning.merge(batchId, running, Boolean::logicalOr);
            }
        }
        for (Map.Entry<Long, Boolean> entry : activeBatchRunning.entrySet()) {
            BatchInflight batch = inflightBatches.get(entry.getKey());
            if (batch == null) {
                continue;
            }
            if (Boolean.TRUE.equals(entry.getValue())) {
                batch.markRunning(statusMs);
            } else {
                batch.markQueued(statusMs);
            }
        }

        // Phase 5: check running requests for anomalies
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
    private void reconcileFinishedWithoutBatchId(long requestId) {
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
        AtomicBoolean removedMember = new AtomicBoolean(false);
        AtomicBoolean protectedMember = new AtomicBoolean(false);
        inflightBatches.computeIfPresent(resolvedBatchId, (id, batch) -> {
            boolean stillOwned = batch.requests().stream()
                    .anyMatch(item -> item.requestId() == requestId);
            if (!stillOwned) {
                return batch;
            }
            Set<Long> protectedRequests = reconciliationRequests.get(id);
            if (protectedRequests != null && protectedRequests.contains(requestId)) {
                // WorkerStatus reaches the endpoint before the scheduler consumes
                // the typed cancel terminal. Preserve an ACK-ambiguous member until
                // that stronger owner settles it.
                protectedMember.set(true);
                return batch;
            }

            List<BatchItem> survivors = batch.requests().stream()
                    .filter(item -> item.requestId() != requestId)
                    .toList();
            long newPredMs = survivors.isEmpty()
                    ? 0
                    : (long) predictor.predictBatchMs(survivors);
            removedMember.set(true);
            inflightRequestCount.addAndGet(-(batch.requests().size() - survivors.size()));
            cachedWaitTimeExpireAtMs = 0;
            if (survivors.isEmpty()) {
                reconciliationRequests.remove(id);
                return null;
            }
            return batch.repack(newPredMs, survivors);
        });

        if (protectedMember.get()) {
            logger.debug("Prefill calibrate: preserve reconciling request reqId={} resolvedBatchId={}",
                    requestId, resolvedBatchId);
        } else if (removedMember.get()) {
            logger.debug("Prefill calibrate: resolved missing batch id reqId={} to batchId={} and removed member",
                    requestId, resolvedBatchId);
        } else {
            logger.debug("Prefill calibrate: missing-batch-id member disappeared during reconciliation "
                            + "reqId={} resolvedBatchId={}",
                    requestId, resolvedBatchId);
        }
    }

    // ==================== Pending Count ====================

    /**
     * Real pending count: total requests the engine will face.
     * Includes master-tracked inflight + batcher queue + engine-reported
     * waiting queue (e.g. traffic not tracked by the current master).
     */
    public long realPendingCount() {
        return inflightRequestCount.get() + batcher.queueSize() + engineWaitingQueryLen;
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
     * Evict inflight batches older than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale prefill entries.
     *
     * @return number of batches evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        return batchEvictor.evictExpired(ttlMs,
                batchId -> !reconciliationRequests.containsKey(batchId));
    }

    /** Protect an ACK-ambiguous batch from age-only eviction. */
    public void beginDispatchReconciliation(long batchId, long requestId) {
        reconciliationRequests.computeIfAbsent(batchId, ignored -> ConcurrentHashMap.newKeySet())
                .add(requestId);
    }

    /** Release one request's reconciliation fence after authoritative settlement. */
    public void endDispatchReconciliation(long batchId, long requestId) {
        reconciliationRequests.computeIfPresent(batchId, (ignored, requests) -> {
            requests.remove(requestId);
            return requests.isEmpty() ? null : requests;
        });
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
    private void reportBatchCompletion(long batchId, BatchInflight batch, Map<String, TaskInfo> finishedTaskInfo) {
        logger.debug("run reportBatchCompletion, batchId: {}, finishedTaskInfo size: {}",
                batchId, finishedTaskInfo.size());
        long actualMs = -1;
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getBatchId() == batchId && task.getExecutionTimeMs() > 0) {
                    actualMs = Math.max(actualMs, task.getExecutionTimeMs());
                }
            }
        }
        if (actualMs < 0) {
            logger.debug("actualMs < 0: {}", actualMs);
            return;
        }

        long predictedMs = batch.predictTimeMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.debug(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, batch.requests().size(), getIp());

        // Feed the actual-vs-predicted timing back into the predictor for future learning.
        predictor.learn(batch.requests(), predictedMs, actualMs);

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

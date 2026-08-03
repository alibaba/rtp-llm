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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
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
        completeBatchMembers(batchId, failedRequestIds);
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
            logger.info("Prefill calibrate: finishedTasks={}, runningTasks={}, inflightBatches={}",
                    finishedSize, runningSize, inflightBatches.size());
        }

        // Phase 1: classify finished requests and clean up non-batch inflight.
        // Non-batch requests use requestId as the inflight key (engine reports
        // them with batch_id=-1).  Remove them immediately to keep
        // realWaitTimeMs() accurate; warn if a finished non-batch request was
        // not tracked in inflight (indicates a bug or stale engine report).
        Map<Long, List<TaskInfo>> finishedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    BatchInflight removed = inflightBatches.remove(task.getRequestId());
                    if (removed == null) {
                        logger.warn("Prefill calibrate: finished non-batch request reqId={} not in inflight", task.getRequestId());
                    } else {
                        inflightRequestCount.addAndGet(-removed.requests().size());
                        cachedWaitTimeExpireAtMs = 0;
                    }
                    continue;
                }
                finishedByBatch.computeIfAbsent(batchId, ignored -> new ArrayList<>()).add(task);
            }
        }

        // Phase 2: retire exactly the members reported finished. The Engine may
        // execute one FlexLB group in several resource-bounded waves.
        for (Map.Entry<Long, List<TaskInfo>> entry : finishedByBatch.entrySet()) {
            long batchId = entry.getKey();
            BatchInflight batch = inflightBatches.get(batchId);
            if (batch == null) {
                logger.debug("batch is null, batchId: {}", batchId);
                continue;
            }
            Set<Long> localRequestIds = new HashSet<>();
            for (BatchItem request : batch.requests()) {
                localRequestIds.add(request.requestId());
            }
            Set<Long> completedRequestIds = new HashSet<>();
            for (TaskInfo t : entry.getValue()) {
                if (localRequestIds.contains(t.getRequestId())) {
                    completedRequestIds.add(t.getRequestId());
                    if (t.getErrorCode() != 0) {
                        logger.warn("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                                batchId, t.getRequestId(), t.getErrorMessage());
                    }
                } else {
                    logger.warn("Prefill calibrate: batchId={} has foreign finished reqId={}; ignoring it",
                            batchId, t.getRequestId());
                }
            }
            if (completedRequestIds.isEmpty()) {
                continue;
            }
            BatchInflight completed = completeBatchMembers(batchId, completedRequestIds);
            if (completed != null) {
                reportBatchCompletion(batchId, completed, finishedTaskInfo);
            }
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
                    logger.warn("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), batchId);
                }
            }
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
        return batchEvictor.evictExpired(ttlMs);
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
        reporter.reportBatcherQueueDepth(RoleType.PREFILL.name(), getIp(), queueSize);
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), getInflightBatchCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), inflightRequestCount.get());
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

        // A split batch has no single completion sample representing the
        // original group, so feeding its last wave back would bias the model.
        if (batch.partiallyCompleted()) {
            logger.debug("skip predictor feedback for split batchId={}", batchId);
            return;
        }

        long predictedMs = batch.predictTimeMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.info(
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

    private BatchInflight completeBatchMembers(long batchId, Set<Long> completedRequestIds) {
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        inflightBatches.computeIfPresent(batchId, (id, old) -> {
            List<BatchItem> survivors = old.requests().stream()
                    .filter(request -> !completedRequestIds.contains(request.requestId()))
                    .toList();
            int removedCount = old.requests().size() - survivors.size();
            if (removedCount == 0) {
                return old;
            }
            inflightRequestCount.addAndGet(-removedCount);
            cachedWaitTimeExpireAtMs = 0;
            if (survivors.isEmpty()) {
                completed.set(old);
                return null;
            }
            return old.repack((long) predictor.predictBatchMs(survivors), survivors);
        });
        return completed.get();
    }

}

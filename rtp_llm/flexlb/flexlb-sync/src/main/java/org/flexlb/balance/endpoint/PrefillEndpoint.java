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
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private volatile PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final AtomicReference<Long> estimatedWaitingTimeMs = new AtomicReference<>(0L);
    private volatile WorkerBatcher batcher;
    private final InflightEvictor<Long, BatchInflight> batchEvictor;
    private final BatchSchedulerReporter reporter;

    /**
     * Engine-reported waiting queue length from the latest WorkerStatus update.
     * Reflects requests queued on the engine side that the master hasn't
     * dispatched yet (e.g. legacy traffic from other masters).
     */
    private volatile long engineWaitingQueryLen = 0;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           BatchDecisionHandler handler,
                           BatchSchedulerReporter reporter) {
        super(status);
        this.reporter = reporter;
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, handler, reporter);
        this.batchEvictor = new InflightEvictor<>(inflightBatches,
                batch -> refreshEstimatedWaitingTime());
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
        batcher.shutdown();
    }

    public int getBatcherQueueSize() {
        return batcher.queueSize();
    }

    public long getBatcherHeadSortKey() {
        return batcher.headSortKey();
    }

    public long getBatcherHeadWaitMs() {
        return batcher.headWaitMs();
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
        inflightBatches.put(batchId, new BatchInflight(batchId, predictMs, requests));
        refreshEstimatedWaitingTime();
    }

    public void releaseBatch(long batchId) {
        BatchInflight removed = inflightBatches.remove(batchId);
        if (removed != null) {
            refreshEstimatedWaitingTime();
        }
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     * @return the new BatchInflight if survivors remain, null if the entire batch was removed
     */
    public BatchInflight repackBatch(long batchId, Set<Long> failedRequestIds) {
        BatchInflight result = inflightBatches.computeIfPresent(batchId, (id, old) -> {
            List<BatchItem> survivors = old.requests().stream()
                    .filter(r -> !failedRequestIds.contains(r.requestId()))
                    .toList();
            if (survivors.isEmpty()) {
                return null; // removes entry from map
            }
            long newPredMs = predictor != null ? predictor.predictBatchMs(survivors) : 0;
            return old.repack(newPredMs, survivors);
        });
        refreshEstimatedWaitingTime();
        return result;
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
    public void calibrate(Map<String, TaskInfo> finishedTaskInfo, Map<String, TaskInfo> runningTaskInfo) {
        if (predictor == null) {
            return;
        }
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
        Set<Long> batchesWithSuccess = new HashSet<>();
        Map<Long, List<TaskInfo>> failedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    BatchInflight removed = inflightBatches.remove(task.getRequestId());
                    if (removed == null) {
                        logger.warn("Prefill calibrate: finished non-batch request reqId={} not in inflight", task.getRequestId());
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
        for (long batchId : batchesWithSuccess) {
            BatchInflight batch = inflightBatches.get(batchId);
            if (batch == null) {
                continue;
            }
            // Defense-in-depth: verify that at least one success task's requestId
            // belongs to this local batch. Mismatch indicates a stale engine report
            // from a previous master epoch with the same batchId.
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
                        + "Likely stale report from previous master epoch. Skipping removal.", batchId);
                continue;
            }
            inflightBatches.remove(batchId);
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
                if (!isCancelError(t)) {
                    logger.warn("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                            batchId, t.getRequestId(), t.getErrorMessage());
                }
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
                    logger.warn("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), batchId);
                }
            }
        }

        // Phase 6: refresh waiting time snapshot
        refreshEstimatedWaitingTime();
    }

    // ==================== Pending Count ====================

    /**
     * Real pending count: total requests the engine will face.
     * Includes master-tracked inflight + batcher queue + engine-reported
     * waiting queue (e.g. traffic from other sources).
     */
    public long realPendingCount() {
        return getInflightRequestCount() + batcher.queueSize() + engineWaitingQueryLen;
    }

    // ==================== Wait Time ====================

    /**
     * Real wait time: estimated time to drain current inflight batches.
     */
    public long realWaitTimeMs() {
        long waitMs = estimateWaitingTimeMs(System.currentTimeMillis());
        estimatedWaitingTimeMs.set(waitMs);
        return waitMs;
    }

    public int getInflightBatchCount() {
        return inflightBatches.size();
    }

    public int getInflightRequestCount() {
        int count = 0;
        for (BatchInflight batch : inflightBatches.values()) {
            count += batch.requests().size();
        }
        return count;
    }

    /**
     * Evict inflight batches older than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale prefill entries.
     *
     * @return number of batches evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        int evicted = batchEvictor.evictExpired(ttlMs);
        if (evicted > 0) {
            refreshEstimatedWaitingTime();
        }
        return evicted;
    }

    @Override
    public long getLoadMetric() {
        return realWaitTimeMs();
    }

    @Override
    public int getLocalTaskCount() {
        return getInflightRequestCount();
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    ConcurrentHashMap<Long, BatchInflight> getInflightBatches() {
        return inflightBatches;
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportBatcherQueueDepth(RoleType.PREFILL.name(), getIp(), getBatcherQueueSize());
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), getInflightBatchCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), getInflightRequestCount());
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportBatchCompletion(long batchId, BatchInflight batch, Map<String, TaskInfo> finishedTaskInfo) {
        long actualMs = -1;
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getBatchId() == batchId && task.getExecutionTimeMs() > 0) {
                    actualMs = Math.max(actualMs, task.getExecutionTimeMs());
                }
            }
        }
        if (actualMs < 0) {
            return;
        }

        long predictedMs = batch.predictTimeMs();
        long gapMs = actualMs - predictedMs;
        logger.info("flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, batch.requests().size(), getIp());

        // Feed the actual-vs-predicted timing back into the predictor for future learning.
        batch.setActualTimeMs(actualMs);
        predictor.learn(batch.requests(), predictedMs, actualMs);

        if (reporter != null) {
            reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
            reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
            reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
        }
    }

    private void refreshEstimatedWaitingTime() {
        estimatedWaitingTimeMs.set(estimateWaitingTimeMs(System.currentTimeMillis()));
    }

    private long estimateWaitingTimeMs(long nowMs) {
        if (inflightBatches.isEmpty()) {
            return 0;
        }
        long totalPredMs = 0;
        long earliestProgressBaseMs = Long.MAX_VALUE;
        for (BatchInflight batch : inflightBatches.values()) {
            totalPredMs += Math.max(0, batch.predictTimeMs());
            earliestProgressBaseMs = Math.min(earliestProgressBaseMs, batch.progressBaseMs());
        }
        if (earliestProgressBaseMs == Long.MAX_VALUE) {
            return 0;
        }
        long elapsedMs = Math.max(0, nowMs - earliestProgressBaseMs);
        return Math.max(0, totalPredMs - elapsedMs);
    }

    private static boolean isCancelError(TaskInfo task) {
        return task.getErrorMessage() != null && task.getErrorMessage().toLowerCase().contains("cancel");
    }

}

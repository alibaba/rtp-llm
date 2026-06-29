package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
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

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private volatile PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final AtomicReference<Long> estimatedWaitingTimeMs = new AtomicReference<>(0L);
    private volatile WorkerBatcher batcher;
    private final InflightEvictor<Long, BatchInflight> batchEvictor;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           BatchDecisionHandler handler) {
        super(status);
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, handler);
        this.batchEvictor = new InflightEvictor<>(inflightBatches,
                batch -> refreshEstimatedWaitingTime());
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config, BatchDecisionHandler handler) {
        return new WorkerBatcher(status.getIpPort(), this, config, handler);
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
        return new PrefillTimePredictor(
                cfg.getCostAlpha0(), cfg.getCostAlpha1(), cfg.getCostAlpha2(),
                cfg.getCostAlpha3(), cfg.getCostAlpha4(), cfg.getCostAlpha5());
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
                        logger.warn("Prefill calibrate: finished non-batch reqId={} not in inflight, errorCode={}",
                                task.getRequestId(), task.getErrorCode());
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

        // Phase 2: any success request → remove entire batch
        for (long batchId : batchesWithSuccess) {
            inflightBatches.remove(batchId);
        }

        // Phase 3: fail-only batches → repack survivors
        for (Map.Entry<Long, List<TaskInfo>> entry : failedByBatch.entrySet()) {
            long batchId = entry.getKey();
            if (batchesWithSuccess.contains(batchId)) {
                continue;
            }
            if (!inflightBatches.containsKey(batchId)) {
                continue;
            }

            List<TaskInfo> failedTasks = entry.getValue();
            Set<Long> failedIds = new HashSet<>();
            for (TaskInfo t : failedTasks) {
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
     */
    public long realPendingCount() {
        return getInflightRequestCount() + batcher.queueSize();
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

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.BatcherSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.BatchRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private volatile PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final AtomicLong totalPredictTimeMs = new AtomicLong();
    private final AtomicReference<Long> estimatedWaitingTimeMs = new AtomicReference<>(0L);
    private volatile WorkerBatcher batcher;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           BatchDecisionHandler handler) {
        super(status);
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, handler);
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config, BatchDecisionHandler handler) {
        String key = status.getIpPort();
        return new WorkerBatcher(key, this, config, handler);
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

    public BatcherSnapshot getBatcherSnapshot() {
        return batcher.snapshot();
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        return new PrefillTimePredictor(
                cfg.getCostAlpha0(), cfg.getCostAlpha1(), cfg.getCostAlpha2(),
                cfg.getCostAlpha3(), cfg.getCostAlpha4(), cfg.getCostAlpha5());
    }

    public void commitBatch(long batchId, long predictMs, List<BatchRequest> requests) {
        inflightBatches.put(batchId, new BatchInflight(batchId, predictMs, requests));
        totalPredictTimeMs.addAndGet(predictMs);
        refreshEstimatedWaitingTime();
    }

    public void releaseBatch(long batchId) {
        BatchInflight removed = inflightBatches.remove(batchId);
        if (removed != null) {
            totalPredictTimeMs.addAndGet(-removed.predictTimeMs());
            refreshEstimatedWaitingTime();
        }
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     * @return the new BatchInflight if survivors remain, null if the entire batch was removed
     */
    public BatchInflight repackBatch(long batchId, Set<Long> failedRequestIds) {
        BatchInflight old = inflightBatches.get(batchId);
        if (old == null) {
            return null;
        }

        List<BatchRequest> survivors = old.requests().stream()
                .filter(r -> !failedRequestIds.contains(r.requestId()))
                .toList();

        long oldPredMs = old.predictTimeMs();
        if (survivors.isEmpty()) {
            inflightBatches.remove(batchId);
            totalPredictTimeMs.addAndGet(-oldPredMs);
            refreshEstimatedWaitingTime();
            return null;
        }

        long newPredMs = predictor != null ? predictor.predictBatchMs(survivors) : 0;
        BatchInflight newBatch = new BatchInflight(batchId, newPredMs, survivors);
        inflightBatches.put(batchId, newBatch);
        totalPredictTimeMs.addAndGet(newPredMs - oldPredMs);
        refreshEstimatedWaitingTime();
        return newBatch;
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

        // Phase 1: classify finished requests by batchId
        Set<Long> batchesWithSuccess = new HashSet<>();
        Map<Long, List<TaskInfo>> failedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    logger.warn("Prefill calibrate: finished request reqId={} has no valid batchId, errorCode={}, error={}",
                            task.getRequestId(), task.getErrorCode(), task.getErrorMessage());
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

        // Phase 4: check running requests for anomalies
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

        // Phase 6: recompute cumulative and refresh snapshot
        recomputeTotalAndRefresh();
    }

    public long getEstimatedWaitingTimeMs() {
        return estimatedWaitingTimeMs.get();
    }

    public long computeDeadlineMs(long seqLen, long hitCache, long sloMs) {
        long predMs = predictor != null ? predictor.estimateMs(seqLen, hitCache) : 0;
        long workerQueueMs = estimatedWaitingTimeMs.get();
        return System.currentTimeMillis() + Math.max(0, sloMs - predMs - workerQueueMs);
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

    @Override
    public long getRunningLoad() {
        return getEstimatedWaitingTimeMs();
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

    AtomicLong getTotalPredictTimeMs() {
        return totalPredictTimeMs;
    }

    private void refreshEstimatedWaitingTime() {
        estimatedWaitingTimeMs.set(Math.max(0, totalPredictTimeMs.get()));
    }

    private synchronized void recomputeTotalAndRefresh() {
        long max = 0;
        for (BatchInflight batch : inflightBatches.values()) {
            max = Math.max(max, batch.predictTimeMs());
        }
        totalPredictTimeMs.set(max);
        refreshEstimatedWaitingTime();
    }

    private static boolean isCancelError(TaskInfo task) {
        return task.getErrorMessage() != null && task.getErrorMessage().toLowerCase().contains("cancel");
    }

}

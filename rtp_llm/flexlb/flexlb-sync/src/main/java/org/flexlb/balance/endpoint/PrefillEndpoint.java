package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.RequestProfile;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
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
    private final AtomicReference<Long> snapshot = new AtomicReference<>(0L);

    public PrefillEndpoint(String ip, int httpPort, int grpcPort, WorkerStatus status, PrefillTimePredictor predictor) {
        super(ip, httpPort, grpcPort, status);
        this.predictor = predictor;
    }

    public void commitBatch(long batchId, long predictMs, List<Long> requestIds, List<RequestProfile> profiles) {
        inflightBatches.put(batchId, new BatchInflight(batchId, predictMs, requestIds, profiles));
        totalPredictTimeMs.addAndGet(predictMs);
        refreshSnapshot();
    }

    public void releaseBatch(long batchId) {
        BatchInflight removed = inflightBatches.remove(batchId);
        if (removed != null) {
            safeDecrement(totalPredictTimeMs, removed.predictTimeMs());
            refreshSnapshot();
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

        List<Long> survivingIds = new ArrayList<>();
        List<RequestProfile> survivingProfiles = new ArrayList<>();
        for (int i = 0; i < old.requestIds().size(); i++) {
            if (!failedRequestIds.contains(old.requestIds().get(i))) {
                survivingIds.add(old.requestIds().get(i));
                survivingProfiles.add(old.profiles().get(i));
            }
        }

        long oldPredMs = old.predictTimeMs();
        if (survivingIds.isEmpty()) {
            inflightBatches.remove(batchId);
            safeDecrement(totalPredictTimeMs, oldPredMs);
            refreshSnapshot();
            return null;
        }

        long newPredMs = predictor != null ? predictor.predictBatchMs(survivingProfiles) : 0;
        BatchInflight newBatch = new BatchInflight(batchId, newPredMs, survivingIds, survivingProfiles);
        inflightBatches.put(batchId, newBatch);
        totalPredictTimeMs.addAndGet(newPredMs - oldPredMs);
        refreshSnapshot();
        return newBatch;
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
                    if (task.getErrorCode() != 0) {
                        logger.warn("Prefill calibrate: finished request reqId={} has no valid batchId, error={}",
                                task.getRequestId(), task.getErrorMessage());
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

        // Phase 5: recompute cumulative and refresh snapshot
        recomputeTotalAndRefresh();
    }

    public long getEstimatedWaitingTimeMs() {
        return snapshot.get();
    }

    public int getInflightBatchCount() {
        return inflightBatches.size();
    }

    public int getInflightRequestCount() {
        int count = 0;
        for (BatchInflight batch : inflightBatches.values()) {
            count += batch.requestIds().size();
        }
        return count;
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    public void setPredictor(PrefillTimePredictor predictor) {
        this.predictor = predictor;
    }

    ConcurrentHashMap<Long, BatchInflight> getInflightBatches() {
        return inflightBatches;
    }

    AtomicLong getTotalPredictTimeMs() {
        return totalPredictTimeMs;
    }

    private void refreshSnapshot() {
        snapshot.set(Math.max(0, totalPredictTimeMs.get()));
    }

    private synchronized void recomputeTotalAndRefresh() {
        long sum = 0;
        for (BatchInflight batch : inflightBatches.values()) {
            sum += batch.predictTimeMs();
        }
        totalPredictTimeMs.set(sum);
        refreshSnapshot();
    }

    private static boolean isCancelError(TaskInfo task) {
        return task.getErrorMessage() != null && task.getErrorMessage().toLowerCase().contains("cancel");
    }

    private static void safeDecrement(AtomicLong value, long delta) {
        value.accumulateAndGet(delta, (current, d) -> Math.max(0, current - d));
    }
}

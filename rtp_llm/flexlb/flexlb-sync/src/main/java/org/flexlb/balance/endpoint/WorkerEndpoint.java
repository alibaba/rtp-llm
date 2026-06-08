package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.RequestProfile;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskStateEnum;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ip;
    private final int httpPort;
    private final int grpcPort;
    private final WorkerStatus status;
    private volatile PrefillTimePredictor predictor;

    private final AtomicLong estimatedWaitingTimeMs = new AtomicLong();
    private final ConcurrentHashMap<Long, Long> inflightBatches = new ConcurrentHashMap<>();

    public WorkerEndpoint(String ip, int httpPort, int grpcPort, WorkerStatus status, PrefillTimePredictor predictor) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.status = status;
        this.predictor = predictor;
    }

    public void commitBatch(long batchId, long batchPredMs) {
        inflightBatches.put(batchId, batchPredMs);
        estimatedWaitingTimeMs.addAndGet(batchPredMs);
    }

    public void releaseBatch(long batchId) {
        Long predMs = inflightBatches.remove(batchId);
        if (predMs != null) {
            safeDecrement(estimatedWaitingTimeMs, predMs);
        }
    }

    public void calibrateWaitingTime() {
        if (predictor == null) {
            return;
        }

        ConcurrentHashMap<Long, TaskInfo> localTaskMap = status.getLocalTaskMap();
        if (localTaskMap.isEmpty()) {
            inflightBatches.clear();
            estimatedWaitingTimeMs.set(0);
            return;
        }

        Map<Long, List<TaskInfo>> batchGroups = new HashMap<>();
        for (TaskInfo task : localTaskMap.values()) {
            if (task.getTaskState() != TaskStateEnum.CONFIRMED) {
                continue;
            }
            batchGroups.computeIfAbsent(task.getBatchId(), k -> new ArrayList<>()).add(task);
        }

        long waitingEstimate = 0;
        for (Map.Entry<Long, List<TaskInfo>> entry : batchGroups.entrySet()) {
            long batchId = entry.getKey();
            List<TaskInfo> tasks = entry.getValue();

            if (batchId == -1) {
                for (TaskInfo task : tasks) {
                    waitingEstimate += predictor.estimateMs(task.getInputLength(), task.getPrefixLength());
                }
            } else {
                List<RequestProfile> profiles = new ArrayList<>(tasks.size());
                for (TaskInfo task : tasks) {
                    profiles.add(new RequestProfile(task.getInputLength(), task.getPrefixLength()));
                }
                waitingEstimate += predictor.predictBatchMs(profiles);
                inflightBatches.remove(batchId);
            }
        }

        long inflightEstimate = 0;
        for (Long predMs : inflightBatches.values()) {
            inflightEstimate += predMs;
        }

        estimatedWaitingTimeMs.set(waitingEstimate + inflightEstimate);
    }

    public long getEstimatedWaitingTimeMs() {
        return estimatedWaitingTimeMs.get();
    }

    public String ipPort() {
        return ip + ":" + httpPort;
    }

    public String getIp() {
        return ip;
    }

    public int getHttpPort() {
        return httpPort;
    }

    public int getGrpcPort() {
        return grpcPort;
    }

    public WorkerStatus getStatus() {
        return status;
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    public void setPredictor(PrefillTimePredictor predictor) {
        this.predictor = predictor;
    }

    ConcurrentHashMap<Long, Long> getInflightBatches() {
        return inflightBatches;
    }

    private static void safeDecrement(AtomicLong value, long delta) {
        value.accumulateAndGet(delta, (current, d) -> Math.max(0, current - d));
    }
}

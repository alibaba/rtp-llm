package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskPhase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final ConcurrentHashMap<Long, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong totalKvTokens = new AtomicLong();
    private final AtomicReference<Long> snapshot = new AtomicReference<>(0L);
    private volatile long reportedKvAvailable;

    public DecodeEndpoint(String ip, int httpPort, int grpcPort, WorkerStatus status) {
        super(ip, httpPort, grpcPort, status);
    }

    public void reserve(long requestId, long kvTokens) {
        inflightRequests.put(requestId, new RequestInflight(requestId, kvTokens));
        totalKvTokens.addAndGet(kvTokens);
        refreshSnapshot();
    }

    public void release(long requestId) {
        RequestInflight removed = inflightRequests.remove(requestId);
        if (removed != null) {
            safeDecrement(totalKvTokens, removed.kvTokens());
            refreshSnapshot();
        }
    }

    /**
     * Full calibration against worker status report.
     */
    public void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo,
                           long latestAvailableKvCacheTokens) {
        this.reportedKvAvailable = latestAvailableKvCacheTokens;

        // Phase 1: process running requests with KV_ALLOCATED phase
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                if (task.getPhase() == TaskPhase.KV_ALLOCATED) {
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed == null) {
                        logger.warn("Decode calibrate: running KV_ALLOCATED request reqId={} not in inflight",
                                task.getRequestId());
                    }
                } else {
                    if (!inflightRequests.containsKey(task.getRequestId())) {
                        logger.warn("Decode calibrate: running request reqId={} phase={} not in inflight",
                                task.getRequestId(), task.getPhase());
                    }
                }
            }
        }

        // Phase 2: process finished non-success requests
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() != 0) {
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed == null && !isCancelError(task)) {
                        logger.warn("Decode calibrate: finished failed request reqId={} not in inflight, error={}",
                                task.getRequestId(), task.getErrorMessage());
                    }
                }
            }

            // Phase 3: process finished success requests
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() == 0) {
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed != null) {
                        logger.debug("Decode calibrate: success request reqId={} still in inflight, " +
                                "KV_ALLOCATED detection may have been skipped", task.getRequestId());
                    }
                }
            }
        }

        // Phase 4: recompute cumulative and refresh snapshot
        recomputeTotalAndRefresh();
    }

    public long getAvailableKvTokens() {
        return snapshot.get();
    }

    public int getInflightCount() {
        return inflightRequests.size();
    }

    ConcurrentHashMap<Long, RequestInflight> getInflightRequests() {
        return inflightRequests;
    }

    AtomicLong getTotalKvTokens() {
        return totalKvTokens;
    }

    private void refreshSnapshot() {
        snapshot.set(Math.max(0, reportedKvAvailable - totalKvTokens.get()));
    }

    private synchronized void recomputeTotalAndRefresh() {
        long sum = 0;
        for (RequestInflight ri : inflightRequests.values()) {
            sum += ri.kvTokens();
        }
        totalKvTokens.set(sum);
        refreshSnapshot();
    }

    private static boolean isCancelError(TaskInfo task) {
        return task.getErrorMessage() != null && task.getErrorMessage().toLowerCase().contains("cancel");
    }

    private static void safeDecrement(AtomicLong value, long delta) {
        value.accumulateAndGet(delta, (current, d) -> Math.max(0, current - d));
    }
}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final ConcurrentHashMap<Long, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong reportedKvAvailable = new AtomicLong();
    private volatile int confirmedRunningCount;
    private final InflightEvictor<Long, RequestInflight> requestEvictor;

    public DecodeEndpoint(WorkerStatus status) {
        super(status);
        this.requestEvictor = new InflightEvictor<>(inflightRequests, req -> {});
    }

    public void reserve(long requestId, long kvTokens) {
        inflightRequests.put(requestId, new RequestInflight(requestId, kvTokens));
    }

    public void release(long requestId) {
        inflightRequests.remove(requestId);
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getRunningTaskInfo(), resp.getFinishedTaskInfo(),
                status.getAvailableKvCacheTokens().get());
    }

    /**
     * Full calibration against worker status report.
     */
    public void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo,
                           long latestAvailableKvCacheTokens) {
        this.reportedKvAvailable.set(latestAvailableKvCacheTokens);

        // Phase 1: process running requests — KV_ALLOCATED or RUNNING means the engine
        // has taken ownership, so we can release our inflight reservation.
        int kvAllocatedRequests = 0;
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    inflightRequests.remove(task.getRequestId());
                    kvAllocatedRequests++;
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

        this.confirmedRunningCount = kvAllocatedRequests;
    }

    // ==================== KV Cache 三视图 ====================

    /**
     * Local inflight KV reservation not yet confirmed by the engine.
     * Computed on demand from the inflight map — no separate counter needed.
     */
    public long inflightKvReserved() {
        long sum = 0;
        for (RequestInflight ri : inflightRequests.values()) {
            sum += ri.kvTokens();
        }
        return sum;
    }

    /**
     * Real KV used: engine-reported used (total - available) + local inflight reservations.
     */
    public long realKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        return reportedUsed + inflightKvReserved();
    }

    /**
     * Real KV available: engine-reported available - local inflight reservations.
     *
     * <p><b>Approximate:</b> reads {@code reportedKvAvailable} and
     * computes {@code inflightKvReserved()} non-atomically — the returned value may reflect a
     * slightly inconsistent snapshot. This is acceptable for scheduling decisions.
     */
    public long realKvAvailable() {
        return Math.max(0, reportedKvAvailable.get() - inflightKvReserved());
    }

    /**
     * Real KV total capacity reported by the engine.
     */
    public long realKvTotal() {
        return status.getTotalKvCacheTokens().get();
    }

    public int getInflightCount() {
        return inflightRequests.size();
    }

    /**
     * Evict inflight requests older than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale decode entries.
     *
     * @return number of entries evicted
     */
    public int evictExpiredRequests(long ttlMs) {
        return requestEvictor.evictExpired(ttlMs);
    }

    public int getTotalLoad() {
        return confirmedRunningCount + inflightRequests.size();
    }

    @Override
    public long getLoadMetric() {
        return getTotalLoad();
    }

    @Override
    public int getLocalTaskCount() {
        return getInflightCount();
    }

    ConcurrentHashMap<Long, RequestInflight> getInflightRequests() {
        return inflightRequests;
    }

    private static boolean isCancelError(TaskInfo task) {
        return task.getErrorMessage() != null && task.getErrorMessage().toLowerCase().contains("cancel");
    }

}

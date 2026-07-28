package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final ConcurrentHashMap<Long, Reservation> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong reportedKvAvailable = new AtomicLong();
    private volatile int confirmedRunningCount;
    private final InflightEvictor<Long, Reservation> requestEvictor;

    public DecodeEndpoint(WorkerStatus status) {
        super(status);
        this.requestEvictor = new InflightEvictor<>(inflightRequests,
                req -> inflightKvReservedTotal.addAndGet(-req.kvTokens()));
    }

    public Lease reserve(long requestId, long kvTokens) {
        return reserve(requestId, kvTokens, null);
    }

    public Lease reserve(long requestId, long kvTokens, BalanceContext routeOwner) {
        Reservation reservation = new Reservation(requestId, kvTokens, routeOwner);
        // Account first so the only transient view is conservative.
        inflightKvReservedTotal.addAndGet(kvTokens);
        Reservation existing = inflightRequests.putIfAbsent(requestId, reservation);
        if (existing != null) {
            inflightKvReservedTotal.addAndGet(-kvTokens);
            throw new IllegalStateException(
                    "decode request already reserved: " + requestId);
        }
        return reservation;
    }

    public Lease leaseFor(long requestId) {
        return inflightRequests.get(requestId);
    }

    public Lease leaseFor(long requestId, BalanceContext routeOwner) {
        Reservation reservation = inflightRequests.get(requestId);
        return reservation != null && reservation.routeOwner == routeOwner
                ? reservation
                : null;
    }

    public boolean release(long requestId) {
        Reservation reservation = inflightRequests.get(requestId);
        return reservation != null && reservation.release();
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getRunningTaskInfo(), resp.getFinishedTaskInfo());
    }

    /**
     * Full calibration against worker status report.
     */
    private void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());

        // Phase 1: process running requests — KV_ALLOCATED or RUNNING means the engine
        // has taken ownership, so we can release our inflight reservation.
        //
        // Two-pass to avoid transient undercount: if we remove from inflightRequests before
        // updating confirmedRunningCount, a task transitioning from inflight to confirmed
        // is briefly counted in neither, which could allow oversubscription. By updating
        // the count first and removing second, the transient window overcounts (conservative).
        int kvAllocatedRequests = 0;
        if (runningTaskInfo != null) {
            // First pass: count and update confirmedRunningCount
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    kvAllocatedRequests++;
                }
            }
        }
        this.confirmedRunningCount = kvAllocatedRequests;

        // Second pass: remove confirmed tasks from inflightRequests
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    releaseForWorkerStatus(task);
                }
            }
        }

        // Phase 2: process finished non-success requests
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() != 0) {
                    if (!releaseForWorkerStatus(task)) {
                        logger.warn("Decode calibrate: failed request reqId={} batchId={} has no matching reservation, error={}",
                                task.getRequestId(), task.getBatchId(), task.getErrorMessage());
                    }
                }
            }

            // Phase 3: process finished success requests
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() == 0) {
                    releaseForWorkerStatus(task);
                }
            }
        }
    }

    // ==================== KV Cache 三视图 ====================

    /**
     * Local inflight KV reservation not yet confirmed by the engine.
     * Maintained as an {@link AtomicLong} counter, updated incrementally on
     * {@link #reserve}, {@link #release}, {@link #calibrate}, and TTL eviction.
     */
    private long inflightKvReserved() {
        return inflightKvReservedTotal.get();
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

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), getInflightCount());
        reporter.reportDecodeTotalLoad(getIp(), getTotalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), inflightKvReserved());
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

    public interface Lease {
        boolean release();

        boolean bindBatch(long batchId);
    }

    private final class Reservation implements Lease, InflightEvictor.TtlTracked {
        private final long requestId;
        private final long kvTokens;
        private final BalanceContext routeOwner;
        private volatile long batchId;
        private volatile long ttlBaseAtMs = System.currentTimeMillis();

        private Reservation(long requestId, long kvTokens, BalanceContext routeOwner) {
            this.requestId = requestId;
            this.kvTokens = kvTokens;
            this.routeOwner = routeOwner;
        }

        @Override
        public boolean release() {
            if (!inflightRequests.remove(requestId, this)) {
                return false;
            }
            inflightKvReservedTotal.addAndGet(-kvTokens);
            return true;
        }

        @Override
        public synchronized boolean bindBatch(long assignedBatchId) {
            if (assignedBatchId <= 0) {
                throw new IllegalArgumentException("batchId must be positive");
            }
            if (inflightRequests.get(requestId) != this) {
                return false;
            }
            if (batchId == 0) {
                batchId = assignedBatchId;
            }
            if (batchId != assignedBatchId) {
                return false;
            }
            ttlBaseAtMs = System.currentTimeMillis();
            return true;
        }

        private boolean releaseForBatch(long reportedBatchId) {
            return reportedBatchId > 0
                    && batchId == reportedBatchId
                    && release();
        }

        @Override
        public long createdAtMs() {
            return ttlBaseAtMs;
        }

        private long kvTokens() {
            return kvTokens;
        }
    }

    private boolean releaseForWorkerStatus(TaskInfo task) {
        Reservation reservation = inflightRequests.get(task.getRequestId());
        return reservation != null && reservation.releaseForBatch(task.getBatchId());
    }

}

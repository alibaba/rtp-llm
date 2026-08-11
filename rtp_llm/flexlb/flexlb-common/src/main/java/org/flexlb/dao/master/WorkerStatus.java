package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

@Data
@Slf4j
public class WorkerStatus {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private RoleType role;
    private String group;
    private String ip;
    private int port;
    private int grpcPort;
    private String site;
    /**
     * Compatibility-only mirror of WorkerStatusPB.available_concurrency.
     * LocalRpcServer currently leaves that protobuf field unset, so this value
     * must not participate in routing, admission control, or batch sizing.
     */
    private Long availableConcurrency;
    /**
     * Authoritative Master-local health state. Engine payload data must never
     * mutate this field directly.
     */
    private final AtomicReference<WorkerLifecycleState> lifecycleState =
            new AtomicReference<>(WorkerLifecycleState.PROBING);
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong totalKvCacheTokens = new AtomicLong();
    private CacheStatus cacheStatus;
    private Map<String, TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1L);

    private double stepLatencyMs;
    private long iterateCount;
    private long dpSize;
    private long tpSize;
    private long dpRank;
    /** Model-level maximum sequence length reported by the Engine. */
    private long maxSeqLen;
    /** Strict aggregate context-token limit for an Engine batch/group. */
    private long maxBatchTokensSize;

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1);
    private AtomicLong statusUpdateIntervalUs = new AtomicLong(0);
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1);
    private AtomicLong lastSelectedTime = new AtomicLong(-1);
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true);
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false);
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false);
    private AtomicLong statusVersion = new AtomicLong(-1L);
    private AtomicLong consecutiveFailures = new AtomicLong(0);
    private final AtomicLong discoveryLastSeenTime = new AtomicLong(-1L);
    private final AtomicLong discoveryUpdateIntervalUs = new AtomicLong(0L);

    /**
     * Absorb dynamic engine payload fields from a gRPC status response.
     * Topology labels ({@code site}, {@code group}) are NOT set here —
     * they are managed externally by the sync runner. Health lifecycle and
     * heartbeat timestamps are also intentionally excluded and are committed
     * only after the response has passed generation and validity checks.
     */
    public void updateFromResponse(WorkerStatusResponse resp) {
        if (resp == null) {
            return;
        }
        this.role = resp.getRole();
        this.availableConcurrency = resp.getAvailableConcurrency();
        this.stepLatencyMs = resp.getStepLatencyMs();
        this.iterateCount = resp.getIterateCount();
        this.dpSize = resp.getDpSize();
        this.tpSize = resp.getTpSize();
        this.dpRank = resp.getDpRank();
        this.maxSeqLen = resp.getMaxSeqLen();
        this.maxBatchTokensSize = resp.getMaxBatchTokensSize();
        this.availableKvCacheTokens.set(resp.getAvailableKvCacheTokens());
        this.totalKvCacheTokens.set(resp.getTotalKvCacheTokens());
        // GetWorkerStatus response does not include cache status; preserve the one
        // set by GrpcCacheStatusCheckRunner to avoid nullifying it on every status sync.
        if (resp.getCacheStatus() != null) {
            this.cacheStatus = resp.getCacheStatus();
        }
        this.runningTaskList = resp.getRunningTaskInfo();
        this.statusVersion.set(resp.getStatusVersion());
        // NOTE: latestFinishedTaskVersion is NOT set here. It is advanced only after
        // calibrate has processed finished tasks, in GrpcWorkerStatusRunner.handleStatusResponse().
        // Setting it here would advance the version before calibrate runs, causing the engine
        // to filter out unprocessed finished tasks on the next poll — leaking inflight entries.
    }

    /**
     * Record a valid status response. Transport or validation failures must not
     * call this method.
     */
    public void recordStatusSuccess() {
        lock.lock();
        try {
            if (!isProbeable()) {
                return;
            }
            this.consecutiveFailures.set(0L);
            updateStatusHeartbeatTime();
        } finally {
            lock.unlock();
        }
    }

    /** Record a failed status attempt without refreshing the success heartbeat. */
    public long recordStatusFailure() {
        lock.lock();
        try {
            if (!isProbeable()) {
                return this.consecutiveFailures.get();
            }
            return this.consecutiveFailures.incrementAndGet();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Record that this address is still present in service discovery.
     *
     * <p>The lifecycle lock linearizes discovery observations with retirement:
     * a generation either records the observation while still probeable, or
     * retirement wins and the observation is rejected. This prevents an older,
     * overlapping discovery round from retiring a generation that a newer round
     * has already observed.
     *
     * @return {@code true} when the observation belongs to this live generation
     */
    public boolean recordDiscoverySeen(long nowUs) {
        lock.lock();
        try {
            if (!isProbeable()) {
                return false;
            }
            long previous = this.discoveryLastSeenTime.getAndSet(nowUs);
            if (previous > 0L && nowUs >= previous) {
                this.discoveryUpdateIntervalUs.set(nowUs - previous);
            }
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** Promote a newly discovered generation after its first valid live status. */
    public boolean tryMarkReady() {
        return this.lifecycleState.compareAndSet(
                WorkerLifecycleState.PROBING, WorkerLifecycleState.READY);
    }

    /**
     * Fence this generation before removing its endpoint and local state.
     * Exactly one competing failure/expiration/discovery path can win.
     */
    public boolean tryBeginRetirement() {
        while (true) {
            WorkerLifecycleState current = this.lifecycleState.get();
            if (current == WorkerLifecycleState.RETIRING
                    || current == WorkerLifecycleState.CLOSED) {
                return false;
            }
            if (this.lifecycleState.compareAndSet(current, WorkerLifecycleState.RETIRING)) {
                return true;
            }
        }
    }

    /** Mark a generation closed after its endpoint retirement has completed. */
    public boolean markClosed() {
        return this.lifecycleState.compareAndSet(
                WorkerLifecycleState.RETIRING, WorkerLifecycleState.CLOSED);
    }

    public WorkerLifecycleState getLifecycleState() {
        return this.lifecycleState.get();
    }

    public boolean isReady() {
        return this.lifecycleState.get() == WorkerLifecycleState.READY;
    }

    public boolean isProbeable() {
        WorkerLifecycleState state = this.lifecycleState.get();
        return state == WorkerLifecycleState.PROBING || state == WorkerLifecycleState.READY;
    }

    /** Compatibility view used by routing and monitoring code. */
    public boolean isAlive() {
        return isReady();
    }

    /**
     * Compatibility setter for tests and manually assembled statuses. Production
     * synchronization code must use the explicit lifecycle transitions instead.
     */
    @Deprecated
    public void setAlive(boolean alive) {
        if (alive) {
            tryMarkReady();
        } else {
            tryBeginRetirement();
        }
    }

    private void updateStatusHeartbeatTime() {
        long nowUs = System.nanoTime() / 1000;
        long prev = this.statusLastUpdateTime.get();
        if (prev > 0) {
            this.statusUpdateIntervalUs.set(nowUs - prev);
        }
        this.statusLastUpdateTime.set(nowUs);
    }

    /**
     * Update resource availability with hysteresis to prevent state oscillation.
     * <p>
     * Hysteresis uses two thresholds: upper and lower (calculated as upper - hysteresisBias%).
     * This creates a band where state doesn't change, preventing rapid toggling.
     * <p>
     * State transitions:
     * - AVAILABLE → UNAVAILABLE: when current metric EXCEEDS upper threshold
     * - UNAVAILABLE → AVAILABLE: when current metric FALLS BELOW lower threshold
     *
     * @param currentMetric current resource metric value
     * @param upperThreshold upper threshold for disabling availability
     * @param hysteresisBias bias percentage for calculating lower threshold (lower = upper - upper * bias / 100)
     * @return the new resource availability state
     */
    public boolean updateResourceAvailabilityWithHysteresis(long currentMetric, long upperThreshold, long hysteresisBias) {
        long lowerThreshold = Math.max(0, upperThreshold - (long)(upperThreshold * hysteresisBias / 100.0));

        if (currentMetric >= upperThreshold) {
            resourceAvailable.compareAndSet(true, false);
        } else if (currentMetric <= lowerThreshold) {
            resourceAvailable.compareAndSet(false, true);
        }
        return resourceAvailable.get();
    }

    /**
     * Get IP:PORT format address
     *
     * @return IP:PORT string
     */
    public String getIpPort() {
        if (ip == null) {
            return null;
        }
        return ip + ":" + port;
    }
}

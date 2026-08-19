package org.flexlb.dao.master;

import lombok.Data;
import org.flexlb.dao.route.RoleType;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Data
public class WorkerStatus {
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
    private volatile boolean alive;
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong totalKvCacheTokens = new AtomicLong();
    private volatile CacheStatus cacheStatus;
    private volatile Map<String, TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1L);
    /** Last status version whose Endpoint and request-lifecycle effects both completed. */
    private AtomicLong lastAppliedStatusVersion = new AtomicLong(-1L);

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
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true);
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false);
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false);
    private AtomicLong statusVersion = new AtomicLong(-1L);
    private AtomicLong consecutiveFailures = new AtomicLong(0);
    /** Last scheduling-load snapshot published by EndpointRegistry for monitoring. */
    private volatile long reportedSchedulingLoad;

    /**
     * Absorb all dynamic engine fields from a gRPC status response.
     * Topology labels ({@code site}, {@code group}) are NOT set here —
     * they are managed externally by the sync runner.
     */
    public void updateFromResponse(WorkerStatusResponse resp) {
        if (resp == null) {
            return;
        }
        this.role = resp.getRole();
        this.alive = resp.isAlive();
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
        // Cache metadata belongs exclusively to GrpcCacheStatusCheckRunner.
        // Status and cache callbacks intentionally run concurrently.
        this.runningTaskList = resp.getRunningTaskInfo();
        this.statusVersion.set(resp.getStatusVersion());
        // NOTE: latestFinishedTaskVersion is NOT set here. It is advanced only after
        // endpoint and request-lifecycle consumers have processed finished tasks.
        // Setting it here would advance the cursor before those consumers run, causing the engine
        // to filter out unprocessed finished tasks on the next poll — leaking inflight entries.

        updateStatusHeartbeatTime();
    }

    /**
     * Record a successful status heartbeat without replacing versioned task data.
     */
    public void refreshStatusHeartbeat(boolean alive) {
        this.alive = alive;
        updateStatusHeartbeatTime();
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

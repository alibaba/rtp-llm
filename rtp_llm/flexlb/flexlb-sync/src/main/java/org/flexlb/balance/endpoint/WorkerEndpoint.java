package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.domain.worker.WorkerStatusResponse;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Primary abstraction for a remote inference worker.  Holds both the
 * static identity (ip, ports) and the dynamic engine state that was
 * previously scattered between {@link WorkerEndpoint} and
 * {@link WorkerStatus}.
 *
 * <p>Strategy and sync layers should interact with the endpoint
 * directly rather than reaching into the internal {@link WorkerStatus}.
 */
public abstract class WorkerEndpoint {

    // ---- identity (immutable) ----
    private final String ip;
    private final int httpPort;
    private final int grpcPort;

    // ---- engine state (mutable, populated by gRPC sync) ----
    private final WorkerStatus status;
    private volatile boolean available = true;

    protected WorkerEndpoint(String ip, int httpPort, int grpcPort, WorkerStatus status) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.status = status;
    }

    // ==================== identity ====================

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

    public boolean isAvailable() {
        return available;
    }

    public void setAvailable(boolean available) {
        this.available = available;
    }

    // ==================== topology ====================

    public long getDpRank() { return status.getDpRank(); }
    public void setDpRank(long v) { status.setDpRank(v); }

    public long getDpSize() { return status.getDpSize(); }
    public void setDpSize(long v) { status.setDpSize(v); }

    public long getTpSize() { return status.getTpSize(); }
    public void setTpSize(long v) { status.setTpSize(v); }

    // ==================== group / role ====================

    public String getGroup() { return status.getGroup(); }
    public void setGroup(String v) { status.setGroup(v); }

    public String getSite() { return status.getSite(); }
    public void setSite(String v) { status.setSite(v); }

    public String getRole() { return status.getRole(); }
    public void setRole(String v) { status.setRole(v); }

    // ==================== liveness / capacity ====================

    public boolean isAlive() { return status.isAlive(); }
    public void setAlive(boolean v) { status.setAlive(v); }

    public Long getAvailableConcurrency() { return status.getAvailableConcurrency(); }
    public void setAvailableConcurrency(Long v) { status.setAvailableConcurrency(v); }

    public double getStepLatencyMs() { return status.getStepLatencyMs(); }
    public void setStepLatencyMs(double v) { status.setStepLatencyMs(v); }

    public long getIterateCount() { return status.getIterateCount(); }
    public void setIterateCount(long v) { status.setIterateCount(v); }

    // ==================== KV cache ====================

    public AtomicLong getAvailableKvCacheTokens() { return status.getAvailableKvCacheTokens(); }
    public AtomicLong getUsedKvCacheTokens() { return status.getUsedKvCacheTokens(); }

    public org.flexlb.dao.master.CacheStatus getCacheStatus() { return status.getCacheStatus(); }
    public void setCacheStatus(org.flexlb.dao.master.CacheStatus v) { status.setCacheStatus(v); }

    // ==================== tasks ====================

    public Map<String, org.flexlb.dao.master.TaskInfo> getRunningTaskList() { return status.getRunningTaskList(); }
    public void setRunningTaskList(Map<String, org.flexlb.dao.master.TaskInfo> v) { status.setRunningTaskList(v); }

    // ==================== resource availability ====================

    public AtomicBoolean getResourceAvailable() { return status.getResourceAvailable(); }
    public boolean isResourceAvailable() { return status.getResourceAvailable().get(); }

    public boolean updateResourceAvailabilityWithHysteresis(long currentMetric, long upperThreshold, long hysteresisBias) {
        return status.updateResourceAvailabilityWithHysteresis(currentMetric, upperThreshold, hysteresisBias);
    }

    // ==================== sync state ====================

    public AtomicLong getStatusVersion() { return status.getStatusVersion(); }
    public AtomicLong getLatestFinishedTaskVersion() { return status.getLatestFinishedTaskVersion(); }
    public AtomicLong getStatusLastUpdateTime() { return status.getStatusLastUpdateTime(); }
    public AtomicLong getStatusUpdateIntervalUs() { return status.getStatusUpdateIntervalUs(); }
    public AtomicBoolean getStatusCheckInProgress() { return status.getStatusCheckInProgress(); }
    public AtomicBoolean getCacheCheckInProgress() { return status.getCacheCheckInProgress(); }

    // ==================== status snapshot ====================

    /**
     * Returns an <b>immutable snapshot</b> of the underlying engine status.
     * Callers may read but must not mutate the returned object.
     */
    public WorkerStatus getStatus() {
        return status.toSnapshot();
    }

    // ==================== gRPC sync entry point ====================

    /**
     * Apply a full gRPC status response to this endpoint.
     * Replaces the piecemeal field writes previously done in
     * {@code GrpcWorkerStatusRunner}.
     */
    public void updateFromGrpcResponse(WorkerStatusResponse resp) {
        if (resp == null) {
            return;
        }
        status.setSite(getSite());
        status.setGroup(getGroup());
        status.setRole(resp.getRole());
        status.setAlive(resp.isAlive());
        status.setAvailableConcurrency(resp.getAvailableConcurrency());
        status.setStepLatencyMs(resp.getStepLatencyMs());
        status.setIterateCount(resp.getIterateCount());
        status.setDpSize(resp.getDpSize());
        status.setTpSize(resp.getTpSize());
        status.setDpRank(resp.getDpRank());
        status.getAvailableKvCacheTokens().set(resp.getAvailableKvCacheTokens());
        status.getStatusVersion().set(resp.getStatusVersion() != null ? resp.getStatusVersion() : -1L);
        status.getLatestFinishedTaskVersion().set(
                resp.getLatestFinishedVersion() != null ? resp.getLatestFinishedVersion() : -1L);
        status.setRunningTaskList(resp.getRunningTaskInfo());

        long nowUs = System.nanoTime() / 1000;
        long prev = status.getStatusLastUpdateTime().get();
        if (prev > 0) {
            status.getStatusUpdateIntervalUs().set(nowUs - prev);
        }
        status.getStatusLastUpdateTime().set(nowUs);
    }

    public void close() {
    }
}

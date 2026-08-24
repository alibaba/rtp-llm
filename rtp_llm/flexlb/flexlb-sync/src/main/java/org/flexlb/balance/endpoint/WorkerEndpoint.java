package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Primary abstraction for a remote inference worker.
 * Holds only a mutable {@link WorkerStatus} reference — all state
 * (identity, engine metrics, topology) is carried by the status object.
 *
 * <p>Callers read dynamic engine state via {@link #getStatus()} and
 * operate on it directly.
 */
public abstract class WorkerEndpoint {

    // ---- the sole mutable holding ----
    protected volatile WorkerStatus status;

    /**
     * Last time this endpoint was selected by a scheduling strategy.
     * Used for CAS-based fairness across concurrent requests.
     * Lives on the endpoint (not WorkerStatus) because {@code status} is replaced
     * on every sync cycle, which would lose the CAS state.
     */
    protected final AtomicLong lastSelectedTime = new AtomicLong(-1);

    public AtomicLong getLastSelectedTime() {
        return lastSelectedTime;
    }

    protected WorkerEndpoint(WorkerStatus status) {
        this.status = status;
    }

    // ==================== identity (delegated to status) ====================

    public String ipPort() {
        return status.getIpPort();
    }

    public String getIp() {
        return status.getIp();
    }

    public int getHttpPort() {
        return status.getPort();
    }

    public int getGrpcPort() {
        return status.getGrpcPort();
    }

    // ==================== status ====================

    /**
     * Returns the underlying {@link WorkerStatus} reference.
     * Callers read dynamic engine state from it; sync logic mutates
     * it in-place via {@link WorkerStatus#updateFromResponse}.
     */
    public WorkerStatus getStatus() {
        return status;
    }

    // ==================== gRPC sync entry point ====================

    /**
     * Replaces the endpoint's internal {@link #status} reference with
     * the one already updated by
     * {@link WorkerStatus#updateFromResponse(WorkerStatusResponse)}.
     * Triggers role-specific per-EP state refresh (ledger counter
     * snapshots) via subclass overrides.
     *
     * <p>Topology labels ({@code site}, {@code group}) are already
     * part of the incoming status — they belong to
     * {@link WorkerStatus}, not to {@link WorkerEndpoint}.
     *
     * @param ws   the updated status (replaces {@link #status})
     * @param resp the raw gRPC response (used by subclasses for task info)
     */
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        this.status = ws;
    }

    public void close() {
        // No resources by default. Stateful endpoints override when needed.
    }

    /**
     * Release EP-level resources tracked for the given request.
     * <p>Default implementation is a no-op. {@link DecodeEndpoint} overrides
     * to release the ledger-side KV reservation; prefill-side terminal
     * settlement lives in the state ledger's single settle exit, so
     * {@link PrefillEndpoint} no longer overrides this.
     *
     * @param requestId the request identifier whose EP-level reservation
     *                  should be released
     */
    public void release(long requestId) {
        // No resources by default. Stateful endpoints override when needed.
    }

}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;

/**
 * Primary abstraction for a remote inference worker.
 * Owns one immutable master-local endpoint generation and a stable
 * {@link WorkerStatus} object for that generation.
 *
 * <p>Callers read dynamic engine state via {@link #getStatus()} and
 * operate on it directly.
 */
public abstract class WorkerEndpoint {

    private final EndpointId endpointId;
    protected final WorkerStatus status;
    private final AtomicReference<EndpointLifecycleState> lifecycleState =
            new AtomicReference<>(EndpointLifecycleState.READY);
    private final ReentrantReadWriteLock operationGate = new ReentrantReadWriteLock(true);

    /**
     * Last time this endpoint was selected by a scheduling strategy.
     * Used for CAS-based fairness across concurrent requests.
     * Lives on the endpoint because fairness is scoped to this immutable local
     * generation and must not leak across retirement/republication.
     */
    protected final AtomicLong lastSelectedTime = new AtomicLong(-1);

    public AtomicLong getLastSelectedTime() {
        return lastSelectedTime;
    }

    protected WorkerEndpoint(EndpointId endpointId, WorkerStatus status) {
        this.endpointId = Objects.requireNonNull(endpointId, "endpointId");
        this.status = Objects.requireNonNull(status, "status");
        if (!endpointId.ipPort().equals(status.getIpPort())) {
            throw new IllegalArgumentException("EndpointId address " + endpointId.ipPort()
                    + " does not match WorkerStatus address " + status.getIpPort());
        }
    }

    public EndpointId getEndpointId() {
        return endpointId;
    }

    public EndpointLifecycleState getLifecycleState() {
        return lifecycleState.get();
    }

    public boolean isReady() {
        return lifecycleState.get() == EndpointLifecycleState.READY;
    }

    /**
     * READY for a new operation, or already covered by this thread's lease.
     * The latter lets a transaction that linearized before RETIRING finish so
     * the retirement write barrier can observe and settle its complete state.
     */
    public boolean isReadyForCurrentOperation() {
        return isReady() || operationGate.getReadHoldCount() > 0;
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
     * Apply one status response only to the still-current READY generation.
     *
     * <p>Topology labels ({@code site}, {@code group}) are already
     * part of the incoming status — they belong to
     * {@link WorkerStatus}, not to {@link WorkerEndpoint}.
     *
     * @param ws   the stable status object owned by this endpoint generation
     * @param resp the raw gRPC response (used by subclasses for task info)
     */
    public final void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        tryOnWorkerStatusUpdate(ws, resp);
    }

    public final boolean tryOnWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        if (ws != status) {
            return false;
        }
        return runIfReady(() -> handleWorkerStatusUpdate(resp));
    }

    protected void handleWorkerStatusUpdate(WorkerStatusResponse resp) {
        // Stateless endpoints have nothing to reconcile.
    }

    /** Run one operation only when this endpoint generation is READY. */
    public final boolean runIfReady(Runnable operation) {
        Objects.requireNonNull(operation, "operation");
        boolean alreadyLeased = operationGate.getReadHoldCount() > 0;
        operationGate.readLock().lock();
        try {
            if (!alreadyLeased && !isReady()) {
                return false;
            }
            operation.run();
            return true;
        } finally {
            operationGate.readLock().unlock();
        }
    }

    /** Run one operation only when READY, returning {@code unavailableValue} otherwise. */
    public final <T> T supplyIfReady(Supplier<T> operation, T unavailableValue) {
        Objects.requireNonNull(operation, "operation");
        boolean alreadyLeased = operationGate.getReadHoldCount() > 0;
        operationGate.readLock().lock();
        try {
            if (!alreadyLeased && !isReady()) {
                return unavailableValue;
            }
            return operation.get();
        } finally {
            operationGate.readLock().unlock();
        }
    }

    /** Linearization point that permanently stops admission to this generation. */
    final boolean beginRetirement() {
        return lifecycleState.compareAndSet(
                EndpointLifecycleState.READY, EndpointLifecycleState.RETIRING);
    }

    /** Signal role-specific background activity before waiting at the operation barrier. */
    void signalRetirement() {
        // Stateless endpoints have no background activity.
    }

    /** Wait until every operation admitted before RETIRING has left its read lease. */
    final void awaitOperationQuiescence() {
        operationGate.writeLock().lock();
        operationGate.writeLock().unlock();
    }

    /** Drain role-specific pending work after the READY-operation barrier. */
    List<BatchItem> drainForRetirement() {
        return List.of();
    }

    /** Clear role-specific local accounting after scheduler-owned state is settled. */
    void clearLocalStateForRetirement() {
        // Stateless endpoints have no local accounting.
    }

    final void completeRetirement() {
        lifecycleState.compareAndSet(
                EndpointLifecycleState.RETIRING, EndpointLifecycleState.CLOSED);
    }

    /**
     * Compatibility close for directly constructed endpoints. Registry-owned
     * endpoints must be retired through {@link EndpointRegistry} so scheduler
     * state participates in the same barrier.
     */
    public final void close() {
        if (!beginRetirement()) {
            return;
        }
        signalRetirement();
        awaitOperationQuiescence();
        try {
            drainForRetirement();
        } finally {
            clearLocalStateForRetirement();
            completeRetirement();
        }
    }

    final void lockOperationRead() {
        operationGate.readLock().lock();
    }

    final void unlockOperationRead() {
        operationGate.readLock().unlock();
    }

    // ==================== monitoring (EP-authoritative) ====================

    /**
     * Role-specific load metric for monitoring.
     * <p>Prefill: estimated queue wait time (ms).
     * <p>Decode: total active task count (confirmed running + inflight).
     */
    public abstract long getLoadMetric();

}

package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;

/**
 * Primary abstraction for a remote inference worker.
 * Holds one immutable-generation {@link WorkerStatus} reference — all state
 * (identity, engine metrics, topology) is carried by the status object.
 *
 * <p>Callers read dynamic engine state via {@link #getStatus()} and
 * operate on it directly.
 */
public abstract class WorkerEndpoint {

    private static final AtomicLong NEXT_LIFECYCLE_ID = new AtomicLong();

    protected final WorkerStatus status;
    /** Stable lock order for one RPC that depends on multiple endpoints. */
    private final long lifecycleId = NEXT_LIFECYCLE_ID.incrementAndGet();
    /**
     * Linearizes only the synchronous RPC-invocation boundary with retirement.
     * Fairness prevents a continuous stream of dispatch readers from starving
     * the rare retirement writer.
     */
    private final ReentrantReadWriteLock generationDispatchFence =
            new ReentrantReadWriteLock(true);
    private boolean retirementBegun;

    /**
     * Last time this endpoint was selected by a scheduling strategy.
     * Used for CAS-based fairness across concurrent requests.
     * Lives on the endpoint because it is Master scheduling state, not an
     * Engine-reported status field.
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
     * Apply a response only to the WorkerStatus generation that created this
     * endpoint. A new generation must create a new endpoint through the
     * registry; rebinding this object would break generation fencing.
     *
     * <p>Topology labels ({@code site}, {@code group}) are already
     * part of the incoming status — they belong to
     * {@link WorkerStatus}, not to {@link WorkerEndpoint}.
     *
     * @param ws   the expected generation
     * @param resp the raw gRPC response (used by subclasses for task info)
     */
    public final void applyWorkerStatusResponse(WorkerStatus ws, WorkerStatusResponse resp) {
        requireGeneration(ws);
        updateFromWorkerStatus(resp);
    }

    protected void updateFromWorkerStatus(WorkerStatusResponse resp) {
        // Simple endpoints have no local ledger to calibrate.
    }

    /**
     * Refresh only liveness anchors already owned by this endpoint generation.
     * Equal-version WorkerStatus responses use this path so active requests do
     * not expire without paying the cost of a full absence/terminal calibration.
     */
    public final void refreshWorkerStatusActivity(WorkerStatus ws, WorkerStatusResponse resp) {
        requireGeneration(ws);
        refreshActivityFromWorkerStatus(resp);
    }

    protected void refreshActivityFromWorkerStatus(WorkerStatusResponse resp) {
        // Stateless endpoints have no local activity anchors.
    }

    private void requireGeneration(WorkerStatus ws) {
        if (ws != status) {
            throw new IllegalArgumentException("WorkerStatus generation mismatch for " + ipPort());
        }
    }

    /**
     * Invoke an asynchronous client's synchronous entry point only while all
     * endpoint generations used by that request are active. Leases are taken
     * in a stable order and released as soon as the client returns its future;
     * they are never held for the RPC lifetime.
     */
    public final <T> T initiateGenerationDispatch(
            Iterable<? extends WorkerEndpoint> dependencies,
            Supplier<T> invocation) {
        Objects.requireNonNull(dependencies, "dependencies");
        Objects.requireNonNull(invocation, "invocation");

        List<WorkerEndpoint> ordered = orderedDependencies(this, dependencies);
        int acquired = 0;
        try {
            for (WorkerEndpoint endpoint : ordered) {
                endpoint.generationDispatchFence.readLock().lock();
                acquired++;
                if (endpoint.retirementBegun) {
                    throw new EndpointRetiredException(endpoint.ipPort());
                }
            }
            return invocation.get();
        } finally {
            for (int i = acquired - 1; i >= 0; i--) {
                ordered.get(i).generationDispatchFence.readLock().unlock();
            }
        }
    }

    private static List<WorkerEndpoint> orderedDependencies(
            WorkerEndpoint owner,
            Iterable<? extends WorkerEndpoint> dependencies) {
        Set<WorkerEndpoint> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<WorkerEndpoint> ordered = new ArrayList<>();
        seen.add(owner);
        ordered.add(owner);
        for (WorkerEndpoint endpoint : dependencies) {
            if (endpoint == null) {
                throw new IllegalArgumentException("dispatch endpoint dependency must not be null");
            }
            if (seen.add(endpoint)) {
                ordered.add(endpoint);
            }
        }
        ordered.sort(Comparator.comparingLong(endpoint -> endpoint.lifecycleId));
        return ordered;
    }

    /** Definite pre-send rejection: the endpoint generation is retired. */
    public static final class EndpointRetiredException extends IllegalStateException {
        public EndpointRetiredException(String ipPort) {
            super("Endpoint generation retired before RPC invocation: " + ipPort);
        }
    }

    /**
     * Fence generation-specific work before this endpoint is unpublished.
     * Resource shutdown belongs to {@link #close()} after the registry mutation.
     */
    final void beginRetirement() {
        generationDispatchFence.writeLock().lock();
        try {
            retirementBegun = true;
        } finally {
            generationDispatchFence.writeLock().unlock();
        }
    }

    public void close() {
        beginRetirement();
        // No resources by default. Stateful endpoints override when needed.
    }

    // ==================== monitoring (EP-authoritative) ====================

    /**
     * Role-specific load metric for monitoring.
     * <p>Prefill: estimated queue wait time (ms).
     * <p>Decode: total active task count (confirmed running + inflight).
     */
    abstract long schedulingLoad();

}

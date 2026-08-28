package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Primary abstraction for a remote inference worker.
 * Holds one immutable-generation {@link WorkerStatus} reference — all state
 * (identity, engine metrics, topology) is carried by the status object.
 *
 * <p>Callers read dynamic engine state via {@link #getStatus()} and
 * operate on it directly.
 */
public abstract class WorkerEndpoint {

    private static final AtomicLong RETIREMENT_THREAD_SEQUENCE =
            new AtomicLong();
    private static final ExecutorService RETIREMENT_EXECUTOR =
            Executors.newFixedThreadPool(4, task -> {
                Thread thread = new Thread(
                        task,
                        "flexlb-endpoint-retirement-"
                                + RETIREMENT_THREAD_SEQUENCE.incrementAndGet());
                thread.setDaemon(true);
                return thread;
            });

    private final WorkerStatus status;
    private final EndpointGenerationLifecycle generationLifecycle;

    /**
     * Last time this endpoint was selected by a scheduling strategy.
     * Used for live-LRU fairness across concurrent requests.
     * Lives on the endpoint (not WorkerStatus) because it belongs to the
     * endpoint generation rather than to an Engine status payload.
     */
    protected final AtomicLong lastSelectedTime = new AtomicLong(-1);

    public AtomicLong getLastSelectedTime() {
        return lastSelectedTime;
    }

    protected WorkerEndpoint(WorkerStatus status) {
        this.status = Objects.requireNonNull(status, "status");
        this.generationLifecycle = new EndpointGenerationLifecycle(
                this::continueRetirementAfterHandoff);
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
     * Callers read only committed dynamic engine state from it.
     */
    public WorkerStatus getStatus() {
        return status;
    }

    /**
     * Reduce one private status observation and publish it at the role's
     * consistency boundary. Any returned scheduler facts were derived from
     * exact endpoint-owned identities and are consumed only after this method
     * has released the role-local lock.
     */
    public EndpointStatusReduction applyPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        if (!prepared.observation().alive()) {
            beginRetirement();
        }
        ws.publishPreparedStatus(prepared);
        return EndpointStatusReduction.none();
    }

    /**
     * Initialize role-local state on a candidate which is not yet routable.
     * Publication is owned by {@link EndpointRegistry} after all reducers
     * succeed.
     */
    public EndpointStatusReduction initializeFromPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        return EndpointStatusReduction.none();
    }

    /**
     * Project active-request ownership from a successful same-version status
     * heartbeat without replaying versioned endpoint state or terminal facts.
     */
    public EndpointStatusReduction observeStatusHeartbeat(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        if (observation.owner() != ws) {
            throw new IllegalArgumentException(
                    "Status observation belongs to another WorkerStatus generation");
        }
        return EndpointStatusReduction.none();
    }

    protected final void requireStatusGeneration(WorkerStatus ws) {
        if (status != ws) {
            throw new IllegalArgumentException(
                    "WorkerStatus generation does not belong to this endpoint");
        }
    }

    /**
     * Pin this exact endpoint generation for one admission handoff.
     *
     * <p>The returned capability remains valid after retirement starts. That is
     * the point of the pin: a capture which linearized before registry detach
     * may finish its already accepted handoff. The owner must close the pin on
     * every selected, rejected, and exceptional path.</p>
     */
    public final GenerationPin tryPinGeneration() {
        EndpointGenerationLifecycle.HandoffPermit permit =
                generationLifecycle.tryAcquireHandoff();
        return permit == null
                ? null
                : new GenerationPin(
                        this,
                        status.getGenerationId(),
                        permit);
    }

    /** Validate an exact, still-open pin before consuming its admission right. */
    public final void requirePinnedGeneration(GenerationPin pin) {
        if (pin == null || pin.endpoint != this
                || pin.generationId != status.getGenerationId()
                || !pin.permit.isOpen()) {
            throw new IllegalArgumentException(
                    "Generation pin does not own this endpoint generation");
        }
    }

    /**
     * Internal generation handoff used by existing batch/route leases whose
     * terminal callback may run on a different thread. It shares the one gate
     * with public route pins but deliberately is not exposed as a route API.
     */
    final EndpointGenerationLifecycle.HandoffPermit
            tryAcquireGenerationHandoff() {
        return generationLifecycle.tryAcquireHandoff();
    }

    protected final boolean isGenerationRetiringOrRetired() {
        return generationLifecycle.isRetiringOrRetired();
    }

    /**
     * Close the admission gate for this exact endpoint generation.
     *
     * <p>This is the non-blocking half of endpoint retirement. Implementations
     * must make every later generation-local ownership acquisition fail closed,
     * but must not wait for existing handoffs, drain queues, or invoke external
     * callbacks. {@link EndpointRegistry} invokes this method while atomically
     * withdrawing the endpoint from its address mapping.</p>
     */
    final void beginRetirement() {
        generationLifecycle.beginRetirement();
    }

    /**
     * Initiate exact-generation drain and cleanup after the admission gate has
     * closed. This method never waits for cleanup already owned by another
     * caller; use {@link #awaitRetirement()} at a composition shutdown barrier.
     */
    public final void close() {
        retireInternal(false);
    }

    /** Initiate unpublished-candidate cleanup without blocking its status lock. */
    final void closeAsynchronously() {
        retireInternal(true);
    }

    private void retireInternal(boolean forceAsynchronous) {
        beginRetirement();

        if (!generationLifecycle.tryClaimCleanup()) {
            // Another exact caller already owns cleanup. Initiation is
            // idempotent and deliberately separate from the wait barrier.
            return;
        }

        // An accepted handoff may finish on a worker callback thread. The last
        // release schedules cleanup on the shared retirement executor so that
        // cleanup never waits on, or joins, its own callback thread.
        if (generationLifecycle.armDrainContinuation()) {
            return;
        }
        if (forceAsynchronous) {
            continueRetirementAfterHandoff();
        } else {
            finishRetirement();
        }
    }

    /**
     * Wait until this exact endpoint generation has completed all local cleanup
     * and retirement callbacks, then rethrow its recorded terminal failure.
     * {@link #close()} must have initiated cleanup first.
     */
    public final void awaitRetirement() {
        generationLifecycle.awaitRetirement();
    }

    private void finishRetirement() {
        generationLifecycle.beginCleanup();
        Throwable retirementFailure = null;
        try {
            closeEndpoint();
        } catch (Throwable failure) {
            retirementFailure = failure;
        } finally {
            generationLifecycle.completeRetirement(retirementFailure);
        }
        rethrowRetirementFailure(retirementFailure);
    }

    private void continueRetirementAfterHandoff() {
        RETIREMENT_EXECUTOR.execute(
                () -> {
                    try {
                        finishRetirement();
                    } catch (Throwable cleanupFailure) {
                        org.flexlb.util.Logger.error(
                                "Endpoint generation cleanup failed after detach: {}#{}",
                                status.getIpPort(),
                                status.getGenerationId(),
                                cleanupFailure);
                    }
                });
    }

    /**
     * Release subclass resources after the shared generation gate has drained.
     * This is invoked exactly once by the retirement owner.
     */
    protected void closeEndpoint() {
        // Stateless roles own no endpoint-local resources.
    }

    private static void rethrowRetirementFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Endpoint generation retirement failed", failure);
        }
    }

    /**
     * Exact, single-owner admission capability for one endpoint generation.
     * The permit is transferable: asynchronous ownership may close it from a
     * completion thread without adding a second lifecycle state.
     */
    public static final class GenerationPin implements AutoCloseable {

        private final WorkerEndpoint endpoint;
        private final long generationId;
        private final EndpointGenerationLifecycle.HandoffPermit permit;

        private GenerationPin(
                WorkerEndpoint endpoint,
                long generationId,
                EndpointGenerationLifecycle.HandoffPermit permit) {
            this.endpoint = endpoint;
            this.generationId = generationId;
            this.permit = permit;
        }

        public WorkerEndpoint endpoint() {
            return endpoint;
        }

        public long generationId() {
            return generationId;
        }

        @Override
        public void close() {
            permit.close();
        }
    }

    // ==================== monitoring (EP-authoritative) ====================

    /**
     * Role-specific observable load metric for monitoring.
     * <p>Prefill: identity-backed committed work (ms), absent while the
     * endpoint cannot publish a coherent complete work view.
     * <p>Decode: total active task count (confirmed running + inflight).
     */
    public abstract OptionalLong getLoadMetric();

}

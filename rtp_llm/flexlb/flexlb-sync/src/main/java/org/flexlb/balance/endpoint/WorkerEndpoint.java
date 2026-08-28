package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

import java.util.Objects;
import java.util.OptionalLong;
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

    private final WorkerStatus status;
    private final EndpointGenerationLifecycle generationLifecycle =
            new EndpointGenerationLifecycle();

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
                        Thread.currentThread(),
                        permit);
    }

    /** Validate an exact, still-open pin before consuming its admission right. */
    public final void requirePinnedGeneration(GenerationPin pin) {
        if (pin == null || pin.endpoint != this
                || pin.generationId != status.getGenerationId()
                || pin.ownerThread != Thread.currentThread()
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
    public final void beginRetirement() {
        generationLifecycle.beginRetirement();
    }

    /**
     * Initiate exact-generation drain and cleanup after the admission gate has
     * closed. This method never waits for cleanup already owned by another
     * caller; use {@link #awaitRetirement()} at a composition shutdown barrier.
     */
    public final void close() {
        retireInternal();
    }

    private void retireInternal() {
        beginRetirement();

        if (!generationLifecycle.tryClaimCleanup()) {
            // Another exact caller already owns cleanup. Initiation is
            // idempotent and deliberately separate from the wait barrier.
            return;
        }

        // Any accepted handoff may have moved to this callback thread. Never
        // guess thread ownership: when the closed gate still has active work,
        // the one claimed cleanup continues on a dedicated daemon.
        if (generationLifecycle.hasActiveHandoffs()) {
            continueRetirementAfterHandoff();
            return;
        }
        finishRetirement();
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
            generationLifecycle.awaitHandoffs();
            closeEndpoint();
        } catch (Throwable failure) {
            retirementFailure = failure;
        } finally {
            generationLifecycle.completeRetirement(retirementFailure);
        }
        rethrowRetirementFailure(retirementFailure);
    }

    private void continueRetirementAfterHandoff() {
        Thread continuation = new Thread(
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
                },
                "flexlb-endpoint-retirement-"
                        + status.getIpPort()
                        + "-" + status.getGenerationId());
        continuation.setDaemon(true);
        continuation.start();
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
     * Route pins are thread-confined: capture, admission, and close must all run
     * on the acquiring thread.
     */
    public static final class GenerationPin implements AutoCloseable {

        private final WorkerEndpoint endpoint;
        private final long generationId;
        private final Thread ownerThread;
        private final EndpointGenerationLifecycle.HandoffPermit permit;

        private GenerationPin(
                WorkerEndpoint endpoint,
                long generationId,
                Thread ownerThread,
                EndpointGenerationLifecycle.HandoffPermit permit) {
            this.endpoint = endpoint;
            this.generationId = generationId;
            this.ownerThread = ownerThread;
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
            if (ownerThread != Thread.currentThread()) {
                throw new IllegalStateException(
                        "Generation pin must close on its capture thread");
            }
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

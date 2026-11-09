package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Primary abstraction for a remote inference worker.
 * Holds one immutable-generation {@link WorkerStatus} reference — all state
 * (identity, engine metrics, topology) is carried by the status object.
 *
 * <p>Callers read dynamic engine state via {@link #getStatus()} and
 * operate on it directly.
 */
public class WorkerEndpoint {
    public static final StatusChange NO_STATUS_CHANGE = new StatusChange.None();
    private static final ExecutorService RETIREMENT_EXECUTOR =
            Executors.newFixedThreadPool(
                    4,
                    Thread.ofPlatform().daemon().name("flexlb-endpoint-retirement-", 1).factory());

    private final WorkerStatus status;
    private final ReentrantLock stateLock = new ReentrantLock();

    private volatile RetirementPhase phase = RetirementPhase.ACCEPTING_REQUESTS;

    private final CompletableFuture<Void> retirementCompleted = new CompletableFuture<>();

    private volatile Thread cleanupThread;

    public WorkerEndpoint(WorkerStatus status) {
        this.status = Objects.requireNonNull(status, "status");
    }

    public WorkerStatus getStatus() {
        return status;
    }

    // ==================== identity (delegated to status) ====================

    /**
     * Returns the logical worker identity, including the engine index.
     * Network calls use the separate IP and transport port accessors.
     */
    public String ipPort() {
        return status.getLogicalIpPort();
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
     * Reduce one private status observation and publish it at the role's
     * consistency boundary. Any returned scheduler facts were derived from
     * exact endpoint-owned identities and are consumed only after this method
     * has released the Endpoint lock.
     */
    public StatusChange applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        if (!prepared.observation().alive()) {
            beginRetirement();
        }
        ws.publishPreparedStatus(prepared);
        return NO_STATUS_CHANGE;
    }

    /**
     * Initialize role-local state on a candidate which is not yet routable.
     * Publication is owned by {@link EndpointRegistry} after all reducers
     * succeed.
     */
    public void initializeFromPreparedStatus(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
    }

    /**
     * Apply active-request facts from a successful same-version status heartbeat
     * without replaying versioned endpoint state or terminal facts.
     *
     * @return request notifications to run after releasing {@link WorkerStatus#lock}
     */
    public StatusChange applyStatusHeartbeat(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        checkArgument(
                observation.owner() == ws,
                "Status observation belongs to another WorkerStatus generation");
        return NO_STATUS_CHANGE;
    }

    /**
     * Publish only facts belonging to this role; subclasses own their notification adapters.
     */
    public void publishStatus(StatusChange change) {
        checkArgument(
                change instanceof StatusChange.None,
                "Status facts belong to another endpoint role");
    }

    /**
     * New requests require both a schedulable Worker and an Endpoint accepting requests.
     * Admission callers recheck this while holding the Endpoint state lock.
     */
    public final boolean acceptsNewRequests() {
        return !isRetired() && status.isAvailableForScheduling();
    }

    public final boolean isRetired() {
        return phase != RetirementPhase.ACCEPTING_REQUESTS;
    }

    public final void close() {
        retireInternal(stateLock.isHeldByCurrentThread());
    }

    // ==================== monitoring (EP-authoritative) ====================

    /**
     * Role-specific observable load metric for monitoring.
     * <p>Prefill: identity-backed committed work (ms), absent while the
     * endpoint cannot publish a coherent complete work view.
     * <p>Decode: total active task count (confirmed running + inflight).
     * Stateful endpoints skip the sample while their state lock is busy.
     */
    public OptionalLong tryGetLoadMetric() {
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        return OptionalLong.of(
                engine.runningTaskList() == null ? 0L : engine.runningTaskList().size());
    }

    protected final void requireStatusGeneration(WorkerStatus ws) {
        checkArgument(status == ws, "WorkerStatus generation does not belong to this endpoint");
    }

    /**
     * Remove request records in the Endpoint state lock, then notify outside it.
     */
    protected void closeEndpoint() {}

    final ReentrantLock stateLock() {
        return stateLock;
    }

    /**
     * Reject later admissions in the same critical section as request record changes.
     */
    final void beginRetirement() {
        stateLock.lock();
        try {
            if (phase == RetirementPhase.ACCEPTING_REQUESTS) phase = RetirementPhase.RETIRING;
        } finally {
            stateLock.unlock();
        }
    }

    final void closeAsynchronously() {
        retireInternal(true);
    }

    /**
     * Finalize registry and cache state after request cleanup and notifications succeed.
     */
    void onRetired(Runnable completion) {
        retirementCompleted.thenRun(Objects.requireNonNull(completion, "completion"));
    }

    void awaitRetirement(long timeoutMs) {
        checkArgument(timeoutMs > 0L, "endpoint retirement timeout must be positive");
        checkState(
                !stateLock.isHeldByCurrentThread() && cleanupThread != Thread.currentThread(),
                "endpoint cleanup cannot wait for itself");
        try {
            com.google.common.util.concurrent.Uninterruptibles.getUninterruptibly(
                    retirementCompleted, timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS);
        } catch (java.util.concurrent.ExecutionException failure) {
            Failures.rethrow(failure.getCause(), "Endpoint retirement failed");
        } catch (java.util.concurrent.TimeoutException timeout) {
            throw new IllegalStateException(
                    "Endpoint retirement timed out: phase=" + phase, timeout);
        }
    }

    static void executeRetirement(Runnable cleanup) {
        RETIREMENT_EXECUTOR.execute(cleanup);
    }

    private void retireInternal(boolean asynchronously) {
        beginRetirement();
        stateLock.lock();
        try {
            if (phase != RetirementPhase.RETIRING) return;
            phase = RetirementPhase.CLEANUP_SCHEDULED;
        } finally {
            stateLock.unlock();
        }
        if (asynchronously) executeRetirement(this::finishRetirement);
        else finishRetirement();
    }

    private void finishRetirement() {
        cleanupThread = Thread.currentThread();
        stateLock.lock();
        try {
            phase = RetirementPhase.CLEANING;
        } finally {
            stateLock.unlock();
        }
        Throwable failure = null;
        try {
            closeEndpoint();
        } catch (Throwable problem) {
            failure = problem;
        } finally {
            stateLock.lock();
            try {
                phase = RetirementPhase.RETIRED;
            } finally {
                stateLock.unlock();
            }
            cleanupThread = null;
            if (failure == null) retirementCompleted.complete(null);
            else retirementCompleted.completeExceptionally(failure);
        }
        Failures.rethrow(failure, "Endpoint retirement failed");
    }

    /**
     * Passive status facts. The sync workflow publishes them after releasing the generation lock.
     */
    public sealed interface StatusChange {
        record None() implements StatusChange {}

        /**
         * Freezes Encoder request observations for status processing outside the endpoint lock.
         */
        record Encoder(java.util.List<EncoderEndpoint.RequestStatus> requests)
                implements StatusChange {
            public Encoder {
                requests = java.util.List.copyOf(requests);
            }
        }

        /**
         * Freezes Prefill request and batch changes together with scheduling and capacity notifications.
         */
        record Prefill(
                java.util.List<PrefillEndpoint.PrefillRequestStatus> requestStatuses,
                java.util.List<PrefillEndpoint.BatchCompletion> batchCompletions,
                boolean schedulingInputsChanged,
                boolean capacityReleased)
                implements StatusChange {
            public Prefill {
                requestStatuses = java.util.List.copyOf(requestStatuses);
                batchCompletions = java.util.List.copyOf(batchCompletions);
            }
        }

        /**
         * Freezes Decode request changes and the placement and dispatch capacity notification flags.
         */
        record Decode(
                java.util.List<DecodeResources.DecodeRequestStatus> requests,
                boolean placementCapacityChanged,
                boolean dispatchCapacityChanged)
                implements StatusChange {
            public Decode {
                requests = java.util.List.copyOf(requests);
            }
        }
    }

    private enum RetirementPhase {
        ACCEPTING_REQUESTS,
        RETIRING,
        CLEANUP_SCHEDULED,
        CLEANING,
        RETIRED
    }
}

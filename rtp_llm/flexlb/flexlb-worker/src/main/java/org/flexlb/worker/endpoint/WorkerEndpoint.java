package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.worker.status.WorkerStatus;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Serializes role-specific request updates and WorkerStatus publication for one worker generation.
 * Request events are delivered after status locks are released; closing stops admission before cleanup.
 */
public class WorkerEndpoint {
    public static final StatusChange NO_STATUS_CHANGE = new StatusChange.None();
    private static final ExecutorService CLEANUP_EXECUTOR =
            Executors.newFixedThreadPool(
                    4, Thread.ofPlatform().daemon().name("flexlb-endpoint-cleanup-", 1).factory());

    private final WorkerStatus status;
    private final ReentrantLock stateLock = new ReentrantLock();

    private volatile EndpointState endpointState = EndpointState.ACCEPTING_REQUESTS;

    private final CompletableFuture<Void> closeCompleted = new CompletableFuture<>();

    public WorkerEndpoint(WorkerStatus status) {
        this.status = Objects.requireNonNull(status, "status");
    }

    /**
     * Remove expired local request records that the scheduler no longer retains.
     * Implementations query retention outside endpoint locks and recheck captured records before removal.
     * Endpoints without local request records have nothing to expire.
     */
    public int evictExpiredRequests(
            long ttlMs, java.util.function.Predicate<String> retainForSchedulerCleanup) {
        return 0;
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
     * has released the DiscoveryEndpoint lock.
     */
    public final StatusChange applyWorkerStatus(
            WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        checkArgument(status == ws, "WorkerStatus generation does not belong to this endpoint");
        var observation = prepared.observation();
        checkArgument(observation.owner() == ws, "Status belongs to another worker generation");
        stateLock.lock();
        try {
            checkState(
                    !hasStoppedAcceptingRequests(),
                    "DiscoveryEndpoint has stopped accepting requests");
            try {
                if (!observation.alive()) stopAdmission();
                var changes = updateStatusLocked(observation);
                ws.publishReducedStatus(prepared);
                return changes;
            } catch (Throwable failure) {
                stopAdmission();
                throw Failures.propagate(
                        failure, "DiscoveryEndpoint status update/publication failed");
            }
        } finally {
            stateLock.unlock();
        }
    }

    public final boolean hasStoppedAcceptingRequests() {
        return endpointState != EndpointState.ACCEPTING_REQUESTS;
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
     * New requests require both a schedulable Worker and an DiscoveryEndpoint accepting requests.
     * Admission callers recheck this while holding the DiscoveryEndpoint state lock.
     */
    public final boolean acceptsNewRequests() {
        return !hasStoppedAcceptingRequests() && status.isAvailableForScheduling();
    }

    // ==================== monitoring (EP-authoritative) ====================

    /**
     * Role-specific observable load metric for monitoring.
     * <p>Prefill reports compute tokens; Decode reports observed and locally unconfirmed request counts.
     * Stateful endpoints skip the sample while their state lock is busy.
     */
    public OptionalLong tryGetLoadMetric() {
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        return OptionalLong.of(
                engine.runningTaskList() == null ? 0L : engine.runningTaskList().size());
    }

    /**
     * Updates role-specific request records while the DiscoveryEndpoint lock is held.
     * The caller publishes the WorkerStatus snapshot before releasing that lock.
     */
    protected StatusChange updateStatusLocked(WorkerStatus.StatusObservation observation) {
        return NO_STATUS_CHANGE;
    }

    /**
     * Delivers an already validated request event outside the status and DiscoveryEndpoint locks.
     */
    protected final void publishWorkerRequest(
            WorkerRoute route,
            DecodeEndpoint.RequestRegistration reservation,
            RoleType role,
            org.flexlb.worker.endpoint.RequestEventType kind,
            long errorCode,
            String detail) {
        var context = route.ctx();
        context.getRequestScheduler()
                .onWorkerRequestEvent(
                        context,
                        new org.flexlb.worker.endpoint.WorkerRequestEvent(
                                route,
                                reservation,
                                role,
                                this,
                                kind,
                                errorCode,
                                detail,
                                System.currentTimeMillis()));
    }

    /**
     * Remove request records in the DiscoveryEndpoint state lock, then notify outside it.
     */
    protected void closeEndpoint() {}

    /**
     * Starts cleanup once and returns its completion without exposing the mutable shared future.
     */
    final java.util.concurrent.CompletionStage<Void> closeAsync() {
        scheduleCleanup();
        return closeCompleted.minimalCompletionStage();
    }

    final ReentrantLock stateLock() {
        return stateLock;
    }

    /**
     * Stops request admission and Worker updates without starting resource cleanup.
     * WorkerStatus locking precedes the DiscoveryEndpoint lock, matching status publication.
     */
    final void stopAcceptingRequests() {
        status.lock.lock();
        try {
            stopAdmission();
            status.stopUpdatesAfterAdmissionClosed();
        } finally {
            status.lock.unlock();
        }
    }

    private void scheduleCleanup() {
        stateLock.lock();
        try {
            if (endpointState != EndpointState.ACCEPTING_REQUESTS
                    && endpointState != EndpointState.NOT_ACCEPTING_REQUESTS) return;
            endpointState = EndpointState.CLEANUP_SCHEDULED;
        } finally {
            stateLock.unlock();
        }
        try {
            CLEANUP_EXECUTOR.execute(this::finishClose);
        } catch (java.util.concurrent.RejectedExecutionException failure) {
            closeCompleted.completeExceptionally(failure);
        }
    }

    private void stopAdmission() {
        stateLock.lock();
        try {
            if (endpointState == EndpointState.ACCEPTING_REQUESTS)
                endpointState = EndpointState.NOT_ACCEPTING_REQUESTS;
        } finally {
            stateLock.unlock();
        }
    }

    private void finishClose() {
        stateLock.lock();
        try {
            endpointState = EndpointState.CLEANING;
        } finally {
            stateLock.unlock();
        }
        Throwable failure = null;
        try {
            stopAcceptingRequests();
            closeEndpoint();
        } catch (Throwable problem) {
            failure = problem;
        } finally {
            stateLock.lock();
            try {
                endpointState = EndpointState.CLOSED;
            } finally {
                stateLock.unlock();
            }
            if (failure == null) closeCompleted.complete(null);
            else closeCompleted.completeExceptionally(failure);
        }
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
         * Freezes Prefill request changes together with scheduling and capacity notifications.
         */
        record Prefill(
                java.util.List<PrefillEndpoint.PrefillRequestStatus> requestStatuses,
                boolean schedulingInputsChanged,
                boolean capacityReleased)
                implements StatusChange {
            public Prefill {
                requestStatuses = java.util.List.copyOf(requestStatuses);
            }
        }

        /**
         * Freezes Decode request changes and whether scheduling should recheck capacity.
         */
        record Decode(
                java.util.List<DecodeResources.DecodeRequestStatus> requests,
                boolean capacityChanged)
                implements StatusChange {
            public Decode {
                requests = java.util.List.copyOf(requests);
            }
        }
    }

    /**
     * Identifies one request's exact resource registration within a worker generation.
     */
    public interface RequestRegistration {
        String requestId();

        long endpointGenerationId();
    }

    /**
     * The local request registration stage, independent of Worker execution reports.
     */
    public enum RegistrationPhase {
        REGISTERED,
        COMMITTED
    }

    public enum ReleaseReason {
        LOCAL_ROLLBACK,
        /**
         * Context revoked an unsent route; release its reservation without fencing a retry.
         */
        COUNTERPART_FINISHED,
        NOT_SENT,
        REMOTE_CLEANUP,
        EXPIRED
    }

    /**
     * Reports the factual outcome of registering resources, without deciding scheduling actions.
     */
    public enum RegistrationStatus {
        REGISTERED,
        VERSION_CONFLICT,
        CAPACITY_FULL,
        ALREADY_REGISTERED,
        UNAVAILABLE
    }

    /**
     * A successful registration carries its exact request record; failures carry no record.
     */
    public record Registration<T>(RegistrationStatus status, T value) {
        public Registration {
            Objects.requireNonNull(status, "status");
            checkArgument(
                    (status == RegistrationStatus.REGISTERED) == (value != null),
                    "only successful registration carries a request record");
        }

        public static <T> Registration<T> rejected(RegistrationStatus status) {
            return new Registration<>(status, null);
        }
    }

    private enum EndpointState {
        ACCEPTING_REQUESTS,
        NOT_ACCEPTING_REQUESTS,
        CLEANUP_SCHEDULED,
        CLEANING,
        CLOSED
    }
}

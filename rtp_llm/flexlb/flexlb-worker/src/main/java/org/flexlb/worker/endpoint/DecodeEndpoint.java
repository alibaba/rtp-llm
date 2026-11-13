package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.DecodeResources.CapacityUsage;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.worker.endpoint.DecodeResources.RegistrationState;
import org.flexlb.worker.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.worker.status.DecodeTaskPhase;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.worker.endpoint.DecodeResources.Capacity.saturatedAddNonNegative;

/**
 * Reserves Decode capacity and reconciles each local request with worker reports.
 * Confirmed cleanup removes the exact reservation and wakes scheduling outside the state lock.
 */
public class DecodeEndpoint extends WorkerEndpoint
        implements EndpointStats.Decode, SelectionRollback {
    private final WorkerRequests requests;
    private final CapacityNotifier capacityNotifier;
    private final Map<String, RequestRecord> registeredRequests = new ConcurrentHashMap<>();

    // Both request and KV reservations remain local until Decode confirms the request.
    private int unconfirmedRequests;
    private long unconfirmedKvTokens;
    private long unconfirmedExpectedKvTokens;
    private volatile int reportedRequestCount;
    private volatile long admissionVersion;

    private volatile DecodeEndpoint.SelectionSnapshot selectionSnapshotCache;

    DecodeEndpoint(
            WorkerStatus status, WorkerRequests requests, CapacityNotifier capacityNotifier) {
        super(status);
        this.requests = java.util.Objects.requireNonNull(requests, "requests");
        this.capacityNotifier =
                java.util.Objects.requireNonNull(capacityNotifier, "capacityNotifier");
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) {
            return;
        }
        var effects = (StatusChange.Decode) change;
        if (effects.capacityChanged()) capacityNotifier.notifyChanged(this);
        for (var observed : effects.requests()) {
            var reservation = observed.reservation();
            var context = requests.findActive(reservation.requestId());
            if (context == null) continue;
            var route = context.snapshot().route();
            if (route == null || route.decodeEndpoint() != this) continue;
            publishWorkerRequest(
                    route,
                    reservation,
                    RoleType.DECODE,
                    workerEventType(observed),
                    observed.errorCode(),
                    "");
        }
    }

    /**
     * Removes an exact selection registration only while the worker has not confirmed execution.
     */
    @Override
    public boolean rollbackSelection(WorkerEndpoint.RequestRegistration exact) {
        return releaseRequest(exact, true);
    }

    /**
     * Releases an exact registration after confirming it was not sent or remote cleanup completed.
     * Sending a cancellation alone is not sufficient.
     */
    public boolean releaseRequest(RequestRegistration exact) {
        return releaseRequest(exact, false);
    }

    /**
     * Removes registrations no longer retained by the scheduler after their last observation exceeds TTL.
     * Both confirmed and unconfirmed requests expire; the return value counts only unconfirmed reservations.
     */
    @Override
    public int evictExpiredRequests(long ttlMs, Predicate<String> schedulerRetainsRequest) {
        List<RequestRecord> expiredRequests = new ArrayList<>();
        stateLock().lock();
        try {
            long nowMs = System.currentTimeMillis();
            for (var request : registeredRequests.values()) {
                if (nowMs - request.observedAtMs > ttlMs) expiredRequests.add(request);
            }
        } finally {
            stateLock().unlock();
        }
        expiredRequests.removeIf(request -> schedulerRetainsRequest.test(request.requestId));
        int expiredUnconfirmedReservations = 0;
        boolean removedAnyRequest = false;
        stateLock().lock();
        try {
            long nowMs = System.currentTimeMillis();
            for (var request : expiredRequests) {
                if (registeredRequests.get(request.requestId) == request
                        && nowMs - request.observedAtMs > ttlMs) {
                    removeRequestLocked(request);
                    if (!request.confirmed) {
                        expiredUnconfirmedReservations++;
                    }
                    removedAnyRequest = true;
                }
            }
            if (removedAnyRequest) {
                admissionVersion++;
            }
        } finally {
            stateLock().unlock();
        }
        if (removedAnyRequest) capacityNotifier.notifyChanged(this);
        return expiredUnconfirmedReservations;
    }

    /**
     * Register against the exact view used for ranking. No remote capacity is promised.
     */
    public WorkerEndpoint.Registration<RequestRegistration> tryRegisterSelection(
            DecodeEndpoint.SelectionSnapshot expected,
            String requestId,
            long hardKv,
            long expectedKv,
            int priority,
            AdmissionCapacity capacity) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests() || expected.generationId() != getStatus().getGenerationId())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.UNAVAILABLE);
            var current = selectionSnapshot();
            if (current.admissionVersion() != expected.admissionVersion()
                    || current.workerStatus() != expected.workerStatus()
                    || current.topology() != expected.topology())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.VERSION_CONFLICT);
            if (!capacity.evaluate(current.usage(), hardKv, expectedKv, Capacity.NONE).isZero())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            if (!requestIdAvailable(requestId))
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.ALREADY_REGISTERED);
            return new WorkerEndpoint.Registration<>(
                    RegistrationStatus.REGISTERED,
                    createReservation(requestId, hardKv, expectedKv, priority));
        } finally {
            stateLock().unlock();
        }
    }

    public DecodeEndpoint.SelectionSnapshot selectionSnapshot() {
        DecodeEndpoint.SelectionSnapshot observed = selectionSnapshotCache;
        if (observed != null
                && observed.admissionVersion() == admissionVersion
                && observed.workerStatus() == getStatus().committedWorkerStatus()
                && observed.topology() == getStatus().topologySnapshot()) {
            return observed;
        }
        stateLock().lock();
        try {
            WorkerStatus.CommittedWorkerStatus committed = getStatus().committedWorkerStatus();
            WorkerStatus.TopologySnapshot topology = getStatus().topologySnapshot();
            var cached = selectionSnapshotCache;
            if (cached == null
                    || cached.admissionVersion() != admissionVersion
                    || cached.workerStatus() != committed
                    || cached.topology() != topology) {
                cached =
                        new DecodeEndpoint.SelectionSnapshot(
                                ipPort(),
                                getStatus().getGenerationId(),
                                topology,
                                committed,
                                admissionVersion,
                                usage(committed.fields()));
                selectionSnapshotCache = cached;
            }
            return cached;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Exact resource obligation; absence also covers closed or replaced reservation identities.
     */
    public RegistrationState registrationState(RequestRegistration exact) {
        if (exact == null || exact.endpointGenerationId() != getStatus().getGenerationId()) {
            return RegistrationState.NOT_REGISTERED;
        }
        stateLock().lock();
        try {
            RequestRecord request = registeredRequests.get(exact.requestId());
            if (request == null || !request.matches(exact)) {
                return RegistrationState.NOT_REGISTERED;
            }
            if (request.confirmed) {
                return RegistrationState.CONFIRMED;
            }
            return RegistrationState.UNCONFIRMED;
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public OptionalLong tryGetLoadMetric() {
        if (!stateLock().tryLock()) return OptionalLong.empty();
        try {
            return OptionalLong.of(reportedRequestCount + unconfirmedRequests);
        } finally {
            stateLock().unlock();
        }
    }

    public AdmissionEvaluation evaluateSelection(
            org.flexlb.worker.endpoint.Request request, int priority, AdmissionCapacity capacity) {
        var snapshot = resourceSnapshot();
        var deficit =
                capacity.evaluate(
                        snapshot.selection().usage(),
                        request.getHardKvTokens(),
                        request.getExpectedKvTokens(),
                        Capacity.NONE);
        return new AdmissionEvaluation(
                snapshot.selection(),
                deficit,
                deficit.isZero()
                        ? null
                        : classifyCapacityFailure(request, priority, snapshot, capacity));
    }

    public ResourceSnapshot resourceSnapshot() {
        stateLock().lock();
        try {
            var active = getStatus().committedEngineObservation().runningTaskList();
            Map<String, DecodeRequestView> captured = new HashMap<>();
            for (var request : registeredRequests.values()) {
                var task = active.get(request.requestId);
                // Execution phase is worker telemetry, not a second local resource state.
                var phase =
                        !request.confirmed
                                ? DecodeTaskPhase.LOCAL_RESERVED
                                : task != null && task.phase() == TaskPhase.RUNNING
                                        ? DecodeTaskPhase.RUNNING
                                        : DecodeTaskPhase.ACCEPTED_NOT_RUNNING;
                captured.put(
                        request.requestId,
                        new DecodeRequestView(
                                request.requestId,
                                request.priority,
                                request.kvTokens,
                                request.expectedKvTokens,
                                phase,
                                request.reservation()));
            }
            return new ResourceSnapshot(selectionSnapshot(), captured);
        } finally {
            stateLock().unlock();
        }
    }

    public EndpointStats.DecodeCounts stats() {
        stateLock().lock();
        try {
            long oldest = Long.MAX_VALUE;
            int accepted = 0;
            int running = 0;
            var active = getStatus().committedEngineObservation().runningTaskList();
            for (RequestRecord request : registeredRequests.values()) {
                long observed = request.observedAtMs;
                if (!request.confirmed) {
                    oldest = Math.min(oldest, observed);
                } else {
                    var task = active.get(request.requestId);
                    if (task != null && task.phase() == TaskPhase.RUNNING) running++;
                    else accepted++;
                }
            }
            long age =
                    oldest == Long.MAX_VALUE
                            ? 0L
                            : Math.max(0L, System.currentTimeMillis() - oldest);
            return new EndpointStats.DecodeCounts(
                    unconfirmedRequests,
                    reportedRequestCount + unconfirmedRequests,
                    unconfirmedExpectedKvTokens,
                    unconfirmedKvTokens,
                    age,
                    accepted,
                    running);
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    protected StatusChange updateStatusLocked(WorkerStatus.StatusObservation observation) {
        var active = observation.engine().runningTaskList();
        var finished = observation.finishedTasks();
        var events = new ArrayList<DecodeRequestStatus>();
        long now = System.currentTimeMillis();
        for (var request : registeredRequests.values()) {
            var completed = finished.get(request.requestId);
            if (completed != null) {
                events.add(
                        DecodeRequestStatus.terminal(request.reservation(), completed.errorCode()));
                removeRequestLocked(request);
                continue;
            }
            var task = active.get(request.requestId);
            if (task != null) {
                if (!request.confirmed) {
                    unconfirmedRequests--;
                    unconfirmedKvTokens -= request.kvTokens;
                    unconfirmedExpectedKvTokens -= request.expectedKvTokens;
                    request.confirmed = true;
                    request.kvTokens = Math.max(0L, task.inputLength());
                    request.expectedKvTokens = request.kvTokens;
                }
                request.observedAtMs = now;
                events.add(DecodeRequestStatus.confirmed(request.reservation()));
            } else if (request.confirmed) {
                removeRequestLocked(request);
            }
        }
        long reported =
                Math.max(
                        active.size(),
                        saturatedAddNonNegative(
                                Math.max(0L, observation.engine().runningQueryLen()),
                                Math.max(0L, observation.engine().waitingQueryLen())));
        reportedRequestCount = (int) Math.min(reported, Integer.MAX_VALUE);
        admissionVersion++;
        return new StatusChange.Decode(events, true);
    }

    // WorkerCleanup and orphan cleanup.

    @Override
    protected void closeEndpoint() {
        var removedReservations = removeRequestsOnClose();
        // Records are already cleared; notification failures must not block registry cleanup.
        for (var reservation : removedReservations) {
            try {
                notifyRequestOfEndpointClose(reservation);
            } catch (RuntimeException notificationFailure) {
                Logger.error(
                        "Decode endpoint close notification failed for request "
                                + reservation.requestId(),
                        notificationFailure);
            }
        }
        try {
            capacityNotifier.notifyChanged(this);
        } catch (RuntimeException notificationFailure) {
            Logger.error(
                    "Decode endpoint capacity notification failed for " + ipPort(),
                    notificationFailure);
        }
    }

    private boolean releaseRequest(
            WorkerEndpoint.RequestRegistration exact, boolean requireUnconfirmed) {
        if (exact == null || exact.endpointGenerationId() != getStatus().getGenerationId())
            return false;
        boolean released;
        stateLock().lock();
        try {
            var request = registeredRequests.get(exact.requestId());
            released = request != null && request.matches(exact);
            checkState(
                    !released || !requireUnconfirmed || !request.confirmed,
                    "route rollback retained Decode resources: worker already confirmed the request");
            if (released) removeRequestLocked(request);
            if (released) admissionVersion++;
        } finally {
            stateLock().unlock();
        }
        if (released) capacityNotifier.notifyChanged(this);
        return released;
    }

    /**
     * Removes the exact local registration and its outstanding contribution once.
     * Confirmed request capacity remains governed by the worker report.
     */
    private void removeRequestLocked(RequestRecord request) {
        if (!registeredRequests.remove(request.requestId, request)) return;
        if (!request.confirmed) {
            unconfirmedRequests--;
            unconfirmedKvTokens -= request.kvTokens;
            unconfirmedExpectedKvTokens -= request.expectedKvTokens;
        }
    }

    private static RequestEventType workerEventType(DecodeRequestStatus observed) {
        if (observed.kind() == DecodeRequestStatus.Kind.TERMINAL) {
            if (observed.errorCode() == 0L) return RequestEventType.COMPLETED;
            return RequestEventType.FAILED;
        }
        return RequestEventType.ALLOCATED;
    }

    private void notifyRequestOfEndpointClose(RequestRegistration reservation) {
        var context = requests.findActive(reservation.requestId());
        if (context == null) return;
        var event =
                new org.flexlb.worker.endpoint.WorkerRequestEvent(
                        context.snapshot().route(),
                        reservation,
                        RoleType.DECODE,
                        this,
                        org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                        0L,
                        "Decode generation closed: " + reservation.endpointGenerationId(),
                        System.currentTimeMillis());
        context.getRequestScheduler().onWorkerRequestEvent(context, event);
    }

    private List<RequestRegistration> removeRequestsOnClose() {
        List<RequestRegistration> removed;
        stateLock().lock();
        try {
            removed = registeredRequests.values().stream().map(RequestRecord::reservation).toList();
            registeredRequests.clear();
            unconfirmedRequests = 0;
            unconfirmedKvTokens = 0L;
            unconfirmedExpectedKvTokens = 0L;
            reportedRequestCount = 0;
            admissionVersion++;
        } finally {
            stateLock().unlock();
        }
        return removed;
    }

    private RequestRegistration createReservation(
            String requestId, long hardKv, long expectedKv, int priority) {
        var request =
                new RequestRecord(
                        getStatus().getGenerationId(), requestId, hardKv, expectedKv, priority);
        registeredRequests.put(requestId, request);
        unconfirmedRequests++;
        unconfirmedKvTokens += hardKv;
        unconfirmedExpectedKvTokens += expectedKv;
        admissionVersion++;
        return request.reservation();
    }

    private boolean requestIdAvailable(String requestId) {
        return !registeredRequests.containsKey(requestId)
                && !getStatus()
                        .committedEngineObservation()
                        .runningTaskList()
                        .containsKey(requestId);
    }

    private CapacityUsage usage(WorkerStatus.EngineObservation fields) {
        long reported =
                fields.totalKvCacheTokens() > 0L
                        ? Math.max(
                                0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens())
                        : 0L;
        return new CapacityUsage(
                engineLoadLocked(),
                Math.max(0L, fields.totalKvCacheTokens()),
                Math.max(0L, fields.availableKvCacheTokens()),
                unconfirmedKvTokens,
                saturatedAddNonNegative(reported, unconfirmedExpectedKvTokens));
    }

    private int engineLoadLocked() {
        return reportedRequestCount + unconfirmedRequests;
    }

    /**
     * Classify only the dimensions that prevent this request from fitting.
     */
    private Response classifyCapacityFailure(
            Request request,
            int priority,
            DecodeResources.ResourceSnapshot snapshot,
            AdmissionCapacity capacity) {
        var routing = snapshot.selection();
        var usage = routing.usage();
        if (usage.totalKvTokens() > 0
                && request.getExpectedKvTokens() > capacity.kvBudget(usage.totalKvTokens())) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        Capacity lower = Capacity.NONE;
        Capacity higher = Capacity.NONE;
        Capacity same = Capacity.NONE;
        for (var entry : snapshot.requests().values()) {
            int occupiedPriority = entry.priority();
            if (!PriorityNormalizer.hasPriority(occupiedPriority)) {
                continue;
            }
            Capacity occupied = entry.placementRelease();
            if (occupiedPriority < priority) {
                lower = lower.plus(occupied);
            } else if (occupiedPriority == priority) {
                same = same.plus(occupied);
            } else {
                higher = higher.plus(occupied);
            }
        }
        // Removing lower-priority occupancy is a counterfactual for attribution,
        // not an authorization to preempt its owners.
        var residual =
                capacity.evaluate(
                        usage, request.getHardKvTokens(), request.getExpectedKvTokens(), lower);
        if (residual.isZero()) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        return classifyProtectedCapacity(residual, higher, same);
    }

    private static Response classifyProtectedCapacity(
            DecodeResources.Capacity residual, Capacity higher, Capacity same) {
        Capacity attributed = higher.plus(same);
        if (residual.requests() > attributed.requests()
                || residual.hardKvTokens() > attributed.hardKvTokens()
                || residual.expectedKvTokens() > attributed.expectedKvTokens()) {
            return Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        }
        boolean higherBlocks =
                residual.requests() > 0 && higher.requests() > 0
                        || residual.hardKvTokens() > 0 && higher.hardKvTokens() > 0
                        || residual.expectedKvTokens() > 0 && higher.expectedKvTokens() > 0;
        return Response.error(
                StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                higherBlocks
                        ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                        : AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    public record AdmissionEvaluation(
            DecodeEndpoint.SelectionSnapshot snapshot,
            org.flexlb.worker.endpoint.DecodeResources.Capacity deficit,
            org.flexlb.worker.endpoint.Response failure) {}

    /**
     * Identifies one Decode registration by object identity within a worker generation.
     */
    public static final class RequestRegistration implements WorkerEndpoint.RequestRegistration {
        private final long endpointGenerationId;
        private final String requestId;

        public RequestRegistration(long endpointGenerationId, String requestId) {
            checkArgument(endpointGenerationId > 0L, "Decode generation must be positive");
            this.endpointGenerationId = endpointGenerationId;
            this.requestId = java.util.Objects.requireNonNull(requestId);
        }

        public long endpointGenerationId() {
            return endpointGenerationId;
        }

        public String requestId() {
            return requestId;
        }
    }

    /**
     * Freezes Decode topology, worker observations and local capacity at one admission version.
     */
    public record SelectionSnapshot(
            String address,
            long generationId,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.CommittedWorkerStatus workerStatus,
            long admissionVersion,
            CapacityUsage usage) {
        public SelectionSnapshot {
            java.util.Objects.requireNonNull(address, "address");
            java.util.Objects.requireNonNull(topology, "topology");
            java.util.Objects.requireNonNull(workerStatus, "workerStatus");
            checkArgument(generationId > 0L, "Decode routing view requires a positive generation");
        }

        public long realKvUsed() {
            return usage.expectedKvUsed();
        }

        public long totalKv() {
            return usage.totalKvTokens();
        }
    }

    /**
     * Tracks an exact Decode registration, KV estimates and the latest worker observation under the
     * endpoint lock.
     */
    private static final class RequestRecord {
        private final String requestId;
        private final RequestRegistration registration;
        private boolean confirmed;
        private final int priority;
        private long kvTokens;
        private long expectedKvTokens;
        private long observedAtMs = System.currentTimeMillis();

        RequestRecord(long generationId, String requestId, long hard, long expected, int priority) {
            this.requestId = requestId;
            this.registration = new RequestRegistration(generationId, requestId);
            this.kvTokens = hard;
            this.expectedKvTokens = expected;
            this.priority = priority;
        }

        RequestRegistration reservation() {
            return registration;
        }

        boolean matches(WorkerEndpoint.RequestRegistration exact) {
            return exact == registration;
        }
    }
}

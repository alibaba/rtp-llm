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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.worker.endpoint.DecodeResources.Capacity.saturatedAddNonNegative;

/**
 * Reserves Decode capacity, tracks priority resource waits and reconciles local requests with worker reports.
 * Confirmed cleanup removes the exact reservation and wakes scheduling outside the state lock.
 */
public class DecodeEndpoint extends WorkerEndpoint
        implements EndpointStats.Decode, SelectionRollback {
    // ResourceWait retains this identity, not the Endpoint and its request ledger.
    private final Object resourceWaitEndpointId = new Object();
    private final java.util.Set<ResourceWait> resourceWaits =
            java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
    private final CapacityNotifier capacityNotifier;
    private final Map<String, RequestRecord> registeredRequests = new ConcurrentHashMap<>();

    // Both request and KV reservations remain local until Decode confirms the request.
    private int unconfirmedRequests;
    private long unconfirmedKvTokens;
    private long unconfirmedExpectedKvTokens;
    private volatile int reportedRequestCount;
    private volatile long admissionVersion;

    private volatile DecodeEndpoint.SelectionSnapshot selectionSnapshotCache;

    DecodeEndpoint(WorkerStatus status, CapacityNotifier capacityNotifier) {
        super(status);
        this.capacityNotifier =
                java.util.Objects.requireNonNull(capacityNotifier, "capacityNotifier");
    }

    public final boolean registerResourceWait(ResourceWait wait) {
        stateLock().lock();
        try {
            return isResourceWaitEffective(wait, System.currentTimeMillis())
                    && resourceWaits.add(wait);
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Checks the wait deadline and revocation together with this endpoint's current availability.
     */
    public final boolean isResourceWaitEffective(ResourceWait wait, long nowMs) {
        checkArgument(
                wait.endpointId == resourceWaitEndpointId,
                "resource wait belongs to another endpoint");
        return !wait.revoked
                && !wait.result.isDone()
                && nowMs < wait.deadlineMs
                && acceptsNewRequests()
                && getStatus().topologySnapshot() == wait.topology;
    }

    public final void removeResourceWait(ResourceWait wait) {
        wait.revoke();
        stateLock().lock();
        try {
            resourceWaits.remove(wait);
        } finally {
            stateLock().unlock();
        }
        completeResourceWait(wait, null);
    }

    public final void tryAcquire(ResourceWait wait) {
        RequestRegistration registration = null;
        boolean ended;
        stateLock().lock();
        try {
            ended =
                    !resourceWaits.contains(wait)
                            || !isResourceWaitEffective(wait, System.currentTimeMillis());
            if (!ended)
                registration =
                        tryRegisterSelection(
                                        wait.request.getRequestId(),
                                        selectionSnapshot(),
                                        wait.hardKv,
                                        wait.expectedKv,
                                        wait.priority,
                                        wait.capacity)
                                .value();
            if (ended || registration != null) resourceWaits.remove(wait);
        } finally {
            stateLock().unlock();
        }
        if (ended || registration != null) completeResourceWait(wait, registration);
    }

    /**
     * Register against the exact view used for ranking. No remote capacity is promised.
     */
    public WorkerEndpoint.Registration<RequestRegistration> tryRegisterSelection(
            String requestId,
            DecodeEndpoint.SelectionSnapshot expected,
            long hardKv,
            long expectedKv,
            int priority,
            AdmissionCapacity capacity) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests() || expected.generationId() != getStatus().getGenerationId())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.UNAVAILABLE);
            var current = selectionSnapshot();
            if (!priorityAllowedLocked(priority))
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            if (!capacity.evaluate(current.usage(), hardKv, expectedKv, Capacity.NONE).isZero())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            if (current.admissionVersion() != expected.admissionVersion()
                    || current.workerStatus() != expected.workerStatus()
                    || current.topology() != expected.topology())
                return WorkerEndpoint.Registration.rejected(RegistrationStatus.VERSION_CONFLICT);
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
     * Confirms the exact local reservation and its feedback receiver without changing worker occupancy.
     */
    public CommitStatus confirmSelection(RequestRegistration registration, WorkerEventSink events) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return CommitStatus.UNAVAILABLE;
            var record = registeredRequests.get(registration.requestId());
            if (record == null || !record.matches(registration) || record.events != null)
                return CommitStatus.INVALID_REGISTRATION;
            record.events = java.util.Objects.requireNonNull(events, "events");
            return CommitStatus.COMMITTED;
        } finally {
            stateLock().unlock();
        }
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
            publishWorkerRequest(
                    reservation,
                    observed.events(),
                    RoleType.DECODE,
                    workerEventType(observed),
                    observed.errorCode(),
                    "");
        }
    }

    /**
     * Removes registrations no longer retained by the scheduler after their last observation exceeds TTL.
     * Returns the number of removed records; confirmed Worker capacity remains governed by Worker reports.
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
        int removedRequests = 0;
        stateLock().lock();
        try {
            long nowMs = System.currentTimeMillis();
            for (var request : expiredRequests) {
                if (registeredRequests.get(request.requestId) == request
                        && nowMs - request.observedAtMs > ttlMs) {
                    removeRequestLocked(request);
                    removedRequests++;
                }
            }
            if (removedRequests > 0) {
                admissionVersion++;
            }
        } finally {
            stateLock().unlock();
        }
        if (removedRequests > 0) capacityNotifier.notifyChanged(this);
        return removedRequests;
    }

    public ResourceWait createResourceWait(
            WorkerRequest request,
            long selectionId,
            int priority,
            long deadlineMs,
            long hardKv,
            long expectedKv,
            AdmissionCapacity capacity) {
        return new ResourceWait(
                resourceWaitEndpointId,
                getStatus().topologySnapshot(),
                request,
                selectionId,
                priority,
                deadlineMs,
                hardKv,
                expectedKv,
                capacity);
    }

    /**
     * Removes an exact selection registration only while the worker has not confirmed execution.
     */
    @Override
    public boolean rollbackSelection(WorkerEndpoint.RequestRegistration exact) {
        var result = removeRegistration(exact, true);
        checkState(
                result != ReleaseResult.RETAINED,
                "route rollback retained Decode resources: worker already confirmed the request");
        return result == ReleaseResult.REMOVED;
    }

    /**
     * Removes exact local bookkeeping after remote cleanup, a known unsent request, or inactivity expiry.
     * Sending a cancellation alone is not sufficient; reported Worker occupancy is retained.
     */
    public boolean releaseRequest(RequestRegistration exact) {
        var result = removeRegistration(exact, false);
        if (result == ReleaseResult.REMOVED) capacityNotifier.notifyChanged(this);
        return result == ReleaseResult.REMOVED;
    }

    /**
     * Removes only an unconfirmed exact registration. Absence completes cleanup; a confirmed
     * registration returns false and remains owned by the remote execution lifecycle.
     */
    public boolean releaseUnconfirmedRequest(RequestRegistration exact) {
        var result = removeRegistration(exact, true);
        if (result == ReleaseResult.REMOVED) capacityNotifier.notifyChanged(this);
        return result != ReleaseResult.RETAINED;
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
            long hardKvTokens, long expectedKvTokens, int priority, AdmissionCapacity capacity) {
        stateLock().lock();
        try {
            var snapshot = selectionSnapshot();
            var deficit =
                    capacity.evaluate(
                            snapshot.usage(), hardKvTokens, expectedKvTokens, Capacity.NONE);
            return new AdmissionEvaluation(
                    snapshot,
                    deficit,
                    deficit.isZero()
                            ? CapacityFailure.NONE
                            : classifyCapacityFailure(
                                    hardKvTokens, expectedKvTokens, priority, snapshot, capacity));
        } finally {
            stateLock().unlock();
        }
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
                        DecodeRequestStatus.terminal(
                                request.reservation(), request.events, completed.errorCode()));
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
                events.add(DecodeRequestStatus.confirmed(request.reservation(), request.events));
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
        closeResourceWaits();
        var removedRequests = removeRequestsOnClose();
        // Records are already cleared; notification failures must not block registry cleanup.
        for (var request : removedRequests) {
            try {
                notifyRequestOfEndpointClose(request);
            } catch (RuntimeException notificationFailure) {
                Logger.error(
                        "Decode endpoint close notification failed for request "
                                + request.requestId,
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

    private boolean priorityAllowedLocked(int priority) {
        long now = System.currentTimeMillis();
        for (var wait : resourceWaits) {
            if (wait.priority > priority && isResourceWaitEffective(wait, now)) return false;
        }
        return true;
    }

    private void closeResourceWaits() {
        java.util.List<ResourceWait> waits;
        stateLock().lock();
        try {
            waits = java.util.List.copyOf(resourceWaits);
            resourceWaits.clear();
        } finally {
            stateLock().unlock();
        }
        for (var wait : waits) completeResourceWait(wait, null);
    }

    /**
     * Completes outside the endpoint lock and returns an undeliverable reservation to this ledger.
     */
    private void completeResourceWait(ResourceWait wait, RequestRegistration registration) {
        if (!wait.result.complete(registration) && registration != null)
            rollbackSelection(registration);
    }

    private ReleaseResult removeRegistration(
            WorkerEndpoint.RequestRegistration exact, boolean requireUnconfirmed) {
        if (exact == null || exact.endpointGenerationId() != getStatus().getGenerationId())
            return ReleaseResult.ABSENT;
        stateLock().lock();
        try {
            var request = registeredRequests.get(exact.requestId());
            if (request == null || !request.matches(exact)) return ReleaseResult.ABSENT;
            if (requireUnconfirmed && request.confirmed) return ReleaseResult.RETAINED;
            removeRequestLocked(request);
            admissionVersion++;
            return ReleaseResult.REMOVED;
        } finally {
            stateLock().unlock();
        }
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

    private void notifyRequestOfEndpointClose(RequestRecord request) {
        var reservation = request.reservation();
        publishWorkerRequest(
                reservation,
                request.events,
                RoleType.DECODE,
                RequestEventType.ENDPOINT_CLOSED,
                0L,
                "Decode generation closed: " + reservation.endpointGenerationId());
    }

    private List<RequestRecord> removeRequestsOnClose() {
        List<RequestRecord> removed;
        stateLock().lock();
        try {
            removed = List.copyOf(registeredRequests.values());
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
    private CapacityFailure classifyCapacityFailure(
            long hardKvTokens,
            long expectedKvTokens,
            int priority,
            SelectionSnapshot snapshot,
            AdmissionCapacity capacity) {
        var usage = snapshot.usage();
        if (usage.totalKvTokens() > 0
                && expectedKvTokens > capacity.kvBudget(usage.totalKvTokens())) {
            return CapacityFailure.INSUFFICIENT_CAPACITY;
        }
        Capacity lower = Capacity.NONE;
        Capacity higher = Capacity.NONE;
        Capacity same = Capacity.NONE;
        for (var entry : registeredRequests.values()) {
            int occupiedPriority = entry.priority;
            if (!PriorityNormalizer.hasPriority(occupiedPriority)) {
                continue;
            }
            Capacity occupied = new Capacity(1L, entry.kvTokens, entry.expectedKvTokens);
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
        var residual = capacity.evaluate(usage, hardKvTokens, expectedKvTokens, lower);
        if (residual.isZero()) {
            return CapacityFailure.INSUFFICIENT_CAPACITY;
        }
        return classifyProtectedCapacity(residual, higher, same);
    }

    private static CapacityFailure classifyProtectedCapacity(
            DecodeResources.Capacity residual, Capacity higher, Capacity same) {
        Capacity attributed = higher.plus(same);
        if (residual.requests() > attributed.requests()
                || residual.hardKvTokens() > attributed.hardKvTokens()
                || residual.expectedKvTokens() > attributed.expectedKvTokens()) {
            return CapacityFailure.UNKNOWN_BLOCKER;
        }
        boolean higherBlocks =
                residual.requests() > 0 && higher.requests() > 0
                        || residual.hardKvTokens() > 0 && higher.hardKvTokens() > 0
                        || residual.expectedKvTokens() > 0 && higher.expectedKvTokens() > 0;
        return higherBlocks
                ? CapacityFailure.BLOCKED_BY_HIGHER_PRIORITY
                : CapacityFailure.BLOCKED_BY_SAME_PRIORITY;
    }

    public record AdmissionEvaluation(
            DecodeEndpoint.SelectionSnapshot snapshot,
            org.flexlb.worker.endpoint.DecodeResources.Capacity deficit,
            CapacityFailure failure) {}

    public enum CapacityFailure {
        NONE,
        INSUFFICIENT_CAPACITY,
        BLOCKED_BY_HIGHER_PRIORITY,
        BLOCKED_BY_SAME_PRIORITY,
        UNKNOWN_BLOCKER
    }

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
    /**
     * Owns Decode capacity and the receiver bound at selection confirmation. Status notifications
     * capture the receiver under the endpoint lock; removed records are no longer modified.
     */
    private static final class RequestRecord {
        private final String requestId;
        private final RequestRegistration registration;
        private boolean confirmed;
        private WorkerEventSink events;
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

        private RequestRegistration reservation() {
            return registration;
        }

        private boolean matches(WorkerEndpoint.RequestRegistration exact) {
            return exact == registration;
        }
    }

    /**
     * Holds one selection's capacity demand, deadline, revocation and result.
     * DecodeEndpoint checks endpoint availability and completes or rolls back the reservation.
     */
    public static final class ResourceWait {
        private volatile boolean revoked;
        private final Object endpointId;
        private final WorkerRequest request;
        private final long selectionId;
        private final WorkerStatus.TopologySnapshot topology;
        private final int priority;
        private final long deadlineMs;
        private final long hardKv;
        private final long expectedKv;
        private final AdmissionCapacity capacity;
        private final CompletableFuture<RequestRegistration> result = new CompletableFuture<>();

        private ResourceWait(
                Object endpointId,
                WorkerStatus.TopologySnapshot topology,
                WorkerRequest request,
                long selectionId,
                int priority,
                long deadlineMs,
                long hardKv,
                long expectedKv,
                AdmissionCapacity capacity) {
            this.endpointId = endpointId;
            this.request = request;
            this.selectionId = selectionId;
            this.topology = topology;
            this.priority = priority;
            this.deadlineMs = deadlineMs;
            this.hardKv = hardKv;
            this.expectedKv = expectedKv;
            this.capacity = capacity;
        }

        public final WorkerRequest request() {
            return request;
        }

        public final long selectionId() {
            return selectionId;
        }

        public final java.util.concurrent.CompletionStage<RequestRegistration> result() {
            return result.minimalCompletionStage();
        }

        public final boolean isDone() {
            return result.isDone();
        }

        /**
         * Removes priority protection immediately; the preemption owner drains the result separately.
         */
        public void revoke() {
            revoked = true;
        }
    }

    private enum ReleaseResult {
        ABSENT,
        RETAINED,
        REMOVED
    }
}

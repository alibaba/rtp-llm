package org.flexlb.worker.endpoint;

import org.flexlb.util.Failures;
import org.flexlb.worker.status.WorkerStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tracks Encoder registrations until worker status observes them, then until completion or cancellation.
 * Its state lock serializes request admission, status changes and removal.
 * Orphan records expire from their last admission or worker observation once the scheduler no
 * longer retains them.
 */
public final class EncoderEndpoint extends WorkerEndpoint implements SelectionRollback {
    private Map<String, Tracked> requests = new HashMap<>();

    private final CapacityNotifier capacityNotifier;

    public EncoderEndpoint(WorkerStatus status, CapacityNotifier capacityNotifier) {
        super(status);
        this.capacityNotifier = capacityNotifier;
    }

    public SelectionSnapshot selectionSnapshot() {
        return selectionSnapshot(true);
    }

    /**
     * Request-count selection does not need to scan token work when the request has no cache estimate.
     */
    public SelectionSnapshot selectionSnapshot(boolean includeTokenWork) {
        return captureSelectionSnapshot(includeTokenWork);
    }

    public Registration<RequestRegistration> tryRegisterSelection(
            String requestId,
            SelectionSnapshot expected,
            boolean includeTokenWork,
            String group,
            int requestLimit,
            long uncachedTokens) {
        java.util.Objects.requireNonNull(requestId, "requestId");
        com.google.common.base.Preconditions.checkArgument(
                uncachedTokens >= 0L, "uncached Encoder tokens must be nonnegative");
        stateLock().lock();
        try {
            if (expected.endpoint() != this || !acceptsNewRequests())
                return Registration.rejected(RegistrationStatus.UNAVAILABLE);
            // The outer lock keeps the captured ranking inputs current through registration.
            var current = captureSelectionSnapshot(includeTokenWork);
            if (group != null && !group.equals(current.group()))
                return Registration.rejected(RegistrationStatus.UNAVAILABLE);
            if (requests.containsKey(requestId))
                return Registration.rejected(RegistrationStatus.ALREADY_REGISTERED);
            if (current.requestCount() >= requestLimit)
                return Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            if (current.requestCount() != expected.requestCount()
                    || includeTokenWork && current.uncachedTokens() != expected.uncachedTokens())
                return Registration.rejected(RegistrationStatus.VERSION_CONFLICT);
            var registration = new RequestRegistration(this, requestId, uncachedTokens);
            requests.put(
                    requestId, new Tracked(registration, null, false, System.currentTimeMillis()));
            return new Registration<>(RegistrationStatus.REGISTERED, registration);
        } finally {
            stateLock().unlock();
        }
    }

    public CommitStatus confirmSelection(RequestRegistration registration, WorkerEventSink events) {
        stateLock().lock();
        try {
            var entry = requests.get(registration.requestId());
            if (!acceptsNewRequests()) return CommitStatus.UNAVAILABLE;
            if (entry == null
                    || entry.registration() != registration
                    || entry.phase() != RegistrationPhase.REGISTERED)
                return CommitStatus.INVALID_REGISTRATION;
            requests.put(
                    registration.requestId(),
                    new Tracked(
                            registration,
                            java.util.Objects.requireNonNull(events, "events"),
                            entry.observed(),
                            entry.lastObservedAtMs()));
            return CommitStatus.COMMITTED;
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public int evictExpiredRequests(
            long ttlMs, java.util.function.Predicate<String> retainForSchedulerCleanup) {
        Map<String, Tracked> candidates;
        stateLock().lock();
        try {
            candidates = new HashMap<>(requests);
        } finally {
            stateLock().unlock();
        }
        candidates.keySet().removeIf(retainForSchedulerCleanup::test);
        int removed = 0;
        stateLock().lock();
        try {
            long now = System.currentTimeMillis();
            for (var entry : candidates.entrySet()) {
                var captured = entry.getValue();
                if (requests.get(entry.getKey()) == captured
                        && now - captured.lastObservedAtMs() > ttlMs) {
                    requests.remove(entry.getKey());
                    removed++;
                }
            }
        } finally {
            stateLock().unlock();
        }
        if (removed > 0) capacityNotifier.notifyChanged(this);
        return removed;
    }

    @Override
    public boolean rollbackSelection(WorkerEndpoint.RequestRegistration registration) {
        return releaseRequest(registration, true);
    }

    public boolean releaseRequest(RequestRegistration registration) {
        return releaseRequest(registration, false);
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) return;
        var observed = (StatusChange.Encoder) change;
        Throwable failure = null;
        for (var request : observed.requests())
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    publishWorkerRequest(
                                            request.registration(),
                                            request.events(),
                                            org.flexlb.config.service.RoleType.ENCODER,
                                            request.kind(),
                                            request.errorCode(),
                                            ""));
        failure = Failures.run(failure, () -> capacityNotifier.notifyChanged(this));
        Failures.rethrow(failure, "Encoder status publication failed");
    }

    @Override
    protected StatusChange updateStatusLocked(WorkerStatus.StatusObservation observation) {
        List<RequestStatus> changes = new ArrayList<>();
        var members = requests.entrySet().iterator();
        while (members.hasNext()) {
            var entry = members.next();
            var tracked = entry.getValue();
            if (observation.activeTasks().containsKey(entry.getKey())) {
                entry.setValue(
                        new Tracked(
                                tracked.registration(),
                                tracked.events(),
                                true,
                                System.currentTimeMillis()));
                if (tracked.phase() == RegistrationPhase.COMMITTED)
                    changes.add(
                            new RequestStatus(
                                    tracked.registration(),
                                    tracked.events(),
                                    RequestEventType.ACTIVE,
                                    0));
            }
            var task = observation.finishedTasks().get(entry.getKey());
            if (task != null) {
                members.remove();
                if (tracked.phase() == RegistrationPhase.COMMITTED)
                    changes.add(
                            new RequestStatus(
                                    tracked.registration(),
                                    tracked.events(),
                                    task.errorCode() == 0
                                            ? RequestEventType.COMPLETED
                                            : RequestEventType.FAILED,
                                    task.errorCode()));
            }
        }
        return new StatusChange.Encoder(changes);
    }

    @Override
    protected void closeEndpoint() {
        Map<String, Tracked> retired;
        stateLock().lock();
        try {
            var result = requests;
            requests = new HashMap<>();
            retired = result;
        } finally {
            stateLock().unlock();
        }
        Throwable failure = null;
        for (var request : retired.values()) {
            if (request.phase() != RegistrationPhase.COMMITTED) continue;
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    publishWorkerRequest(
                                            request.registration(),
                                            request.events(),
                                            getStatus().getRole(),
                                            RequestEventType.ENDPOINT_CLOSED,
                                            0L,
                                            "Encoder endpoint retired: " + ipPort()));
        }
        failure = Failures.run(failure, () -> capacityNotifier.notifyChanged(this));
        Failures.rethrow(failure, "Encoder retirement publication failed");
    }

    private boolean releaseRequest(
            WorkerEndpoint.RequestRegistration registration, boolean requireUncommitted) {
        boolean removed = false;
        stateLock().lock();
        try {
            var current = requests.get(registration.requestId());
            if (current != null
                    && current.registration() == registration
                    && (!requireUncommitted || current.phase() == RegistrationPhase.REGISTERED)) {
                requests.remove(registration.requestId());
                removed = true;
            }
        } finally {
            stateLock().unlock();
        }
        if (removed && !requireUncommitted) capacityNotifier.notifyChanged(this);
        return removed;
    }

    private SelectionSnapshot captureSelectionSnapshot(boolean includeWork) {
        WorkerStatus.EngineObservation engine;
        long pending = 0L;
        long work = 0L;
        boolean available;
        String group;
        stateLock().lock();
        try {
            engine = getStatus().committedEngineObservation();
            available = acceptsNewRequests() && getStatus().isAlive();
            group = getStatus().getGroup();
            for (var tracked : requests.values()) {
                if (!tracked.observed()) {
                    pending++;
                    if (includeWork
                            && !engine.runningTaskList()
                                    .containsKey(tracked.registration().requestId()))
                        work =
                                com.google.common.math.LongMath.saturatedAdd(
                                        work, tracked.uncachedTokens());
                }
            }
        } finally {
            stateLock().unlock();
        }
        long count =
                com.google.common.math.LongMath.saturatedAdd(
                        com.google.common.math.LongMath.saturatedAdd(
                                Math.max(0L, engine.runningQueryLen()),
                                Math.max(0L, engine.waitingQueryLen())),
                        pending);
        if (includeWork)
            for (var task : engine.runningTaskList().values())
                work =
                        com.google.common.math.LongMath.saturatedAdd(
                                work, Math.max(0L, task.inputLength()));
        return new SelectionSnapshot(this, available, group, count, work);
    }

    /**
     * Captures request count and requested token work against one engine observation.
     * Token work is zero when the request has no cache estimate and ranking only uses request count.
     */
    public record SelectionSnapshot(
            EncoderEndpoint endpoint,
            boolean available,
            String group,
            long requestCount,
            long uncachedTokens) {
        public String address() {
            return endpoint.ipPort();
        }
    }

    /**
     * Identifies one Encoder selection and its uncached token work within the owning endpoint.
     */
    public record RequestRegistration(
            EncoderEndpoint endpoint, String requestId, long uncachedTokens)
            implements WorkerEndpoint.RequestRegistration {
        @Override
        public long endpointGenerationId() {
            return endpoint.getStatus().getGenerationId();
        }
    }

    /**
     * Retains an Encoder registration, its confirmed event receiver and the last worker observation.
     */
    private record Tracked(
            RequestRegistration registration,
            WorkerEventSink events,
            boolean observed,
            long lastObservedAtMs) {
        long uncachedTokens() {
            return registration.uncachedTokens();
        }

        RegistrationPhase phase() {
            return events == null ? RegistrationPhase.REGISTERED : RegistrationPhase.COMMITTED;
        }
    }

    public record RequestStatus(
            RequestRegistration registration,
            WorkerEventSink events,
            RequestEventType kind,
            long errorCode) {}
}

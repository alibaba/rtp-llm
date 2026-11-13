package org.flexlb.worker.endpoint;

import org.flexlb.util.Failures;
import org.flexlb.worker.status.WorkerStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tracks Encoder routes until worker status observes them, then until completion or cancellation.
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
        return captureSelectionSnapshot(true);
    }

    /**
     * Request-count selection does not need to scan token work when the request has no cache estimate.
     */
    public SelectionSnapshot selectionSnapshot(org.flexlb.worker.endpoint.Request request) {
        return captureSelectionSnapshot(request.getEncoderCacheHitLen() != null);
    }

    public Registration<RequestRegistration> tryRegisterSelection(
            String requestId, int requestLimit, long uncachedTokens) {
        java.util.Objects.requireNonNull(requestId, "requestId");
        com.google.common.base.Preconditions.checkArgument(
                uncachedTokens >= 0L, "uncached Encoder tokens must be nonnegative");
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return Registration.rejected(RegistrationStatus.UNAVAILABLE);
            if (requests.containsKey(requestId))
                return Registration.rejected(RegistrationStatus.ALREADY_REGISTERED);
            if (requestCountLocked() >= requestLimit)
                return Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            var registration = new RequestRegistration(this, requestId, uncachedTokens);
            requests.put(
                    requestId, new Tracked(registration, null, false, System.currentTimeMillis()));
            return new Registration<>(RegistrationStatus.REGISTERED, registration);
        } finally {
            stateLock().unlock();
        }
    }

    public boolean commitSelection(WorkerRoute route) {
        stateLock().lock();
        try {
            var registration = route.encoderRegistration();
            var entry = requests.get(route.requestId());
            if (!acceptsNewRequests()
                    || entry == null
                    || entry.registration() != registration
                    || entry.phase() != RegistrationPhase.REGISTERED) return false;
            requests.put(
                    route.requestId(),
                    new Tracked(registration, route, entry.observed(), entry.lastObservedAtMs()));
            return true;
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

    public void releaseRequest(WorkerRoute route) {
        var registration = route.encoderRegistration();
        if (registration != null) releaseRequest(registration);
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
                                            request.route(),
                                            null,
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
                                tracked.route(),
                                true,
                                System.currentTimeMillis()));
                if (tracked.phase() == RegistrationPhase.COMMITTED)
                    changes.add(
                            new RequestStatus(
                                    tracked.route(),
                                    org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                    0));
            }
            var task = observation.finishedTasks().get(entry.getKey());
            if (task != null) {
                members.remove();
                if (tracked.phase() == RegistrationPhase.COMMITTED)
                    changes.add(
                            new RequestStatus(
                                    tracked.route(),
                                    task.errorCode() == 0
                                            ? org.flexlb.worker.endpoint.RequestEventType.COMPLETED
                                            : org.flexlb.worker.endpoint.RequestEventType.FAILED,
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
            var route = request.route();
            if (request.phase() != RegistrationPhase.COMMITTED) continue;
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    route.ctx()
                                            .getRequestScheduler()
                                            .onWorkerRequestEvent(
                                                    route.ctx(),
                                                    new org.flexlb.worker.endpoint
                                                            .WorkerRequestEvent(
                                                            route,
                                                            null,
                                                            org.flexlb.config.service.RoleType
                                                                    .ENCODER,
                                                            this,
                                                            org.flexlb.worker.endpoint
                                                                    .RequestEventType
                                                                    .ENDPOINT_CLOSED,
                                                            0L,
                                                            "Encoder endpoint retired: " + ipPort(),
                                                            System.currentTimeMillis())));
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
        if (removed) capacityNotifier.notifyChanged(this);
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

    private long requestCountLocked() {
        var engine = getStatus().committedEngineObservation();
        long pending = 0;
        for (var request : requests.values()) if (!request.observed()) pending++;
        return Math.max(0, engine.runningQueryLen())
                + Math.max(0, engine.waitingQueryLen())
                + pending;
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
     * Retains an Encoder registration, its optional committed route and the last worker observation.
     */
    private record Tracked(
            RequestRegistration registration,
            WorkerRoute route,
            boolean observed,
            long lastObservedAtMs) {
        long uncachedTokens() {
            return registration.uncachedTokens();
        }

        RegistrationPhase phase() {
            return route == null ? RegistrationPhase.REGISTERED : RegistrationPhase.COMMITTED;
        }
    }

    public record RequestStatus(
            WorkerRoute route, org.flexlb.worker.endpoint.RequestEventType kind, long errorCode) {}
}

package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.strategy.EncoderSelectionStrategy;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tracks Encoder routes until worker status observes them, then until completion or cancellation.
 * The worker generation lock protects both admission and status changes.
 */
public final class EncoderEndpoint extends WorkerEndpoint
        implements EncoderSelectionStrategy.Candidate {
    private record Tracked(RequestRoute route, boolean observed, long uncachedTokens) {}

    public record RequestStatus(
            RequestRoute route, ContextResult.EngineKind kind, long errorCode) {}

    private final Map<String, Tracked> requests = new HashMap<>();
    private final PlacementAvailability availability;

    public EncoderEndpoint(WorkerStatus status, PlacementAvailability availability) {
        super(status);
        this.availability = availability;
    }

    @Override
    public String address() {
        return ipPort();
    }

    @Override
    public String group() {
        return getStatus().topologySnapshot().group();
    }

    @Override
    public boolean available() {
        return !isRetired() && getStatus().isAvailableForScheduling() && getStatus().isAlive();
    }

    @Override
    public long requestCount() {
        var lock = getStatus().lock;
        lock.lock();
        try {
            var engine = getStatus().committedEngineObservation();
            return Math.max(0, engine.runningQueryLen())
                    + Math.max(0, engine.waitingQueryLen())
                    + requests.values().stream().filter(request -> !request.observed()).count();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public long uncachedTokens() {
        var lock = getStatus().lock;
        lock.lock();
        try {
            var active = getStatus().committedEngineObservation().runningTaskList();
            long total = 0;
            for (var task : active.values())
                total =
                        com.google.common.math.LongMath.saturatedAdd(
                                total, Math.max(0, task.inputLength()));
            for (var request : requests.values()) {
                if (!request.observed() && !active.containsKey(request.route().requestId()))
                    total =
                            com.google.common.math.LongMath.saturatedAdd(
                                    total, request.uncachedTokens());
            }
            return total;
        } finally {
            lock.unlock();
        }
    }

    public boolean commit(GenerationPin pin, RequestRoute route, int requestLimit) {
        requirePinnedGeneration(pin);
        var lock = getStatus().lock;
        lock.lock();
        try {
            if (requests.containsKey(route.requestId()) || requestCount() >= requestLimit)
                return false;
            var request = route.ctx().data().getRequest();
            long hit =
                    request.getEncoderCacheHitLen() == null
                            ? 0
                            : Math.max(0, request.getEncoderCacheHitLen());
            requests.put(
                    route.requestId(),
                    new Tracked(route, false, Math.max(0, request.getSeqLen() - hit)));
            return true;
        } finally {
            lock.unlock();
        }
    }

    public void release(RequestRoute route) {
        boolean removed = false;
        var lock = getStatus().lock;
        lock.lock();
        try {
            var current = requests.get(route.requestId());
            if (current != null && current.route() == route) {
                requests.remove(route.requestId());
                removed = true;
            }
        } finally {
            lock.unlock();
        }
        if (removed) availability.changed(this);
    }

    @Override
    public StatusChange applyPreparedStatus(
            WorkerStatus status, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(status);
        if (!prepared.observation().alive()) beginRetirement();
        status.publishPreparedStatus(prepared);
        return collectStatus(prepared.observation(), true);
    }

    @Override
    public StatusChange applyStatusHeartbeat(
            WorkerStatus status, WorkerStatus.StatusObservation observation) {
        super.applyStatusHeartbeat(status, observation);
        return collectStatus(observation, false);
    }

    private StatusChange collectStatus(
            WorkerStatus.StatusObservation observation, boolean includeFinished) {
        if (!getStatus().lock.isHeldByCurrentThread())
            throw new IllegalStateException("worker lock required for status updates");
        List<RequestStatus> changes = new ArrayList<>();
        for (var task : observation.activeTasks().values()) {
            var tracked = requests.get(task.requestId());
            if (tracked == null) continue;
            requests.put(
                    task.requestId(), new Tracked(tracked.route(), true, tracked.uncachedTokens()));
            changes.add(new RequestStatus(tracked.route(), ContextResult.EngineKind.ACTIVE, 0));
        }
        if (includeFinished)
            for (var task : observation.finishedTasks().values()) {
                var tracked = requests.remove(task.requestId());
                if (tracked != null)
                    changes.add(
                            new RequestStatus(
                                    tracked.route(),
                                    task.errorCode() == 0
                                            ? ContextResult.EngineKind.COMPLETED
                                            : ContextResult.EngineKind.FAILED,
                                    task.errorCode()));
            }
        return new StatusChange.Encoder(changes);
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) return;
        var observed = (StatusChange.Encoder) change;
        Throwable failure = null;
        for (var request : observed.requests())
            failure = Failures.run(failure, () -> publish(request));
        failure = Failures.run(failure, () -> availability.changed(this));
        Failures.rethrow(failure, "Encoder status publication failed");
    }

    private void publish(RequestStatus status) {
        var route = status.route();
        route.ctx()
                .data()
                .getLifecycle()
                .onEngineEvidence(
                        route.ctx(),
                        new ContextResult.EngineEvidence(
                                route,
                                null,
                                org.flexlb.dao.route.RoleType.ENCODER,
                                status.kind(),
                                status.errorCode(),
                                "",
                                System.currentTimeMillis()));
    }

    @Override
    protected void closeEndpoint() {
        List<RequestStatus> retired;
        var lock = getStatus().lock;
        lock.lock();
        try {
            retired =
                    requests.values().stream()
                            .map(
                                    request ->
                                            new RequestStatus(
                                                    request.route(),
                                                    ContextResult.EngineKind.RETIRED,
                                                    0))
                            .toList();
            requests.clear();
        } finally {
            lock.unlock();
        }
        publishStatus(new StatusChange.Encoder(retired));
    }
}

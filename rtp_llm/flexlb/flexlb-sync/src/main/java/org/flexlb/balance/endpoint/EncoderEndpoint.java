package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.CapacityNotifier;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.strategy.EncoderSelectionStrategy;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tracks Encoder routes until worker status observes them, then until completion or cancellation.
 * Its state lock serializes request admission, status changes and removal.
 */
public final class EncoderEndpoint extends WorkerEndpoint
        implements EncoderSelectionStrategy.Candidate {
    private Map<String, Tracked> requests = new HashMap<>();

    private final CapacityNotifier capacityNotifier;

    public EncoderEndpoint(WorkerStatus status, CapacityNotifier capacityNotifier) {
        super(status);
        this.capacityNotifier = capacityNotifier;
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
        return acceptsNewRequests() && getStatus().isAlive();
    }

    @Override
    public long requestCount() {
        stateLock().lock();
        try {
            return requestCountLocked();
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public long uncachedTokens() {
        Map<String, WorkerStatus.TaskObservation> active;
        long localTokens;
        stateLock().lock();
        try {
            active = getStatus().committedEngineObservation().runningTaskList();
            localTokens = 0;
            for (var request : requests.values()) {
                if (!request.observed() && !active.containsKey(request.route().requestId()))
                    localTokens =
                            com.google.common.math.LongMath.saturatedAdd(
                                    localTokens, request.uncachedTokens());
            }
        } finally {
            stateLock().unlock();
        }
        long total = localTokens;
        for (var task : active.values())
            total =
                    com.google.common.math.LongMath.saturatedAdd(
                            total, Math.max(0, task.inputLength()));
        return total;
    }

    public boolean commit(RouteResult route, int requestLimit, long uncachedTokens) {
        com.google.common.base.Preconditions.checkArgument(
                uncachedTokens >= 0L, "uncached Encoder tokens must be nonnegative");
        var tracked = new Tracked(route, false, uncachedTokens);
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return false;
            if (requests.containsKey(route.requestId()) || requestCountLocked() >= requestLimit)
                return false;
            requests.put(route.requestId(), tracked);
            return true;
        } finally {
            stateLock().unlock();
        }
    }

    public void release(RouteResult route) {
        boolean removed = false;
        stateLock().lock();
        try {
            var current = requests.get(route.requestId());
            if (current != null && current.route() == route) {
                requests.remove(route.requestId());
                removed = true;
            }
        } finally {
            stateLock().unlock();
        }
        if (removed) capacityNotifier.notifyChanged(this);
    }

    @Override
    public StatusChange applyWorkerStatus(
            WorkerStatus status, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(status);
        List<RequestStatus> changes = new ArrayList<>();
        stateLock().lock();
        try {
            com.google.common.base.Preconditions.checkState(
                    !isRetired(), "Encoder endpoint is retired");
            if (!prepared.observation().alive()) beginRetirement();
            collectStatusLocked(prepared.observation(), changes);
            status.publishReducedStatus(prepared);
        } finally {
            stateLock().unlock();
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
        failure = Failures.run(failure, () -> capacityNotifier.notifyChanged(this));
        Failures.rethrow(failure, "Encoder status publication failed");
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
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    route.ctx()
                                            .data()
                                            .getRequestScheduler()
                                            .onEndpointRetired(
                                                    route.ctx(),
                                                    new ContextResult.EndpointRetirement(
                                                            route,
                                                            null,
                                                            org.flexlb.dao.route.RoleType.ENCODER,
                                                            this,
                                                            "Encoder endpoint retired: " + ipPort(),
                                                            System.currentTimeMillis())));
        }
        failure = Failures.run(failure, () -> capacityNotifier.notifyChanged(this));
        Failures.rethrow(failure, "Encoder retirement publication failed");
    }

    private long requestCountLocked() {
        var engine = getStatus().committedEngineObservation();
        long pending = 0;
        for (var request : requests.values()) if (!request.observed()) pending++;
        return Math.max(0, engine.runningQueryLen())
                + Math.max(0, engine.waitingQueryLen())
                + pending;
    }

    private void collectStatusLocked(
            WorkerStatus.StatusObservation observation, List<RequestStatus> changes) {
        // Only locally tracked requests need reconciliation; reports may include many other tasks.
        for (var entry : requests.entrySet()) {
            if (!observation.activeTasks().containsKey(entry.getKey())) continue;
            var tracked = entry.getValue();
            if (!tracked.observed())
                entry.setValue(new Tracked(tracked.route(), true, tracked.uncachedTokens()));
            changes.add(
                    new RequestStatus(
                            tracked.route(), ContextResult.WorkerRequestEventType.ACTIVE, 0));
        }
        var members = requests.entrySet().iterator();
        while (members.hasNext()) {
            var entry = members.next();
            var task = observation.finishedTasks().get(entry.getKey());
            if (task != null) {
                var tracked = entry.getValue();
                members.remove();
                changes.add(
                        new RequestStatus(
                                tracked.route(),
                                task.errorCode() == 0
                                        ? ContextResult.WorkerRequestEventType.COMPLETED
                                        : ContextResult.WorkerRequestEventType.FAILED,
                                task.errorCode()));
            }
        }
    }

    private void publish(RequestStatus status) {
        var route = status.route();
        route.ctx()
                .data()
                .getRequestScheduler()
                .onWorkerRequestEvent(
                        route.ctx(),
                        new ContextResult.WorkerRequestEvent(
                                route,
                                null,
                                org.flexlb.dao.route.RoleType.ENCODER,
                                status.kind(),
                                status.errorCode(),
                                "",
                                System.currentTimeMillis()));
    }

    private record Tracked(RouteResult route, boolean observed, long uncachedTokens) {}

    public record RequestStatus(
            RouteResult route, ContextResult.WorkerRequestEventType kind, long errorCode) {}
}

package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Predicate;

/**
 * Shared identity directory; request decisions remain with the exact context and its owner.
 */
@Component
public final class RequestRepository implements org.flexlb.worker.endpoint.WorkerRequests {
    private final Object registrationLock = new Object();
    private final EnumMap<RequestPhase, ConcurrentMap<String, RequestContext>> activeByPhase =
            new EnumMap<>(RequestPhase.class);
    private final EnumMap<RequestPhase, ConcurrentMap<String, ArchivedRequest>> terminalByPhase =
            new EnumMap<>(RequestPhase.class);

    private volatile boolean closed;
    private volatile boolean stopping;

    public RequestRepository() {
        for (RequestPhase phase : RequestPhase.values()) {
            activeByPhase.put(phase, new ConcurrentHashMap<>());
            terminalByPhase.put(phase, new ConcurrentHashMap<>());
        }
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedRate = 60_000L)
    public void maintainExpiration() {
        if (!closed) expireTerminalRecords(System.currentTimeMillis() - 10_000L);
    }

    public AbstractRequestScheduler ownerOf(String requestId) {
        return ownerOf(requestId, RequestPhase.GENERATION);
    }

    public AbstractRequestScheduler ownerOf(String requestId, RequestPhase phase) {
        RequestContext context = findActive(requestId, phase);
        if (context != null) {
            return context.getRequestScheduler();
        }
        var terminal = terminalByPhase.get(phase).get(requestId);
        return terminal == null ? null : terminal.owner();
    }

    public RequestContext findActive(String requestId) {
        return findActive(requestId, RequestPhase.GENERATION);
    }

    public RequestContext findActive(String requestId, RequestPhase phase) {
        return activeByPhase.get(phase).get(requestId);
    }

    public boolean retainsIdentity(String requestId) {
        return activeByPhase.values().stream().anyMatch(map -> map.containsKey(requestId))
                || terminalByPhase.values().stream().anyMatch(map -> map.containsKey(requestId));
    }

    public boolean isClosed() {
        return closed;
    }

    public long oldestLiveRequestAgeMs() {
        long oldest = Long.MAX_VALUE;
        for (RequestContext context : snapshotActive()) {
            oldest = Math.min(oldest, context.snapshot().createdAtMs());
        }
        return oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
    }

    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> snapshots = new ArrayList<>(liveRequestCount());
        for (RequestContext context :
                snapshotActive(
                        request -> request.snapshot().phase() != RequestContext.Phase.CLOSED)) {
            synchronized (context) {
                if (isCurrent(context)
                        && context.snapshot().phase() != RequestContext.Phase.CLOSED) {
                    snapshots.add(RequestState.from(context.snapshot()));
                }
            }
        }
        snapshots.sort(
                (left, right) -> {
                    int createdOrder = Long.compare(left.createdAtMs(), right.createdAtMs());
                    return createdOrder != 0
                            ? createdOrder
                            : left.requestId().compareTo(right.requestId());
                });
        return List.copyOf(snapshots);
    }

    public int liveRequestCount() {
        return activeByPhase.values().stream().mapToInt(Map::size).sum();
    }

    public RequestState getRequestState(String requestId) {
        return getRequestState(requestId, RequestPhase.GENERATION);
    }

    public RequestState getRequestState(String requestId, RequestPhase phase) {
        RequestContext context = findActive(requestId, phase);
        RequestState terminal = findTerminal(requestId, phase);
        RequestState state =
                context != null
                        ? RequestState.from(context.snapshot())
                        : terminal == null ? null : terminal;
        return state;
    }

    public boolean closeRegistration() {
        synchronized (registrationLock) {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        }
    }

    public void stopAccepting() {
        synchronized (registrationLock) {
            stopping = true;
        }
    }

    /**
     * Wait only for accepted response futures; resource settlement follows separately.
     */
    public boolean awaitResponses(long deadlineNanos) {
        if (!stopping) throw new IllegalStateException("stop accepting requests before draining");
        for (var context : snapshotActive(request -> !request.getFuture().isDone())) {
            var response = context.getFuture();
            if (response.isDone()) continue;
            long remaining =
                    deadlineNanos == Long.MAX_VALUE
                            ? Long.MAX_VALUE
                            : deadlineNanos - System.nanoTime();
            if (remaining <= 0L) return false;
            try {
                response.get(remaining, java.util.concurrent.TimeUnit.NANOSECONDS);
            } catch (java.util.concurrent.ExecutionException
                    | java.util.concurrent.CancellationException completed) {
                // A failed or cancelled response is finished; the scheduler still owns request
                // cleanup.
            } catch (java.util.concurrent.TimeoutException timeout) {
                return false;
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return true;
    }

    RequestState findTerminal(String requestId) {
        return findTerminal(requestId, RequestPhase.GENERATION);
    }

    RequestState findTerminal(String requestId, RequestPhase phase) {
        var archived = terminalByPhase.get(phase).get(requestId);
        return archived == null ? null : archived.state();
    }

    RequestState findTerminal(
            String requestId, RequestPhase phase, AbstractRequestScheduler owner) {
        var archived = terminalByPhase.get(phase).get(requestId);
        return archived != null && archived.owner() == owner ? archived.state() : null;
    }

    List<RequestContext> snapshotActive() {
        return snapshotActive(request -> true);
    }

    /**
     * Filters before collecting request references. Request state may change after filtering;
     * callers must recheck it before committing changes.
     */
    List<RequestContext> snapshotActive(Predicate<RequestContext> filter) {
        return activeByPhase.values().stream()
                .flatMap(map -> map.values().stream())
                .filter(filter)
                .toList();
    }

    int pendingRequestCount() {
        int pending = 0;
        for (var requests : activeByPhase.values()) {
            for (var request : requests.values()) {
                if (request.snapshot().phase() == RequestContext.Phase.OPEN) {
                    pending++;
                }
            }
        }
        return pending;
    }

    boolean isStopping() {
        return stopping;
    }

    boolean isCurrent(RequestContext exact) {
        return exact != null
                && activeByPhase.get(exact.getPhase()).get(exact.getRequestId()) == exact;
    }

    RegistrationResult register(
            RequestContext context,
            AbstractRequestScheduler scheduler,
            RequestFuture response,
            long inactivityTimeoutMs,
            StrategyErrorType timeoutError) {
        java.util.Objects.requireNonNull(context);
        java.util.Objects.requireNonNull(scheduler);
        java.util.Objects.requireNonNull(response);
        java.util.Objects.requireNonNull(timeoutError, "timeout error required");
        if (inactivityTimeoutMs <= 0L) {
            throw new IllegalArgumentException("positive request inactivity timeout required");
        }
        synchronized (registrationLock) {
            if (closed || stopping) {
                return RegistrationResult.CLOSED;
            }
            synchronized (context) {
                String id = context.getRequestId();
                var activeRequests = activeByPhase.get(context.getPhase());
                var terminalRecords = terminalByPhase.get(context.getPhase());
                if (activeRequests.containsKey(id) || terminalRecords.containsKey(id)) {
                    return RegistrationResult.DUPLICATE_ID;
                }
                if (context.getRequestScheduler() != null
                        || context.snapshot().requestId() != null) {
                    return RegistrationResult.CONTEXT_BOUND;
                }
                context.register(
                        scheduler,
                        response,
                        System.currentTimeMillis(),
                        inactivityTimeoutMs,
                        timeoutError);
                activeRequests.put(id, context);
            }
            return RegistrationResult.REGISTERED;
        }
    }

    void archive(RequestContext exact, RequestState terminal) {
        synchronized (registrationLock) {
            if (activeByPhase.get(exact.getPhase()).get(exact.getRequestId()) != exact) {
                return;
            }
            terminalByPhase
                    .get(exact.getPhase())
                    .put(
                            exact.getRequestId(),
                            new ArchivedRequest(terminal, exact.getRequestScheduler()));
            activeByPhase.get(exact.getPhase()).remove(exact.getRequestId(), exact);
            exact.releaseSchedulingInput();
        }
    }

    void expireTerminalRecords(long before) {
        terminalByPhase
                .values()
                .forEach(
                        map ->
                                map.forEach(
                                        (id, record) ->
                                                removeExactTerminal(record.state(), before)));
    }

    boolean removeExactTerminal(RequestState exact, long before) {
        if (exact == null || exact.updatedAtMs() >= before) {
            return false;
        }
        synchronized (registrationLock) {
            var archived = terminalByPhase.get(exact.requestPhase()).get(exact.requestId());
            if (archived == null || archived.state() != exact) {
                return false;
            }
            return terminalByPhase.get(exact.requestPhase()).remove(exact.requestId(), archived);
        }
    }

    private record ArchivedRequest(RequestState state, AbstractRequestScheduler owner) {}

    enum RegistrationResult {
        REGISTERED,
        DUPLICATE_ID,
        CONTEXT_BOUND,
        CLOSED
    }
}

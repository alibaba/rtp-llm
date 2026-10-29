package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RequestPhase;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Shared identity directory; request decisions remain with the exact context and its owner.
 */
@Component
public final class RequestRepository {
    private final Object registrationLock = new Object();
    private final ConcurrentMap<RequestKey, RequestContext> activeRequests =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<RequestKey, TerminalRecord> terminalRecords =
            new ConcurrentHashMap<>();
    private volatile boolean closed;
    private volatile boolean stopping;

    enum RegistrationResult {
        REGISTERED,
        DUPLICATE_ID,
        CONTEXT_BOUND,
        INPUT_REQUIRED,
        CLOSED
    }

    private record RequestKey(String requestId, RequestPhase phase) {}

    private static RequestKey key(RequestContext context) {
        return new RequestKey(context.data().getRequestId(), context.data().getRequestPhase());
    }

    public record TerminalRecord(
            RequestState state, AbstractRequestScheduler owner, RequestPhase requestPhase) {
        public TerminalRecord(RequestState state, AbstractRequestScheduler owner) {
            this(state, owner, RequestPhase.GENERATION);
        }
    }

    RegistrationResult register(
            RequestContext context, AbstractRequestScheduler scheduler, RequestFuture response) {
        java.util.Objects.requireNonNull(context);
        java.util.Objects.requireNonNull(scheduler);
        java.util.Objects.requireNonNull(response);
        synchronized (registrationLock) {
            if (closed || stopping) {
                return RegistrationResult.CLOSED;
            }
            synchronized (context) {
                var data = context.data();
                synchronized (data) {
                    var id = key(context);
                    if (activeRequests.containsKey(id) || terminalRecords.containsKey(id)) {
                        return RegistrationResult.DUPLICATE_ID;
                    }
                    if (data.getScheduler() != null || context.snapshot().key() != null) {
                        return RegistrationResult.CONTEXT_BOUND;
                    }
                    if (data.getRequestPhase() != RequestPhase.ENCODER
                            && data.getConfig().getDispatcher().requiresGenerateInput()
                            && !data.hasGenerateInput()) {
                        return RegistrationResult.INPUT_REQUIRED;
                    }
                    long timeout =
                            data.getConfig().getRequestLifecycle().getRequest().getTimeoutMs();
                    if (timeout <= 0L) {
                        throw new IllegalArgumentException(
                                "positive request inactivity timeout required");
                    }
                    if (!data.getConfig().getDispatcher().requiresGenerateInput()) {
                        data.discardGenerateInput();
                    }
                    data.register(scheduler, response);
                    context.register(
                            data.getRequestId(),
                            System.currentTimeMillis(),
                            data.getRequestExpiresAtMs(),
                            timeout);
                    activeRequests.put(id, context);
                }
            }
            return RegistrationResult.REGISTERED;
        }
    }

    public RequestContext findActive(String requestId) {
        return findActive(requestId, RequestPhase.GENERATION);
    }

    public RequestContext findActive(String requestId, RequestPhase phase) {
        return activeRequests.get(new RequestKey(requestId, phase));
    }

    public TerminalRecord findTerminal(String requestId) {
        return findTerminal(requestId, RequestPhase.GENERATION);
    }

    public TerminalRecord findTerminal(String requestId, RequestPhase phase) {
        return terminalRecords.get(new RequestKey(requestId, phase));
    }

    public AbstractRequestScheduler ownerOf(String requestId) {
        return ownerOf(requestId, RequestPhase.GENERATION);
    }

    public AbstractRequestScheduler ownerOf(String requestId, RequestPhase phase) {
        RequestContext context = findActive(requestId, phase);
        if (context != null) {
            return context.data().getScheduler();
        }
        TerminalRecord terminal = findTerminal(requestId, phase);
        return terminal == null ? null : terminal.owner();
    }

    public List<RequestContext> snapshotActive() {
        return List.copyOf(activeRequests.values());
    }

    public boolean retainsIdentity(String requestId) {
        return activeRequests.containsKey(new RequestKey(requestId, RequestPhase.GENERATION))
                || terminalRecords.containsKey(new RequestKey(requestId, RequestPhase.GENERATION));
    }

    public boolean isCurrent(RequestContext exact) {
        return exact != null && activeRequests.get(key(exact)) == exact;
    }

    boolean closeRegistration() {
        synchronized (registrationLock) {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        }
    }

    void stopAccepting() {
        synchronized (registrationLock) {
            stopping = true;
        }
    }

    boolean isStopping() {
        return stopping;
    }

    boolean isClosed() {
        return closed;
    }

    void archive(RequestContext exact, RequestState terminal) {
        synchronized (registrationLock) {
            if (activeRequests.get(key(exact)) != exact) {
                return;
            }
            terminalRecords.put(
                    key(exact),
                    new TerminalRecord(
                            terminal, exact.data().getScheduler(), exact.data().getRequestPhase()));
            activeRequests.remove(key(exact), exact);
            exact.data().releaseSchedulingInput();
        }
    }

    boolean removeExactTerminal(TerminalRecord exact, long before) {
        if (exact == null || exact.state().updatedAtMs() >= before) {
            return false;
        }
        synchronized (registrationLock) {
            if (terminalRecords.get(new RequestKey(exact.state().requestId(), exact.requestPhase()))
                    != exact) {
                return false;
            }
            return terminalRecords.remove(
                    new RequestKey(exact.state().requestId(), exact.requestPhase()), exact);
        }
    }

    void expireTerminalRecords(long before) {
        terminalRecords.forEach((id, record) -> removeExactTerminal(record, before));
    }

    public record TrackedRequestStats(
            int generationRequests, int encoderRequests, long oldestAgeMs) {}

    public TrackedRequestStats trackedRequestStats() {
        int generation = 0;
        int encoder = 0;
        long oldest = Long.MAX_VALUE;
        for (RequestContext context : activeRequests.values()) {
            var snapshot = context.snapshot();
            if (snapshot.phase() == ContextResult.Phase.CLOSED || !isCurrent(context)) {
                continue;
            }
            if (context.data().getRequestPhase() == RequestPhase.ENCODER) {
                encoder++;
            } else {
                generation++;
            }
            oldest = Math.min(oldest, snapshot.createdAtMs());
        }
        return new TrackedRequestStats(
                generation,
                encoder,
                oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest));
    }

    public int liveRequestCount() {
        return activeRequests.size();
    }

    /**
     * Count accepted QUEUE requests before delivery handoff, not just global queue entries.
     */
    public int pendingDeliveryRequestCount() {
        int pending = 0;
        for (RequestContext context : activeRequests.values()) {
            var state = context.snapshot();
            if (context.data().getConfig().isQueue()
                    && state.phase() == ContextResult.Phase.OPEN
                    && state.delivery() == null) {
                pending++;
            }
        }
        return pending;
    }

    public long oldestLiveRequestAgeMs() {
        long oldest = Long.MAX_VALUE;
        for (RequestContext context : activeRequests.values()) {
            oldest = Math.min(oldest, context.snapshot().createdAtMs());
        }
        return oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
    }

    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> snapshots = new ArrayList<>(activeRequests.size());
        for (Map.Entry<RequestKey, RequestContext> candidate : activeRequests.entrySet()) {
            RequestContext context = candidate.getValue();
            synchronized (context) {
                if (activeRequests.get(candidate.getKey()) == context
                        && context.snapshot().phase() != ContextResult.Phase.CLOSED) {
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

    public RequestState getRequestState(String requestId, long expectedBatchId) {
        return getRequestState(requestId, expectedBatchId, RequestPhase.GENERATION);
    }

    public RequestState getRequestState(
            String requestId, long expectedBatchId, RequestPhase phase) {
        RequestContext context = findActive(requestId, phase);
        TerminalRecord terminal = findTerminal(requestId, phase);
        RequestState state =
                context != null
                        ? RequestState.from(context.snapshot())
                        : terminal == null ? null : terminal.state();
        return state != null && state.matchesBatch(expectedBatchId) ? state : null;
    }
}

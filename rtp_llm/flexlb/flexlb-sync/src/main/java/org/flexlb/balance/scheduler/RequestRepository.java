package org.flexlb.balance.scheduler;

import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/** Shared identity directory; request decisions remain with the exact context and its owner. */
@Component
public final class RequestRepository {
    private final Object registrationLock = new Object();
    private final ConcurrentMap<Long, RequestContext> activeRequests = new ConcurrentHashMap<>();
    private final ConcurrentMap<Long, TerminalRecord> terminalRecords = new ConcurrentHashMap<>();
    private volatile boolean closed;

    enum RegistrationResult { REGISTERED, DUPLICATE_ID, CONTEXT_BOUND, CLOSED }
    public record TerminalRecord(RequestState state, AbstractRequestScheduler owner) { }

    RegistrationResult register(RequestContext context, AbstractRequestScheduler scheduler, RequestContext.RequestFuture response) {
        java.util.Objects.requireNonNull(context, "context");
        java.util.Objects.requireNonNull(scheduler, "scheduler");
        java.util.Objects.requireNonNull(response, "response");
        synchronized (registrationLock) {
            if (closed || !scheduler.runtime.isAccepting()) { return RegistrationResult.CLOSED; }
            if (activeRequests.containsKey(context.getRequestId()) || terminalRecords.containsKey(context.getRequestId())) {
                return RegistrationResult.DUPLICATE_ID;
            }
            synchronized (context) {
                if (context.scheduler() != null || context.getFuture() instanceof RequestContext.RequestFuture) {
                    return RegistrationResult.CONTEXT_BOUND;
                }
                context.activate(response);
                context.bindScheduler(scheduler);
                activeRequests.put(context.getRequestId(), context);
            }
            return RegistrationResult.REGISTERED;
        }
    }

    public RequestContext findActive(long requestId) { return activeRequests.get(requestId); }
    public TerminalRecord findTerminal(long requestId) { return terminalRecords.get(requestId); }
    public AbstractRequestScheduler ownerOf(long requestId) {
        RequestContext context = findActive(requestId);
        if (context != null) { return context.scheduler(); }
        TerminalRecord terminal = findTerminal(requestId);
        return terminal == null ? null : terminal.owner();
    }
    public List<RequestContext> snapshotActive() { return List.copyOf(activeRequests.values()); }
    public boolean retainsIdentity(long requestId) {
        return activeRequests.containsKey(requestId) || terminalRecords.containsKey(requestId);
    }
    public boolean isCurrent(RequestContext exact) {
        return exact != null && activeRequests.get(exact.getRequestId()) == exact;
    }
    boolean closeRegistration() {
        synchronized (registrationLock) {
            if (closed) { return false; }
            closed = true;
            return true;
        }
    }
    boolean isClosed() { return closed; }
    void archive(RequestContext exact, RequestState terminal) {
        synchronized (registrationLock) {
            if (activeRequests.get(exact.getRequestId()) != exact) { return; }
            terminalRecords.put(exact.getRequestId(), new TerminalRecord(terminal, exact.scheduler()));
            activeRequests.remove(exact.getRequestId(), exact);
        }
    }
    boolean removeExactTerminal(TerminalRecord exact, long before) {
        if (exact == null || exact.state().updatedAtMs() >= before) { return false; }
        synchronized (registrationLock) {
            if (terminalRecords.get(exact.state().requestId()) != exact) { return false; }
            return terminalRecords.remove(exact.state().requestId(), exact);
        }
    }
    void expireTerminalRecords(long before) {
        terminalRecords.forEach((id, record) -> removeExactTerminal(record, before));
    }
    public int liveRequestCount() {
        return activeRequests.size();
    }
    /** Count accepted QUEUE requests before delivery handoff, not just global queue entries. */
    public int pendingDeliveryRequestCount() {
        int pending = 0;
        for (RequestContext context : activeRequests.values()) {
            if (context.awaitsDelivery()) { pending++; }
        }
        return pending;
    }
    public long oldestLiveRequestAgeMs() {
        long oldest = Long.MAX_VALUE;
        for (RequestContext context : activeRequests.values()) {
            oldest = Math.min(oldest, context.createdAtMs());
        }
        return oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
    }
    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> snapshots = new ArrayList<>(activeRequests.size());
        for (Map.Entry<Long, RequestContext> candidate : activeRequests.entrySet()) {
            RequestContext context = candidate.getValue();
            synchronized (context) {
                if (activeRequests.get(candidate.getKey()) == context && context.isLiveGeneration()) {
                    snapshots.add(context.snapshot());
                }
            }
        }
        snapshots.sort((left, right) -> {
            int createdOrder = Long.compare(left.createdAtMs(), right.createdAtMs());
            return createdOrder != 0 ? createdOrder : Long.compare(left.requestId(), right.requestId());
        });
        return List.copyOf(snapshots);
    }
    public RequestState getRequestState(long requestId, long expectedBatchId) {
        RequestContext context = activeRequests.get(requestId);
        TerminalRecord terminal = terminalRecords.get(requestId);
        RequestState state = context != null ? context.snapshot() : terminal == null ? null : terminal.state();
        return state != null && state.matchesBatch(expectedBatchId) ? state : null;
    }
}

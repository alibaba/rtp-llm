package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.dao.route.RequestPhase;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/** Shared identity directory; request decisions remain with the exact context and its owner. */
@Component
public final class RequestRepository {
    @org.springframework.scheduling.annotation.Scheduled(fixedRate = 60_000L)
    public void maintainExpiration() {
        if (!closed) expireTerminalRecords(System.currentTimeMillis() - 10_000L);
    }

    private final Object registrationLock = new Object();
    private final EnumMap<RequestPhase, ConcurrentMap<String, RequestContext>> activeByPhase =
            new EnumMap<>(RequestPhase.class);
    private final EnumMap<RequestPhase, ConcurrentMap<String, TerminalRecord>> terminalByPhase =
            new EnumMap<>(RequestPhase.class);

    public RequestRepository() {
        for (RequestPhase phase : RequestPhase.values()) {
            activeByPhase.put(phase, new ConcurrentHashMap<>());
            terminalByPhase.put(phase, new ConcurrentHashMap<>());
        }
    }

    private volatile boolean closed;
    private volatile boolean stopping;

    enum RegistrationResult {
        REGISTERED,
        DUPLICATE_ID,
        CONTEXT_BOUND,
        INPUT_REQUIRED,
        CLOSED
    }

    public record TerminalRecord(RequestState state, RequestLifecycle owner) {}

    RegistrationResult register(
            RequestContext context, RequestLifecycle scheduler, RequestFuture response) {
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
                    String id = data.getRequestId();
                    var activeRequests = activeByPhase.get(data.getPhase());
                    var terminalRecords = terminalByPhase.get(data.getPhase());
                    if (activeRequests.containsKey(id) || terminalRecords.containsKey(id)) {
                        return RegistrationResult.DUPLICATE_ID;
                    }
                    if (data.getLifecycle() != null || context.snapshot().key() != null) {
                        return RegistrationResult.CONTEXT_BOUND;
                    }
                    if (data.getPhase() == RequestPhase.GENERATION
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
                    if (data.getPhase() == RequestPhase.ENCODER
                            || !data.getConfig().getDispatcher().requiresGenerateInput()) {
                        data.discardGenerateInput();
                    }
                    data.register(scheduler, response);
                    context.register(
                            id, System.currentTimeMillis(), data.getRequestExpiresAtMs(), timeout);
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
        return activeByPhase.get(phase).get(requestId);
    }

    public TerminalRecord findTerminal(String requestId) {
        return findTerminal(requestId, RequestPhase.GENERATION);
    }

    public TerminalRecord findTerminal(String requestId, RequestPhase phase) {
        return terminalByPhase.get(phase).get(requestId);
    }

    public RequestLifecycle ownerOf(String requestId) {
        return ownerOf(requestId, RequestPhase.GENERATION);
    }

    public RequestLifecycle ownerOf(String requestId, RequestPhase phase) {
        RequestContext context = findActive(requestId, phase);
        if (context != null) {
            return context.data().getLifecycle();
        }
        TerminalRecord terminal = findTerminal(requestId, phase);
        return terminal == null ? null : terminal.owner();
    }

    public List<RequestContext> snapshotActive() {
        return activeByPhase.values().stream().flatMap(map -> map.values().stream()).toList();
    }

    public boolean retainsIdentity(String requestId) {
        return activeByPhase.values().stream().anyMatch(map -> map.containsKey(requestId))
                || terminalByPhase.values().stream().anyMatch(map -> map.containsKey(requestId));
    }

    public boolean isCurrent(RequestContext exact) {
        return exact != null
                && activeByPhase.get(exact.data().getPhase()).get(exact.data().getRequestId())
                        == exact;
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

    /** Wait only for accepted response futures; resource settlement follows separately. */
    public boolean awaitResponses(long deadlineNanos) {
        if (!stopping) throw new IllegalStateException("stop accepting requests before draining");
        for (var context : snapshotActive()) {
            var response = context.data().getFuture();
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
                // A failed or cancelled response is finished; lifecycle still owns settlement.
            } catch (java.util.concurrent.TimeoutException timeout) {
                return false;
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return true;
    }

    public boolean isStopping() {
        return stopping;
    }

    public boolean isClosed() {
        return closed;
    }

    void archive(RequestContext exact, RequestState terminal) {
        synchronized (registrationLock) {
            if (activeByPhase.get(exact.data().getPhase()).get(exact.data().getRequestId())
                    != exact) {
                return;
            }
            terminalByPhase
                    .get(exact.data().getPhase())
                    .put(
                            exact.data().getRequestId(),
                            new TerminalRecord(terminal, exact.data().getLifecycle()));
            activeByPhase.get(exact.data().getPhase()).remove(exact.data().getRequestId(), exact);
            exact.data().releaseSchedulingInput();
        }
    }

    boolean removeExactTerminal(TerminalRecord exact, long before) {
        if (exact == null || exact.state().updatedAtMs() >= before) {
            return false;
        }
        synchronized (registrationLock) {
            if (terminalByPhase.get(exact.state().requestPhase()).get(exact.state().requestId())
                    != exact) {
                return false;
            }
            return terminalByPhase
                    .get(exact.state().requestPhase())
                    .remove(exact.state().requestId(), exact);
        }
    }

    void expireTerminalRecords(long before) {
        terminalByPhase
                .values()
                .forEach(map -> map.forEach((id, record) -> removeExactTerminal(record, before)));
    }

    public int liveRequestCount() {
        return activeByPhase.values().stream().mapToInt(Map::size).sum();
    }

    /** Count accepted QUEUE requests before delivery handoff, not just global queue entries. */
    public int pendingDeliveryRequestCount() {
        int pending = 0;
        for (RequestContext context : snapshotActive()) {
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
        for (RequestContext context : snapshotActive()) {
            oldest = Math.min(oldest, context.snapshot().createdAtMs());
        }
        return oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
    }

    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> snapshots = new ArrayList<>(liveRequestCount());
        for (RequestContext context : snapshotActive()) {
            synchronized (context) {
                if (isCurrent(context)
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

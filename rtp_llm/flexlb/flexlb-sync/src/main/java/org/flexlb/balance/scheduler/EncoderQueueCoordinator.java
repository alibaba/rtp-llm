package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Orders Encoder route decisions independently of the Generation queue.
 * A single decision thread commits each local selection before reading the next request,
 * so locally pending requests count toward the configured worker limit.
 */
final class EncoderQueueCoordinator implements AutoCloseable {

    private final DefaultRouter router;
    private final RequestRegistry requests;
    private final OrderedRequestQueue queue;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final Set<GlobalQueueEntry> blocked = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Thread decisionThread;
    private boolean closed;
    private long capacityVersion;

    EncoderQueueCoordinator(DefaultRouter router, RequestRegistry requests, boolean priorityOrdering) {
        this.router = Objects.requireNonNull(router, "router");
        this.requests = Objects.requireNonNull(requests, "requests");
        this.queue = new OrderedRequestQueue(priorityOrdering);
        decisionThread = new Thread(this::runDecisions, "flexlb-encoder-decision");
        decisionThread.setDaemon(true);
        requests.attachEncoderQueue(this);
        decisionThread.start();
    }

    boolean offer(BalanceContext context, CompletableFuture<Response> future) {
        GlobalQueueEntry entry = new GlobalQueueEntry(context, future,
                PriorityNormalizer.isValid(context.getPriority())
                        ? context.getPriority() : PriorityNormalizer.DEFAULT_PRIORITY);
        lock.lock();
        try {
            if (closed) {
                return false;
            }
            queue.add(entry);
            future.whenComplete((ignored, failure) -> remove(entry));
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    int size() {
        lock.lock();
        try {
            return queue.size();
        } finally {
            lock.unlock();
        }
    }

    int blockedSize() {
        lock.lock();
        try {
            return blocked.size();
        } finally {
            lock.unlock();
        }
    }

    void capacityChanged() {
        lock.lock();
        try {
            capacityVersion++;
            for (GlobalQueueEntry entry : blocked) {
                queue.markRequestReadyForRetry(entry);
            }
            blocked.clear();
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void runDecisions() {
        try {
            while (true) {
                GlobalQueueEntry entry = nextRequest();
                if (entry == null) {
                    return;
                }
                decide(entry);
            }
        } catch (Throwable failure) {
            Logger.error("Encoder queue decision thread failed", failure);
        } finally {
            close();
        }
    }

    private GlobalQueueEntry nextRequest() {
        lock.lock();
        try {
            while (!closed) {
                List<GlobalQueueEntry> ready = queue.scanForPlanningCandidates(1,
                        Math.max(1, queue.size()), entry -> !entry.removed && !entry.future.isDone()
                                && !blocked.contains(entry));
                if (!ready.isEmpty()) {
                    return ready.getFirst();
                }
                changed.await();
            }
            return null;
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            return null;
        } finally {
            lock.unlock();
        }
    }

    private void decide(GlobalQueueEntry entry) {
        if (entry.future.isDone()) {
            remove(entry);
            return;
        }
        if (entry.context.requestExpired(System.currentTimeMillis())) {
            requests.cancelRequest(entry.context.getRequestId(), 0L,
                    CancelReason.DEADLINE_EXCEEDED, RequestPhase.ENCODER);
            remove(entry);
            return;
        }
        try {
            long observedCapacityVersion = capacityVersion();
            PlacementResult<SelectedRole, PlacementKey> selection = router.selectEncoder(entry.context);
            entry.context.setSchedulingDiagnostics(selection.diagnostics());
            switch (selection.status()) {
                case SUCCESS -> publishRoute(entry, selection.value());
                case BLOCKED -> park(entry, observedCapacityVersion);
                case REJECTED -> fail(entry, selection.failure());
                case CLOSED -> fail(entry, Response.error(StrategyErrorType.REQUEST_CANCELLED));
            }
        } catch (RuntimeException failure) {
            Logger.warn("Encoder QUEUE admission failed: request_id={}", entry.context.getRequestId(), failure);
            fail(entry, Response.error(StrategyErrorType.DISPATCH_FAILED));
        }
    }

    private void publishRoute(GlobalQueueEntry entry, SelectedRole selected) {
        try (selected; var pin = selected.takeGenerationPin()) {
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(selected.serverStatus()));
            if (!requests.claimEncoderRoute(entry.context.getRequestId(), entry.future, pin)
                    || !requests.publishEncoderRoute(entry.context.getRequestId(), entry.future, response)) {
                fail(entry, Response.error(StrategyErrorType.REQUEST_CANCELLED));
            }
        }
    }

    private long capacityVersion() {
        lock.lock();
        try {
            return capacityVersion;
        } finally {
            lock.unlock();
        }
    }

    private void park(GlobalQueueEntry entry, long observedCapacityVersion) {
        lock.lock();
        try {
            if (!entry.removed && !entry.future.isDone()) {
                if (capacityVersion == observedCapacityVersion) {
                    blocked.add(entry);
                } else {
                    queue.markRequestReadyForRetry(entry);
                    changed.signal();
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private void fail(GlobalQueueEntry entry, Response response) {
        requests.publishDecisionResponseAsync(entry.context.getRequestId(), entry.future,
                response, RequestPhase.ENCODER);
    }

    private void remove(GlobalQueueEntry entry) {
        lock.lock();
        try {
            blocked.remove(entry);
            queue.remove(entry);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void close() {
        List<GlobalQueueEntry> abandoned;
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            abandoned = queue.drain();
            blocked.clear();
            changed.signalAll();
        } finally {
            lock.unlock();
        }
        for (GlobalQueueEntry entry : abandoned) {
            fail(entry, Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                    "request scheduler is shutting down"));
        }
        if (Thread.currentThread() != decisionThread) {
            try {
                decisionThread.join(1_000L);
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        }
    }
}

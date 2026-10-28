package org.flexlb.mock.engine;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.Supplier;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;

/** Serial schedule/Forward/commit boundary. Network delivery never runs under the scheduling lock. */
public final class EngineLoop implements AutoCloseable {
    private final Object lock;
    private final Scheduler scheduler;
    private final ComputeModel model;
    private final Role role;
    private final String identity;
    private final SimulatedCompute compute = new SimulatedCompute();
    private final ExecutorService notifications = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledExecutorService maintenance = Executors.newSingleThreadScheduledExecutor(
        Thread.ofPlatform().daemon().name("mock-engine-maintenance").factory());
    private Pending pending;
    private boolean closed;
    private int delivering;
    private RuntimeException failure;

    private static final class Pending {
        final Scheduler.Work work;
        SimulatedCompute.Execution execution;
        Pending(Scheduler.Work work) { this.work = work; }
    }

    public EngineLoop(Object lock, Scheduler scheduler, ComputeModel model, Role role, String identity) {
        this.lock = Objects.requireNonNull(lock);
        this.scheduler = Objects.requireNonNull(scheduler);
        this.model = Objects.requireNonNull(model);
        this.role = Objects.requireNonNull(role);
        this.identity = Objects.requireNonNull(identity);
        // Capacity can be released outside this Engine. Polling is bounded and does not retry unchanged capacity.
        maintenance.scheduleWithFixedDelay(this::wake, 50, 50, TimeUnit.MILLISECONDS);
    }

    public Scheduler.Request submit(Scheduler.Input input, Scheduler.Listener listener) {
        return change(() -> { ensureOpen(); return scheduler.submit(input, listener); });
    }
    public void dependencyCompleted(Scheduler.Request request, Scheduler.Dependency dependency) {
        change(() -> { scheduler.dependencyCompleted(request, dependency); return null; });
    }
    public void finish(Scheduler.Request request, RequestFailure failure) {
        change(() -> { scheduler.finish(request, failure); return null; });
    }
    public RequestState.Use readReady(Scheduler.Request request) {
        var use = change(() -> { ensureOpen(); return scheduler.readReady(request); });
        return () -> change(() -> { use.close(); return null; });
    }
    public List<Scheduler.Request> requests() { synchronized (lock) { checkFailure(); return scheduler.requests(); } }
    public QueueSnapshot queues() { synchronized (lock) { checkFailure(); return scheduler.queues(); } }
    public String scrapeMetrics() { synchronized (lock) { checkFailure(); return scheduler.scrapeMetrics(); } }

    private <T> T change(Supplier<T> operation) {
        try {
            synchronized (lock) { return operation.get(); }
        } finally { wake(); }
    }

    public void wake() {
        synchronized (lock) {
            if (!closed) {
                try {
                    scheduler.maintain();
                    if (pending != null && scheduler.cancelled(pending.work)) {
                        var cancelled = pending;
                        pending = null;
                        cancelled.execution.cancel();
                        scheduler.failed(cancelled.work, new RequestFailure(RequestFailure.Reason.CANCELLED, "All batch members stopped"));
                    }
                    while (pending == null && !closed) {
                        var work = scheduler.schedule();
                        if (work == null) break;
                        var next = new Pending(work);
                        pending = next;
                        try {
                            var sample = model.plan(role, identity, work.batch());
                            next.execution = compute.submit(sample);
                            next.execution.completion().thenAccept(outcome -> complete(next, outcome));
                        } catch (RuntimeException error) {
                            pending = null;
                            scheduler.failed(work, error);
                        }
                    }
                } catch (RuntimeException error) { fail(error); }
            }
            dispatch();
        }
    }

    private void complete(Pending execution, SimulatedCompute.Outcome outcome) {
        synchronized (lock) {
            if (pending != execution) return;
            pending = null;
            try { scheduler.completed(execution.work, outcome); }
            catch (RuntimeException error) { fail(error); }
        }
        wake();
    }

    private void dispatch() {
        for (var notification : scheduler.takeNotifications()) {
            delivering++;
            notifications.execute(() -> {
                try { notification.run(); }
                catch (RuntimeException error) { synchronized (lock) { fail(error); } }
                finally {
                    synchronized (lock) { delivering--; }
                    wake();
                }
            });
        }
        if (closed && delivering == 0 && scheduler.requests().isEmpty()) notifications.shutdown();
    }

    private void ensureOpen() {
        checkFailure();
        if (closed) throw new RequestFailure(RequestFailure.Reason.CLOSED, "Engine is closed");
    }
    private void checkFailure() { if (failure != null) throw failure; }
    private void fail(RuntimeException error) {
        if (failure == null) failure = error;
        stop();
    }
    private void stop() {
        if (closed) return;
        closed = true;
        maintenance.shutdownNow();
        scheduler.close();
        if (pending != null) {
            var current = pending;
            pending = null;
            if (current.execution != null) current.execution.cancel();
            scheduler.failed(current.work, new RequestFailure(RequestFailure.Reason.CLOSED, "Engine closed during Forward"));
        }
        compute.close();
    }

    @Override public void close() {
        synchronized (lock) { stop(); dispatch(); }
    }
}

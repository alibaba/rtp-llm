package org.flexlb.mock.engine;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.Supplier;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.compute.*;

/** Event/deadline driven execution. Only short state transitions hold the scheduling lock. */
public final class EngineLoop implements AutoCloseable {
    private final Object lock;
    private final java.util.concurrent.locks.ReentrantLock signal = new java.util.concurrent.locks.ReentrantLock();
    private final java.util.concurrent.locks.Condition changed = signal.newCondition();
    private final Scheduler scheduler;
    private final KvCache cache;
    private final ComputeModel model;
    private final Role role;
    private final String identity;
    private final SimulatedCompute compute = new SimulatedCompute();
    private final ExecutorService notifications = Executors.newVirtualThreadPerTaskExecutor();
    private Pending pending;
    private volatile boolean closed;
    private boolean dirty;
    private int delivering;
    private RuntimeException failure;

    private static final class Pending {
        final Scheduler.Work work;
        SimulatedCompute.Execution execution;
        Pending(Scheduler.Work work) { this.work = work; }
    }

    public EngineLoop(Object lock, Scheduler scheduler, KvCache cache, ComputeModel model, Role role, String identity) {
        this.lock = Objects.requireNonNull(lock);
        this.scheduler = Objects.requireNonNull(scheduler);
        this.cache = Objects.requireNonNull(cache);
        this.model = Objects.requireNonNull(model);
        this.role = Objects.requireNonNull(role);
        this.identity = Objects.requireNonNull(identity);
        cache.onCapacityChange(this::wake);
        Thread.ofVirtual().name("mock-engine-" + identity).start(this::run);
    }

    public Scheduler.Request input(Scheduler.Input input, Scheduler.Listener listener) {
        return change(() -> { ensureOpen(); return scheduler.input(input, listener); });
    }
    public void control(Scheduler.Request request, Scheduler.Control event) {
        change(() -> { scheduler.control(request, event); return null; });
    }
    public void transferCompleted(Scheduler.Request request, Scheduler.TransferResult result) {
        change(() -> { scheduler.transferCompleted(request, result); return null; });
    }
    public void finish(Scheduler.Request request, RequestFailure failure) {
        change(() -> { scheduler.finish(request, failure); return null; });
    }
    public RequestState.Use beginTransfer(Scheduler.Request request) {
        var use = change(() -> { ensureOpen(); return scheduler.beginTransfer(request); });
        return () -> change(() -> { use.close(); return null; });
    }
    public Scheduler.Status status(long latestFinishedVersion) {
        synchronized (lock) { checkFailure(); return scheduler.status(latestFinishedVersion); }
    }
    public String scrapeMetrics() { synchronized (lock) { checkFailure(); return scheduler.scrapeMetrics(); } }

    private <T> T change(Supplier<T> operation) {
        try { synchronized (lock) { return operation.get(); } }
        finally {
            synchronized (lock) { if (closed) dispatch(); }
            wake();
        }
    }

    /** A signal is retained even if it arrives before the worker starts waiting. */
    public void wake() {
        signal.lock();
        try { dirty = true; changed.signal(); } finally { signal.unlock(); }
    }

    private void run() {
        long deadline = Long.MAX_VALUE;
        try {
            while (true) {
                signal.lock();
                try {
                    while (!dirty && !closed) {
                        if (deadline == Long.MAX_VALUE) changed.await();
                        else {
                            long remaining = deadline - System.nanoTime();
                            if (remaining <= 0) break;
                            changed.awaitNanos(remaining);
                        }
                    }
                    dirty = false;
                } finally { signal.unlock(); }
                synchronized (lock) {
                    if (!closed) {
                        try { advance(); }
                        catch (RuntimeException error) { fail(error); }
                    }
                    dispatch();
                    if (closed) return;
                    deadline = scheduler.nextDeadline();
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            synchronized (lock) { fail(new IllegalStateException("Engine loop interrupted", error)); dispatch(); }
        } catch (RuntimeException error) {
            synchronized (lock) { fail(error); dispatch(); }
        }
    }

    private void advance() {
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
    }

    private void complete(Pending execution, SimulatedCompute.Outcome outcome) {
        synchronized (lock) {
            if (pending != execution) return;
            pending = null;
            try { scheduler.computeCompleted(execution.work, outcome); }
            catch (RuntimeException error) { fail(error); }
            if (closed) dispatch();
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
                    synchronized (lock) { delivering--; if (closed) dispatch(); }
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
        cache.onCapacityChange(() -> {});
        scheduler.close();
        if (pending != null) {
            var current = pending;
            pending = null;
            if (current.execution != null) current.execution.cancel();
            scheduler.failed(current.work, new RequestFailure(RequestFailure.Reason.CLOSED, "Engine closed during Forward"));
        }
        compute.close();
        wake();
    }

    @Override public void close() { synchronized (lock) { stop(); dispatch(); } }
}

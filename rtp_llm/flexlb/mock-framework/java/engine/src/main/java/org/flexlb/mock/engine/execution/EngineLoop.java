package org.flexlb.mock.engine.execution;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.Supplier;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.forward.*;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.RequestState;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.flexlb.mock.engine.scheduler.TaskHistory;

/** Enqueue, schedule, Forward and output processing; waits for events outside the scheduling lock. */
public final class EngineLoop implements AutoCloseable {
    private final Object lock;
    private final java.util.concurrent.locks.ReentrantLock signal = new java.util.concurrent.locks.ReentrantLock();
    private final java.util.concurrent.locks.Condition changed = signal.newCondition();
    private final EngineScheduler scheduler;
    private final TaskHistory finished;
    private final KvCacheManager cache;
    private final ComputeModel model;
    private final Role role;
    private final String identity;
    private final SimulatedCompute compute = new SimulatedCompute();
    private final ExecutorService notifications = Executors.newVirtualThreadPerTaskExecutor();
    private ForwardExecution inFlightForward;
    private volatile boolean closed;
    private boolean dirty;
    private int notifying;
    private volatile RuntimeException failure;

    private static final class ForwardExecution {
        final EngineScheduler.Work work;
        SimulatedCompute.Execution execution;
        ForwardExecution(EngineScheduler.Work work) { this.work = work; }
    }

    public EngineLoop(Object lock, EngineScheduler.Factory schedulers, TaskHistory finished, KvCacheManager cache, ComputeModel model, Role role, String identity) {
        this(lock, schedulers.create(lock, cache, finished), finished, cache, model, role, identity);
    }

    public EngineLoop(Object lock, EngineScheduler scheduler, TaskHistory finished, KvCacheManager cache,
                         ComputeModel model, Role role, String identity) {
        this.lock = Objects.requireNonNull(lock);
        this.finished = Objects.requireNonNull(finished);
        this.scheduler = Objects.requireNonNull(scheduler);
        this.cache = Objects.requireNonNull(cache);
        this.model = Objects.requireNonNull(model);
        this.role = Objects.requireNonNull(role);
        this.identity = Objects.requireNonNull(identity);
        cache.onChange(this::wakeUp);
        Thread.ofVirtual().name("mock-engine-" + identity).start(this::loop);
    }

    public EngineScheduler.Request enqueue(EngineScheduler.Input input, EngineScheduler.Listener listener) {
        return withSchedulerLock(() -> { ensureOpen(); return scheduler.enqueue(input, listener); });
    }
    public List<EngineScheduler.InputResult> enqueue(List<EngineScheduler.InputItem> batch) {
        return withSchedulerLock(() -> { ensureOpen(); return scheduler.enqueue(List.copyOf(batch)); });
    }
    /** Apply protocol-owned generation updates; the Scheduler owns their state transitions. */
    public void updateRequest(EngineScheduler.Request request, EngineScheduler.Control event) {
        withSchedulerLock(() -> { scheduler.control(request, event); return null; });
    }
    public void onKvTransferCompleted(EngineScheduler.Request request, EngineScheduler.TransferResult result) {
        withSchedulerLock(() -> { scheduler.transferCompleted(request, result); return null; });
    }
    public void finishRequest(EngineScheduler.Request request, RequestFailure failure) {
        withSchedulerLock(() -> { scheduler.finish(request, failure); return null; });
    }
    public RequestState.Use beginKvTransfer(EngineScheduler.Request request) {
        var use = withSchedulerLock(() -> { ensureOpen(); return scheduler.beginTransfer(request); });
        return () -> withSchedulerLock(() -> { use.close(); return null; });
    }
    public EngineScheduler.Status status(long latestFinishedVersion) {
        synchronized (lock) {
            checkFailure();
            var counts = scheduler.queues();
            var tasks = scheduler.tasks();
            return finished.snapshot(latestFinishedVersion, tasks, counts.waiting(), counts.running());
        }
    }
    public void checkHealthy() { ensureOpen(); }

    private <T> T withSchedulerLock(Supplier<T> operation) {
        try { synchronized (lock) { return operation.get(); } }
        finally {
            synchronized (lock) { if (closed) dispatchCallbacks(); }
            wakeUp();
        }
    }

    /** A signal is retained even if it arrives before the worker starts waiting. */
    public void wakeUp() {
        signal.lock();
        try { dirty = true; changed.signal(); } finally { signal.unlock(); }
    }

    private void loop() {
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
                        try { step(); }
                        catch (RuntimeException error) { failEngine(error); }
                    }
                    dispatchCallbacks();
                    if (closed) return;
                    deadline = scheduler.nextDeadline();
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            synchronized (lock) { failEngine(new IllegalStateException("Engine loop interrupted", error)); dispatchCallbacks(); }
        } catch (RuntimeException error) {
            synchronized (lock) { failEngine(error); dispatchCallbacks(); }
        }
    }

    private void step() {
        scheduler.prepareKvAndCheckTimeouts();
        if (inFlightForward != null && scheduler.isForwardCancelled(inFlightForward.work)) {
            var cancelled = inFlightForward;
            inFlightForward = null;
            cancelled.execution.cancel();
            scheduler.failForward(cancelled.work, new RequestFailure(RequestFailure.Reason.CANCELLED, "All batch members stopped"));
        }
        while (inFlightForward == null && !closed) {
            var work = scheduler.schedule();
            if (work == null) break;
            forward(work);
        }
    }

    /** One scheduled batch enters one simulated model Forward, matching the real Engine step flow. */
    private void forward(EngineScheduler.Work work) {
        var next = new ForwardExecution(work);
        inFlightForward = next;
        try {
            var sample = model.plan(role, identity, work.batch());
            next.execution = compute.submit(sample);
            next.execution.completion().thenAccept(outcome -> processOutputs(next, outcome));
        } catch (RuntimeException error) {
            inFlightForward = null;
            scheduler.failForward(work, error);
        }
    }

    private void processOutputs(ForwardExecution execution, SimulatedCompute.Outcome outcome) {
        synchronized (lock) {
            if (inFlightForward != execution) return;
            inFlightForward = null;
            try { scheduler.processOutputs(execution.work, outcome); }
            catch (RuntimeException error) { failEngine(error); }
            if (closed) dispatchCallbacks();
        }
        wakeUp();
    }

    private void dispatchCallbacks() {
        for (var notification : scheduler.takeNotifications()) {
            notifying++;
            notifications.execute(() -> {
                try { notification.run(); }
                catch (RuntimeException error) { synchronized (lock) { failEngine(error); } }
                finally {
                    synchronized (lock) { notifying--; if (closed) dispatchCallbacks(); }
                    wakeUp();
                }
            });
        }
        if (closed && notifying == 0 && scheduler.requests().isEmpty()) notifications.shutdown();
    }

    private void ensureOpen() {
        checkFailure();
        if (closed) throw new RequestFailure(RequestFailure.Reason.CLOSED, "Engine is closed");
    }
    private void checkFailure() { if (failure != null) throw failure; }
    private void failEngine(RuntimeException error) {
        if (failure == null) failure = error;
        stopLoop();
    }
    private void stopLoop() {
        if (closed) return;
        closed = true;
        cache.onChange(() -> {});
        scheduler.close();
        if (inFlightForward != null) {
            var current = inFlightForward;
            inFlightForward = null;
            if (current.execution != null) current.execution.cancel();
            scheduler.failForward(current.work, new RequestFailure(RequestFailure.Reason.CLOSED, "Engine closed during Forward"));
        }
        compute.close();
        wakeUp();
    }

    @Override public void close() { synchronized (lock) { stopLoop(); dispatchCallbacks(); } }
}

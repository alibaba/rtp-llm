package org.flexlb.mock.engine;

import java.util.List;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
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
    private final TaskHistory finished = new TaskHistory();
    private final Map<String, Admission> admissions = new LinkedHashMap<>();

    /** Opaque pre-submission identity. Execution state stays in the Scheduler. */
    public final class Admission {
        private final String identity;
        private final Scheduler.Task task;
        private boolean pending = true;
        private Admission(String identity, Scheduler.Task task) { this.identity = identity; this.task = task; }
    }
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

    public EngineLoop(Object lock, Scheduler.Factory schedulers, KvCache cache, ComputeModel model, Role role, String identity) {
        this.lock = Objects.requireNonNull(lock);
        this.scheduler = Objects.requireNonNull(schedulers.create(lock, cache, finished));
        this.cache = Objects.requireNonNull(cache);
        this.model = Objects.requireNonNull(model);
        this.role = Objects.requireNonNull(role);
        this.identity = Objects.requireNonNull(identity);
        cache.onChange(this::wake);
        Thread.ofVirtual().name("mock-engine-" + identity).start(this::run);
    }

    public Admission beginAdmission(String identity, Scheduler.Task task) {
        synchronized (lock) {
            ensureOpen();
            if (admissions.containsKey(identity) || scheduler.requests().stream().anyMatch(r -> r.input().identity().equals(identity)))
                throw new RequestFailure(RequestFailure.Reason.DUPLICATE, "Active request identity already exists");
            var admission = new Admission(identity, Objects.requireNonNull(task));
            admissions.put(identity, admission);
            return admission;
        }
    }

    public Scheduler.Request input(Admission admission, Scheduler.Input input, Scheduler.Listener listener) {
        return change(() -> {
            ensureOpen(); requirePending(admission, input);
            var request = scheduler.input(input, listener);
            submitted(admission);
            return request;
        });
    }

    public List<Scheduler.InputResult> input(List<Admission> pending, List<Scheduler.InputItem> batch) {
        return change(() -> {
            ensureOpen();
            if (pending.size() != batch.size()) throw new IllegalArgumentException("Admission count differs from batch");
            var eligible = new ArrayList<Scheduler.InputItem>();
            var positions = new ArrayList<Integer>();
            var results = new ArrayList<Scheduler.InputResult>(java.util.Collections.nCopies(batch.size(), null));
            for (int i = 0; i < pending.size(); i++) {
                try {
                    requirePending(pending.get(i), batch.get(i).input());
                    eligible.add(batch.get(i)); positions.add(i);
                } catch (RequestFailure failure) { results.set(i, new Scheduler.InputResult(null, failure)); }
            }
            if (!eligible.isEmpty()) {
                var accepted = scheduler.input(List.copyOf(eligible));
                for (int i = 0; i < accepted.size(); i++) {
                    int position = positions.get(i);
                    if (accepted.get(i).failure() == null) submitted(pending.get(position));
                    results.set(position, accepted.get(i));
                }
            }
            return List.copyOf(results);
        });
    }

    /** Only a still-pending admission may publish a pre-execution terminal record. */
    public void finishAdmission(Admission admission, RequestFailure failure) {
        synchronized (lock) {
            if (!admission.pending || admissions.get(admission.identity) != admission) return;
            submitted(admission);
            var t = admission.task;
            finished.add(new Scheduler.Task(t.requestId(), t.batchId(), t.inputLength(), 0, t.priority(),
                Scheduler.ReportedPhase.PENDING, 0, 0, 0, System.currentTimeMillis(), failure));
        }
    }

    private void requirePending(Admission admission, Scheduler.Input input) {
        if (!admission.pending || admissions.get(admission.identity) != admission)
            throw new RequestFailure(RequestFailure.Reason.CANCELLED, "Admission already ended");
        if (!admission.identity.equals(input.identity())) throw new IllegalArgumentException("Admission identity changed");
    }
    private void submitted(Admission admission) {
        admissions.remove(admission.identity, admission);
        admission.pending = false;
    }

    public Scheduler.Request input(Scheduler.Input input, Scheduler.Listener listener) {
        return change(() -> { ensureOpen(); return scheduler.input(input, listener); });
    }
    public List<Scheduler.InputResult> input(List<Scheduler.InputItem> batch) {
        return change(() -> { ensureOpen(); return scheduler.input(List.copyOf(batch)); });
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
        synchronized (lock) {
            checkFailure();
            var counts = scheduler.queues();
            var tasks = new ArrayList<>(scheduler.tasks());
            admissions.values().forEach(admission -> tasks.add(admission.task));
            return finished.snapshot(latestFinishedVersion, tasks, counts.waiting(), counts.running());
        }
    }
    public void checkHealthy() { synchronized (lock) { ensureOpen(); } }
    public List<io.prometheus.client.Collector.MetricFamilySamples> collectMetrics() {
        synchronized (lock) { checkFailure(); return scheduler.collectMetrics(); }
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
        cache.onChange(() -> {});
        for (var admission : List.copyOf(admissions.values()))
            finishAdmission(admission, new RequestFailure(RequestFailure.Reason.CLOSED, "Engine closed before submission"));
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

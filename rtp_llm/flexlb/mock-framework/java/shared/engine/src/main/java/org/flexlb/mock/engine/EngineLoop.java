package org.flexlb.mock.engine;

import java.util.Objects;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;

/** One Engine's serial scheduling/Forward boundary. No protocol or queue representation lives here. */
public final class EngineLoop<T> implements AutoCloseable {
    private final Object lock;
    private final Scheduler<T> scheduler;
    private final ComputeModel model;
    private final Role role;
    private final String identity;
    private final SimulatedCompute compute = new SimulatedCompute();
    private Pending pending;
    private boolean pumping, closed;

    private final class Pending {
        final Scheduler.Work<T> work;
        long started;
        SimulatedCompute.Execution execution;
        NoiseModel.Sample sample;
        Pending(Scheduler.Work<T> work) { this.work = work; }
    }

    public EngineLoop(Object lock, Scheduler<T> scheduler, ComputeModel model, Role role, String identity) {
        this.lock = Objects.requireNonNull(lock); this.scheduler = Objects.requireNonNull(scheduler);
        this.model = Objects.requireNonNull(model); this.role = Objects.requireNonNull(role);
        this.identity = Objects.requireNonNull(identity);
    }

    public void wake() {
        synchronized (lock) {
            if (closed || pumping || pending != null) return;
            pumping = true;
            try {
                while (!closed && pending == null) {
                    var work = scheduler.schedule();
                    if (work == null) return;
                    var next = new Pending(work); pending = next;
                    try {
                        next.sample = model.plan(role, identity, work.batch());
                        next.started = System.nanoTime();
                        next.execution = compute.submit(next.sample);
                        next.execution.completion().thenAccept(outcome -> complete(next, outcome));
                    } catch (RuntimeException error) {
                        pending = null;
                        scheduler.failed(work.owner(), error);
                    }
                }
            } finally { pumping = false; }
        }
    }

    private void complete(Pending current, SimulatedCompute.Outcome outcome) {
        synchronized (lock) {
            if (pending != current) return;
            pending = null;
            pumping = true;
            try { scheduler.completed(current.work.owner(), new SimulatedCompute.Outcome(
                outcome.state(), outcome.sample(), Math.max(outcome.elapsedNanos(), System.nanoTime() - current.started))); }
            catch (RuntimeException error) { scheduler.failed(current.work.owner(), error); }
            finally { pumping = false; }
            wake();
        }
    }

    /** The owner is a batch/session object, never a reusable request ID. */
    public void cancel(T owner) {
        synchronized (lock) {
            if (pending != null && pending.work.owner() == owner && pending.execution != null) {
                var current = pending;
                current.execution.cancel();
                complete(current, new SimulatedCompute.Outcome(SimulatedCompute.State.CANCELLED,
                    current.sample, Math.max(0, System.nanoTime() - current.started)));
            }
        }
    }

    @Override public void close() {
        synchronized (lock) {
            if (closed) return;
            closed = true;
            if (pending != null) cancel(pending.work.owner());
            compute.close();
        }
    }
}

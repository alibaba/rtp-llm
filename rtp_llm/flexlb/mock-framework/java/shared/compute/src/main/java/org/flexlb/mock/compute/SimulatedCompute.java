package org.flexlb.mock.compute;

import java.util.*;
import java.util.concurrent.*;

/** Cancellable waiting only. Engine owns request state, cache commits and resource release. */
public final class SimulatedCompute implements AutoCloseable {
    public enum State { COMPLETED, CANCELLED }
    public record Outcome(State state, NoiseModel.Sample sample, long elapsedNanos) {}
    private final ScheduledThreadPoolExecutor timer;
    private final ExecutorService notifications = Executors.newVirtualThreadPerTaskExecutor();
    private final Set<Execution> active = ConcurrentHashMap.newKeySet();
    private final Object lifecycle = new Object();
    private boolean closed;

    public SimulatedCompute() {
        timer = new ScheduledThreadPoolExecutor(1, Thread.ofPlatform().daemon().name("mock-compute-timer").factory());
        timer.setRemoveOnCancelPolicy(true);
        timer.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    public final class Execution {
        private final NoiseModel.Sample sample;
        private final long started = System.nanoTime();
        private boolean terminal;
        private final CompletableFuture<Outcome> completion = new CompletableFuture<>();
        private ScheduledFuture<?> scheduled;
        private Execution(NoiseModel.Sample sample) { this.sample = sample; }
        public CompletionStage<Outcome> completion() { return completion.minimalCompletionStage(); }
        public boolean cancel() { return finish(State.CANCELLED); }
        private boolean finish(State terminal) {
            synchronized (lifecycle) {
                if (this.terminal) return false;
                this.terminal = true;
                if (scheduled != null) scheduled.cancel(false);
                var outcome = new Outcome(terminal, sample, Math.max(0, System.nanoTime() - started));
                // Never invoke business callbacks on the timer or close caller thread.
                active.remove(this);
                notifications.execute(() -> completion.complete(outcome));
                if (closed && active.isEmpty()) notifications.shutdown();
                return true;
            }
        }
    }

    public Execution submit(NoiseModel.Sample sample) {
        Objects.requireNonNull(sample);
        long nanos = sample.planned().toNanos();
        if (nanos < 0) throw new IllegalArgumentException("Negative computation duration");
        synchronized (lifecycle) {
            if (closed) throw new IllegalStateException("Compute executor is closed");
            var execution = new Execution(sample);
            active.add(execution);
            try {
                execution.scheduled = timer.schedule(() -> execution.finish(State.COMPLETED), nanos, TimeUnit.NANOSECONDS);
            } catch (RuntimeException error) {
                execution.cancel(); throw error;
            }
            return execution;
        }
    }

    public int activeCount() { return active.size(); }

    @Override public void close() {
        synchronized (lifecycle) {
            if (closed) return;
            closed = true;
            timer.shutdownNow();
            List.copyOf(active).forEach(Execution::cancel);
            notifications.shutdown();
        }
    }
}

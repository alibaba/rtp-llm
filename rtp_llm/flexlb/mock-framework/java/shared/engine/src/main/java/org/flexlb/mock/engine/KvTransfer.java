package org.flexlb.mock.engine;

import java.util.*;
import java.util.concurrent.*;

/** Asynchronous zero-time KV transfer model. Resource use lasts until execution quiesces. */
public final class KvTransfer implements AutoCloseable {
    public enum State { COMPLETED, CANCELLED, FAILED }
    public record Outcome(State state, RuntimeException error) {}
    private final Object lock = new Object();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Set<Execution> active = new HashSet<>();
    private boolean closed;

    public final class Execution {
        private final RequestState.Use use;
        private final CompletableFuture<Outcome> completion = new CompletableFuture<>();
        private boolean cancelled, terminal;
        private Execution(RequestState.Use use) { this.use = use; }
        public CompletionStage<Outcome> completion() { return completion.minimalCompletionStage(); }
        public boolean cancel() {
            synchronized (lock) {
                if (terminal) return false;
                cancelled = true;
                return true;
            }
        }
        private void run() {
            State state;
            synchronized (lock) { terminal = true; state = cancelled ? State.CANCELLED : State.COMPLETED; }
            RuntimeException failure = null;
            try { use.close(); }
            catch (RuntimeException error) { state = State.FAILED; failure = error; }
            synchronized (lock) { active.remove(this); }
            completion.complete(new Outcome(state, failure));
        }
    }

    /** Ownership transfers only on a successful return. The zero-time model executes asynchronously. */
    public Execution submit(RequestState.Use use) {
        Objects.requireNonNull(use);
        synchronized (lock) {
            if (closed) throw new IllegalStateException("Transfer executor is closed");
            var execution = new Execution(use);
            active.add(execution);
            try { workers.execute(execution::run); }
            catch (RuntimeException error) { active.remove(execution); throw error; }
            return execution;
        }
    }
    @Override public void close() {
        synchronized (lock) {
            if (!closed) {
                closed = true;
                active.forEach(Execution::cancel);
                workers.shutdown();
            }
        }
        try {
            if (!workers.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KV transfer did not quiesce");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted closing KV transfer", error);
        }
    }
}

package org.flexlb.mock.engine;

import java.util.Objects;
import java.util.function.Consumer;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;

/** Mutated under the owning scheduling domain's lock; snapshots are immutable. */
public final class RequestState {
    public enum Phase { QUEUED, CAPACITY, LOADING, TRANSFER, RUNNING, OUTPUT, FORWARD, READY, TERMINAL }
    public record Snapshot(Phase phase, boolean terminating, int generated, int iterations,
                           long executionNanos) {}
    public interface Use extends AutoCloseable { @Override void close(); }
    private final long deadline;
    private final Consumer<RequestFailure> finished;
    private AutoCloseable resource;
    private Runnable publish;
    private int uses, generated, iterations;
    private long executionNanos;
    private Phase phase = Phase.QUEUED;
    private boolean terminating, delivered;
    private RequestFailure failure;

    public RequestState(long deadline, Consumer<RequestFailure> finished) {
        this.deadline = deadline; this.finished = Objects.requireNonNull(finished);
    }
    public boolean terminating() { return terminating; }
    public boolean expired() { return System.nanoTime() >= deadline; }
    public void phase(Phase phase) { if (!terminating) this.phase = Objects.requireNonNull(phase); }
    public Snapshot snapshot() { return new Snapshot(phase, terminating, generated, iterations, executionNanos); }
    public void attach(AutoCloseable resource, Runnable publish) {
        if (terminating || this.resource != null) throw new IllegalStateException("Request cannot acquire another resource owner");
        this.resource = Objects.requireNonNull(resource); this.publish = Objects.requireNonNull(publish);
    }
    public void transferred(int generated) {
        if (terminating || generated < this.generated) throw new IllegalStateException("Invalid transferred progress");
        this.generated = generated;
    }
    public Use forward() {
        if (terminating) throw new IllegalStateException("Terminating request cannot enter Forward");
        phase = Phase.FORWARD;
        return pin();
    }
    public Use pin() {
        if (terminating) throw new IllegalStateException("Terminating request cannot acquire a resource use");
        uses++;
        return new Use() {
            private boolean closed;
            @Override public void close() {
                if (closed) return;
                closed = true; uses--;
                releaseIfReady();
            }
        };
    }
    public void commit(int tokens, long elapsedNanos) {
        if (terminating || phase != Phase.FORWARD || tokens < 0 || elapsedNanos < 0)
            throw new IllegalStateException("Invalid Forward commit");
        generated = Math.addExact(generated, tokens); iterations++;
        executionNanos = Math.addExact(executionNanos, elapsedNanos);
    }
    public void finish(RequestFailure failure) {
        if (terminating) return;
        terminating = true;
        this.failure = failure == null && expired() ? new RequestFailure(TIMEOUT, "Request expired before commit") : failure;
        releaseIfReady();
    }
    private void releaseIfReady() {
        if (!terminating || delivered || uses != 0) return;
        delivered = true; phase = Phase.TERMINAL;
        try {
            if (resource != null && failure == null) publish.run();
        } catch (Exception error) { failedCleanup(error); }
        finally {
            try { if (resource != null) resource.close(); }
            catch (Exception error) { failedCleanup(error); }
            resource = null; publish = null;
        }
        finished.accept(failure);
    }
    private void failedCleanup(Exception error) {
        if (failure == null || failure.reason() != INTERNAL)
            failure = new RequestFailure(INTERNAL, "Request resource finalization failed", error);
        else failure.addSuppressed(error);
    }
}

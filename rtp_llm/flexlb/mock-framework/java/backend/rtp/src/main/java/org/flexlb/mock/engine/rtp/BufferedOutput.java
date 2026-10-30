package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import java.util.ArrayDeque;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB;

/** Delivery is independent of scheduling, KV ownership and the native P/D exchange. */
final class BufferedOutput {
    private final ArrayDeque<GenerateOutputsPB> pending = new ArrayDeque<>();
    private final Executor executor;
    private final long expires;
    private ServerCallStreamObserver<GenerateOutputsPB> response;
    private boolean attached, draining, terminal, delivered, cancelled;
    private Throwable failure;

    BufferedOutput(Executor executor, long retainMillis) {
        this.executor = executor;
        expires = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(retainMillis);
    }
    synchronized boolean expired() { return !attached && System.nanoTime() >= expires; }
    synchronized boolean finished() { return terminal; }
    synchronized boolean discardable() { return terminal && (delivered || cancelled); }
    void attach(ServerCallStreamObserver<GenerateOutputsPB> observer, Runnable onCancel) {
        synchronized (this) {
            if (attached) { observer.onError(Status.ALREADY_EXISTS.withDescription("Fetch already attached").asRuntimeException()); return; }
            if (expired()) { observer.onError(Status.NOT_FOUND.withDescription("Fetch output expired").asRuntimeException()); return; }
            attached = true;
            response = observer;
        }
        observer.setOnCancelHandler(() -> {
            synchronized (this) { cancelled = true; pending.clear(); }
            onCancel.run();
        });
        observer.setOnReadyHandler(this::schedule);
        if (observer.isCancelled()) {
            synchronized (this) { cancelled = true; pending.clear(); }
            onCancel.run();
        }
        schedule();
    }
    void add(GenerateOutputsPB value) {
        synchronized (this) { if (terminal) throw new IllegalStateException("Output after terminal"); if (!cancelled) pending.addLast(value); }
        schedule();
    }
    void finish(Throwable error) {
        synchronized (this) {
            if (terminal) return;
            terminal = true; failure = error;
            if (error != null) pending.clear();
        }
        schedule();
    }
    private synchronized void schedule() {
        if (draining || response == null || delivered || cancelled) return;
        draining = true;
        try { executor.execute(this::drain); }
        catch (java.util.concurrent.RejectedExecutionException closing) { drain(); }
        catch (RuntimeException error) { draining = false; throw error; }
    }
    private void drain() {
        try {
            while (true) {
                GenerateOutputsPB next;
                Throwable error;
                synchronized (this) {
                    if (cancelled || response.isCancelled()) { cancelled = true; pending.clear(); draining = false; return; }
                    if (!pending.isEmpty()) {
                        if (!response.isReady()) { draining = false; return; }
                        next = pending.removeFirst(); error = null;
                    } else {
                        if (!terminal) { draining = false; return; }
                        delivered = true; next = null; error = failure;
                    }
                }
                if (next != null) response.onNext(next);
                else { if (error == null) response.onCompleted(); else response.onError(error); return; }
            }
        } catch (RuntimeException disconnected) {
            synchronized (this) { cancelled = true; pending.clear(); draining = false; }
        }
    }
}

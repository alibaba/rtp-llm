package org.flexlb.mock.engine.service;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import java.nio.ByteOrder;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Constant-size synthetic output progress. Only the sender materializes a bounded token frame. */
final class GeneratedOutput {
    static final int MAX_FRAME_TOKENS = 4096;
    private final Executor executor;
    private final long expires;
    private final ScheduledExecutorService deadlines;
    private final Consumer<GeneratedOutput> onExpired;
    private ScheduledFuture<?> deliveryTimeout;
    private ServerCallStreamObserver<GenerateOutputsPB> response;
    private long requestId;
    private int tokenId, generated, sent;
    private AuxInfoPB metadata = AuxInfoPB.getDefaultInstance();
    private boolean allocateReady, allocateSent, loadReady, loadSent, generationFinished, finalSent;
    private boolean attached, draining, terminal, delivered, cancelled;
    private Throwable failure;

    GeneratedOutput(Executor executor, ScheduledExecutorService deadlines, long retainMillis, Consumer<GeneratedOutput> onExpired) {
        this.executor = executor;
        this.deadlines = deadlines;
        this.onExpired = onExpired;
        expires = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(retainMillis);
    }
    synchronized boolean expired() { return !attached && System.nanoTime() >= expires; }
    synchronized boolean finished() { return terminal; }
    synchronized boolean discardable() { return terminal && (delivered || cancelled); }

    void attach(ServerCallStreamObserver<GenerateOutputsPB> observer, Runnable onCancel) {
        synchronized (this) {
            if (attached) { observer.onError(Status.ALREADY_EXISTS.withDescription("Fetch already attached").asRuntimeException()); return; }
            if (expired()) { observer.onError(Status.NOT_FOUND.withDescription("Fetch output expired").asRuntimeException()); return; }
            attached = true; response = observer;
        }
        Runnable cancel = () -> {
            synchronized (this) { cancelled = true; cancelTimeout(); }
            onCancel.run();
        };
        observer.setOnCancelHandler(cancel);
        observer.setOnReadyHandler(this::schedule);
        if (observer.isCancelled()) cancel.run();
        schedule();
    }

    // D starts after P has already delivered its first token.
    synchronized void startAt(int length) { generated = sent = length; }

    void advance(long id, int token, int length, AuxInfoPB info, boolean finished) {
        synchronized (this) {
            if (terminal) throw new IllegalStateException("Progress after terminal");
            if (length < generated) throw new IllegalArgumentException("Output length moved backwards");
            requestId = id; tokenId = token; generated = length; metadata = info;
            generationFinished |= finished;
        }
        schedule();
    }

    /** P relays synthetic D frames by retaining only cumulative length and fixed metadata. */
    void add(GenerateOutputsPB value) {
        if (value.hasFlattenOutput()) {
            var output = value.getFlattenOutput();
            var info = output.getAuxInfo(0);
            int token;
            synchronized (this) { token = tokenId; }
            if (!output.getOutputIds().getInt32Data().isEmpty())
                token = output.getOutputIds().getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).getInt();
            advance(value.getRequestId(), token, info.getOutputLen(), info, output.getFinished(0));
        } else {
            synchronized (this) {
                if (terminal) throw new IllegalStateException("Acknowledgement after terminal");
                if (value.getSupportsPrefillCompletion()) allocateReady = true;
                else { requestId = value.getRequestId(); loadReady = true; }
            }
            schedule();
        }
    }

    void finish(Throwable error) {
        synchronized (this) {
            if (terminal) return;
            terminal = true; failure = error;
            if (!delivered && !cancelled) {
                try { deliveryTimeout = deadlines.schedule(this::expireDelivery, 10, TimeUnit.MINUTES); }
                catch (java.util.concurrent.RejectedExecutionException closing) {
                    if (failure == null) failure = Status.UNAVAILABLE.withDescription("Output transport closing").asRuntimeException();
                }
            }
        }
        schedule();
    }

    private void expireDelivery() {
        synchronized (this) {
            if (delivered || cancelled) return;
            failure = Status.DEADLINE_EXCEEDED.withDescription("Output delivery expired after computation completed").asRuntimeException();
        }
        // Registry cleanup is independent of a blocked transport writer. Do not cancel computation.
        onExpired.accept(this);
        schedule();
    }

    private void cancelTimeout() {
        if (deliveryTimeout != null) { deliveryTimeout.cancel(false); deliveryTimeout = null; }
    }

    private void schedule() {
        synchronized (this) {
            if (draining || response == null || delivered || cancelled) return;
            draining = true;
        }
        try { executor.execute(this::drain); }
        catch (java.util.concurrent.RejectedExecutionException closing) {
            // Node shutdown may race the final callback. Never send inline on its caller.
            Thread.startVirtualThread(this::drain);
        }
    }

    private void drain() {
        try {
            while (true) {
                int end = 0, width = 0, token = 0, control = 0;
                long id = 0;
                AuxInfoPB info = null;
                boolean last = false, complete = false;
                Throwable error = null;
                synchronized (this) {
                    if (cancelled || response.isCancelled()) { cancelled = true; draining = false; cancelTimeout(); return; }
                    if (failure != null) { error = failure; delivered = true; cancelTimeout(); }
                    else if (allocateReady && !allocateSent) control = 1;
                    else if (loadReady && !loadSent) { control = 2; id = requestId; }
                    else if (sent < generated || generationFinished && !finalSent) {
                        width = Math.min(MAX_FRAME_TOKENS, generated - sent);
                        end = sent + width; token = tokenId; id = requestId; info = metadata;
                        last = generationFinished && end == generated;
                    } else if (terminal) { complete = true; delivered = true; cancelTimeout(); }
                    else { draining = false; return; }
                    if (error == null && !complete && !response.isReady()) { draining = false; return; }
                }
                if (error != null) { response.onError(error); return; }
                if (complete) { response.onCompleted(); return; }
                GenerateOutputsPB value = switch (control) {
                    case 1 -> GenerateOutputsPB.newBuilder().setSupportsPrefillCompletion(true).build();
                    case 2 -> GenerateOutputsPB.newBuilder().setRequestId(id).setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build();
                    default -> RpcMessages.token(id, token, end, width, info, last);
                };
                response.onNext(value);
                synchronized (this) {
                    if (control == 1) allocateSent = true;
                    else if (control == 2) loadSent = true;
                    else { sent = end; finalSent |= last; }
                }
            }
        } catch (RuntimeException disconnected) {
            synchronized (this) { cancelled = true; draining = false; cancelTimeout(); }
        }
    }
}

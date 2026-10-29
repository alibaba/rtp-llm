package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.flexlb.mock.engine.rtp.transfer.*;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.mock.engine.*;

/** Mock-only token handoff. A source is a live computed request, never an LRU lookup. */
final class PrefillTransfer extends PrefillTransferGrpc.PrefillTransferImplBase implements AutoCloseable {
    private record Key(String client, long request) {}
    private static final class Entry {
        final TransferRequest request;
        final CompletableFuture<LocalPrefill.Result> ready = new CompletableFuture<>();
        final AtomicBoolean reader = new AtomicBoolean();
        LocalPrefill.Session session;
        volatile KvTransfer.Execution execution;
        Entry(TransferRequest request) { this.request = request; }
    }
    final class Source implements AutoCloseable {
        private final Entry entry;
        Source(Entry entry) { this.entry = entry; }
        CompletionStage<LocalPrefill.Result> ready() { return entry.ready.minimalCompletionStage(); }
        void preempt() { entry.session.preempt(); }
        void expire() { entry.session.expire(); }
        void acknowledged() { requests.transferred(entry.session); }
        @Override public void close() { cancel(entry); }
    }
    private final KvTransfer model = new KvTransfer();
    private boolean closed;
    private final LocalPrefill requests;
    private final ConcurrentMap<Key, Entry> active = new ConcurrentHashMap<>();

    PrefillTransfer(LocalPrefill requests) { this.requests = requests; }

    void validate(GenerateInputPB input) { requests.validate(input, true); }

    Source start(TransferRequest request, long deadline) {
        var key = key(request); requests.validate(request.getInput(), true); var entry = new Entry(request);
        synchronized (entry) {
            synchronized (active) {
                if (closed) throw Status.UNAVAILABLE.withDescription("Transfer service closed").asRuntimeException();
                if (active.putIfAbsent(key, entry) != null)
                    throw Status.ALREADY_EXISTS.withDescription("Prefill transfer already exists").asRuntimeException();
            }
            try {
                var session = requests.submit(request.getInput(), deadline, true); entry.session = session;
                session.termination.thenAccept(result -> {
                    active.remove(key, entry);
                    synchronized (entry) { entry.ready.complete(result); }
                });
                session.completion().thenAccept(result -> {
                    synchronized (entry) {
                        if (!result.status().isOk()) return; // Termination callback owns failure delivery.
                        if (!session.request.snapshot().terminating()) entry.ready.complete(result);
                    }
                });
                return new Source(entry);
            } catch (RuntimeException error) { active.remove(key, entry); throw error; }
        }
    }

    @Override public void read(TransferRequest request, StreamObserver<TransferMetadata> observer) {
        var response = (ServerCallStreamObserver<TransferMetadata>) observer; Entry entry;
        try {
            entry = active.get(key(request));
            if (entry == null) throw Status.NOT_FOUND.withDescription("No live Prefill source").asRuntimeException();
            synchronized (entry) {
                if (entry.session == null) throw Status.NOT_FOUND.withDescription("Prefill preparation failed").asRuntimeException();
                if (!entry.request.getInput().getTokenIdsList().equals(request.getInput().getTokenIdsList()))
                    throw Status.FAILED_PRECONDITION.withDescription("Prefill/Decode prompt differs").asRuntimeException();
                var original = entry.request.getInput().getGenerateConfig();
                var incoming = request.getInput().getGenerateConfig();
                if (original.getMinNewTokens() != incoming.getMinNewTokens() || original.getMaxNewTokens() != incoming.getMaxNewTokens()
                        || original.getIgnoreEos() != incoming.getIgnoreEos())
                    throw Status.FAILED_PRECONDITION.withDescription("Prefill/Decode generation limits differ").asRuntimeException();
                if (!entry.reader.compareAndSet(false, true))
                    throw Status.ALREADY_EXISTS.withDescription("Prefill transfer already has a reader").asRuntimeException();
            }
        } catch (StatusRuntimeException error) { response.onError(error); return; }
        response.setOnCancelHandler(() -> cancel(entry));
        entry.ready.thenAccept(result -> {
            if (response.isCancelled()) { entry.session.cancel(); return; }
            if (!result.status().isOk()) { response.onError(RtpErrors.failure(result.status(), result.engineCode())); return; }
            var metadata = TransferMetadata.newBuilder().setInputTokens(entry.request.getInput().getTokenIdsCount())
                .setPrefillFinished(result.output().getFlattenOutput().getFinished(0)).build();
            RequestState.Use use;
            try { use = requests.beginTransfer(entry.session); }
            catch (RuntimeException error) { cancel(entry); response.onError(error); return; }
            KvTransfer.Execution execution;
            try { execution = model.submit(use); }
            catch (RuntimeException error) { use.close(); cancel(entry); response.onError(error); return; }
            entry.execution = execution;
            if (response.isCancelled() || entry.session.request.snapshot().terminating()) execution.cancel();
            execution.completion().thenAccept(outcome -> {
                var failure = outcome.state() == KvTransfer.State.COMPLETED ? null : new RequestFailure(
                    outcome.state() == KvTransfer.State.CANCELLED ? RequestFailure.Reason.CANCELLED : RequestFailure.Reason.INTERNAL,
                    "KV transfer " + outcome.state(), outcome.error());
                try {
                    requests.transferCompleted(entry.session, new RtpTransferResult(metadata.getPrefillFinished(), failure));
                    if (response.isCancelled()) return;
                    if (failure != null) throw RtpErrors.failure(failure);
                    if (entry.session.request.snapshot().terminating()) throw Status.CANCELLED.asRuntimeException();
                    response.onNext(metadata); response.onCompleted();
                } catch (RuntimeException error) { cancel(entry); if (!response.isCancelled()) response.onError(error); }
            });
        });
    }

    private void cancel(Entry entry) {
        if (entry.session != null) entry.session.cancel();
        var execution = entry.execution;
        if (execution != null) execution.cancel();
    }
    @Override public void close() {
        java.util.List<Entry> entries;
        synchronized (active) {
            if (closed) return;
            closed = true;
            entries = java.util.List.copyOf(active.values());
        }
        for (var entry : entries) synchronized (entry) { cancel(entry); }
        model.close();
    }

    private static Key key(TransferRequest request) {
        if (request.getClientId().isBlank() || !request.hasInput())
            throw Status.INVALID_ARGUMENT.withDescription("Transfer requires client identity and native input").asRuntimeException();
        return new Key(request.getClientId(), request.getInput().getRequestId());
    }
    static long deadline(Context context) {
        if (context.getDeadline() == null) return Long.MAX_VALUE;
        try { return Math.addExact(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS)); }
        catch (ArithmeticException error) { return context.getDeadline().isExpired() ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
}

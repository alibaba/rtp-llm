package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.flexlb.mock.engine.rtp.transfer.*;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;

/** Mock-only P cache-store boundary. A source is a live computed request, never an LRU lookup. */
final class PrefillTransfer extends PrefillTransferGrpc.PrefillTransferImplBase {
    private record Key(String client, long request) {}
    private static final class Entry {
        final TransferRequest request;
        final TransferMetadata metadata;
        final CompletableFuture<LocalPrefill.Result> ready = new CompletableFuture<>();
        final AtomicBoolean reader = new AtomicBoolean();
        LocalPrefill.Session session;
        Entry(TransferRequest request, TransferMetadata metadata) { this.request = request; this.metadata = metadata; }
    }
    final class Source implements AutoCloseable {
        private final Entry entry;
        Source(Entry entry) { this.entry = entry; }
        int stopAfter() { return entry.metadata.getStopAfter(); }
        CompletionStage<LocalPrefill.Result> ready() { return entry.ready.minimalCompletionStage(); }
        void preempt() { entry.session.preempt(); }
        void expire() { entry.session.expire(); }
        void acknowledged() { requests.transferred(entry.session); }
        @Override public void close() { entry.session.cancel(); }
    }
    private final RtpConfig config;
    private final LocalPrefill requests;
    private final ConcurrentMap<Key, Entry> active = new ConcurrentHashMap<>();

    PrefillTransfer(RtpConfig config, LocalPrefill requests) { this.config = config; this.requests = requests; }

    void validate(GenerateInputPB input) { requests.validate(input, true); }

    Source start(TransferRequest request, long deadline) {
        var key = key(request); layout(request); requests.validate(request.getInput(), true); var entry = new Entry(request, metadata(request));
        synchronized (entry) {
            if (active.putIfAbsent(key, entry) != null)
                throw Status.ALREADY_EXISTS.withDescription("Prefill transfer already exists").asRuntimeException();
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
            layout(request); entry = active.get(key(request));
            if (entry == null) throw Status.NOT_FOUND.withDescription("No live Prefill source").asRuntimeException();
            synchronized (entry) {
                if (entry.session == null) throw Status.NOT_FOUND.withDescription("Prefill preparation failed").asRuntimeException();
                if (!entry.request.getInput().getTokenIdsList().equals(request.getInput().getTokenIdsList()))
                    throw Status.FAILED_PRECONDITION.withDescription("Prefill/Decode prompt differs").asRuntimeException();
                if (!entry.reader.compareAndSet(false, true))
                    throw Status.ALREADY_EXISTS.withDescription("Prefill transfer already has a reader").asRuntimeException();
            }
        } catch (StatusRuntimeException error) { response.onError(error); return; }
        response.setOnCancelHandler(entry.session::cancel);
        entry.ready.thenAccept(result -> {
            if (response.isCancelled()) { entry.session.cancel(); return; }
            if (!result.status().isOk()) { response.onError(RtpErrors.failure(result.status(), result.engineCode())); return; }
            try (var use = requests.readReady(entry.session)) {
                if (response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
                response.onNext(entry.metadata); response.onCompleted();
            } catch (RuntimeException error) { entry.session.cancel(); response.onError(error); }
        });
    }

    private void layout(TransferRequest request) {
        if (request.getBlockSize() != config.blockSize())
            throw Status.FAILED_PRECONDITION.withDescription("Prefill/Decode block size differs").asRuntimeException();
    }
    private static Key key(TransferRequest request) {
        if (request.getClientId().isBlank() || !request.hasInput())
            throw Status.INVALID_ARGUMENT.withDescription("Transfer requires client identity and native input").asRuntimeException();
        return new Key(request.getClientId(), request.getInput().getRequestId());
    }
    private TransferMetadata metadata(TransferRequest request) {
        return TransferMetadata.newBuilder().setBlockSize(config.blockSize()).setInputTokens(request.getInput().getTokenIdsCount())
            .addAllCacheKeys(PrefixKeys.completeBlocks(request.getInput(), config.blockSize()))
            .setStopAfter(config.stoppingModel().length(Long.toString(request.getInput().getRequestId()),
                request.getInput().getGenerateConfig().getMinNewTokens(), request.getInput().getGenerateConfig().getMaxNewTokens(),
                request.getInput().getGenerateConfig().getIgnoreEos())).build();
    }
    static long deadline(Context context) {
        if (context.getDeadline() == null) return Long.MAX_VALUE;
        try { return Math.addExact(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS)); }
        catch (ArithmeticException error) { return context.getDeadline().isExpired() ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
}

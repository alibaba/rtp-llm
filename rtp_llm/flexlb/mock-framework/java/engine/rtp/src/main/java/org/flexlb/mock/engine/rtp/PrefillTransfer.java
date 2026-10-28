package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.mock.engine.rtp.transfer.*;

/** Mock-only P cache-store boundary. A source is a live computed request, never a key lookup in LRU. */
final class PrefillTransfer extends PrefillTransferGrpc.PrefillTransferImplBase {
    private record Key(String client, long request) {}
    private static final class Entry {
        final TransferRequest request;
        final CompletableFuture<LocalPrefill.Result> ready = new CompletableFuture<>();
        final AtomicBoolean reader = new AtomicBoolean();
        volatile LocalPrefill.Session session;
        Entry(TransferRequest request) { this.request = request; }
    }
    private final RtpConfig config;
    private final LocalPrefill requests;
    private final ConcurrentMap<Key, Entry> active = new ConcurrentHashMap<>();

    PrefillTransfer(RtpConfig config, LocalPrefill requests) { this.config = config; this.requests = requests; }

    @Override public void prepare(TransferRequest request, StreamObserver<TransferMetadata> observer) {
        var response = (ServerCallStreamObserver<TransferMetadata>) observer;
        var context = Context.current();
        Key key;
        try {
            key = key(request);
            layout(request);
            if (context.getDeadline() == null && request.getInput().getGenerateConfig().getTimeoutMs() <= 0)
                throw Status.INVALID_ARGUMENT.withDescription("Prepare requires a request or RPC deadline for retained KV").asRuntimeException();
        } catch (StatusRuntimeException error) { response.onError(error); return; }
        var entry = new Entry(request);
        synchronized (entry) {
            if (active.putIfAbsent(key, entry) != null) {
                response.onError(Status.ALREADY_EXISTS.withDescription("Prefill transfer already exists").asRuntimeException()); return;
            }
            response.setOnCancelHandler(() -> {
                var session = entry.session;
                if (session != null) {
                    if (context.getDeadline() != null && context.getDeadline().isExpired()) session.expire();
                    else session.cancel();
                }
            });
            try {
                var session = requests.submit(request.getInput(), deadline(context), true); entry.session = session;
                session.termination.whenComplete((result, error) -> {
                    active.remove(key, entry);
                    entry.ready.complete(result);
                    synchronized (entry) {
                        if (response.isCancelled()) return;
                        if (result.status().isOk()) response.onCompleted();
                        else response.onError(RtpErrors.failure(result.status(), result.engineCode()));
                    }
                });
                session.completion().thenAccept(result -> {
                    synchronized (entry) {
                        if (response.isCancelled() || !result.status().isOk()) return;
                        try { requests.requireReady(session); response.onNext(metadata(request)); entry.ready.complete(result); }
                        catch (StatusRuntimeException ended) { /* Termination callback owns the terminal response. */ }
                        catch (RuntimeException error) { session.cancel(); }
                    }
                });
                if (response.isCancelled()) session.cancel();
            } catch (StatusRuntimeException error) { active.remove(key, entry); response.onError(error); }
        }
    }

    @Override public void read(TransferRequest request, StreamObserver<TransferMetadata> observer) {
        var response = (ServerCallStreamObserver<TransferMetadata>) observer;
        Entry entry;
        try {
            layout(request);
            entry = active.get(key(request));
            if (entry == null)
                throw Status.NOT_FOUND.withDescription("No live Prefill source for this request").asRuntimeException();
            synchronized (entry) { // Prepare publishes its session before this reader can inspect it.
                if (entry.session == null)
                    throw Status.NOT_FOUND.withDescription("Prefill preparation failed").asRuntimeException();
                if (!entry.request.getInput().getTokenIdsList().equals(request.getInput().getTokenIdsList()))
                    throw Status.FAILED_PRECONDITION.withDescription("Prefill/Decode prompt differs").asRuntimeException();
                if (!entry.reader.compareAndSet(false, true))
                    throw Status.ALREADY_EXISTS.withDescription("Prefill transfer already has a reader").asRuntimeException();
            }
        } catch (StatusRuntimeException error) { response.onError(error); return; }
        response.setOnCancelHandler(entry.session::cancel);
        var session = entry.session;
        // No waiting thread: the P Forward/deadline owner completes this future.
        entry.ready.thenAccept(result -> {
            if (response.isCancelled()) { session.cancel(); return; }
            if (!result.status().isOk()) { response.onError(RtpErrors.failure(result.status(), result.engineCode())); return; }
            try {
                requests.consume(session, () -> {
                    if (response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
                    response.onNext(metadata(entry.request)); response.onCompleted();
                });
            } catch (StatusRuntimeException error) { session.cancel(); response.onError(error); }
            catch (RuntimeException error) { session.cancel(); response.onError(Status.INTERNAL.withCause(error).asRuntimeException()); }
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
            .addAllCacheKeys(PrefixKeys.completeBlocks(request.getInput(), config.blockSize())).build();
    }
    private static long deadline(Context context) {
        if (context.getDeadline() == null) return Long.MAX_VALUE;
        try { return Math.addExact(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS)); }
        catch (ArithmeticException error) { return context.getDeadline().isExpired() ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
}

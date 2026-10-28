package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.rtp.transfer.TransferRequest;

/** P owns one end-to-end native request; the local Forward owner retains KV only through LOAD. */
final class PrefillGeneration implements AutoCloseable {
    private final RtpConfig config;
    private final PrefillTransfer transfer;
    private final RpcChannels channels = new RpcChannels();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-pd-deadline").factory());
    private final Map<Long, Call> active = new HashMap<>();
    private record Fence(long expires, boolean preempted) {}
    private static final long REGISTRY_TTL = TimeUnit.MINUTES.toNanos(10);
    private final Map<Long, Fence> fences = new LinkedHashMap<>();
    private String address;
    private boolean closed;

    PrefillGeneration(RtpConfig config, PrefillTransfer transfer) {
        this.config = config; this.transfer = transfer; deadlines.setRemoveOnCancelPolicy(true);
    }
    synchronized void address(String address) { this.address = address; }

    void generate(GenerateInputPB input, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        try { start(input, response, 0); }
        catch (StatusRuntimeException error) { response.onError(error); }
    }

    private Call start(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, long attachMillis) {
        DecodeAdmission.validateOutput(input);
        transfer.validate(input);
        if (input.getGenerateConfig().getMaxNewTokens() <= 1 || !input.getGenerateConfig().getCanUsePdSeparation())
            throw Status.UNIMPLEMENTED.withDescription("Batch and multi-token requests require P/D separation").asRuntimeException();
        var targets = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.DECODE).toList();
        if (targets.size() != 1 || targets.getFirst().getIp().isBlank() || targets.getFirst().getGrpcPort() <= 0 || targets.getFirst().getGrpcPort() > 65535)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Exactly one valid Decode role address is required"), 605);
        var target = targets.getFirst(); String host = target.getIp().contains(":") ? "[" + target.getIp() + "]" : target.getIp();
        Call call;
        synchronized (this) {
            if (closed) throw Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException();
            sweepFences();
            if (response == null && fences.containsKey(input.getRequestId()) && fences.get(input.getRequestId()).preempted()) throw preempted();
            if (active.containsKey(input.getRequestId())) throw Status.ALREADY_EXISTS.withDescription("Active Prefill request id").asRuntimeException();
            if (active.size() >= TaskHistory.LIMIT)
                throw Status.RESOURCE_EXHAUSTED.withDescription("Mock P/D request registry full").asRuntimeException();
            call = new Call(input, response, host + ":" + target.getGrpcPort(), attachMillis);
            active.put(input.getRequestId(), call);
            if (call.deferred) remember(input.getRequestId(), false);
        }
        synchronized (call) {
            if (call.terminal) return call;
            if (response != null) call.attach(response);
            if (call.terminal) return call;
            try {
                if (call.deadline != Long.MAX_VALUE) call.timeout = deadlines.schedule(() -> call.finish(
                    RtpErrors.failure(Status.DEADLINE_EXCEEDED.withDescription("P/D request budget expired"), 603)),
                    Math.max(0, call.deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                call.worker = workers.submit(call::run);
            } catch (RejectedExecutionException closing) { call.finish(Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException()); }
        }
        return call;
    }

    CompletableFuture<Void> enqueue(GenerateInputPB input, long attachMillis) {
        try { return start(input, null, attachMillis).admitted; }
        catch (RuntimeException error) { return CompletableFuture.failedFuture(error); }
    }

    void fetch(long id, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        Call call; Fence fence;
        synchronized (this) { call = active.get(id); fence = fences.get(id); }
        if (call == null || !call.deferred) {
            response.onError(fence != null && fence.preempted() && fence.expires() > System.nanoTime() ? preempted()
                : Status.NOT_FOUND.withDescription("No fetchable request").asRuntimeException());
            return;
        }
        synchronized (call) {
            if (call.terminal || !call.admitted.isDone() || call.admitted.isCompletedExceptionally()) {
                response.onError(call.priorityPreempted ? preempted() : Status.NOT_FOUND.withDescription("No fetchable request").asRuntimeException()); return;
            }
            if (call.response != null) { response.onError(Status.ALREADY_EXISTS.withDescription("Fetch already attached").asRuntimeException()); return; }
            call.attach(response);
        }
    }

    CancelStatusPB cancel(long id) {
        if (id <= 0) throw Status.INVALID_ARGUMENT.withDescription("Cancel requires a positive request_id").asRuntimeException();
        Call call;
        synchronized (this) {
            if (closed) return CancelStatusPB.CANCEL_STATUS_NOT_FOUND;
            sweepFences();
            call = active.get(id);
            if (call == null || !call.deferred) {
                var fence = fences.get(id);
                if (fence != null) return fence.preempted() ? CancelStatusPB.CANCEL_STATUS_TOMBSTONED : CancelStatusPB.CANCEL_STATUS_NOT_FOUND;
                if (call != null) return CancelStatusPB.CANCEL_STATUS_NOT_FOUND;
                remember(id, true);
                return CancelStatusPB.CANCEL_STATUS_TOMBSTONED;
            }
        }
        synchronized (call) {
            if (call.terminal) return CancelStatusPB.CANCEL_STATUS_NOT_FOUND;
            call.priorityPreempted = true;
            synchronized (this) { remember(id, true); }
            if (call.source != null) call.source.preempt();
            call.finish(preempted());
            return CancelStatusPB.CANCEL_STATUS_ACCEPTED;
        }
    }
    // Registry monitor held; refresh insertion order so expiry never scans live history.
    private void remember(long id, boolean preempted) {
        fences.remove(id);
        fences.put(id, new Fence(System.nanoTime() + REGISTRY_TTL, preempted));
    }
    private void sweepFences() {
        long now = System.nanoTime();
        var entries = fences.values().iterator();
        while (entries.hasNext()) {
            if (entries.next().expires() > now) break;
            entries.remove();
        }
    }
    private static StatusRuntimeException preempted() {
        return RtpErrors.failure(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429);
    }

    private final class Call {
        final GenerateInputPB input;
        ServerCallStreamObserver<GenerateOutputsPB> response;
        final boolean deferred;
        final long attachMillis;
        final CompletableFuture<Void> admitted = new CompletableFuture<>();
        final String target;
        final long started = System.nanoTime(), deadline;
        final Context.CancellableContext context;
        PrefillTransfer.Source source;
        Future<?> worker;
        ScheduledFuture<?> timeout, attachment;
        boolean terminal, priorityPreempted;
        Call(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, String target, long attachMillis) {
            this.input = input; this.target = target; this.deferred = response == null;
            this.attachMillis = attachMillis > 0 ? attachMillis : TimeUnit.MINUTES.toMillis(10);
            context = (deferred ? Context.ROOT : Context.current()).withCancellation();
            deadline = Math.min(deferred ? Long.MAX_VALUE : PrefillTransfer.deadline(Context.current()), input.getGenerateConfig().getTimeoutMs() <= 0 ? Long.MAX_VALUE
                : started + TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs()));
        }
        synchronized void attach(ServerCallStreamObserver<GenerateOutputsPB> response) {
            this.response = response;
            if (attachment != null) attachment.cancel(false);
            response.setOnCancelHandler(() -> finish(Status.CANCELLED.asRuntimeException()));
            response.setOnReadyHandler(() -> { synchronized (this) { notifyAll(); } });
            notifyAll();
            if (response.isCancelled()) finish(Status.CANCELLED.asRuntimeException());
        }
        void run() { context.run(this::exchange); }
        void exchange() {
            try {
                var inbox = new Inbox();
                var stub = RpcServiceGrpc.newStub(channels.get(target));
                if (deadline != Long.MAX_VALUE) stub = stub.withDeadlineAfter(Math.max(1, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                var stream = stub.remoteGenerate(inbox);
                var allocationInput = input.toBuilder().clearGroupId().clearGroupSize();
                allocationInput.getGenerateConfigBuilder().clearGroupTimeout();
                if (deadline != Long.MAX_VALUE) allocationInput.getGenerateConfigBuilder().setTimeoutMs((int) Math.min(Integer.MAX_VALUE,
                    Math.max(1, TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime()))));
                stream.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(input.getRequestId()).setClientId(config.id())
                    .setInput(allocationInput).addPeerAddrs(address).build());
                var allocated = inbox.next();
                if (allocated == null || allocated.hasFlattenOutput() || allocated.hasErrorInfo() && allocated.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid ALLOCATE acknowledgement").asRuntimeException();
                synchronized (this) {
                    if (terminal) return;
                    source = transfer.start(TransferRequest.newBuilder().setClientId(config.id()).setInput(input).setBlockSize(config.blockSize()).build(), deadline);
                    if (deferred) attachment = deadlines.schedule(() -> {
                        synchronized (this) {
                            if (response == null) finish(RtpErrors.failure(Status.DEADLINE_EXCEEDED.withDescription("Fetch attachment expired"), 603));
                        }
                    }, attachMillis, TimeUnit.MILLISECONDS);
                    admitted.complete(null);
                    while (!terminal && response == null) wait();
                    if (terminal) return;
                }
                stream.onNext(frame(RemoteStage.LOAD));
                var first = source.ready().toCompletableFuture().get();
                if (!first.status().isOk()) throw RtpErrors.failure(first.status(), first.engineCode());
                long ttft = System.nanoTime() - started;
                int prefillReuse = first.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen();
                send(LocalResponse.token(input, input.getTokenIds(input.getTokenIdsCount() - 1), 1, 1, prefillReuse, ttft, source.stopAfter() == 1, true));
                var loaded = inbox.next();
                if (loaded == null || !loaded.hasErrorInfo() || loaded.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid LOAD acknowledgement").asRuntimeException();
                if (source.stopAfter() == 1) stream.onNext(frame(RemoteStage.PREFILL_COMPLETE));
                else stream.onNext(frame(RemoteStage.GENERATE).toBuilder().setFirstGenerateTokenId(input.getTokenIds(input.getTokenIdsCount() - 1)).build());
                stream.onCompleted();
                boolean finished = source.stopAfter() == 1;
                for (GenerateOutputsPB output; (output = inbox.next()) != null;) {
                    if (finished || output.getRequestId() != input.getRequestId() || !output.hasFlattenOutput() || output.getFlattenOutput().getFinishedCount() != 1)
                        throw Status.DATA_LOSS.withDescription("Invalid Decode output stream").asRuntimeException();
                    var merged = output.toBuilder();
                    for (var aux : merged.getFlattenOutputBuilder().getAuxInfoBuilderList()) {
                        aux.setDecodeTotalReuseLen(aux.getTotalReuseLen()).setDecodeLocalReuseLen(aux.getLocalReuseLen())
                            .setPrefillTotalReuseLen(prefillReuse).setPrefillLocalReuseLen(prefillReuse)
                            .setTotalReuseLen(prefillReuse).setLocalReuseLen(prefillReuse).setPrefixLen(prefillReuse)
                            .setFirstTokenCostTimeUs((int) Math.min(Integer.MAX_VALUE, ttft / 1000)).setPdSep(true);
                    }
                    finished = output.getFlattenOutput().getFinished(0); send(merged.build());
                }
                if (!finished) throw Status.DATA_LOSS.withDescription("Decode ended without a final output").asRuntimeException();
                finish(null);
            } catch (InterruptedException error) { Thread.currentThread().interrupt(); finish(Status.CANCELLED.withCause(error).asRuntimeException()); }
            catch (ExecutionException error) { finish(Status.INTERNAL.withCause(error.getCause()).asRuntimeException()); }
            catch (RuntimeException error) { finish(error); }
        }
        GenerateRequestPB frame(RemoteStage stage) { return GenerateRequestPB.newBuilder().setStage(stage).setRequestId(input.getRequestId()).setClientId(config.id()).build(); }
        synchronized void send(GenerateOutputsPB output) throws InterruptedException {
            while (!terminal && !response.isCancelled() && !response.isReady()) wait();
            if (terminal || response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
            response.onNext(output);
        }
        synchronized void finish(Throwable error) {
            if (terminal) return;
            terminal = true; notifyAll();
            if (timeout != null) timeout.cancel(false);
            if (attachment != null) attachment.cancel(false);
            if (!admitted.isDone()) admitted.completeExceptionally(error == null ? Status.CANCELLED.asRuntimeException() : error);
            context.cancel(null);
            if (source != null) {
                if (error != null && Status.fromThrowable(error).getCode() == Status.Code.DEADLINE_EXCEEDED) source.expire();
                else source.close();
            }
            if (worker != null) worker.cancel(true);
            synchronized (PrefillGeneration.this) {
                active.remove(input.getRequestId(), this);
                if (deferred) {
                    var old = fences.get(input.getRequestId());
                    remember(input.getRequestId(), old != null && old.preempted());
                }
            }
            if (response != null && !response.isCancelled()) {
                if (error == null) response.onCompleted();
                else response.onError(error);
            }
        }
    }
    private static final class Inbox implements ClientResponseObserver<GenerateRequestPB, GenerateOutputsPB> {
        final BlockingQueue<Object> values = new LinkedBlockingQueue<>(2);
        final AtomicReference<ClientCallStreamObserver<GenerateRequestPB>> stream = new AtomicReference<>();
        @Override public void beforeStart(ClientCallStreamObserver<GenerateRequestPB> request) { stream.set(request); request.disableAutoRequestWithInitial(1); }
        @Override public void onNext(GenerateOutputsPB value) { values.add(value); }
        @Override public void onError(Throwable error) { values.add(error); }
        @Override public void onCompleted() { values.add(Boolean.TRUE); }
        GenerateOutputsPB next() throws InterruptedException {
            var value = values.take();
            if (value instanceof Throwable error) throw Status.fromThrowable(error).asRuntimeException(Status.trailersFromThrowable(error));
            if (value instanceof GenerateOutputsPB output) { stream.get().request(1); return output; }
            return null;
        }
    }
    @Override public void close() {
        List<Call> owned;
        synchronized (this) { closed = true; owned = List.copyOf(active.values()); }
        for (var call : owned) call.finish(Status.UNAVAILABLE.withDescription("Prefill node closed").asRuntimeException());
        workers.shutdownNow(); deadlines.shutdownNow(); channels.close();
    }
}

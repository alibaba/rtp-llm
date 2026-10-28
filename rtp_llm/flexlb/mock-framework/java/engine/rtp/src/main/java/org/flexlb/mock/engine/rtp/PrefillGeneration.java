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
    private String address;
    private boolean closed;

    PrefillGeneration(RtpConfig config, PrefillTransfer transfer) {
        this.config = config; this.transfer = transfer; deadlines.setRemoveOnCancelPolicy(true);
    }
    synchronized void address(String address) { this.address = address; }

    void generate(GenerateInputPB input, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        Call call;
        try {
            DecodeAdmission.validateOutput(input);
            transfer.validate(input);
            if (input.getGenerateConfig().getMaxNewTokens() <= 1 || !input.getGenerateConfig().getCanUsePdSeparation())
                throw Status.UNIMPLEMENTED.withDescription("Multi-token requests require can_use_pd_separation").asRuntimeException();
            var targets = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.DECODE).toList();
            if (targets.size() != 1 || targets.getFirst().getIp().isBlank() || targets.getFirst().getGrpcPort() <= 0 || targets.getFirst().getGrpcPort() > 65535)
                throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Exactly one valid Decode role address is required"), 605);
            var target = targets.getFirst(); String host = target.getIp().contains(":") ? "[" + target.getIp() + "]" : target.getIp();
            synchronized (this) {
                if (closed) throw Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException();
                if (active.containsKey(input.getRequestId())) throw Status.ALREADY_EXISTS.withDescription("Active Prefill request id").asRuntimeException();
                if (active.size() >= TaskHistory.LIMIT) throw Status.RESOURCE_EXHAUSTED.withDescription("Mock P/D request limit reached").asRuntimeException();
                call = new Call(input, response, host + ":" + target.getGrpcPort());
                active.put(input.getRequestId(), call);
            }
        } catch (StatusRuntimeException error) { response.onError(error); return; }
        synchronized (call) {
            if (call.terminal) return;
            response.setOnCancelHandler(() -> call.finish(Status.CANCELLED.asRuntimeException()));
            response.setOnReadyHandler(() -> { synchronized (call) { call.notifyAll(); } });
            try {
                if (call.deadline != Long.MAX_VALUE) call.timeout = deadlines.schedule(() -> call.finish(
                    RtpErrors.failure(Status.DEADLINE_EXCEEDED.withDescription("P/D request budget expired"), 603)),
                    Math.max(0, call.deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                call.worker = workers.submit(call::run);
                if (response.isCancelled()) call.finish(Status.CANCELLED.asRuntimeException());
            } catch (RejectedExecutionException closing) { call.finish(Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException()); }
        }
    }

    private final class Call {
        final GenerateInputPB input;
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final String target;
        final long started = System.nanoTime(), deadline;
        final Context.CancellableContext context = Context.current().withCancellation();
        PrefillTransfer.Source source;
        Future<?> worker;
        ScheduledFuture<?> timeout;
        boolean terminal;
        Call(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, String target) {
            this.input = input; this.response = response; this.target = target;
            deadline = Math.min(PrefillTransfer.deadline(Context.current()), input.getGenerateConfig().getTimeoutMs() <= 0 ? Long.MAX_VALUE
                : started + TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs()));
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
                }
                stream.onNext(frame(RemoteStage.LOAD));
                var first = source.ready().toCompletableFuture().get();
                if (!first.status().isOk()) throw RtpErrors.failure(first.status(), first.engineCode());
                long ttft = System.nanoTime() - started;
                int prefillReuse = first.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen();
                send(LocalResponse.token(input, input.getTokenIds(input.getTokenIdsCount() - 1), 1, prefillReuse, ttft, false, true));
                var loaded = inbox.next();
                if (loaded == null || !loaded.hasErrorInfo() || loaded.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid LOAD acknowledgement").asRuntimeException();
                stream.onNext(frame(RemoteStage.GENERATE).toBuilder().setFirstGenerateTokenId(input.getTokenIds(input.getTokenIdsCount() - 1)).build());
                stream.onCompleted();
                boolean finished = false;
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
            context.cancel(null);
            if (source != null) source.close();
            if (worker != null) worker.cancel(true);
            synchronized (PrefillGeneration.this) { active.remove(input.getRequestId(), this); }
            if (!response.isCancelled()) {
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

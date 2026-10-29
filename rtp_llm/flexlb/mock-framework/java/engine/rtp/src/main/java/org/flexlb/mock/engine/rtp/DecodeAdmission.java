package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.List;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.DeviceCache;
import org.flexlb.mock.engine.rtp.transfer.*;

/** Native ALLOCATE/LOAD/GENERATE protocol. Queue membership and progress stay behind Scheduler. */
final class DecodeAdmission implements AutoCloseable {
    private final RtpConfig config;
    private final EngineLoop engine;
    private final TaskHistory history = new TaskHistory();
    private final ExecutorService transport = Executors.newVirtualThreadPerTaskExecutor();
    private final RpcChannels channels = new RpcChannels();
    private boolean closed;

    DecodeAdmission(RtpConfig config, DeviceCache cache, Scheduler.Factory schedulers) {
        this.config = config;
        engine = new EngineLoop(this, schedulers.create(this, cache), config.compute(), config.role(), config.id());
    }

    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }

    private final class Call implements StreamObserver<GenerateRequestPB>, Scheduler.Listener {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final Object wire = new Object();
        final long started = System.nanoTime();
        final long rpcDeadline;
        GenerateInputPB input;
        String client;
        Scheduler.Request request;
        volatile boolean terminal;
        boolean acknowledged, loading, loaded, generating;
        List<String> peers;
        Context.CancellableContext source;
        int token, stopAfter;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE
                : RtpTasks.after(started, context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            response.setOnReadyHandler(() -> {
                synchronized (DecodeAdmission.this) {
                    if (!terminal && request != null) engine.dependencyCompleted(request, Scheduler.Signal.OUTPUT_WRITABLE);
                }
            });
            response.setOnCancelHandler(() -> {
                boolean expired = context.getDeadline() != null && context.getDeadline().isExpired();
                stop(this, expired ? Status.DEADLINE_EXCEEDED.withDescription("Decode RPC deadline expired")
                    : Status.CANCELLED.withDescription("Decode RPC cancelled"), expired ? 603 : 8100);
            });
        }
        @Override public void onNext(GenerateRequestPB frame) { accept(this, frame); }
        @Override public void onError(Throwable error) { stop(this, Status.CANCELLED.withDescription("Decode stream closed"), 8100); }
        @Override public void onCompleted() {
            synchronized (DecodeAdmission.this) {
                if (!generating) stop(this, Status.CANCELLED.withDescription("P closed before generation"), 8100);
            }
        }
        @Override public boolean writable() { return response.isReady(); }
        @Override public boolean cancelled() { return response.isCancelled(); }
        @Override public void allocated(Scheduler.Request request) { acknowledge(this); }
        @Override public void output(Scheduler.Request request, Scheduler.Output progress) {
            var output = LocalResponse.token(input, token, progress.generated(), progress.committed(), request.reused(),
                System.nanoTime() - started, progress.finished(), true).toBuilder();
            output.getFlattenOutputBuilder().getAuxInfoBuilder(0).setIterCount(progress.iterations() + 1);
            send(this, output.build());
        }
        @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
            history.add(RtpTasks.finished(request, failure));
            ended(this, RtpErrors.status(failure), RtpErrors.code(failure));
        }
    }

    private synchronized void accept(Call call, GenerateRequestPB frame) {
        if (call.terminal) return;
        try {
            if (closed) throw Status.UNAVAILABLE.withDescription("Decode node is closing").asRuntimeException();
            if (call.input == null) {
                validate(frame);
                call.input = frame.getInput();
                call.client = frame.getClientId();
                call.peers = List.copyOf(frame.getPeerAddrsList());
                call.request = engine.submit(RtpTasks.input(call.input, call.client + "/" + call.input.getRequestId(),
                    call.started, call.rpcDeadline, false, config.blockSize()), call);
                return;
            }
            if (frame.getRequestId() != call.input.getRequestId()
                    || !frame.getClientId().isEmpty() && !frame.getClientId().equals(call.client))
                throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate identity changed").asRuntimeException();
            if (frame.getStage() == RemoteStage.LOAD && call.acknowledged && !call.loading && !call.loaded) {
                startLoad(call);
            } else if (frame.getStage() == RemoteStage.PREFILL_COMPLETE && call.loaded && !call.generating) {
                if (call.stopAfter != 1 || !frame.getClientId().equals(call.client))
                    throw Status.FAILED_PRECONDITION.withDescription("Invalid Prefill-only completion").asRuntimeException();
                engine.dependencyCompleted(call.request, Scheduler.Signal.PREFILL_COMPLETE);
            } else if (frame.getStage() == RemoteStage.GENERATE && call.loaded && !call.generating) {
                if (frame.getProposeTokenIdsCount() > 0 || frame.getPositionIdsCount() > 0 || frame.hasProposeProbs() || frame.hasProposeHidden())
                    throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                call.generating = true;
                call.token = frame.getFirstGenerateTokenId();
                engine.dependencyCompleted(call.request, Scheduler.Signal.GENERATE);
            } else {
                throw Status.FAILED_PRECONDITION.withDescription("Expected ALLOCATE, LOAD, then GENERATE or PREFILL_COMPLETE").asRuntimeException();
            }
        } catch (RequestFailure failure) { stop(call, RtpErrors.status(failure), RtpErrors.code(failure)); }
        catch (StatusRuntimeException error) { stop(call, error.getStatus(), error.getStatus().getCode() == Status.Code.INVALID_ARGUMENT ? 605 : 0); }
        catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode frame failed: " + error.getMessage()), 0); }
    }
    private void validate(GenerateRequestPB request) {
        if (request.getStage() != RemoteStage.ALLOCATE || !request.hasInput() || request.getClientId().isBlank()
                || request.getInput().getRequestId() != request.getRequestId())
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid ALLOCATE frame or identity"), 605);
        var input = request.getInput(); var generation = input.getGenerateConfig();
        RtpRequestValidation.supported(input);
        if (generation.getPrefillOnly())
            throw Status.UNIMPLEMENTED.withDescription("Decode does not support prefill_only").asRuntimeException();
        if ((long) input.getTokenIdsCount() + generation.getMaxNewTokens() >= config.maxSeqLen())
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Mock requires input + output below max_seq_len"), 605);
        if (input.getTokenIdsCount() < 1 || input.getTokenIdsCount() >= config.maxSeqLen()
                || generation.getTimeoutMs() < 0 || generation.getMaxNewTokens() < 1 || generation.getMinNewTokens() < 0
                || generation.getMinNewTokens() > generation.getMaxNewTokens() || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid decode allocation settings"), 605);
    }

    private void acknowledge(Call call) {
        synchronized (this) {
            if (!live(call)) return;
            call.acknowledged = true;
        }
        try { send(call, GenerateOutputsPB.newBuilder().setSupportsPrefillCompletion(true).build()); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("ALLOCATE acknowledgement failed"), 8100); }
    }

    private void startLoad(Call call) {
        if (call.peers.size() != 1 || call.peers.getFirst().isBlank())
            throw Status.FAILED_PRECONDITION.withDescription("Mock LOAD requires one P transfer address").asRuntimeException();
        call.loading = true;
        var source = Context.current().withCancellation();
        call.source = source;
        long budget = Math.min(TimeUnit.MILLISECONDS.toNanos(config.loadTimeoutMs()), call.request.input().deadline() == Long.MAX_VALUE
            ? Long.MAX_VALUE : Math.max(1, call.request.input().deadline() - System.nanoTime()));
        transport.execute(() -> {
            try {
                source.run(() -> PrefillTransferGrpc.newStub(channels.get(call.peers.getFirst())).withDeadlineAfter(budget, TimeUnit.NANOSECONDS)
                    .read(TransferRequest.newBuilder().setClientId(call.client).setInput(call.input).setBlockSize(config.blockSize()).build(),
                        new StreamObserver<>() {
                            private TransferMetadata metadata;
                            @Override public void onNext(TransferMetadata value) { metadata = value; }
                            @Override public void onError(Throwable error) {
                                var status = Status.fromThrowable(error);
                                stop(call, status.augmentDescription("P-to-D LOAD failed"), status.getCode() == Status.Code.DEADLINE_EXCEEDED ? 603 : 8212);
                            }
                            @Override public void onCompleted() { loaded(call, metadata); }
                        }));
            } catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Cannot start KV load"), 8212); }
        });
    }

    private void loaded(Call call, TransferMetadata metadata) {
        synchronized (this) {
            if (!live(call)) return;
            var keys = PrefixKeys.completeBlocks(call.input, config.blockSize());
            if (metadata == null || metadata.getBlockSize() != config.blockSize()
                    || metadata.getStopAfter() < Math.max(1, call.input.getGenerateConfig().getMinNewTokens())
                    || metadata.getStopAfter() > call.input.getGenerateConfig().getMaxNewTokens()
                    || call.input.getGenerateConfig().getIgnoreEos() && metadata.getStopAfter() != call.input.getGenerateConfig().getMaxNewTokens()
                    || metadata.getInputTokens() != call.input.getTokenIdsCount() || !metadata.getCacheKeysList().equals(keys)) {
                stop(call, Status.FAILED_PRECONDITION.withDescription("P-to-D KV layout differs"), 8212);
                return;
            }
            try { engine.dependencyCompleted(call.request, new Scheduler.Loaded(metadata.getStopAfter())); }
            catch (RequestFailure failure) { stop(call, RtpErrors.status(failure), RtpErrors.code(failure)); return; }
            call.loading = false;
            call.loaded = true;
            call.stopAfter = metadata.getStopAfter();
            call.source.cancel(null);
            call.source = null;
        }
        try {
            send(call, GenerateOutputsPB.newBuilder().setRequestId(call.input.getRequestId())
                .setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build());
        } catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("LOAD acknowledgement failed"), 8100); }
    }

    private boolean live(Call call) {
        if (call.terminal || call.request.snapshot().terminating()) return false;
        if (System.nanoTime() >= call.request.input().deadline()) {
            stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode request expired"), 603);
            return false;
        }
        if (call.response.isCancelled()) {
            stop(call, Status.CANCELLED.withDescription("Decode RPC cancelled"), 8100);
            return false;
        }
        return true;
    }

    private void send(Call call, GenerateOutputsPB output) {
        synchronized (call.wire) {
            if (call.terminal || call.response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
            call.response.onNext(output);
        }
    }

    private synchronized void stop(Call call, Status status, long code) {
        if (call.terminal) return;
        if (call.request == null) ended(call, status, code);
        else engine.finish(call.request, RtpErrors.scheduling(status, code));
    }
    private void ended(Call call, Status status, long code) {
        synchronized (this) {
            if (call.terminal) return;
            call.terminal = true;
            if (call.source != null) { call.source.cancel(null); call.source = null; }
        }
        Runnable reply = () -> {
            synchronized (call.wire) {
                if (!call.response.isCancelled()) {
                    if (status.isOk()) call.response.onCompleted();
                    else call.response.onError(RtpErrors.failure(status, code));
                }
            }
        };
        try { transport.execute(reply); }
        catch (RejectedExecutionException closing) { reply.run(); }
    }

    synchronized TaskHistory.Snapshot snapshot(long latest) {
        var counts = engine.queues();
        return history.snapshot(latest, engine.requests().stream().map(r -> RtpTasks.task(r).build()).toList(),
            counts.waiting() + counts.loading() + counts.ready(), counts.running());
    }
    String scrapeMetrics() { return engine.scrapeMetrics(); }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            engine.close();
        }
        channels.close();
        transport.shutdown();
    }
}

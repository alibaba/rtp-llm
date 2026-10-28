package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.engine.rtp.transfer.*;

/** RemoteGenerate adapter. LOAD readiness requires data from a live, computed P source. */
final class DecodeAdmission implements AutoCloseable {
    private record Key(String client, long request) {}
    private final RtpConfig config;
    private final RtpScheduler scheduler;
    private final TaskHistory history = new TaskHistory();
    private final ExecutorService callbacks = Executors.newVirtualThreadPerTaskExecutor();
    private final RpcChannels channels = new RpcChannels();
    private boolean closed;

    DecodeAdmission(RtpConfig config, DeviceCache cache) {
        this.config = config; scheduler = new RtpScheduler(this, config, cache);
    }
    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }
    final class Call implements StreamObserver<GenerateRequestPB>, RtpScheduler.Listener {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final long started = System.nanoTime();
        final long rpcDeadline;
        Key key;
        GenerateInputPB input;
        RtpScheduler.Request request;
        boolean terminal, acknowledged, loading, loaded, generating;
        List<String> peers;
        Context.CancellableContext source;
        int token, stopAfter;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE
                : RtpTasks.after(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            response.setOnReadyHandler(() -> resume(this));
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
        @Override public void allocated(RtpScheduler.Request request) { callbacks.execute(() -> acknowledge(this)); }
        @Override public void output(RtpScheduler.Request request, int committed, boolean finished) {
            if (response.isCancelled()) throw RtpErrors.failure(Status.CANCELLED.withDescription("Decode cancelled"), 8100);
            var progress = request.snapshot();
            var output = LocalResponse.token(input, token, progress.generated(), committed, request.reused,
                System.nanoTime() - started, finished, true).toBuilder();
            output.getFlattenOutputBuilder().getAuxInfoBuilder(0).setIterCount(progress.iterations() + 1);
            response.onNext(output.build());
        }
        @Override public void terminal(RtpScheduler.Request request, RequestFailure failure) {
            history.add(RtpTasks.finished(request, failure));
            ended(this, RtpErrors.status(failure), RtpErrors.code(failure));
        }
    }
    private synchronized void accept(Call call, GenerateRequestPB frame) {
        if (call.terminal) return;
        try {
            if (closed) throw Status.UNAVAILABLE.withDescription("Decode node is closing").asRuntimeException();
            if (call.input != null) {
                if (frame.getRequestId() != call.key.request())
                    throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate identity changed").asRuntimeException();
                if (!frame.getClientId().isEmpty() && !frame.getClientId().equals(call.key.client()))
                    throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate client identity changed").asRuntimeException();
                if (frame.getStage() == RemoteStage.LOAD && call.acknowledged && !call.loading && !call.loaded) {
                    load(call); return;
                }
                if (call.loaded && !call.generating && frame.getStage() == RemoteStage.PREFILL_COMPLETE) {
                    if (call.stopAfter != 1 || !frame.getClientId().equals(call.key.client()))
                        throw Status.FAILED_PRECONDITION.withDescription("Invalid Prefill-only completion").asRuntimeException();
                    call.request.state.transferred(1); stop(call, Status.OK, 0); return;
                }
                if (call.loaded && !call.generating && frame.getStage() == RemoteStage.GENERATE) {
                    if (frame.getProposeTokenIdsCount() > 0 || frame.getPositionIdsCount() > 0 || frame.hasProposeProbs() || frame.hasProposeHidden())
                        throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                    call.generating = true; call.token = frame.getFirstGenerateTokenId();
                    scheduler.generate(call.request, call.stopAfter); return;
                }
                throw Status.FAILED_PRECONDITION.withDescription("Expected LOAD after ALLOCATE acknowledgement").asRuntimeException();
            }
            validate(frame);
            call.input = frame.getInput(); call.key = new Key(frame.getClientId(), frame.getRequestId()); call.peers = frame.getPeerAddrsList();
            call.request = scheduler.submit(RtpTasks.input(call.input, call.key.client() + "/" + call.key.request(),
                call.started, call.rpcDeadline, false, config.blockSize()), call);
            if (call.response.isCancelled()) stop(call, Status.CANCELLED.withDescription("Decode cancelled before admission"), 8100);
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

    private synchronized void acknowledge(Call call) {
        if (call.terminal) return;
        if (System.nanoTime() >= call.request.input.deadline()) {
            stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode expired before ALLOCATE acknowledgement"), 603); return;
        }
        if (call.response.isCancelled()) { stop(call, Status.CANCELLED.withDescription("Decode RPC cancelled"), 8100); return; }
        try { call.acknowledged = true; call.response.onNext(GenerateOutputsPB.newBuilder().setSupportsPrefillCompletion(true).build()); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("Decode acknowledgement failed"), 8100); }
    }

    private void load(Call call) {
        if (call.peers.size() != 1 || call.peers.getFirst().isBlank())
            throw Status.FAILED_PRECONDITION.withDescription("Mock LOAD requires one P transfer address in ALLOCATE.peer_addrs").asRuntimeException();
        call.loading = true;
        call.source = Context.current().withCancellation();
        long budget = Math.min(TimeUnit.MILLISECONDS.toNanos(config.loadTimeoutMs()), call.request.input.deadline() == Long.MAX_VALUE
            ? Long.MAX_VALUE : Math.max(1, call.request.input.deadline() - System.nanoTime()));
        call.source.run(() -> PrefillTransferGrpc.newStub(channels.get(call.peers.getFirst())).withDeadlineAfter(budget, TimeUnit.NANOSECONDS).read(
            TransferRequest.newBuilder().setClientId(call.key.client()).setInput(call.input).setBlockSize(config.blockSize()).build(), new StreamObserver<>() {
                private TransferMetadata metadata;
                @Override public void onNext(TransferMetadata value) { metadata = value; }
                @Override public void onError(Throwable error) {
                    var status = Status.fromThrowable(error);
                    stop(call, status.augmentDescription("P-to-D LOAD failed"), status.getCode() == Status.Code.DEADLINE_EXCEEDED ? 603 : 8212);
                }
                @Override public void onCompleted() { loaded(call, metadata); }
            }));
    }

    private synchronized void loaded(Call call, TransferMetadata metadata) {
        if (call.terminal) return;
        if (System.nanoTime() >= call.request.input.deadline()) { stop(call, Status.DEADLINE_EXCEEDED.withDescription("LOAD exceeded request budget"), 603); return; }
        if (call.response.isCancelled()) { stop(call, Status.CANCELLED.withDescription("LOAD cancelled"), 8100); return; }
        var keys = PrefixKeys.completeBlocks(call.input, config.blockSize());
        if (metadata == null || metadata.getBlockSize() != config.blockSize()
                || metadata.getStopAfter() < Math.max(1, call.input.getGenerateConfig().getMinNewTokens())
                || metadata.getStopAfter() > call.input.getGenerateConfig().getMaxNewTokens()
                || call.input.getGenerateConfig().getIgnoreEos() && metadata.getStopAfter() != call.input.getGenerateConfig().getMaxNewTokens()
                || metadata.getInputTokens() != call.input.getTokenIdsCount() || !metadata.getCacheKeysList().equals(keys)) {
            stop(call, Status.FAILED_PRECONDITION.withDescription("P-to-D KV layout differs"), 8212); return;
        }
        call.loading = false; call.loaded = true; call.stopAfter = metadata.getStopAfter();
        call.source.cancel(null); call.source = null;
        try {
            // The transferred prefix is not a historical D cache hit. Keep the lease private until terminal cleanup.
            call.response.onNext(GenerateOutputsPB.newBuilder().setRequestId(call.key.request())
                .setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build());
        } catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("LOAD acknowledgement failed"), 8100); }
    }

    private synchronized void resume(Call call) {
        if (!call.terminal && call.request != null) scheduler.resume(call.request);
    }
    private synchronized void stop(Call call, Status status, long code) {
        if (call.terminal) return;
        if (call.request == null) ended(call, status, code);
        else scheduler.finish(call.request, RtpErrors.scheduling(status, code));
    }
    private void ended(Call call, Status status, long code) {
        if (call.terminal) return;
        call.terminal = true;
        if (call.source != null) { call.source.cancel(null); call.source = null; }
        Runnable reply = () -> {
            if (!call.response.isCancelled()) {
                if (status.isOk()) call.response.onCompleted();
                else call.response.onError(RtpErrors.failure(status, code));
            }
        };
        try { callbacks.execute(reply); }
        catch (RejectedExecutionException closing) { reply.run(); }
    }
    synchronized TaskHistory.Snapshot snapshot(long latest) {
        var counts = scheduler.queues();
        return history.snapshot(latest, scheduler.requests().stream().map(r -> RtpTasks.task(r).build()).toList(),
            counts.waiting(), counts.running());
    }
    String scrapeMetrics() { return scheduler.scrapeMetrics(); }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true; scheduler.close();
        }
        channels.close(); callbacks.shutdown();
    }
}

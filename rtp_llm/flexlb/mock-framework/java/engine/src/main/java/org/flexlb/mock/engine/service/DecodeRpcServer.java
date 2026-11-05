package org.flexlb.mock.engine.service;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.execution.RtpEngineLoop;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpControl;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpTransferResult;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.EngineScheduler;

/** Native ALLOCATE/LOAD/GENERATE protocol. Queue membership and progress stay behind EngineScheduler. */
final class DecodeRpcServer implements AutoCloseable {
    private final RtpRpcService.Config config;
    private final RtpEngineLoop engine;
    private final Object executionLock;
    private final Runnable recordCancellation;
    private final ExecutorService transport = Executors.newVirtualThreadPerTaskExecutor();
    private final SimulatedKvLoad transfer = new SimulatedKvLoad();
    private enum Stage { ALLOCATING, ALLOCATED, LOADING, LOADED, GENERATING, FINISHING }
    private boolean closed;

    DecodeRpcServer(RtpRpcService.Config config, Object executionLock, RtpEngineLoop engine, Runnable recordCancellation) {
        this.config = config;
        this.executionLock = executionLock;
        this.engine = engine;
        this.recordCancellation = recordCancellation;
    }

    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }

    private final class Call implements StreamObserver<GenerateRequestPB>, EngineScheduler.Listener {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final GeneratedOutput delivery;
        final long started = System.nanoTime();
        final long rpcDeadline;
        GenerateInputPB input;
        String client;
        EngineScheduler.Request request;
        volatile boolean terminal;
        Stage stage = Stage.ALLOCATING;
        SimulatedKvLoad.Execution transfer;
        int token;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE
                : RpcMessages.after(started, context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            delivery = new GeneratedOutput(transport, TimeUnit.MINUTES.toMillis(10));
            delivery.attach(response, () -> {
                boolean expired = context.getDeadline() != null && context.getDeadline().isExpired();
                disconnected(this, expired ? Status.DEADLINE_EXCEEDED.withDescription("Decode RPC deadline expired")
                    : Status.CANCELLED.withDescription("Decode RPC cancelled"), expired ? 603 : 8100);
            });
        }
        @Override public void onNext(GenerateRequestPB frame) { accept(this, frame); }
        @Override public void onError(Throwable error) { disconnected(this, Status.CANCELLED.withDescription("Decode stream closed"), 8100); }
        @Override public void onCompleted() {
            synchronized (executionLock) {
                if (stage != Stage.GENERATING && stage != Stage.FINISHING) stop(this, Status.CANCELLED.withDescription("P closed before generation"), 8100);
            }
        }
        @Override public void allocated(EngineScheduler.Request request) { acknowledge(this); }
        @Override public void progressed(EngineScheduler.Request request) {
            var progress = request.progress();
            int elapsedUs = (int) Math.min(Integer.MAX_VALUE, (System.nanoTime() - started) / 1000);
            var info = AuxInfoPB.newBuilder().setInputLen(input.getTokenIdsCount())
                .setIterCount(progress.iterations() + 1).setPrefixLen(request.reused()).setTotalReuseLen(request.reused())
                .setLocalReuseLen(request.reused() - request.remoteReused()).setRemoteReuseLen(request.remoteReused()).setPdSep(true)
                .setCostTimeUs(elapsedUs).setFirstTokenCostTimeUs(elapsedUs);
            delivery.advance(input.getRequestId(), token, progress.generated(), info.build(), progress.finished());
        }
        @Override public void terminal(EngineScheduler.Request request, RequestFailure failure) {
            ended(this, RpcErrors.status(failure), RpcErrors.code(failure));
        }
    }

    private void accept(Call call, GenerateRequestPB frame) {
        synchronized (executionLock) {
            if (call.terminal) return;
            try {
                if (closed) throw Status.UNAVAILABLE.withDescription("Decode node is closing").asRuntimeException();
                if (call.input == null) {
                    validate(frame);
                    call.input = frame.getInput();
                    call.client = frame.getClientId();
                    call.request = engine.input(RpcMessages.input(new org.flexlb.mock.protocol.MediaInput.Prepared(call.input, frame.getMultimodalCacheSpansList()), call.client + "/" + call.input.getRequestId(),
                        call.started, RpcMessages.deadline(call.input, call.started, call.rpcDeadline), false, config.blockSize(), config.cacheKeySeed()), call);
                    return;
                }
                if (frame.getRequestId() != call.input.getRequestId()
                        || !frame.getClientId().isEmpty() && !frame.getClientId().equals(call.client))
                    throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate identity changed").asRuntimeException();
                if (frame.getStage() == RemoteStage.LOAD && call.stage == Stage.ALLOCATED) {
                    startLoad(call);
                } else if (frame.getStage() == RemoteStage.PREFILL_COMPLETE && call.stage == Stage.LOADED) {
                    if (!frame.getClientId().equals(call.client) || call.input.getGenerateConfig().getMinNewTokens() > 1
                            || call.input.getGenerateConfig().getIgnoreEos() && call.input.getGenerateConfig().getMaxNewTokens() != 1)
                        throw Status.FAILED_PRECONDITION.withDescription("Invalid Prefill-only completion").asRuntimeException();
                    call.stage = Stage.FINISHING;
                    engine.control(call.request, RtpControl.PREFILL_COMPLETE);
                } else if (frame.getStage() == RemoteStage.GENERATE && call.stage == Stage.LOADED) {
                    if (frame.hasPrefillKvHandoff() && (!frame.getClientId().equals(call.client)
                            || frame.getPrefillKvHandoff().getComputedKvEnd() != call.input.getTokenIdsCount()))
                        throw Status.INVALID_ARGUMENT.withDescription("Prefill KV handoff does not match loaded prompt").asRuntimeException();
                    if (frame.getProposeTokenIdsCount() > 0 || frame.getPositionIdsCount() > 0 || frame.hasProposeProbs() || frame.hasProposeHidden())
                        throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                    if (call.input.getGenerateConfig().getMaxNewTokens() == 1)
                        throw Status.FAILED_PRECONDITION.withDescription("Single-token request is already complete at Prefill").asRuntimeException();
                    call.stage = Stage.GENERATING;
                    call.token = frame.getFirstGenerateTokenId();
                    call.delivery.startAt(1);
                    engine.control(call.request, new RtpControl.Generate(call.token));
                } else {
                    throw Status.FAILED_PRECONDITION.withDescription("Expected ALLOCATE, LOAD, then GENERATE or PREFILL_COMPLETE").asRuntimeException();
                }
            } catch (RequestFailure failure) {
                stop(call, failure.reason() == RequestFailure.Reason.INVALID ? Status.INVALID_ARGUMENT.withDescription(failure.getMessage()) : RpcErrors.status(failure),
                    failure.reason() == RequestFailure.Reason.INVALID ? 605 : RpcErrors.code(failure));
            }
            catch (StatusRuntimeException error) { stop(call, error.getStatus(), error.getStatus().getCode() == Status.Code.INVALID_ARGUMENT ? 605 : 0); }
            catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode frame failed: " + error.getMessage()), 0); }
        }
    }
    private void validate(GenerateRequestPB request) {
        if (request.getStage() != RemoteStage.ALLOCATE || !request.hasInput() || request.getClientId().isBlank()
                || request.getInput().getRequestId() != request.getRequestId())
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid ALLOCATE frame or identity"), 605);
        var input = request.getInput(); var generation = input.getGenerateConfig();
        RpcMessages.supported(input);
        if (org.flexlb.mock.protocol.MediaInput.present(input))
            throw Status.FAILED_PRECONDITION.withDescription("Decode ALLOCATE requires the prepared P input").asRuntimeException();
        if (input.getTokenIdsCount() < 1
                || generation.getTimeoutMs() < 0 || generation.getMaxNewTokens() < 1 || generation.getMinNewTokens() < 0
                || generation.getMinNewTokens() > generation.getMaxNewTokens() || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid decode allocation settings"), 605);
    }

    private void acknowledge(Call call) {
        synchronized (executionLock) {
            if (!live(call)) return;
            call.stage = Stage.ALLOCATED;
        }
        try { send(call, GenerateOutputsPB.newBuilder().setSupportsPrefillCompletion(true).build()); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("ALLOCATE acknowledgement failed"), 8100); }
    }

    private void startLoad(Call call) {
        var use = engine.beginTransfer(call.request);
        call.stage = Stage.LOADING;
        try {
            call.transfer = transfer.submit(use);
            call.transfer.completion().thenAccept(outcome -> loaded(call, outcome));
        } catch (RuntimeException error) { use.close(); throw error; }
    }

    private void loaded(Call call, SimulatedKvLoad.Outcome outcome) {
        synchronized (executionLock) {
            if (!live(call)) return;
            var failure = outcome.state() == SimulatedKvLoad.State.COMPLETED ? null : new RequestFailure(
                outcome.state() == SimulatedKvLoad.State.CANCELLED ? RequestFailure.Reason.CANCELLED : RequestFailure.Reason.INTERNAL,
                "KV transfer " + outcome.state(), outcome.error());
            try {
                engine.transferCompleted(call.request, new RtpTransferResult(false, failure));
                if (failure != null) return;
                call.stage = Stage.LOADED;
                send(call, GenerateOutputsPB.newBuilder().setRequestId(call.input.getRequestId())
                    .setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build());
            } catch (RequestFailure error) { stop(call, RpcErrors.status(error), RpcErrors.code(error)); }
            catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("LOAD acknowledgement failed"), 8100); }
        }
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
        synchronized (executionLock) {
            if (call.terminal || call.response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
            call.delivery.add(output);
        }
    }

    private void disconnected(Call call, Status status, long code) {
        synchronized (executionLock) {
            // Once GENERATE is accepted, D computes independently of its P output consumer.
            // Scheduler deadlines, explicit engine cancellation and node shutdown still own termination.
            if (call.stage == Stage.GENERATING || call.stage == Stage.FINISHING) return;
            stop(call, status, code);
        }
    }

    private void stop(Call call, Status status, long code) {
        synchronized (executionLock) {
            if (call.terminal) return;
            if (call.request == null) ended(call, status, code);
            else engine.finish(call.request, RpcErrors.scheduling(status, code));
        }
    }
    private void ended(Call call, Status status, long code) {
        synchronized (executionLock) {
            if (call.terminal) return;
            call.terminal = true;
            if (status.getCode() == Status.Code.CANCELLED) recordCancellation.run();
            if (call.transfer != null) call.transfer.cancel();
            call.delivery.finish(status.isOk() ? null : RpcErrors.failure(status, code));
        }
    }

    EngineScheduler.Status snapshot(long latestFinished) { return engine.status(latestFinished); }
    void checkHealthy() { engine.checkHealthy(); }
    @Override public void close() {
        synchronized (executionLock) {
            if (closed) return;
            closed = true;
            engine.close();
        }
        transfer.close();
        transport.shutdown();
    }
}

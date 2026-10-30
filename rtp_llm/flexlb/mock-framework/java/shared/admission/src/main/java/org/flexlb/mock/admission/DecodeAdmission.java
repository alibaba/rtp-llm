package org.flexlb.mock.admission;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.KvCache;

/** Native ALLOCATE/LOAD/GENERATE protocol. Queue membership and progress stay behind Scheduler. */
final class DecodeAdmission implements AutoCloseable {
    private final AdmissionConfig config;
    private final EngineLoop engine;
    private final ExecutorService transport = Executors.newVirtualThreadPerTaskExecutor();
    private final KvTransfer transfer = new KvTransfer();
    private enum Stage { ALLOCATING, ALLOCATED, LOADING, LOADED, GENERATING, FINISHING }
    private boolean closed;

    DecodeAdmission(AdmissionConfig config, KvCache cache, Scheduler.Factory schedulers) {
        this.config = config;
        engine = new EngineLoop(this, schedulers.create(this, cache), cache, config.compute(), config.role(), config.id());
    }

    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }

    private final class Call implements StreamObserver<GenerateRequestPB>, Scheduler.Listener {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final BufferedOutput delivery;
        final long started = System.nanoTime();
        final long rpcDeadline;
        GenerateInputPB input;
        String client;
        Scheduler.Request request;
        volatile boolean terminal;
        Stage stage = Stage.ALLOCATING;
        KvTransfer.Execution transfer;
        int token;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE
                : RpcTasks.after(started, context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            delivery = new BufferedOutput(transport, TimeUnit.MINUTES.toMillis(10));
            delivery.attach(response, () -> {
                boolean expired = context.getDeadline() != null && context.getDeadline().isExpired();
                stop(this, expired ? Status.DEADLINE_EXCEEDED.withDescription("Decode RPC deadline expired")
                    : Status.CANCELLED.withDescription("Decode RPC cancelled"), expired ? 603 : 8100);
            });
        }
        @Override public void onNext(GenerateRequestPB frame) { accept(this, frame); }
        @Override public void onError(Throwable error) { stop(this, Status.CANCELLED.withDescription("Decode stream closed"), 8100); }
        @Override public void onCompleted() {
            synchronized (DecodeAdmission.this) {
                if (stage != Stage.GENERATING && stage != Stage.FINISHING) stop(this, Status.CANCELLED.withDescription("P closed before generation"), 8100);
            }
        }
        @Override public boolean cancelled() { return response.isCancelled(); }
        @Override public void allocated(Scheduler.Request request) { acknowledge(this); }
        @Override public void output(Scheduler.Request request, Scheduler.Output progress) {
            var output = LocalResponse.token(input, token, progress.generated(), progress.committed(), request.reused(),
                System.nanoTime() - started, progress.finished(), true).toBuilder();
            output.getFlattenOutputBuilder().getAuxInfoBuilder(0).setIterCount(progress.iterations() + 1);
            send(this, output.build());
        }
        @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
            ended(this, RpcErrors.status(failure), RpcErrors.code(failure));
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
                call.request = engine.input(RpcTasks.input(call.input, call.client + "/" + call.input.getRequestId(),
                    call.started, call.rpcDeadline, false, config.blockSize()), call);
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
                engine.control(call.request, Control.PREFILL_COMPLETE);
            } else if (frame.getStage() == RemoteStage.GENERATE && call.stage == Stage.LOADED) {
                if (frame.getProposeTokenIdsCount() > 0 || frame.getPositionIdsCount() > 0 || frame.hasProposeProbs() || frame.hasProposeHidden())
                    throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                if (call.input.getGenerateConfig().getMaxNewTokens() == 1)
                    throw Status.FAILED_PRECONDITION.withDescription("Single-token request is already complete at Prefill").asRuntimeException();
                call.stage = Stage.GENERATING;
                call.token = frame.getFirstGenerateTokenId();
                engine.control(call.request, Control.GENERATE);
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
    private void validate(GenerateRequestPB request) {
        if (request.getStage() != RemoteStage.ALLOCATE || !request.hasInput() || request.getClientId().isBlank()
                || request.getInput().getRequestId() != request.getRequestId())
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid ALLOCATE frame or identity"), 605);
        var input = request.getInput(); var generation = input.getGenerateConfig();
        RequestValidation.supported(input);
        if (org.flexlb.mock.multimodal.MediaInput.present(input))
            throw Status.FAILED_PRECONDITION.withDescription("Decode ALLOCATE requires the prepared P input").asRuntimeException();
        if (input.getTokenIdsCount() < 1
                || generation.getTimeoutMs() < 0 || generation.getMaxNewTokens() < 1 || generation.getMinNewTokens() < 0
                || generation.getMinNewTokens() > generation.getMaxNewTokens() || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid decode allocation settings"), 605);
    }

    private void acknowledge(Call call) {
        synchronized (this) {
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

    private synchronized void loaded(Call call, KvTransfer.Outcome outcome) {
        if (!live(call)) return;
        var failure = outcome.state() == KvTransfer.State.COMPLETED ? null : new RequestFailure(
            outcome.state() == KvTransfer.State.CANCELLED ? RequestFailure.Reason.CANCELLED : RequestFailure.Reason.INTERNAL,
            "KV transfer " + outcome.state(), outcome.error());
        try {
            engine.transferCompleted(call.request, new TransferResult(false, failure));
            if (failure != null) return;
            call.stage = Stage.LOADED;
            send(call, GenerateOutputsPB.newBuilder().setRequestId(call.input.getRequestId())
                .setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build());
        } catch (RequestFailure error) { stop(call, RpcErrors.status(error), RpcErrors.code(error)); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("LOAD acknowledgement failed"), 8100); }
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

    private synchronized void send(Call call, GenerateOutputsPB output) {
        if (call.terminal || call.response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
        call.delivery.add(output);
    }

    private synchronized void stop(Call call, Status status, long code) {
        if (call.terminal) return;
        if (call.request == null) ended(call, status, code);
        else engine.finish(call.request, RpcErrors.scheduling(status, code));
    }
    private synchronized void ended(Call call, Status status, long code) {
        if (call.terminal) return;
        call.terminal = true;
        if (call.transfer != null) call.transfer.cancel();
        call.delivery.finish(status.isOk() ? null : RpcErrors.failure(status, code));
    }

    Scheduler.Status snapshot(long latestFinished) { return engine.status(latestFinished); }
    String scrapeMetrics() { return engine.scrapeMetrics(); }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            engine.close();
        }
        transfer.close();
        transport.shutdown();
    }
}

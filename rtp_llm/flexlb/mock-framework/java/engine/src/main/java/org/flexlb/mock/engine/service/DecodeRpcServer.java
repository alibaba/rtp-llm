package org.flexlb.mock.engine.service;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.execution.EngineLoop;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpControl;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpTransferResult;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.flexlb.mock.engine.scheduler.RequestState;
import org.flexlb.mock.protocol.PromptCacheKeys;
import org.flexlb.mock.protocol.MultimodalProtocol;

/**
 * Handles the native ALLOCATE/LOAD/GENERATE protocol for Decode.
 */
final class DecodeRpcServer implements AutoCloseable {
    private final RtpRpcService.Config config;
    private final EngineLoop engine;
    private final Object executionLock;
    private final Runnable recordCancellation;
    private final java.util.function.Supplier<Runnable> trackRequest;
    private final ExecutorService transport = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-decode-output-deadline").factory());
    private final SimulatedKvLoad kvLoader = new SimulatedKvLoad();

    private boolean closed;

    DecodeRpcServer(RtpRpcService.Config config, Object executionLock, EngineLoop engine, Runnable recordCancellation, java.util.function.Supplier<Runnable> trackRequest) {
        this.trackRequest=trackRequest;
        deadlines.setRemoveOnCancelPolicy(true);
        this.config = config;
        this.executionLock = executionLock;
        this.engine = engine;
        this.recordCancellation = recordCancellation;
    }

    @Override public void close() {
        synchronized (executionLock) {
            if (closed) return;
            closed = true;
            engine.close();
        }
        kvLoader.close();
        deadlines.shutdownNow();
        transport.shutdown();
    }

    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }

    EngineScheduler.Status snapshot(long latestFinished) { return engine.status(latestFinished); }

    void checkHealthy() { engine.checkHealthy(); }

    private void accept(Call call, GenerateRequestPB frame) {
        try {
            EngineScheduler.Input prepared = null;
            if (frame.getStage() == RemoteStage.ALLOCATE) {
                validate(frame);
                var input = frame.getInput();
                var sequence = PromptCacheKeys.fromDecodePrompt(frame.getMockDecodePrompt(), config.physicalBlockSize());
                prepared = RpcMessages.toSchedulerInput(input, frame.getClientId() + "/" + input.getRequestId(), call.started,
                    RpcMessages.deadline(input, call.started, call.rpcDeadline), false, sequence);
            }
            synchronized (executionLock) {
                if (call.terminal) return;
                if (closed) throw Status.UNAVAILABLE.withDescription("Decode node is closing").asRuntimeException();
                if (call.input == null) {
                    if (prepared == null) throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Expected ALLOCATE frame"), 605);
                    call.input = frame.getInput();
                    call.client = frame.getClientId();
                    call.request = engine.enqueue(prepared, call);
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
                    engine.updateRequest(call.request, RtpControl.PREFILL_COMPLETE);
                } else if (frame.getStage() == RemoteStage.GENERATE && call.stage == Stage.LOADED) {
                    if (frame.hasPrefillKvHandoff() && (!frame.getClientId().equals(call.client)
                            || frame.getPrefillKvHandoff().getComputedKvEnd() != call.request.input().tokens()))
                        throw Status.INVALID_ARGUMENT.withDescription("Prefill KV handoff does not match loaded prompt").asRuntimeException();
                    if (frame.getProposeTokenIdsCount() > 0 || frame.getPositionIdsCount() > 0 || frame.hasProposeProbs() || frame.hasProposeHidden())
                        throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                    if (call.input.getGenerateConfig().getMaxNewTokens() == 1)
                        throw Status.FAILED_PRECONDITION.withDescription("Single-token request is already complete at Prefill").asRuntimeException();
                    call.stage = Stage.GENERATING;
                    call.delivery.startAt(1);
                    engine.updateRequest(call.request, new RtpControl.Generate(frame.getFirstGenerateTokenId()));
                } else {
                    throw Status.FAILED_PRECONDITION.withDescription("Expected ALLOCATE, LOAD, then GENERATE or PREFILL_COMPLETE").asRuntimeException();
                }
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
        RpcMessages.supported(input);
        if (MultimodalProtocol.present(input))
            throw Status.FAILED_PRECONDITION.withDescription("Decode ALLOCATE requires the prepared P input").asRuntimeException();
        if (!request.hasMockDecodePrompt() || input.getTokenIdsCount() != 0 || request.getMultimodalCacheSpansCount() != 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid Decode prompt"), 605);
        if (generation.getTimeoutMs() < 0 || generation.getMaxNewTokens() < 1 || generation.getMinNewTokens() < 0
                || generation.getMinNewTokens() > generation.getMaxNewTokens() || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid decode allocation settings"), 605);
    }

    private void startLoad(Call call) {
        var use = engine.beginKvTransfer(call.request);
        call.stage = Stage.LOADING;
        try {
            call.kvLoad = kvLoader.submit(use);
            call.kvLoad.completion().thenAccept(outcome -> loaded(call, outcome));
        } catch (RuntimeException error) { use.close(); throw error; }
    }

    private void loaded(Call call, SimulatedKvLoad.Outcome outcome) {
        synchronized (executionLock) {
            if (!live(call)) return;
            var failure = outcome.state() == SimulatedKvLoad.State.COMPLETED ? null : new RequestFailure(
                outcome.state() == SimulatedKvLoad.State.CANCELLED ? RequestFailure.Reason.CANCELLED : RequestFailure.Reason.INTERNAL,
                "KV transfer " + outcome.state(), outcome.error());
            try {
                engine.onKvTransferCompleted(call.request, new RtpTransferResult(failure));
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

    private void stop(Call call, Status status, long code) {
        synchronized (executionLock) {
            if (call.terminal) return;
            if (call.request == null) ended(call, status, code);
            else engine.abortRequest(call.request, RpcErrors.scheduling(status, code));
        }
    }

    private void ended(Call call, Status status, long code) {
        synchronized (executionLock) {
            if (call.terminal) return;
            call.terminal = true;
            call.rpcCompleted.run();
            if (status.getCode() == Status.Code.CANCELLED) recordCancellation.run();
            if (call.kvLoad != null) call.kvLoad.cancel();
            call.delivery.finish(status.isOk() ? null : RpcErrors.failure(status, code));
        }
    }

    private void send(Call call, GenerateOutputsPB output) {
        synchronized (executionLock) {
            if (call.terminal || call.response.isCancelled()) throw Status.CANCELLED.asRuntimeException();
            call.delivery.add(output);
        }
    }

    private void acknowledge(Call call) {
        synchronized (executionLock) {
            if (!live(call)) return;
            call.stage = Stage.ALLOCATED;
        }
        try { send(call, GenerateOutputsPB.newBuilder().setSupportsPrefillCompletion(true).build()); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("ALLOCATE acknowledgement failed"), 8100); }
    }

    private void disconnected(Call call, Status status, long code) {
        call.rpcCompleted.run();
        synchronized (executionLock) {
            // Once GENERATE is accepted, D computes independently of its P output consumer.
            // Scheduler deadlines, explicit engine cancellation and node shutdown still own termination.
            if (call.stage == Stage.GENERATING || call.stage == Stage.FINISHING) return;
            stop(call, status, code);
        }
    }

    private enum Stage { ALLOCATING, ALLOCATED, LOADING, LOADED, GENERATING, FINISHING }

    /**
     * Tracks one Decode RPC through allocation, loading and generation.
     */
    private final class Call implements StreamObserver<GenerateRequestPB>, EngineScheduler.Listener {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final GenerateResponseStream delivery;
        final Runnable rpcCompleted;
        final long started = System.nanoTime();
        final long rpcDeadline;
        GenerateInputPB input;
        String client;
        EngineScheduler.Request request;
        volatile boolean terminal;
        Stage stage = Stage.ALLOCATING;
        SimulatedKvLoad.Execution kvLoad;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response;
            rpcCompleted=trackRequest.get();
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE
                : RpcMessages.addNanosClamped(started, context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            delivery = new GenerateResponseStream(transport, deadlines, TimeUnit.MINUTES.toMillis(10), ignored -> {});
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
            var info = AuxInfoPB.newBuilder().setInputLen(request.input().tokens())
                .setIterCount(progress.iterations() + 1).setPrefixLen(request.reused()).setTotalReuseLen(request.reused())
                .setLocalReuseLen(request.reused()).setPdSep(true)
                .setCostTimeUs(elapsedUs).setFirstTokenCostTimeUs(elapsedUs);
            delivery.advance(input.getRequestId(), 0, progress.generated(), info.build(), progress.finished());
        }
        @Override public void terminal(EngineScheduler.Request request, RequestFailure failure) {
            ended(this, RpcErrors.status(failure), RpcErrors.code(failure));
        }
    }

    /**
     * Simulates zero-time KV loading asynchronously and holds resources until it quiesces.
     */
    static final class SimulatedKvLoad implements AutoCloseable {
        private final Object lock = new Object();
        private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
        private final Set<Execution> active = new HashSet<>();

        private boolean closed;

        /** Ownership transfers only on a successful return. The zero-time model executes asynchronously. */
        public Execution submit(RequestState.Use use) {
            Objects.requireNonNull(use);
            synchronized (lock) {
                if (closed) throw new IllegalStateException("Transfer executor is closed");
                var execution = new Execution(use);
                active.add(execution);
                try { workers.execute(execution::run); }
                catch (RuntimeException error) { active.remove(execution); throw error; }
                return execution;
            }
        }

        @Override public void close() {
            synchronized (lock) {
                if (!closed) {
                    closed = true;
                    active.forEach(Execution::cancel);
                    workers.shutdown();
                }
            }
            try {
                if (!workers.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KV transfer did not quiesce");
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted closing KV transfer", error);
            }
        }

        public enum State { COMPLETED, CANCELLED, FAILED }

        public record Outcome(State state, RuntimeException error) {}

        /**
         * Tracks cancellation and releases the KV reference before reporting completion.
         */
        public final class Execution {
            private final RequestState.Use use;
            private final CompletableFuture<Outcome> completion = new CompletableFuture<>();
            private boolean cancelled, terminal;
            private Execution(RequestState.Use use) { this.use = use; }
            public CompletionStage<Outcome> completion() { return completion.minimalCompletionStage(); }
            public boolean cancel() {
                synchronized (lock) {
                    if (terminal) return false;
                    cancelled = true;
                    return true;
                }
            }
            private void run() {
                State state;
                synchronized (lock) { terminal = true; state = cancelled ? State.CANCELLED : State.COMPLETED; }
                RuntimeException failure = null;
                try { use.close(); }
                catch (RuntimeException error) { state = State.FAILED; failure = error; }
                synchronized (lock) { active.remove(this); }
                completion.complete(new Outcome(state, failure));
            }
        }
    }
}

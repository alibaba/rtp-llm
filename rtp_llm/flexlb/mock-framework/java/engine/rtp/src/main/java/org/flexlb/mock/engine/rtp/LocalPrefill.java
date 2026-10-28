package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;

/** Native LocalGenerate boundary. Scheduling and network delivery use separate execution contexts. */
final class LocalPrefill implements AutoCloseable {
    static final int LIMIT = RtpScheduler.LIMIT;
    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    private final RtpConfig config;
    private final EngineLoop engine;
    private final TaskHistory history = new TaskHistory();

    LocalPrefill(RtpConfig config, DeviceCache cache) {
        this.config = config;
        engine = new EngineLoop(this, new RtpScheduler(this, config, cache), config.compute(), config.role(), config.id());
    }

    final class Session implements Scheduler.Listener {
        final GenerateInputPB input;
        final boolean retained;
        final CompletableFuture<Result> result = new CompletableFuture<>();
        final CompletableFuture<Result> termination = new CompletableFuture<>();
        Scheduler.Request request;

        Session(GenerateInputPB input, boolean retained) { this.input = input; this.retained = retained; }
        CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled local generation"), 0); }
        void preempt() { finish(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429); }
        void expire() { finish(Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired"), 603); }
        private void finish(Status status, long code) { engine.finish(request, RtpErrors.scheduling(status, code)); }
        private Result success(Scheduler.Request request) {
            return new Result(LocalResponse.oneToken(input, request.reused(), System.nanoTime() - request.input().started()), Status.OK, 0);
        }
        @Override public void output(Scheduler.Request request, Scheduler.Output output) {
            if (retained) result.complete(success(request));
        }
        @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
            history.add(RtpTasks.finished(request, failure));
            var ended = failure == null ? success(request) : new Result(null, RtpErrors.status(failure), RtpErrors.code(failure));
            result.complete(ended);
            termination.complete(ended);
        }
    }

    Session submit(GenerateInputPB input) { return submit(input, Long.MAX_VALUE); }
    Session submit(GenerateInputPB input, long rpcDeadline) { return submit(input, rpcDeadline, false); }
    synchronized Session submit(GenerateInputPB input, long rpcDeadline, boolean retained) {
        validate(input, retained);
        var session = new Session(input, retained);
        try {
            session.request = engine.submit(RtpTasks.input(input, Long.toString(input.getRequestId()), System.nanoTime(),
                rpcDeadline, retained, config.blockSize()), session);
        } catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
        return session;
    }

    void validate(GenerateInputPB input, boolean retained) {
        if (config.role() != Role.prefill || (retained ? input.getGenerateConfig().getMaxNewTokens() < 1 : input.getGenerateConfig().getMaxNewTokens() != 1))
            throw Status.UNIMPLEMENTED.withDescription("Local generation supports one token; transfer prepares only the Prefill token").asRuntimeException();
        var generation = input.getGenerateConfig();
        RtpRequestValidation.supported(input);
        if (input.getTokenIdsCount() < 1 || (long) input.getTokenIdsCount() + 1 > config.maxSeqLen()
                || config.maxBatchTokens() > 0 && input.getTokenIdsCount() > config.maxBatchTokens()
                || generation.getTimeoutMs() < 0 || generation.getMinNewTokens() < 0 || generation.getMinNewTokens() > generation.getMaxNewTokens()
                || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid input length or local generation settings"), 605);
    }

    RequestState.Use readReady(Session session) {
        try { return engine.readReady(session.request); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    void transferred(Session session) {
        try { engine.dependencyCompleted(session.request, Scheduler.Signal.TRANSFER_ACKNOWLEDGED); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    synchronized TaskHistory.Snapshot snapshot(long latestFinished) {
        var counts = engine.queues();
        return history.snapshot(latestFinished, engine.requests().stream().map(r -> RtpTasks.task(r).build()).toList(),
            counts.waiting() + counts.loading() + counts.ready(), counts.running());
    }
    String scrapeMetrics() { return engine.scrapeMetrics(); }
    @Override public void close() { engine.close(); }
}

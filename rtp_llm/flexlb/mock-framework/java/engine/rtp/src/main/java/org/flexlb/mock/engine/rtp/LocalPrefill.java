package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;

/** LocalGenerate adapter; the scheduler owns request progress and KV through completion or transfer. */
final class LocalPrefill implements AutoCloseable {
    static final int LIMIT = RtpScheduler.LIMIT;
    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    private final RtpConfig config;
    private final RtpScheduler scheduler;
    private final ExecutorService replies = Executors.newVirtualThreadPerTaskExecutor();
    private final TaskHistory history = new TaskHistory();

    LocalPrefill(RtpConfig config, DeviceCache cache) {
        this.config = config; scheduler = new RtpScheduler(this, config, cache);
    }
    final class Session implements RtpScheduler.Listener {
        final GenerateInputPB input;
        final boolean retained;
        final CompletableFuture<Result> result = new CompletableFuture<>();
        final CompletableFuture<Result> termination = new CompletableFuture<>();
        RtpScheduler.Request request;
        Session(GenerateInputPB input, boolean retained) { this.input = input; this.retained = retained; }
        CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled local generation"), 0); }
        void preempt() { finish(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429); }
        void expire() { finish(Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired"), 603); }
        private void finish(Status status, long code) { scheduler.finish(request, RtpErrors.scheduling(status, code)); }
        private Result success(RtpScheduler.Request request) {
            return new Result(LocalResponse.oneToken(input, request.reused, System.nanoTime() - request.input.started()), Status.OK, 0);
        }
        @Override public void output(RtpScheduler.Request request, int committed, boolean finished) {
            if (retained) { var ready = success(request); replies.execute(() -> result.complete(ready)); }
        }
        @Override public void terminal(RtpScheduler.Request request, RequestFailure failure) {
            history.add(RtpTasks.finished(request, failure));
            var ended = failure == null ? success(request) : new Result(null, RtpErrors.status(failure), RtpErrors.code(failure));
            replies.execute(() -> { result.complete(ended); termination.complete(ended); });
        }
    }
    Session submit(GenerateInputPB input) { return submit(input, Long.MAX_VALUE); }
    synchronized Session submit(GenerateInputPB input, long rpcDeadline) { return submit(input, rpcDeadline, false); }
    synchronized Session submit(GenerateInputPB input, long rpcDeadline, boolean retained) {
        validate(input, retained);
        var session = new Session(input, retained);
        try {
            session.request = scheduler.submit(RtpTasks.input(input, Long.toString(input.getRequestId()), System.nanoTime(),
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

    synchronized void requireReady(Session session) {
        try { scheduler.requireReady(session.request); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    synchronized void consume(Session session, Runnable deliver) {
        try { scheduler.consume(session.request, deliver); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    synchronized TaskHistory.Snapshot snapshot(long latestFinished) {
        var counts = scheduler.queues();
        return history.snapshot(latestFinished, scheduler.requests().stream().map(r -> RtpTasks.task(r).build()).toList(),
            counts.waiting(), counts.running());
    }
    String scrapeMetrics() { return scheduler.scrapeMetrics(); }
    @Override public synchronized void close() { scheduler.close(); replies.shutdown(); }
}

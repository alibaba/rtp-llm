package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.KvCache;

/** Native LocalGenerate boundary. Scheduling and network delivery use separate execution contexts. */
final class LocalPrefill implements AutoCloseable {
    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    private final RtpConfig config;
    private final EngineLoop engine;

    LocalPrefill(RtpConfig config, KvCache cache, Scheduler.Factory schedulers) {
        this.config = config;
        engine = new EngineLoop(this, schedulers.create(this, cache), cache, config.compute(), config.role(), config.id());
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
        private Result success(Scheduler.Request request, boolean finished) {
            return new Result(LocalResponse.token(input, input.getTokenIds(input.getTokenIdsCount() - 1), 1, 1, request.reused(), System.nanoTime() - request.input().started(), finished, false), Status.OK, 0);
        }
        @Override public void output(Scheduler.Request request, Scheduler.Output output) {
            if (retained) result.complete(success(request, output.finished()));
        }
        @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
            var ended = failure == null ? success(request, true) : new Result(null, RtpErrors.status(failure), RtpErrors.code(failure));
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
            session.request = engine.input(RtpTasks.input(input, Long.toString(input.getRequestId()), System.nanoTime(),
                rpcDeadline, retained, config.blockSize()), session);
        } catch (RequestFailure failure) {
            throw rpcFailure(failure);
        }
        return session;
    }

    record Submission(GenerateInputPB input, long deadline) {}
    record Admission(Session session, RuntimeException failure) {}

    synchronized List<Admission> submit(List<Submission> batch) {
        var sessions = new ArrayList<Session>();
        var items = new ArrayList<Scheduler.InputItem>();
        var positions = new ArrayList<Integer>();
        var results = new ArrayList<Admission>(Collections.nCopies(batch.size(), null));
        for (int i = 0; i < batch.size(); i++) {
            var item = batch.get(i);
            try {
                validate(item.input(), true);
                var session = new Session(item.input(), true);
                var input = RtpTasks.input(item.input(), Long.toString(item.input().getRequestId()),
                    System.nanoTime(), item.deadline(), true, config.blockSize());
                sessions.add(session); positions.add(i); items.add(new Scheduler.InputItem(input, session));
            } catch (RuntimeException failure) { results.set(i, new Admission(null, failure)); }
        }
        var admitted = engine.input(items);
        for (int i = 0; i < admitted.size(); i++) {
            var result = admitted.get(i); var session = sessions.get(i);
            if (result.failure() == null) {
                session.request = result.request(); results.set(positions.get(i), new Admission(session, null));
            } else results.set(positions.get(i), new Admission(null, rpcFailure(result.failure())));
        }
        return List.copyOf(results);
    }

    private static io.grpc.StatusRuntimeException rpcFailure(RequestFailure failure) {
        return failure.reason() == RequestFailure.Reason.INVALID
            ? RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription(failure.getMessage()), 605) : RtpErrors.failure(failure);
    }

    void validate(GenerateInputPB input, boolean retained) {
        if (config.role() != Role.prefill || (retained ? input.getGenerateConfig().getMaxNewTokens() < 1 : input.getGenerateConfig().getMaxNewTokens() != 1))
            throw Status.UNIMPLEMENTED.withDescription("Local generation supports one token; transfer prepares only the Prefill token").asRuntimeException();
        var generation = input.getGenerateConfig();
        RtpRequestValidation.supported(input);
        if (input.getTokenIdsCount() < 1
                || generation.getTimeoutMs() < 0 || generation.getMinNewTokens() < 0 || generation.getMinNewTokens() > generation.getMaxNewTokens()
                || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid input length or local generation settings"), 605);
    }

    RequestState.Use beginTransfer(Session session) {
        try { return engine.beginTransfer(session.request); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    void transferCompleted(Session session, RtpTransferResult result) {
        try { engine.transferCompleted(session.request, result); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    void transferred(Session session) {
        try { engine.control(session.request, RtpControl.TRANSFER_ACKNOWLEDGED); }
        catch (RequestFailure failure) { throw RtpErrors.failure(failure); }
    }
    Scheduler.Status snapshot(long latestFinished) { return engine.status(latestFinished); }
    String scrapeMetrics() { return engine.scrapeMetrics(); }
    @Override public void close() { engine.close(); }
}

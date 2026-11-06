package org.flexlb.mock.engine.service;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpControl;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.execution.RtpEngineLoop;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.RequestState;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.flexlb.mock.engine.scheduler.TaskHistory;
import org.flexlb.mock.protocol.MediaInput;

/** P protocol sessions, input preparation and output delivery. EngineScheduler owns local execution and KV. */
final class PrefillRpcServer implements AutoCloseable {
    private final RtpRpcService.Config config;
    private final Object executionLock;
    private final Runnable recordCancellation;
    private final RtpEngineLoop engine;
    private final TaskHistory finished;
    // Local request lifetime; output/Fetch sessions may outlive these entries. Guarded by executionLock.
    private final Map<Long, Session> sessions = new LinkedHashMap<>();
    private final org.flexlb.mock.protocol.VitClient vision;
    private final RpcChannels channels = new RpcChannels();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-pd-deadline").factory());
    private final Map<Long, Call> active = new HashMap<>();
    private final Map<Long, GeneratedOutput> outputs = new LinkedHashMap<>();
    private record Fence(long expires, boolean preempted) {}
    private static final long REGISTRY_TTL = TimeUnit.MINUTES.toNanos(10);
    private final Map<Long, Fence> fences = new LinkedHashMap<>();
    private long nextOutputSweep;
    private volatile boolean closed;

    PrefillRpcServer(RtpRpcService.Config config, Object executionLock, TaskHistory finished, RtpEngineLoop engine, Runnable recordCancellation) {
        this.config = config;
        this.executionLock = executionLock;
        this.finished = finished;
        this.engine = engine;
        this.recordCancellation = recordCancellation;
        vision = new org.flexlb.mock.protocol.VitClient(config.imageToken());
        deadlines.setRemoveOnCancelPolicy(true);
    }

    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    final class Session implements EngineScheduler.Listener {
        GenerateInputPB input;
        final long started, deadline;
        final long id;
        boolean ended;
        final boolean retained;
        final CompletableFuture<Result> result = new CompletableFuture<>();
        final CompletableFuture<Result> termination = new CompletableFuture<>();
        EngineScheduler.Request request;

        Session(GenerateInputPB input, boolean retained, long started, long deadline) {
            this.input = input; this.retained = retained; this.started = started; this.deadline = deadline;
            id = input.getRequestId();
            synchronized (executionLock) {
                if (closed) throw Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException();
                engine.checkHealthy();
                if (sessions.containsKey(id)) throw Status.ALREADY_EXISTS.withDescription("Active request identity already exists").asRuntimeException();
                sessions.put(id, this);
            }
        }
        CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled local generation"), 0); }
        void preempt() { finish(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429); }
        void expire() { finish(Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired"), 603); }
        private void finish(Status status, long code) { finish(RpcErrors.scheduling(status, code)); }
        private void fail(Throwable error) { finish(RpcErrors.scheduling(error)); }
        private void finish(RequestFailure failure) {
            synchronized (executionLock) {
                if (ended) return;
                if (request != null) { engine.finish(request, failure); return; }
                var task = RpcMessages.pending(input);
                finished.add(new EngineScheduler.Task(id, task.batchId(), task.inputLength(), 0, task.priority(),
                    EngineScheduler.ReportedPhase.PENDING, 0, 0, 0, System.currentTimeMillis(), failure, false));
                ended = true;
                sessions.remove(id, this);
            }
            complete(null, failure);
        }
        private Result success(EngineScheduler.Request request, boolean finished) {
            return new Result(RpcMessages.token(input, RpcMessages.sampledToken(input), 1, 1, request.reused(), request.remoteReused(), System.nanoTime() - request.input().started(), finished, false), Status.OK, 0);
        }
        @Override public void progressed(EngineScheduler.Request request) {
            var output = request.progress();
            if (retained) result.complete(success(request, output.finished()));
        }
        @Override public void terminal(EngineScheduler.Request request, RequestFailure failure) {
            synchronized (executionLock) {
                if (ended) return;
                ended = true;
                sessions.remove(id, this);
            }
            complete(request, failure);
        }
        private void complete(EngineScheduler.Request request, RequestFailure failure) {
            var ended = failure == null ? success(request, true) : new Result(null, RpcErrors.status(failure), RpcErrors.code(failure));
            result.complete(ended);
            termination.complete(ended);
        }
    }

    Session submit(GenerateInputPB input) { return submit(input, Long.MAX_VALUE); }
    Session submit(GenerateInputPB input, long rpcDeadline) { return submit(input, rpcDeadline, false); }
    Session submit(GenerateInputPB input, long rpcDeadline, boolean retained) {
        return submit(input, rpcDeadline, retained, System.nanoTime());
    }
    Session submit(GenerateInputPB input, long rpcDeadline, boolean retained, long started) {
        return submit(MediaInput.text(input), rpcDeadline, retained, started);
    }
    Session submit(MediaInput.Prepared prepared, long rpcDeadline, boolean retained, long started) {
        var input = prepared.input();
        validate(input, retained);
        requirePrepared(input);
        var session = new Session(input, retained, started, RpcMessages.deadline(input, started, rpcDeadline));
        return submit(session, prepared);
    }
    Session submit(Session session, MediaInput.Prepared prepared) {
        try {
            requirePrepared(prepared.input());
            synchronized (executionLock) {
                requirePending(session, prepared.input());
                session.input = prepared.input();
                session.request = engine.input(RpcMessages.input(prepared, Long.toString(session.input.getRequestId()), session.started,
                    session.deadline, session.retained, config.blockSize(), config.cacheKeySeed()), session);
            }
        } catch (RuntimeException error) {
            var nativeError = error instanceof RequestFailure f ? rpcFailure(f) : error;
            session.fail(nativeError);
            throw nativeError;
        }
        return session;
    }

    record Submission(MediaInput.Prepared prepared, long deadline, Session session) {
        Submission(GenerateInputPB input, long deadline) { this(MediaInput.text(input), deadline, null); }
        GenerateInputPB input() { return prepared.input(); }
    }
    record Admission(Session session, RuntimeException failure) {}

    List<Admission> submit(List<Submission> batch) {
        var items = new ArrayList<EngineScheduler.InputItem>();
        var admittedSessions = new ArrayList<Session>();
        var positions = new ArrayList<Integer>();
        var results = new ArrayList<Admission>(Collections.nCopies(batch.size(), null));
        synchronized (executionLock) {
            for (int i = 0; i < batch.size(); i++) {
                var item = batch.get(i);
                Session session = item.session();
                try {
                    if (session == null) validate(item.input(), true);
                    requirePrepared(item.input());
                    long started = System.nanoTime();
                    if (session == null) session = new Session(item.input(), true, started,
                        RpcMessages.deadline(item.input(), started, item.deadline()));
                    requirePending(session, item.input());
                    var input = RpcMessages.input(item.prepared(), Long.toString(session.id),
                        session.started, session.deadline, true, config.blockSize(), config.cacheKeySeed());
                    session.input = item.input();
                    admittedSessions.add(session); positions.add(i); items.add(new EngineScheduler.InputItem(input, session));
                } catch (RuntimeException error) {
                    var failure = error instanceof RequestFailure f ? rpcFailure(f) : error;
                    if (session != null) session.fail(failure);
                    results.set(i, new Admission(null, failure));
                }
            }
            if (!items.isEmpty()) {
                List<EngineScheduler.InputResult> admitted;
                try { admitted = engine.input(items); }
                catch (RuntimeException error) {
                    admittedSessions.forEach(session -> session.fail(error));
                    throw error;
                }
                for (int i = 0; i < admitted.size(); i++) {
                    var result = admitted.get(i); var session = admittedSessions.get(i);
                    if (result.failure() == null) {
                        session.request = result.request(); results.set(positions.get(i), new Admission(session, null));
                    } else {
                        var failure = rpcFailure(result.failure());
                        session.fail(failure);
                        results.set(positions.get(i), new Admission(null, failure));
                    }
                }
            }
        }
        return List.copyOf(results);
    }

    private void requirePending(Session session, GenerateInputPB input) {
        if (session.ended || session.request != null || sessions.get(session.id) != session)
            throw new RequestFailure(RequestFailure.Reason.CANCELLED, "Admission already ended");
        if (session.id != input.getRequestId()) throw new IllegalArgumentException("Admission identity changed");
    }

    private static io.grpc.StatusRuntimeException rpcFailure(RequestFailure failure) {
        return failure.reason() == RequestFailure.Reason.INVALID
            ? RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription(failure.getMessage()), 605) : RpcErrors.failure(failure);
    }

    private static void requirePrepared(GenerateInputPB input) {
        if (org.flexlb.mock.protocol.MediaInput.present(input))
            throw Status.INVALID_ARGUMENT.withDescription("Prepare multimodal input before EngineScheduler submission").asRuntimeException();
    }

    void validate(GenerateInputPB input, boolean retained) {
        if (config.role() != Role.prefill || (retained ? input.getGenerateConfig().getMaxNewTokens() < 1 : input.getGenerateConfig().getMaxNewTokens() != 1))
            throw Status.UNIMPLEMENTED.withDescription("Local generation supports one token; transfer prepares only the Prefill token").asRuntimeException();
        var generation = input.getGenerateConfig();
        RpcMessages.supported(input);
        if (input.getTokenIdsCount() < 1
                || generation.getTimeoutMs() < 0 || generation.getMinNewTokens() < 0 || generation.getMinNewTokens() > generation.getMaxNewTokens()
                || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid input length or local generation settings"), 605);
    }

    RequestState.Use beginTransfer(Session session) {
        try { return engine.beginTransfer(session.request); }
        catch (RequestFailure failure) { throw RpcErrors.failure(failure); }
    }
    private void acknowledgeLoad(Session session) {
        try { engine.control(session.request, RtpControl.TRANSFER_ACKNOWLEDGED); }
        catch (RequestFailure failure) { throw RpcErrors.failure(failure); }
    }
    EngineScheduler.Status snapshot(long latestFinished) {
        synchronized (executionLock) {
            var execution = engine.status(latestFinished);
            var tasks = new ArrayList<>(execution.tasks());
            sessions.values().stream().filter(session -> session.request == null)
                .forEach(session -> tasks.add(RpcMessages.pending(session.input)));
            return new EngineScheduler.Status(tasks, execution.finished(), execution.finishedVersion(), execution.waiting(), execution.running());
        }
    }
    void checkHealthy() { engine.checkHealthy(); }

    void generate(GenerateInputPB input, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        if (input.getGenerateConfig().getMaxNewTokens() > 1) {
            try { start(input, response, 0); }
            catch (RuntimeException error) { response.onError(error); }
            return;
        }
        Session session;
        var context = Context.current();
        try {
            validate(input, false);
            long started = System.nanoTime();
            session = new Session(input, false, started, RpcMessages.deadline(input, started, rpcDeadline()));
        } catch (RuntimeException error) { response.onError(error instanceof RequestFailure f ? rpcFailure(f) : error); return; }
        response.setOnCancelHandler(() -> {
            if (context.getDeadline() != null && context.getDeadline().isExpired()) session.expire();
            else session.cancel();
        });
        try {
            if (response.isCancelled()) session.cancel();
            else submit(session, vision.prepare(input, session.deadline));
        } catch (RuntimeException error) {
            session.fail(error);
        }
        session.completion().thenAccept(result -> {
            if (response.isCancelled()) return;
            if (!result.status().isOk()) { response.onError(RpcErrors.failure(result.status(), result.engineCode())); return; }
            try { response.onNext(result.output()); response.onCompleted(); }
            catch (RuntimeException error) { session.cancel(); }
        });
    }

    private Call start(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, long attachMillis) {
        return start(input, response, attachMillis, false);
    }
    private Call start(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, long attachMillis, boolean grouped) {
        validate(input, true);
        // Batch admission already selects the PD protocol, even for a single output token.
        if (!grouped && !input.getGenerateConfig().getCanUsePdSeparation())
            throw Status.UNIMPLEMENTED.withDescription("Non-separated multi-token generation is not supported").asRuntimeException();
        var targets = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.DECODE).toList();
        if (targets.size() != 1 || targets.getFirst().getIp().isBlank() || targets.getFirst().getGrpcPort() <= 0 || targets.getFirst().getGrpcPort() > 65535)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Exactly one valid Decode role address is required"), 605);
        var target = targets.getFirst(); String host = target.getIp().contains(":") ? "[" + target.getIp() + "]" : target.getIp();
        Call call;
        synchronized (this) {
            if (closed) throw Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException();
            sweepFences();
            if (response == null && fences.containsKey(input.getRequestId()) && fences.get(input.getRequestId()).preempted()) throw preempted();
            if (active.containsKey(input.getRequestId())) throw Status.ALREADY_EXISTS.withDescription("Active Prefill request id").asRuntimeException();
            long now = System.nanoTime();
            if (now >= nextOutputSweep) {
                outputs.entrySet().removeIf(e -> e.getValue().expired() || e.getValue().discardable());
                nextOutputSweep = now + TimeUnit.SECONDS.toNanos(1);
            }
            if (outputs.containsKey(input.getRequestId())) throw Status.ALREADY_EXISTS.withDescription("Retained output request id").asRuntimeException();
            call = new Call(input, response, host + ":" + target.getGrpcPort(), attachMillis, grouped);
            if (call.deferred) outputs.put(input.getRequestId(), call.output);
            active.put(input.getRequestId(), call);
            if (call.deferred) remember(input.getRequestId(), false);
        }
        synchronized (call) {
            if (call.terminal) return call;
            if (response != null) call.attach(response);
            if (call.terminal) return call;
            try {
                if (call.deadline != Long.MAX_VALUE) call.timeout = deadlines.schedule(() -> call.finish(
                    RpcErrors.failure(Status.DEADLINE_EXCEEDED.withDescription("P/D request budget expired"), 603)),
                    Math.max(0, call.deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                call.worker = workers.submit(call::run);
            } catch (RejectedExecutionException closing) { call.finish(Status.UNAVAILABLE.withDescription("Prefill node closing").asRuntimeException()); }
        }
        return call;
    }

    List<CompletableFuture<Void>> enqueue(List<GenerateInputPB> inputs, long attachMillis) {
        var calls = new ArrayList<Call>(); var results = new ArrayList<CompletableFuture<Void>>();
        for (var input : inputs) {
            try {
                var call = start(input, null, attachMillis, true); calls.add(call); results.add(call.admitted);
            } catch (RuntimeException error) { results.add(CompletableFuture.failedFuture(error)); }
        }
        var prepared = calls.stream().map(call -> call.prepared.handle((unused, error) -> null)).toArray(CompletableFuture[]::new);
        CompletableFuture.allOf(prepared).thenRunAsync(() -> submitGroup(calls), workers)
            .exceptionally(error -> { calls.forEach(call -> call.finish(error)); return null; });
        return List.copyOf(results);
    }
    private void submitGroup(List<Call> calls) {
        var live = new ArrayList<Call>(); var inputs = new ArrayList<Submission>();
        for (var call : calls) synchronized (call) {
            if (!call.terminal && !call.prepared.isCompletedExceptionally()) {
                live.add(call); inputs.add(new Submission(call.media, call.deadline, call.source));
            }
        }
        if (live.isEmpty()) return;
        var results = submit(inputs);
        for (int i = 0; i < live.size(); i++) {
            var call = live.get(i); var result = results.get(i);
            synchronized (call) {
                if (result.failure() != null) call.finish(result.failure());
                else if (call.terminal) result.session().cancel();
                else {
                    call.batchSource.complete(result.session());
                }
            }
        }
    }

    private GeneratedOutput newOutput(long id, long attachMillis) {
        // Retain only the registry key, never the Call or its input token data.
        return new GeneratedOutput(workers, deadlines, attachMillis > 0 ? attachMillis : TimeUnit.MINUTES.toMillis(10), expired -> {
            synchronized (this) { outputs.remove(id, expired); }
        });
    }

    void fetch(long id, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        GeneratedOutput output; Fence fence;
        synchronized (this) { output = outputs.get(id); fence = fences.get(id); }
        if (fence != null && fence.preempted() && fence.expires() > System.nanoTime()) {
            response.onError(preempted()); return;
        }
        if (output == null || output.expired()) {
            response.onError(Status.NOT_FOUND.withDescription("No fetchable request").asRuntimeException()); return;
        }
        output.attach(response, () -> {});
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
            call.source.preempt();
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
        return RpcErrors.failure(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429);
    }

    private final class Call {
        GenerateInputPB input;
        final GeneratedOutput output;
        final boolean deferred;
        final CompletableFuture<Void> admitted = new CompletableFuture<>(), prepared = new CompletableFuture<>();
        final CompletableFuture<Session> batchSource;
        final String target;
        final long started, deadline;
        final Context.CancellableContext context;
        org.flexlb.mock.protocol.MediaInput.Prepared media;
        final Session source;
        RequestState.Use transferUse;
        Future<?> worker;
        ScheduledFuture<?> timeout;
        boolean terminal, priorityPreempted;
        Call(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, String target, long attachMillis, boolean grouped) {
            this.input = input; this.target = target; this.deferred = response == null;
            batchSource = grouped ? new CompletableFuture<>() : null;
            output = newOutput(input.getRequestId(), attachMillis);
            context = (deferred ? Context.ROOT : Context.current()).withCancellation();
            started = System.nanoTime();
            deadline = RpcMessages.deadline(input, started, deferred ? Long.MAX_VALUE : rpcDeadline());
            source = new Session(input, true, started, deadline);
        }
        void attach(ServerCallStreamObserver<GenerateOutputsPB> response) {
            output.attach(response, () -> finish(Status.CANCELLED.asRuntimeException()));
        }
        void run() { context.run(this::exchange); }
        void exchange() {
            try {
                checkActive();
                media = vision.prepare(input, deadline);
                checkActive();
                input = media.input();
                var inbox = new Inbox(error -> {
                    synchronized (this) { if (!source.result.isDone()) finish(error); }
                });
                var stub = RpcServiceGrpc.newStub(channels.get(target));
                if (deadline != Long.MAX_VALUE) stub = stub.withDeadlineAfter(Math.max(1, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                var stream = stub.remoteGenerate(inbox);
                var allocationInput = input.toBuilder().clearGroupId().clearGroupSize();
                allocationInput.getGenerateConfigBuilder().clearGroupTimeout();
                if (deadline != Long.MAX_VALUE) allocationInput.getGenerateConfigBuilder().setTimeoutMs((int) Math.min(Integer.MAX_VALUE,
                    Math.max(1, TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime()))));
                stream.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(input.getRequestId()).setClientId(config.id())
                    .setInput(allocationInput).addAllMultimodalCacheSpans(media.spans()).build());
                var allocated = inbox.next();
                if (allocated == null || allocated.hasFlattenOutput() || allocated.hasErrorInfo() && allocated.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid ALLOCATE acknowledgement").asRuntimeException();
                if (batchSource != null) {
                    prepared.complete(null);
                    batchSource.get();
                }
                synchronized (this) {
                    if (terminal) return;
                    if (batchSource == null)
                        submit(source, media);
                    admitted.complete(null);
                }
                if (inbox.failure != null) throw Status.fromThrowable(inbox.failure)
                    .asRuntimeException(Status.trailersFromThrowable(inbox.failure));
                var first = source.completion().toCompletableFuture().get();
                if (!first.status().isOk()) throw RpcErrors.failure(first.status(), first.engineCode());
                synchronized (this) {
                    if (terminal) return;
                    transferUse = beginTransfer(source);
                }
                stream.onNext(frame(RemoteStage.LOAD));
                long ttft = System.nanoTime() - started;
                boolean prefillFinished = first.output().getFlattenOutput().getFinished(0);
                var prefillAux = first.output().getFlattenOutput().getAuxInfo(0);
                int prefillReuse = prefillAux.getTotalReuseLen();
                int prefillRemoteReuse = prefillAux.getRemoteReuseLen();
                send(RpcMessages.token(input, RpcMessages.sampledToken(input), 1, 1,
                    prefillReuse, prefillRemoteReuse, ttft, prefillFinished, true));
                var loaded = inbox.next();
                if (loaded == null || !loaded.hasErrorInfo() || loaded.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid LOAD acknowledgement").asRuntimeException();
                synchronized (this) {
                    if (terminal) return;
                    acknowledgeLoad(source);
                    transferUse.close(); transferUse = null;
                }
                if (prefillFinished) stream.onNext(frame(RemoteStage.PREFILL_COMPLETE));
                else stream.onNext(frame(RemoteStage.GENERATE).toBuilder().setFirstGenerateTokenId(RpcMessages.sampledToken(input)).build());
                stream.onCompleted();
                boolean finished = prefillFinished;
                for (GenerateOutputsPB output; (output = inbox.next()) != null;) {
                    if (finished || output.getRequestId() != input.getRequestId() || !output.hasFlattenOutput() || output.getFlattenOutput().getFinishedCount() != 1)
                        throw Status.DATA_LOSS.withDescription("Invalid Decode output stream").asRuntimeException();
                    var merged = output.toBuilder();
                    for (var aux : merged.getFlattenOutputBuilder().getAuxInfoBuilderList()) {
                        aux.setDecodeTotalReuseLen(aux.getTotalReuseLen()).setDecodeLocalReuseLen(aux.getLocalReuseLen())
                            .setDecodeRemoteReuseLen(aux.getRemoteReuseLen())
                            .setPrefillTotalReuseLen(prefillReuse).setPrefillLocalReuseLen(prefillAux.getLocalReuseLen())
                            .setPrefillRemoteReuseLen(prefillRemoteReuse)
                            .setTotalReuseLen(prefillReuse).setLocalReuseLen(prefillAux.getLocalReuseLen())
                            .setRemoteReuseLen(prefillRemoteReuse).setPrefixLen(prefillReuse)
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
        synchronized void checkActive() {
            if (terminal || context.isCancelled() || Thread.currentThread().isInterrupted())
                throw Status.CANCELLED.withDescription("Prefill request cancelled before downstream preparation").asRuntimeException();
            if (System.nanoTime() >= deadline)
                throw Status.DEADLINE_EXCEEDED.withDescription("Prefill request expired before downstream preparation").asRuntimeException();
        }
        GenerateRequestPB frame(RemoteStage stage) {
            var frame = GenerateRequestPB.newBuilder().setStage(stage).setRequestId(input.getRequestId()).setClientId(config.id());
            if (stage == RemoteStage.GENERATE) frame.setPrefillKvHandoff(PrefillKvHandoffPB.newBuilder().setComputedKvEnd(input.getTokenIdsCount()));
            return frame.build();
        }
        void send(GenerateOutputsPB value) { output.add(value); }
        synchronized void finish(Throwable error) {
            if (terminal) return;
            terminal = true;
            if (error != null && Status.fromThrowable(error).getCode() == Status.Code.CANCELLED)
                recordCancellation.run();
            if (timeout != null) timeout.cancel(false);
            Throwable failure = error == null ? Status.CANCELLED.asRuntimeException() : error;
            prepared.completeExceptionally(failure);
            if (batchSource != null) batchSource.completeExceptionally(failure);
            if (!admitted.isDone()) admitted.completeExceptionally(failure);
            context.cancel(null);
            if (error != null) source.fail(error);
            else source.cancel();
            if (transferUse != null) { transferUse.close(); transferUse = null; }
            if (worker != null) worker.cancel(true);
            synchronized (PrefillRpcServer.this) {
                active.remove(input.getRequestId(), this);
                if (deferred) {
                    var old = fences.get(input.getRequestId());
                    remember(input.getRequestId(), old != null && old.preempted());
                }
            }
            output.finish(error);
        }
    }
    private static long rpcDeadline() {
        var deadline = Context.current().getDeadline();
        return deadline == null ? Long.MAX_VALUE : RpcMessages.after(System.nanoTime(), deadline.timeRemaining(TimeUnit.NANOSECONDS));
    }
    private static final class Inbox implements ClientResponseObserver<GenerateRequestPB, GenerateOutputsPB> {
        final BlockingQueue<Object> values = new LinkedBlockingQueue<>(2);
        volatile Throwable failure;
        final java.util.function.Consumer<Throwable> failed;
        Inbox(java.util.function.Consumer<Throwable> failed) { this.failed = failed; }
        final AtomicReference<ClientCallStreamObserver<GenerateRequestPB>> stream = new AtomicReference<>();
        @Override public void beforeStart(ClientCallStreamObserver<GenerateRequestPB> request) { stream.set(request); request.disableAutoRequestWithInitial(1); }
        @Override public void onNext(GenerateOutputsPB value) { values.add(value); }
        @Override public void onError(Throwable error) { failure = error; values.add(error); failed.accept(error); }
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
        synchronized (this) { outputs.clear(); }
        List<Session> preparing;
        synchronized (executionLock) {
            preparing = sessions.values().stream().filter(session -> session.request == null).toList();
        }
        for (var session : preparing) session.finish(Status.UNAVAILABLE.withDescription("Prefill node closed"), 0);
        workers.shutdownNow(); deadlines.shutdownNow(); vision.close(); channels.close(); engine.close();
    }

    /** Node-owned connections shared by Prefill sessions. */
    private static final class RpcChannels implements AutoCloseable {
        private final Map<String, ManagedChannel> channels = new HashMap<>();
        private boolean closed;
        synchronized ManagedChannel get(String address) {
            if (closed) throw Status.UNAVAILABLE.withDescription("RPC connections closed").asRuntimeException();
            return channels.computeIfAbsent(address, key -> ManagedChannelBuilder.forTarget(key).usePlaintext().build());
        }
        @Override public void close() {
            List<ManagedChannel> owned;
            synchronized (this) { closed = true; owned = List.copyOf(channels.values()); channels.clear(); }
            owned.forEach(ManagedChannel::shutdownNow);
            for (var channel : owned) try { channel.awaitTermination(3, TimeUnit.SECONDS); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); return; }
        }
    }
}

package org.flexlb.mock.admission;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.multimodal.MediaInput;
import org.flexlb.mock.api.model.Types.Role;

/** P protocol sessions, input preparation and output delivery. Scheduler owns local execution and KV. */
final class PrefillAdmission implements AutoCloseable {
    private final AdmissionConfig config;
    private final Object executionLock = new Object();
    private final EngineLoop engine;
    private final org.flexlb.mock.multimodal.MediaClient vision;
    private final RpcChannels channels = new RpcChannels();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-pd-deadline").factory());
    private final Map<Long, Call> active = new HashMap<>();
    private final Map<Long, BufferedOutput> outputs = new LinkedHashMap<>();
    private record Fence(long expires, boolean preempted) {}
    private static final int MAX_ACTIVE_REQUESTS = 1024;
    private static final long REGISTRY_TTL = TimeUnit.MINUTES.toNanos(10);
    private final Map<Long, Fence> fences = new LinkedHashMap<>();
    private long nextOutputSweep;
    private boolean closed;

    PrefillAdmission(AdmissionConfig config, KvCache cache, Scheduler.Factory schedulers) {
        this.config = config;
        engine = new EngineLoop(executionLock, schedulers, cache, config.compute(), config.role(), config.id());
        vision = new org.flexlb.mock.multimodal.MediaClient(config.imageToken());
        deadlines.setRemoveOnCancelPolicy(true);
    }

    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    final class Session implements Scheduler.Listener {
        GenerateInputPB input;
        final long started, deadline;
        final EngineLoop.Admission admission;
        final boolean retained;
        final CompletableFuture<Result> result = new CompletableFuture<>();
        final CompletableFuture<Result> termination = new CompletableFuture<>();
        Scheduler.Request request;

        Session(GenerateInputPB input, boolean retained, long started, long deadline) {
            this.input = input; this.retained = retained; this.started = started; this.deadline = deadline;
            try { admission = engine.beginAdmission(Long.toString(input.getRequestId()), RpcTasks.pending(input)); }
            catch (RequestFailure failure) { throw rpcFailure(failure); }
        }
        CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled local generation"), 0); }
        void preempt() { finish(Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429); }
        void expire() { finish(Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired"), 603); }
        private void finish(Status status, long code) { finish(RpcErrors.scheduling(status, code)); }
        private void fail(Throwable error) { finish(RpcErrors.scheduling(error)); }
        private void finish(RequestFailure failure) {
            synchronized (executionLock) {
                if (request != null) { engine.finish(request, failure); return; }
                engine.finishAdmission(admission, failure);
            }
            terminal(null, failure);
        }
        private Result success(Scheduler.Request request, boolean finished) {
            return new Result(LocalResponse.token(input, LocalResponse.sampledToken(input), 1, 1, request.reused(), System.nanoTime() - request.input().started(), finished, false), Status.OK, 0);
        }
        @Override public void output(Scheduler.Request request, Scheduler.Output output) {
            if (retained) result.complete(success(request, output.finished()));
        }
        @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
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
        var session = new Session(input, retained, started, RpcTasks.deadline(input, started, rpcDeadline));
        return submit(session, prepared);
    }
    Session submit(Session session, MediaInput.Prepared prepared) {
        try {
            requirePrepared(prepared.input());
            synchronized (executionLock) {
                session.input = prepared.input();
                session.request = engine.input(session.admission, RpcTasks.input(prepared, Long.toString(session.input.getRequestId()), session.started,
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

    synchronized List<Admission> submit(List<Submission> batch) {
        var sessions = new ArrayList<Session>();
        var items = new ArrayList<Scheduler.InputItem>();
        var positions = new ArrayList<Integer>();
        var results = new ArrayList<Admission>(Collections.nCopies(batch.size(), null));
        for (int i = 0; i < batch.size(); i++) {
            var item = batch.get(i);
            Session session = item.session();
            try {
                if (session == null) validate(item.input(), true);
                requirePrepared(item.input());
                long started = System.nanoTime();
                if (session == null) session = new Session(item.input(), true, started,
                    RpcTasks.deadline(item.input(), started, item.deadline()));
                session.input = item.input();
                var input = RpcTasks.input(item.prepared(), Long.toString(item.input().getRequestId()),
                    session.started, session.deadline, true, config.blockSize(), config.cacheKeySeed());
                sessions.add(session); positions.add(i); items.add(new Scheduler.InputItem(input, session));
            } catch (RuntimeException failure) {
                if (session != null) session.fail(failure);
                results.set(i, new Admission(null, failure));
            }
        }
        List<Scheduler.InputResult> admitted;
        synchronized (executionLock) {
            try { admitted = engine.input(sessions.stream().map(s -> s.admission).toList(), items); }
            catch (RuntimeException error) {
                sessions.forEach(s -> s.fail(error));
                throw error;
            }
            for (int i = 0; i < admitted.size(); i++) {
                var result = admitted.get(i); var session = sessions.get(i);
                if (result.failure() == null) {
                    session.request = result.request(); results.set(positions.get(i), new Admission(session, null));
                } else {
                    session.fail(rpcFailure(result.failure()));
                    results.set(positions.get(i), new Admission(null, rpcFailure(result.failure())));
                }
            }
        }
        return List.copyOf(results);
    }

    private static io.grpc.StatusRuntimeException rpcFailure(RequestFailure failure) {
        return failure.reason() == RequestFailure.Reason.INVALID
            ? RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription(failure.getMessage()), 605) : RpcErrors.failure(failure);
    }

    private static void requirePrepared(GenerateInputPB input) {
        if (org.flexlb.mock.multimodal.MediaInput.present(input))
            throw Status.INVALID_ARGUMENT.withDescription("Prepare multimodal input before Scheduler submission").asRuntimeException();
    }

    void validate(GenerateInputPB input, boolean retained) {
        if (config.role() != Role.prefill || (retained ? input.getGenerateConfig().getMaxNewTokens() < 1 : input.getGenerateConfig().getMaxNewTokens() != 1))
            throw Status.UNIMPLEMENTED.withDescription("Local generation supports one token; transfer prepares only the Prefill token").asRuntimeException();
        var generation = input.getGenerateConfig();
        RequestValidation.supported(input);
        if (input.getTokenIdsCount() < 1
                || generation.getTimeoutMs() < 0 || generation.getMinNewTokens() < 0 || generation.getMinNewTokens() > generation.getMaxNewTokens()
                || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RpcErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid input length or local generation settings"), 605);
    }

    RequestState.Use beginTransfer(Session session) {
        try { return engine.beginTransfer(session.request); }
        catch (RequestFailure failure) { throw RpcErrors.failure(failure); }
    }
    void transferCompleted(Session session, TransferResult result) {
        try { engine.transferCompleted(session.request, result); }
        catch (RequestFailure failure) { throw RpcErrors.failure(failure); }
    }
    void transferred(Session session) {
        try { engine.control(session.request, Control.TRANSFER_ACKNOWLEDGED); }
        catch (RequestFailure failure) { throw RpcErrors.failure(failure); }
    }
    Scheduler.Status snapshot(long latestFinished) { return engine.status(latestFinished); }
    void checkHealthy() { engine.checkHealthy(); }
    java.util.List<io.prometheus.client.Collector.MetricFamilySamples> collectMetrics() { return engine.collectMetrics(); }
    String scrapeMetrics() { return engine.scrapeMetrics(); }

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
            session = new Session(input, false, started, RpcTasks.deadline(input, started, rpcDeadline()));
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
            if (active.size() >= MAX_ACTIVE_REQUESTS)
                throw Status.RESOURCE_EXHAUSTED.withDescription("Mock P/D request registry full").asRuntimeException();
            long now = System.nanoTime();
            if (now >= nextOutputSweep) {
                outputs.entrySet().removeIf(e -> e.getValue().expired() || e.getValue().discardable());
                nextOutputSweep = now + TimeUnit.SECONDS.toNanos(1);
            }
            if (outputs.size() >= 65536) {
                var oldest = outputs.entrySet().stream().filter(e -> e.getValue().finished()).findFirst();
                if (oldest.isEmpty()) throw Status.RESOURCE_EXHAUSTED.withDescription("Mock output registry full").asRuntimeException();
                outputs.remove(oldest.get().getKey());
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

    void fetch(long id, StreamObserver<GenerateOutputsPB> observer) {
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        BufferedOutput output; Fence fence;
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
        final BufferedOutput output;
        final boolean deferred;
        final CompletableFuture<Void> admitted = new CompletableFuture<>(), prepared = new CompletableFuture<>();
        final CompletableFuture<Session> batchSource;
        final String target;
        final long started, deadline;
        final Context.CancellableContext context;
        org.flexlb.mock.multimodal.MediaInput.Prepared media;
        final Session source;
        RequestState.Use transferUse;
        Future<?> worker;
        ScheduledFuture<?> timeout;
        boolean terminal, priorityPreempted;
        Call(GenerateInputPB input, ServerCallStreamObserver<GenerateOutputsPB> response, String target, long attachMillis, boolean grouped) {
            this.input = input; this.target = target; this.deferred = response == null;
            batchSource = grouped ? new CompletableFuture<>() : null;
            output = new BufferedOutput(workers, attachMillis > 0 ? attachMillis : TimeUnit.MINUTES.toMillis(10));
            context = (deferred ? Context.ROOT : Context.current()).withCancellation();
            started = System.nanoTime();
            deadline = RpcTasks.deadline(input, started, deferred ? Long.MAX_VALUE : rpcDeadline());
            source = new Session(input, true, started, deadline);
        }
        void attach(ServerCallStreamObserver<GenerateOutputsPB> response) {
            output.attach(response, () -> finish(Status.CANCELLED.asRuntimeException()));
        }
        void run() { context.run(this::exchange); }
        void exchange() {
            try {
                media = vision.prepare(input, deadline);
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
                int prefillReuse = first.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen();
                send(LocalResponse.token(input, LocalResponse.sampledToken(input), 1, 1, prefillReuse, ttft, prefillFinished, true));
                var loaded = inbox.next();
                if (loaded == null || !loaded.hasErrorInfo() || loaded.getErrorInfo().getErrorCodeValue() != 0)
                    throw Status.DATA_LOSS.withDescription("Invalid LOAD acknowledgement").asRuntimeException();
                synchronized (this) {
                    if (terminal) return;
                    transferCompleted(source, new TransferResult(prefillFinished, null));
                    transferred(source);
                    transferUse.close(); transferUse = null;
                }
                if (prefillFinished) stream.onNext(frame(RemoteStage.PREFILL_COMPLETE));
                else stream.onNext(frame(RemoteStage.GENERATE).toBuilder().setFirstGenerateTokenId(LocalResponse.sampledToken(input)).build());
                stream.onCompleted();
                boolean finished = prefillFinished;
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
        GenerateRequestPB frame(RemoteStage stage) {
            var frame = GenerateRequestPB.newBuilder().setStage(stage).setRequestId(input.getRequestId()).setClientId(config.id());
            if (stage == RemoteStage.GENERATE) frame.setPrefillKvHandoff(PrefillKvHandoffPB.newBuilder().setComputedKvEnd(input.getTokenIdsCount()));
            return frame.build();
        }
        void send(GenerateOutputsPB value) { output.add(value); }
        synchronized void finish(Throwable error) {
            if (terminal) return;
            terminal = true;
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
            synchronized (PrefillAdmission.this) {
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
        return deadline == null ? Long.MAX_VALUE : RpcTasks.after(System.nanoTime(), deadline.timeRemaining(TimeUnit.NANOSECONDS));
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
        workers.shutdownNow(); deadlines.shutdownNow(); vision.close(); channels.close(); engine.close();
    }
}

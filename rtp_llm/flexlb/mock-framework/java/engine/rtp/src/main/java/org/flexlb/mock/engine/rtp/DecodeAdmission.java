package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.metrics.EngineMetrics;
import org.flexlb.mock.engine.rtp.transfer.*;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.api.model.Types.Role;

/** D-side ALLOCATE/LOAD owner. A LOAD ACK requires data from a live, computed P source. */
final class DecodeAdmission implements AutoCloseable {
    private record Key(String client, long request) {}
    private final RtpConfig config;
    private final DeviceCache cache;
    private final Map<Key, Call> active = new HashMap<>();
    private final TaskHistory history = new TaskHistory();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final ExecutorService callbacks = Executors.newVirtualThreadPerTaskExecutor();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-decode-deadline").factory());
    private final EngineMetrics metrics;
    private final RpcChannels channels = new RpcChannels();
    private final SimulatedCompute compute = new SimulatedCompute();
    private final ArrayDeque<Call> ready = new ArrayDeque<>();
    private Call running;
    private boolean closed;

    DecodeAdmission(RtpConfig config, DeviceCache cache) {
        this.config = config; this.cache = cache;
        metrics = new EngineMetrics("rtp", "decode", config.id(), this::metricState);
        deadlines.setRemoveOnCancelPolicy(true);
    }

    StreamObserver<GenerateRequestPB> open(StreamObserver<GenerateOutputsPB> observer) {
        return new Call((ServerCallStreamObserver<GenerateOutputsPB>) observer, Context.current());
    }

    private final class Call implements StreamObserver<GenerateRequestPB> {
        final ServerCallStreamObserver<GenerateOutputsPB> response;
        final Context context;
        final long started = System.nanoTime();
        final long rpcDeadline;
        Key key;
        GenerateInputPB input;
        long deadline = Long.MAX_VALUE;
        DeviceCache.Lease lease;
        Future<?> allocation;
        ScheduledFuture<?> timeout;
        boolean terminal, acknowledged, loading, loaded;
        List<String> peers;
        Context.CancellableContext source;
        boolean generating, backpressured;
        int generated, token, stopAfter, iterations;
        DecodeStepModel.Step step;
        long executionNanos, stepStarted;
        SimulatedCompute.Execution execution;
        int reused;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response; this.context = context;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE : after(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            response.setOnReadyHandler(() -> resume(this));
            response.setOnCancelHandler(() -> {
                boolean expired = context.getDeadline() != null && context.getDeadline().isExpired();
                stop(this, expired ? Status.DEADLINE_EXCEEDED.withDescription("Decode RPC deadline expired")
                    : Status.CANCELLED.withDescription("Decode RPC cancelled"), expired ? 603 : 8100);
            });
        }
        @Override public void onNext(GenerateRequestPB request) { accept(this, request); }
        @Override public void onError(Throwable error) { stop(this, Status.CANCELLED.withDescription("Decode stream closed"), 8100); }
        @Override public void onCompleted() {
            synchronized (DecodeAdmission.this) {
                if (!generating) stop(this, Status.CANCELLED.withDescription("P closed before generation"), 8100);
            }
        }
    }

    private synchronized void accept(Call call, GenerateRequestPB request) {
        if (call.terminal) return;
        try {
            if (closed) throw Status.UNAVAILABLE.withDescription("Decode node is closing").asRuntimeException();
            if (call.input != null) {
                if (request.getRequestId() != call.key.request())
                    throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate identity changed").asRuntimeException();
                if (!request.getClientId().isEmpty() && !request.getClientId().equals(call.key.client()))
                    throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate client identity changed").asRuntimeException();
                if (request.getStage() == RemoteStage.LOAD && call.acknowledged && !call.loading && !call.loaded) {
                    load(call); return;
                }
                if (call.loaded && !call.generating && request.getStage() == RemoteStage.PREFILL_COMPLETE) {
                    if (call.stopAfter != 1 || !request.getClientId().equals(call.key.client()))
                        throw Status.FAILED_PRECONDITION.withDescription("Invalid Prefill-only completion").asRuntimeException();
                    call.generated = 1; stop(call, Status.OK, 0); return;
                }
                if (call.loaded && !call.generating && request.getStage() == RemoteStage.GENERATE) {
                    if (request.getProposeTokenIdsCount() > 0 || request.getPositionIdsCount() > 0 || request.hasProposeProbs() || request.hasProposeHidden())
                        throw Status.UNIMPLEMENTED.withDescription("Speculative/position handoff is not supported").asRuntimeException();
                    call.generating = true; call.generated = 1; call.token = request.getFirstGenerateTokenId();
                    if (call.stopAfter == 1) {
                        call.response.onNext(LocalResponse.token(call.input, call.token, 1, 1, call.reused, System.nanoTime() - call.started, true, true));
                        stop(call, Status.OK, 0);
                    } else { ready.addLast(call); advance(); }
                    return;
                }
                throw Status.FAILED_PRECONDITION.withDescription("Expected LOAD after ALLOCATE acknowledgement").asRuntimeException();
            }
            validate(request);
            var key = new Key(request.getClientId(), request.getRequestId());
            if (active.containsKey(key)) throw Status.ALREADY_EXISTS.withDescription("Active decode request already exists").asRuntimeException();
            if (active.size() >= TaskHistory.LIMIT) throw Status.RESOURCE_EXHAUSTED.withDescription("Mock decode reservation limit reached").asRuntimeException();
            call.input = request.getInput(); call.key = key; call.peers = request.getPeerAddrsList();
            call.deadline = Math.min(call.rpcDeadline, call.input.getGenerateConfig().getTimeoutMs() == 0 ? Long.MAX_VALUE
                : after(call.started, TimeUnit.MILLISECONDS.toNanos(call.input.getGenerateConfig().getTimeoutMs())));
            active.put(key, call); metrics.started();
            if (call.deadline != Long.MAX_VALUE)
                call.timeout = deadlines.schedule(() -> stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode request budget expired"), 603),
                    Math.max(0, call.deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
            if (System.nanoTime() >= call.deadline) {
                stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode expired before admission"), 603); return;
            }
            if (call.response.isCancelled()) {
                stop(call, Status.CANCELLED.withDescription("Decode cancelled before admission"), 8100); return;
            }
            call.allocation = workers.submit(() -> allocate(call));
        } catch (StatusRuntimeException error) { stop(call, error.getStatus(), error.getStatus().getCode() == Status.Code.INVALID_ARGUMENT ? 605 : 0); }
        catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode frame failed: " + error.getMessage()), 0); }
    }

    private void validate(GenerateRequestPB request) {
        if (request.getStage() != RemoteStage.ALLOCATE || !request.hasInput() || request.getClientId().isBlank()
                || request.getInput().getRequestId() != request.getRequestId())
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid ALLOCATE frame or identity"), 605);
        var input = request.getInput(); var generation = input.getGenerateConfig();
        if (input.getMultimodalInputsCount() > 0 || input.hasV41Inputs() || input.hasMultimodalTokenLayout()
                || generation.hasAdapterName() || !generation.getUniqueKey().isEmpty() || generation.getPrefillOnly()
                || generation.getNumBeams() > 1 || generation.getNumReturnSequences() > 1 || generation.getVariableNumBeamsCount() > 0)
            throw Status.UNIMPLEMENTED.withDescription("Decode reservation currently supports single-sequence text only").asRuntimeException();
        validateOutput(input);
        if ((long) input.getTokenIdsCount() + generation.getMaxNewTokens() >= config.maxSeqLen())
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Mock requires input + output below max_seq_len"), 605);
        if (input.getTokenIdsCount() < 1 || input.getTokenIdsCount() >= config.maxSeqLen()
                || generation.getTimeoutMs() < 0 || generation.getMaxNewTokens() < 1 || generation.getMinNewTokens() < 0
                || generation.getMinNewTokens() > generation.getMaxNewTokens() || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid decode allocation settings"), 605);
    }

    private void allocate(Call call) {
        long observed = Long.MIN_VALUE;
        try {
            while (true) {
                synchronized (this) {
                    if (call.terminal) return;
                    if (System.nanoTime() >= call.deadline) {
                        stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode expired while waiting for capacity"), 603); return;
                    }
                    long current = cache.allocationGeneration();
                    if (current != observed) {
                        boolean reuse = config.reuseCache() && config.deviceCache() && !config.disableDecodeFirstReuse()
                            && call.input.getGenerateConfig().getReuseCache();
                        var full = PrefixKeys.completeBlocks(call.input, config.blockSize());
                        var keys = reuse ? full.subList(0, Math.min(full.size(), (call.input.getTokenIdsCount() - 1) / config.blockSize())) : List.<Long>of();
                        int blocks = (int) (((long) call.input.getTokenIdsCount() + config.blockSize() - 1) / config.blockSize());
                        var admission = cache.acquire(blocks, keys); observed = current;
                        if (admission.lease() != null) {
                            call.lease = admission.lease(); call.reused = call.lease.reusedBlocks() * config.blockSize();
                            metrics.cacheLookup(call.input.getTokenIdsCount(), call.reused);
                            callbacks.execute(() -> acknowledge(call));
                            return;
                        }
                        if (admission.failure() == DeviceCache.Failure.PERMANENT) {
                            stop(call, Status.RESOURCE_EXHAUSTED.withDescription("Decode demand exceeds usable KV capacity"), 602); return;
                        }
                    }
                }
                // Timed wakes service interruption/deadlines; allocation is retried only after generation changes.
                cache.awaitCapacityChange(observed, 50);
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt(); stop(call, Status.CANCELLED.withDescription("Decode allocation interrupted"), 8100);
        } catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode allocation failed: " + error.getMessage()), 0); }
    }

    private synchronized void acknowledge(Call call) {
        if (call.terminal) return;
        if (System.nanoTime() >= call.deadline) {
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
        long budget = Math.min(TimeUnit.MILLISECONDS.toNanos(config.loadTimeoutMs()), call.deadline == Long.MAX_VALUE
            ? Long.MAX_VALUE : Math.max(1, call.deadline - System.nanoTime()));
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
        if (System.nanoTime() >= call.deadline) { stop(call, Status.DEADLINE_EXCEEDED.withDescription("LOAD exceeded request budget"), 603); return; }
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

    static void validateOutput(GenerateInputPB input) {
        var g = input.getGenerateConfig();
        if (g.getReturnPromptLogits() || g.getReturnHiddenStates() || g.getReturnLogits() || g.getReturnAllHiddenStates()
                || g.getReturnAllProbs() || g.getReturnSoftmaxProbs() || g.getReturnCumLogProbs() || g.getCalculateLoss() != 0
                || g.getStopWordsList().getRowsCount() > 0 || g.getSpEdit() || g.getSpInputLookup())
            throw Status.UNIMPLEMENTED.withDescription("Extra outputs, stop words and speculative generation are not supported").asRuntimeException();
    }

    private synchronized void resume(Call call) {
        if (!call.terminal && call.backpressured) { call.backpressured = false; ready.addLast(call); advance(); }
    }

    private boolean advancing;
    /** One simulated execution slot, fair round-robin by round; capacity waits do not block other calls. */
    private void advance() {
        if (advancing) return;
        advancing = true;
        try {
            while (!closed && running == null && !ready.isEmpty()) {
                var call = ready.removeFirst();
                if (call.terminal) continue;
                if (!call.response.isReady()) { call.backpressured = true; continue; }
                if (System.nanoTime() >= call.deadline) { stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode expired in queue"), 603); continue; }
                try {
                    if (call.step == null) call.step = config.decodeModel().sample(Long.toString(call.input.getRequestId()), call.iterations,
                        call.input.getGenerateConfig().getMaxNewTokens() - call.generated);
                    int contextTokens = Math.addExact(Math.addExact(call.input.getTokenIdsCount(), call.generated), call.step.proposedTokens());
                    int blocks = (int) (((long) contextTokens + config.blockSize() - 1) / config.blockSize());
                    var failure = call.lease.growTo(blocks);
                    if (failure == DeviceCache.Failure.PERMANENT) { stop(call, Status.RESOURCE_EXHAUSTED.withDescription("Decode growth exceeds capacity"), 602); continue; }
                    if (failure != null) {
                        call.allocation = workers.submit(() -> grow(call, blocks)); continue;
                    }
                    running = call; call.stepStarted = System.nanoTime();
                    var request = new ForwardBatch.Request(call.key.client() + "/" + call.key.request(), call.input.getTokenIdsCount(),
                        call.reused, call.input.getGenerateConfig().getMaxNewTokens(), call.generated, call.step.verifyTokens(), call.input.getPriority(), List.of());
                    var batch = new ForwardBatch(call.key.client() + "/" + call.key.request() + "/" + call.generated,
                        List.of(new ForwardBatch.Partition(0, List.of(request))));
                    call.execution = compute.submit(config.compute().plan(Role.decode, config.id(), batch));
                    call.execution.completion().thenAccept(outcome -> stepped(call, outcome));
                } catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode step failed: " + error.getMessage()), 0); }
            }
        } finally { advancing = false; }
    }

    private void grow(Call call, int blocks) {
        try {
            long observed = Long.MIN_VALUE;
            while (true) {
                synchronized (this) {
                    if (call.terminal) return;
                    long generation = cache.allocationGeneration();
                    if (observed != generation) {
                        var failure = call.lease.growTo(blocks); observed = cache.allocationGeneration();
                        if (failure == null) { ready.addLast(call); advance(); return; }
                        if (failure == DeviceCache.Failure.PERMANENT) { stop(call, Status.RESOURCE_EXHAUSTED.withDescription("Decode growth exceeds capacity"), 602); return; }
                    }
                }
                cache.awaitCapacityChange(observed, 50);
            }
        } catch (InterruptedException error) { Thread.currentThread().interrupt(); stop(call, Status.CANCELLED.withDescription("Decode growth interrupted"), 8100); }
        catch (RuntimeException error) { stop(call, Status.INTERNAL.withDescription("Decode growth failed: " + error.getMessage()), 0); }
    }

    private synchronized void stepped(Call call, SimulatedCompute.Outcome outcome) {
        if (call.terminal) return;
        if (outcome.state() != SimulatedCompute.State.COMPLETED) { stop(call, Status.CANCELLED.withDescription("Decode compute cancelled"), 8100); return; }
        if (System.nanoTime() >= call.deadline) { stop(call, Status.DEADLINE_EXCEEDED.withDescription("Decode expired during Forward"), 603); return; }
        if (call.response.isCancelled()) { stop(call, Status.CANCELLED.withDescription("Decode cancelled"), 8100); return; }
        int committed = Math.min(call.step.committedTokens(), call.stopAfter - call.generated);
        call.executionNanos += outcome.elapsedNanos(); call.generated += committed; call.iterations++; running = null;
        metrics.forwardCompleted(outcome.elapsedNanos(), call.step.verifyTokens(), call.step.verifyTokens(), committed);
        call.step = null;
        boolean finished = call.generated >= call.stopAfter;
        try {
            var output = LocalResponse.token(call.input, call.token, call.generated, committed, call.reused, System.nanoTime() - call.started, finished, true).toBuilder();
            output.getFlattenOutputBuilder().getAuxInfoBuilder(0).setIterCount(call.iterations + 1);
            call.response.onNext(output.build());
        }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("Decode output consumer closed"), 8100); return; }
        if (finished) stop(call, Status.OK, 0);
        else { ready.addLast(call); advance(); }
    }

    private synchronized void stop(Call call, Status status, long code) {
        if (call.terminal) return;
        call.terminal = true;
        if (call.timeout != null) call.timeout.cancel(false);
        if (call.allocation != null) call.allocation.cancel(true);
        if (call.source != null) { call.source.cancel(null); call.source = null; }
        if (call.execution != null) call.execution.cancel();
        ready.remove(call);
        if (running == call) running = null;
        if (call.lease != null) {
            if (status.isOk() && config.reuseCache() && config.deviceCache() && call.input.getGenerateConfig().getReuseCache())
                call.lease.complete(PrefixKeys.completeBlocks(call.input, config.blockSize()));
            else call.lease.close();
        }
        if (call.input != null) {
            active.remove(call.key, call);
            var finished = task(call).setEndTimeMs(System.currentTimeMillis()).setIterateCount(call.iterations);
            if (!status.isOk()) finished.setErrorInfo(ErrorDetailsPB.newBuilder()
                .setErrorCode(code == 0 ? status.getCode().value() : code).setErrorMessage(status.getDescription()));
            history.add(finished.build());
            metrics.finished(switch (status.getCode()) {
                case OK -> EngineMetrics.Outcome.success;
                case CANCELLED -> EngineMetrics.Outcome.cancelled;
                case DEADLINE_EXCEEDED -> EngineMetrics.Outcome.timeout;
                default -> EngineMetrics.Outcome.error;
            });
        }
        Runnable reply = () -> {
            if (!call.response.isCancelled()) {
                if (status.isOk()) call.response.onCompleted();
                else call.response.onError(RtpErrors.failure(status, code));
            }
        };
        try { callbacks.execute(reply); }
        catch (RejectedExecutionException closing) { reply.run(); }
        advance();
    }

    synchronized TaskHistory.Snapshot snapshot(long latest) {
        return history.snapshot(latest, active.values().stream().map(c -> task(c).build()).toList(), active.size() - (running == null ? 0 : 1), running == null ? 0 : 1);
    }
    private TaskInfoPB.Builder task(Call call) {
        long now = System.nanoTime();
        long executing = call.executionNanos + (running == call ? Math.max(0, now - call.stepStarted) : 0);
        return TaskInfoPB.newBuilder().setRequestId(call.input.getRequestId()).setInputLength(call.input.getTokenIdsCount())
            .setPrefixLength(call.reused).setPriority(call.input.getPriority()).setDpRank(0).setIsWaiting(running != call)
            .setPhase(running == call ? TaskPhase.TASK_PHASE_RUNNING : call.lease == null ? TaskPhase.TASK_PHASE_RECEIVED : TaskPhase.TASK_PHASE_KV_ALLOCATED)
            .setWaitingTimeMs(TimeUnit.NANOSECONDS.toMillis(Math.max(0, now - call.started - executing)))
            .setExecutionTimeMs(TimeUnit.NANOSECONDS.toMillis(executing));
    }
    String scrapeMetrics() { return metrics.scrape(); }
    private synchronized EngineMetrics.State metricState() {
        var kv = cache.snapshot(0, false); long size = config.blockSize();
        return new EngineMetrics.State(active.size() - (running == null ? 0 : 1), running == null ? 0 : 1, kv.totalBlocks() * size, kv.availableBlocks() * size,
            kv.freeBlocks() * size, kv.heldBlocks() * size);
    }
    private static long after(long now, long nanos) {
        try { return Math.addExact(now, nanos); }
        catch (ArithmeticException overflow) { return nanos < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            for (var call : List.copyOf(active.values())) stop(call, Status.UNAVAILABLE.withDescription("Decode node closed"), 0);
        }
        workers.shutdownNow(); deadlines.shutdownNow(); compute.close(); channels.close(); callbacks.shutdown();
    }
}

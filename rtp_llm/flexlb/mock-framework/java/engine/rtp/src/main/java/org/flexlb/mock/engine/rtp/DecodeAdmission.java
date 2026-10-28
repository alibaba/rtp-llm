package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.metrics.EngineMetrics;
import org.flexlb.mock.engine.rtp.transfer.*;

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
        ManagedChannel source;
        int reused;

        Call(ServerCallStreamObserver<GenerateOutputsPB> response, Context context) {
            this.response = response; this.context = context;
            rpcDeadline = context.getDeadline() == null ? Long.MAX_VALUE : after(System.nanoTime(), context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
            response.setOnCancelHandler(() -> {
                boolean expired = context.getDeadline() != null && context.getDeadline().isExpired();
                stop(this, expired ? Status.DEADLINE_EXCEEDED.withDescription("Decode RPC deadline expired")
                    : Status.CANCELLED.withDescription("Decode RPC cancelled"), expired ? 603 : 8100);
            });
        }
        @Override public void onNext(GenerateRequestPB request) { accept(this, request); }
        @Override public void onError(Throwable error) { stop(this, Status.CANCELLED.withDescription("Decode stream closed"), 8100); }
        @Override public void onCompleted() { stop(this, Status.CANCELLED.withDescription("P closed before LOAD/generation"), 8100); }
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
                if (call.loaded && request.getStage() == RemoteStage.GENERATE)
                    throw Status.UNIMPLEMENTED.withDescription("Decode token generation is not implemented").asRuntimeException();
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
        try { call.acknowledged = true; call.response.onNext(GenerateOutputsPB.getDefaultInstance()); }
        catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("Decode acknowledgement failed"), 8100); }
    }

    private void load(Call call) {
        if (call.peers.size() != 1 || call.peers.getFirst().isBlank())
            throw Status.FAILED_PRECONDITION.withDescription("Mock LOAD requires one P transfer address in ALLOCATE.peer_addrs").asRuntimeException();
        call.loading = true;
        call.source = ManagedChannelBuilder.forTarget(call.peers.getFirst()).usePlaintext().build();
        long budget = Math.min(TimeUnit.MILLISECONDS.toNanos(config.loadTimeoutMs()), call.deadline == Long.MAX_VALUE
            ? Long.MAX_VALUE : Math.max(1, call.deadline - System.nanoTime()));
        PrefillTransferGrpc.newStub(call.source).withDeadlineAfter(budget, TimeUnit.NANOSECONDS).read(
            TransferRequest.newBuilder().setClientId(call.key.client()).setInput(call.input).setBlockSize(config.blockSize()).build(), new StreamObserver<>() {
                private TransferMetadata metadata;
                @Override public void onNext(TransferMetadata value) { metadata = value; }
                @Override public void onError(Throwable error) {
                    var status = Status.fromThrowable(error);
                    stop(call, status.augmentDescription("P-to-D LOAD failed"), status.getCode() == Status.Code.DEADLINE_EXCEEDED ? 603 : 8212);
                }
                @Override public void onCompleted() { loaded(call, metadata); }
            });
    }

    private synchronized void loaded(Call call, TransferMetadata metadata) {
        if (call.terminal) return;
        if (System.nanoTime() >= call.deadline) { stop(call, Status.DEADLINE_EXCEEDED.withDescription("LOAD exceeded request budget"), 603); return; }
        if (call.response.isCancelled()) { stop(call, Status.CANCELLED.withDescription("LOAD cancelled"), 8100); return; }
        var keys = PrefixKeys.completeBlocks(call.input, config.blockSize());
        if (metadata == null || metadata.getBlockSize() != config.blockSize()
                || metadata.getInputTokens() != call.input.getTokenIdsCount() || !metadata.getCacheKeysList().equals(keys)) {
            stop(call, Status.FAILED_PRECONDITION.withDescription("P-to-D KV layout differs"), 8212); return;
        }
        call.loading = false; call.loaded = true;
        call.source.shutdownNow(); call.source = null;
        try {
            // The transferred prefix is not a historical D cache hit. Keep the lease private until terminal cleanup.
            call.response.onNext(GenerateOutputsPB.newBuilder().setRequestId(call.key.request())
                .setErrorInfo(RpcErrorPB.newBuilder().setErrorCodeValue(0)).build());
        } catch (RuntimeException error) { stop(call, Status.CANCELLED.withDescription("LOAD acknowledgement failed"), 8100); }
    }

    private synchronized void stop(Call call, Status status, long code) {
        if (call.terminal) return;
        call.terminal = true;
        if (call.timeout != null) call.timeout.cancel(false);
        if (call.allocation != null) call.allocation.cancel(true);
        if (call.source != null) { call.source.shutdownNow(); call.source = null; }
        if (call.lease != null) call.lease.close(); // Partial P transfer is not committed as a reusable D prefix on cancellation.
        if (call.input != null) {
            active.remove(call.key, call);
            history.add(task(call).setEndTimeMs(System.currentTimeMillis()).setErrorInfo(ErrorDetailsPB.newBuilder()
                .setErrorCode(code == 0 ? status.getCode().value() : code).setErrorMessage(status.getDescription())).build());
            metrics.finished(switch (status.getCode()) {
                case CANCELLED -> EngineMetrics.Outcome.cancelled;
                case DEADLINE_EXCEEDED -> EngineMetrics.Outcome.timeout;
                default -> EngineMetrics.Outcome.error;
            });
        }
        Runnable reply = () -> { if (!call.response.isCancelled()) call.response.onError(RtpErrors.failure(status, code)); };
        try { callbacks.execute(reply); }
        catch (RejectedExecutionException closing) { reply.run(); } // A frame can arrive on an unaccepted stream during server shutdown.
    }

    synchronized TaskHistory.Snapshot snapshot(long latest) {
        return history.snapshot(latest, active.values().stream().map(c -> task(c).build()).toList(), active.size(), 0);
    }
    private TaskInfoPB.Builder task(Call call) {
        return TaskInfoPB.newBuilder().setRequestId(call.input.getRequestId()).setInputLength(call.input.getTokenIdsCount())
            .setPrefixLength(call.reused).setPriority(call.input.getPriority()).setDpRank(0).setIsWaiting(true)
            .setPhase(call.lease == null ? TaskPhase.TASK_PHASE_RECEIVED : TaskPhase.TASK_PHASE_KV_ALLOCATED)
            .setWaitingTimeMs(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - call.started));
    }
    String scrapeMetrics() { return metrics.scrape(); }
    private synchronized EngineMetrics.State metricState() {
        var kv = cache.snapshot(0, false); long size = config.blockSize();
        return new EngineMetrics.State(active.size(), 0, kv.totalBlocks() * size, kv.availableBlocks() * size,
            kv.freeBlocks() * size, kv.heldBlocks() * size);
    }
    private static long after(long now, long nanos) {
        try { return Math.addExact(now, nanos); }
        catch (ArithmeticException overflow) { return nanos < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
    }
    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        for (var call : List.copyOf(active.values())) stop(call, Status.UNAVAILABLE.withDescription("Decode node closed"), 0);
        workers.shutdownNow(); deadlines.shutdownNow(); callbacks.shutdown();
    }
}

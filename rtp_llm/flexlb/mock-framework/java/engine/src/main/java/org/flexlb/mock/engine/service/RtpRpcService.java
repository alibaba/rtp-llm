package org.flexlb.mock.engine.service;

import io.grpc.Status;
import io.prometheus.client.Collector.MetricFamilySamples;
import io.prometheus.client.CollectorRegistry;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.flexlb.mock.metrics.MetricTags;
import org.flexlb.mock.metrics.EventMetrics;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.forward.ComputeModel;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.execution.EngineLoop;
import org.flexlb.mock.engine.scheduler.TaskHistory;

/**
 * Native RPC adapter; P and D owners manage request state and KV independently.
 */
public final class RtpRpcService extends RpcServiceGrpc.RpcServiceImplBase implements AutoCloseable {
    private final CollectorRegistry rpcRegistry = new CollectorRegistry();
    private final EventMetrics rpcMetrics;
    private final Map<String, String> metricTags;
    private final Config config;
    private final KvCacheManager cache;
    private final TaskHistory finished;
    private final PrefillRpcServer prefill;
    private final DecodeRpcServer decode;

    private final boolean groups;

    private final AtomicLong onflightRequests = new AtomicLong();
    private final AtomicLong statusVersion = new AtomicLong();

    private volatile boolean closed;

    public RtpRpcService(Config config, KvCacheManager cache, Object executionLock, TaskHistory finished,
                         EngineLoop engine, boolean groups) {
        this.config = config;
        this.groups = groups;
        this.cache = cache; this.finished = finished;
        var tags = new HashMap<>(MetricTags.fromEnvironment(System.getenv()));
        tags.put("dp_rank", "0");
        metricTags = Map.copyOf(tags);
        rpcMetrics = new EventMetrics(config.role().name(), metricTags);
        rpcMetrics.register(rpcRegistry);
        prefill = config.role() == Role.prefill
            ? new PrefillRpcServer(config, executionLock, finished, engine, this::recordCancellation, this::trackRequest) : null;
        decode = prefill == null ? new DecodeRpcServer(config, executionLock, engine, this::recordCancellation, this::trackRequest) : null;
    }

    public void close() {
        closed = true;
        RuntimeException failure = null;
        for (AutoCloseable component : new AutoCloseable[]{prefill, decode}) {
            if (component == null) continue;
            try { component.close(); }
            catch (Exception error) {
                if (failure == null) failure = new IllegalStateException("Cannot close RTP component", error);
                else failure.addSuppressed(error);
            }
        }
        if (failure != null) throw failure;
    }

    @Override public void executeFunction(FunctionRequestPB input, StreamObserver<FunctionResponsePB> response) {
        if (input.hasRemoteRequest()) {
            var operation = input.getRemoteRequest().getOp();
            if (operation == RemoteOpType.REMOTE_OPERATION_READ || operation == RemoteOpType.REMOTE_OPERATION_WRITE) {
                // No SDK or external I/O was submitted: failure is explicitly buffer-safe.
                response.onNext(FunctionResponsePB.newBuilder().setRemoteResponse(RemoteOperationResponsePB.newBuilder()
                    .setTransferStatus(RemoteTransferStatusPB.REMOTE_TRANSFER_STATUS_FAILED_BUFFER_SAFE).setPayloadBufferSafe(true)).build());
                response.onCompleted(); return;
            }
        }
        response.onError(Status.UNIMPLEMENTED.withDescription("KVCM payload SDK and metadata backend operations are not simulated").asRuntimeException());
    }

    @Override public StreamObserver<GenerateRequestPB> remoteGenerate(StreamObserver<GenerateOutputsPB> observer) {
        return decode == null ? super.remoteGenerate(observer) : decode.open(observer);
    }

    @Override public void enqueueBatch(EnqueueBatchRequestPB request, StreamObserver<EnqueueBatchResponsePB> response) {
        if (prefill == null || !groups) { super.enqueueBatch(request, response); return; }
        PrefillEnqueueHandler.enqueue(prefill, request, response);
    }

    @Override public void enqueueGroup(EnqueueGroupRequestPB request, StreamObserver<EnqueueBatchResponsePB> response) {
        if (prefill == null || !groups) { super.enqueueGroup(request, response); return; }
        var slot = EnqueueBatchDpSlotPB.newBuilder().setDpRank(request.getDpRank());
        for (var item : request.getRequestsList()) {
            var external = EnqueueBatchExternalInputPB.newBuilder();
            if (item.hasInput()) external.setInput(item.getInput());
            slot.addRequests(external);
        }
        PrefillEnqueueHandler.enqueue(prefill, EnqueueBatchRequestPB.newBuilder().setBatchId(request.getBatchId())
            .setFetchAttachTimeoutMs(request.getFetchAttachTimeoutMs()).addDpSlots(slot).build(), response);
    }

    @Override public void fetchResponse(FetchRequestPB request, StreamObserver<GenerateOutputsPB> response) {
        if (prefill == null) { super.fetchResponse(request, response); return; }
        prefill.fetch(request.getRequestId(), response);
    }

    @Override public void cancel(CancelRequestPB request, StreamObserver<CancelResponsePB> response) {
        if (prefill == null) { super.cancel(request, response); return; }
        reply(response, () -> CancelResponsePB.newBuilder().setStatus(prefill.cancel(request.getRequestId())).build());
    }

    @Override public void generateStreamCall(GenerateInputPB request, StreamObserver<GenerateOutputsPB> observer) {
        if (prefill == null) { super.generateStreamCall(request, observer); return; }
        prefill.generate(request, observer);
    }

    @Override public void checkHealth(EmptyPB request, StreamObserver<CheckHealthResponsePB> response) {
        reply(response, () -> { checkHealthy(); return CheckHealthResponsePB.newBuilder().setHealth("OK").build(); });
    }

    public void checkHealthy() {
        if (closed) throw new IllegalStateException("Engine closed");
        if (prefill == null) decode.checkHealthy(); else prefill.checkHealthy();
    }

    @Override public void getWorkerStatus(StatusVersionPB request, StreamObserver<WorkerStatusPB> response) {
        statusReply("worker_status", response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), false);
            String role = config.role().name().toUpperCase(java.util.Locale.ROOT);
            long micros = Math.multiplyExact(System.currentTimeMillis(), 1000);
            long version = statusVersion.updateAndGet(previous -> Math.max(previous + 1, micros));
            var tasks = prefill == null ? decode.snapshot(request.getLatestFinishedVersion()) : prefill.snapshot(request.getLatestFinishedVersion());
            long collected = System.nanoTime();
            finished.reportGaps();
            var result = WorkerStatusPB.newBuilder().setRole("RoleType." + role)
                .setRoleType(RoleTypePB.valueOf("ROLE_TYPE_" + role)).setAlive(true).setKvCacheGroupMode(KvCacheGroupModePB.KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY)
                .setBlockSize(config.physicalBlockSize()).setBlockHashLookaheadTokens(0).setCacheMatchRollbackBlocks(0)
                .setDpSize(1).setDpRank(0).setTpSize(config.cpSize()).setPrecision("FP16")
                .setStatusVersion(version).setMaxSeqLen(config.maxSeqLen()).setMaxBatchTokensSize(config.maxBatchTokens())
                .setTotalKvCache(tokens(snapshot.totalBlocks())).setAvailableKvCache(tokens(snapshot.availableBlocks()))
                .addAllRunningTaskInfo(tasks.tasks().stream().map(RpcMessages::task).toList())
                .addAllFinishedTaskList(tasks.finished().stream().map(RpcMessages::task).toList()).setLatestFinishedVersion(tasks.finishedVersion())
                .build();
            return new StatusResult<>(result, collected);
        });
    }

    @Override public void getCacheStatus(CacheVersionPB request, StreamObserver<CacheStatusPB> response) {
        statusReply("cache_status", response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), request.getNeedCacheKeys());
            // Real measures cache collection only, before protobuf construction.
            long collected = System.nanoTime();
            var result = CacheStatusPB.newBuilder().setTotalKvCache(tokens(snapshot.totalBlocks()))
                .setAvailableKvCache(tokens(snapshot.availableBlocks())).setBlockSize(config.tokensPerCacheKey()).setVersion(snapshot.version());
            if (decode != null) result.setMockDecodeCacheSeed(config.cacheKeySeed());
            for (long key : snapshot.keys()) result.addCacheKeyList(key);
            return new StatusResult<>(result.build(), collected);
        });
    }

    public List<MetricFamilySamples> collectMetrics() {
        return MetricTags.collect(rpcRegistry, metricTags);
    }

    // Real samples at GenerateContext destruction, before its AtomicGuard decrements.
    private Runnable trackRequest() {
        onflightRequests.incrementAndGet();
        var completed = new java.util.concurrent.atomic.AtomicBoolean();
        return () -> {
            if (completed.compareAndSet(false, true))
                rpcMetrics.observe("rtp_llm_rpc_onflight_request", onflightRequests.getAndDecrement());
        };
    }

    private void recordCancellation() {
        rpcMetrics.increment("rtp_llm_rpc_cancel_qps");
    }

    private <T> void statusReply(String kind, StreamObserver<T> observer, Supplier<StatusResult<T>> read) {
        long started = System.nanoTime();
        T value = null;
        RuntimeException failure = null;
        long completed;
        try {
            if (closed) throw Status.UNAVAILABLE.withDescription("RTP node is closed").asRuntimeException();
            var result = read.get(); value = result.value(); completed = result.collected();
        } catch (RuntimeException error) { failure = error; completed = System.nanoTime(); }
        long elapsed = completed - started;
        rpcMetrics.increment("rtp_llm_rpc_" + kind + "_qps");
        rpcMetrics.observe("rtp_llm_rpc_" + kind + "_total_rt_us", elapsed / 1000.0);
        if (failure != null) {
            observer.onError(failure instanceof io.grpc.StatusRuntimeException ? failure
                : Status.INTERNAL.withCause(failure).withDescription(failure.getMessage()).asRuntimeException());
        } else { observer.onNext(value); observer.onCompleted(); }
    }

    private long tokens(int blocks) { return (long) blocks * config.tokensPerCacheKey(); }

    private <T> void reply(StreamObserver<T> observer, Supplier<T> read) {
        if (closed) { observer.onError(Status.UNAVAILABLE.withDescription("RTP node is closed").asRuntimeException()); return; }
        T value;
        try { value = read.get(); }
        catch (io.grpc.StatusRuntimeException error) { observer.onError(error); return; }
        catch (RuntimeException error) { observer.onError(Status.INTERNAL.withCause(error).withDescription(error.getMessage()).asRuntimeException()); return; }
        observer.onNext(value); observer.onCompleted();
    }

    /**
     * Validates the identity, limits, and cache-key layout of an RTP endpoint.
     */
    public record Config(String id, Role role, int physicalBlockSize, int maxSeqLen,
                         int maxBatchTokens, ComputeModel compute, int imageToken, long cacheKeySeed, int cpSize) {
        public Config {
            if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                    || physicalBlockSize < 1 || cpSize < 1 || maxSeqLen < 1 || maxBatchTokens < 0)
                throw new IllegalArgumentException("Invalid RTP endpoint configuration");
            Objects.requireNonNull(compute);
        }

        public int tokensPerCacheKey() { return Math.multiplyExact(physicalBlockSize, cpSize); }

        public static Config from(RtpConfig config) {
            return new Config(config.id(), config.role(), config.physicalBlockSize(), config.maxSeqLen(),
                config.maxBatchTokens(), config.compute(), config.imageToken(), config.cacheKeySeed(), config.cpSize());
        }
    }

    private record StatusResult<T>(T value, long collected) {}
}

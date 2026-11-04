package org.flexlb.mock.engine.service;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.execution.RtpEngineLoop;
import org.flexlb.mock.engine.scheduler.TaskHistory;

/** Native RPC adapter; P and D owners manage request state and KV independently. */
public final class RtpRpcService extends RpcServiceGrpc.RpcServiceImplBase implements AutoCloseable {
    private final RtpServiceConfig config;
    private final KvCache cache;
    private final PrefillRpcServer prefill;
    private final DecodeRpcServer decode;
    private final boolean groups;
    private final AtomicLong statusVersion = new AtomicLong();
    private volatile boolean closed;

    public RtpRpcService(RtpServiceConfig config, KvCache cache, Object executionLock, TaskHistory finished,
                         RtpEngineLoop engine, boolean groups) {
        this.config = config;
        this.groups = groups;
        this.cache = cache;
        prefill = config.role() == org.flexlb.mock.api.model.Types.Role.prefill
            ? new PrefillRpcServer(config, executionLock, finished, engine) : null;
        decode = prefill == null ? new DecodeRpcServer(config, executionLock, engine) : null;
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
    public void checkHealthy() {
        if (closed) throw new IllegalStateException("Engine closed");
        if (prefill == null) decode.checkHealthy(); else prefill.checkHealthy();
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
        PrefillBatch.enqueue(prefill, request, response);
    }

    @Override public void enqueueGroup(EnqueueGroupRequestPB request, StreamObserver<EnqueueBatchResponsePB> response) {
        if (prefill == null || !groups) { super.enqueueGroup(request, response); return; }
        var slot = EnqueueBatchDpSlotPB.newBuilder().setDpRank(request.getDpRank());
        for (var item : request.getRequestsList()) {
            var external = EnqueueBatchExternalInputPB.newBuilder();
            if (item.hasInput()) external.setInput(item.getInput());
            slot.addRequests(external);
        }
        PrefillBatch.enqueue(prefill, EnqueueBatchRequestPB.newBuilder().setBatchId(request.getBatchId())
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

    @Override public void getWorkerStatus(StatusVersionPB request, StreamObserver<WorkerStatusPB> response) {
        reply(response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), false);
            String role = config.role().name().toUpperCase(java.util.Locale.ROOT);
            long micros = Math.multiplyExact(System.currentTimeMillis(), 1000);
            long version = statusVersion.updateAndGet(previous -> Math.max(previous + 1, micros));
            var tasks = prefill == null ? decode.snapshot(request.getLatestFinishedVersion()) : prefill.snapshot(request.getLatestFinishedVersion());
            return WorkerStatusPB.newBuilder().setRole("RoleType." + role)
                .setRoleType(RoleTypePB.valueOf("ROLE_TYPE_" + role)).setAlive(true).setKvCacheGroupMode(KvCacheGroupModePB.KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY)
                .setDpSize(config.dpSize()).setDpRank(0).setTpSize(config.tpSize()).setPrecision("FP16")
                .setStatusVersion(version).setMaxSeqLen(config.maxSeqLen()).setMaxBatchTokensSize(config.maxBatchTokens())
                .setTotalKvCache(tokens(snapshot.totalBlocks())).setAvailableKvCache(tokens(snapshot.availableBlocks()))
                .addAllRunningTaskInfo(tasks.tasks().stream().map(RpcTasks::task).toList())
                .addAllFinishedTaskList(tasks.finished().stream().map(RpcTasks::task).toList()).setLatestFinishedVersion(tasks.finishedVersion())
                .build();
        });
    }

    @Override public void getCacheStatus(CacheVersionPB request, StreamObserver<CacheStatusPB> response) {
        reply(response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), request.getNeedCacheKeys());
            var result = CacheStatusPB.newBuilder().setTotalKvCache(tokens(snapshot.totalBlocks()))
                .setAvailableKvCache(tokens(snapshot.availableBlocks())).setBlockSize(config.blockSize()).setVersion(snapshot.version());
            snapshot.keys().forEach(key -> result.putCacheKeys(key, true));
            return result.build();
        });
    }

    private long tokens(int blocks) { return (long) blocks * config.blockSize(); }
    private <T> void reply(StreamObserver<T> observer, Supplier<T> read) {
        if (closed) { observer.onError(Status.UNAVAILABLE.withDescription("RTP node is closed").asRuntimeException()); return; }
        T value;
        try { value = read.get(); }
        catch (io.grpc.StatusRuntimeException error) { observer.onError(error); return; }
        catch (RuntimeException error) { observer.onError(Status.INTERNAL.withCause(error).withDescription(error.getMessage()).asRuntimeException()); return; }
        observer.onNext(value); observer.onCompleted();
    }
}

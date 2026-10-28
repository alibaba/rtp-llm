package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import io.grpc.stub.ServerCallStreamObserver;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;

/** Native RPC adapter; P and D owners manage request state and KV independently. */
public final class RtpRpcService extends RpcServiceGrpc.RpcServiceImplBase {
    private final RtpConfig config;
    private final DeviceCache cache;
    private final LocalPrefill requests;
    private final DecodeAdmission decode;
    private final AtomicLong statusVersion = new AtomicLong();
    private volatile boolean closed;

    public RtpRpcService(RtpConfig config, DeviceCache cache) {
        this.config = config; this.cache = cache;
        requests = config.role() == org.flexlb.mock.api.model.Types.Role.prefill ? new LocalPrefill(config, cache) : null;
        decode = requests == null ? new DecodeAdmission(config, cache) : null;
    }
    public void close() { closed = true; if (requests != null) requests.close(); if (decode != null) decode.close(); }
    String scrapeMetrics() { return requests == null ? decode.scrapeMetrics() : requests.scrapeMetrics(); }

    @Override public StreamObserver<GenerateRequestPB> remoteGenerate(StreamObserver<GenerateOutputsPB> observer) {
        return decode == null ? super.remoteGenerate(observer) : decode.open(observer);
    }

    @Override public void generateStreamCall(GenerateInputPB request, StreamObserver<GenerateOutputsPB> observer) {
        if (requests == null) { super.generateStreamCall(request, observer); return; }
        var response = (ServerCallStreamObserver<GenerateOutputsPB>) observer;
        // Cancellation can happen before submit returns; capture the exact session, never just its reusable id.
        var session = new java.util.concurrent.atomic.AtomicReference<LocalPrefill.Session>();
        var context = io.grpc.Context.current();
        Runnable cancel = () -> {
            var current = session.get();
            if (current != null) {
                if (context.getDeadline() != null && context.getDeadline().isExpired()) current.expire();
                else current.cancel();
            }
        };
        response.setOnCancelHandler(cancel);
        LocalPrefill.Session accepted;
        long deadline = Long.MAX_VALUE;
        if (context.getDeadline() != null) {
            long now = System.nanoTime(), remaining = context.getDeadline().timeRemaining(java.util.concurrent.TimeUnit.NANOSECONDS);
            try { deadline = Math.addExact(now, remaining); }
            catch (ArithmeticException overflow) { deadline = remaining < 0 ? Long.MIN_VALUE : Long.MAX_VALUE; }
        }
        try { accepted = requests.submit(request, deadline); }
        catch (io.grpc.StatusRuntimeException error) { response.onError(error); return; }
        session.set(accepted);
        if (response.isCancelled()) cancel.run();
        accepted.completion().thenAccept(result -> {
            if (response.isCancelled()) return;
            if (!result.status().isOk()) { response.onError(RtpErrors.failure(result.status(), result.engineCode())); return; }
            try { response.onNext(result.output()); response.onCompleted(); }
            catch (RuntimeException error) { accepted.cancel(); }
        });
    }

    @Override public void checkHealth(EmptyPB request, StreamObserver<CheckHealthResponsePB> response) {
        reply(response, () -> CheckHealthResponsePB.newBuilder().setHealth("OK").build());
    }

    @Override public void getWorkerStatus(StatusVersionPB request, StreamObserver<WorkerStatusPB> response) {
        reply(response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), false);
            String role = config.role().name().toUpperCase(java.util.Locale.ROOT);
            long micros = Math.multiplyExact(System.currentTimeMillis(), 1000);
            long version = statusVersion.updateAndGet(previous -> Math.max(previous + 1, micros));
            var tasks = requests == null ? decode.snapshot(request.getLatestFinishedVersion()) : requests.snapshot(request.getLatestFinishedVersion());
            return WorkerStatusPB.newBuilder().setRole("RoleType." + role)
                .setRoleType(RoleTypePB.valueOf("ROLE_TYPE_" + role)).setAlive(true)
                .setDpSize(1).setDpRank(0).setTpSize(config.tpSize()).setPrecision("FP16")
                .setStatusVersion(version).setMaxSeqLen(config.maxSeqLen()).setMaxBatchTokensSize(config.maxBatchTokens())
                .setTotalKvCache(tokens(snapshot.totalBlocks())).setAvailableKvCache(tokens(snapshot.availableBlocks()))
                .addAllRunningTaskInfo(tasks.tasks()).addAllFinishedTaskList(tasks.finished()).setLatestFinishedVersion(tasks.finishedVersion())
                .setWaitingQueryLen(tasks.waiting()).setRunningQueryLen(tasks.running())
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

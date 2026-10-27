package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;

/** Native control-plane RPCs only. Unimplemented generation methods retain gRPC UNIMPLEMENTED. */
public final class RtpStatusService extends RpcServiceGrpc.RpcServiceImplBase {
    private final RtpConfig config;
    private final DeviceCache cache;
    private final AtomicLong statusVersion = new AtomicLong();
    private volatile boolean closed;

    public RtpStatusService(RtpConfig config, DeviceCache cache) { this.config = config; this.cache = cache; }
    public void close() { closed = true; }

    @Override public void checkHealth(EmptyPB request, StreamObserver<CheckHealthResponsePB> response) {
        reply(response, () -> CheckHealthResponsePB.newBuilder().setHealth("OK").build());
    }

    @Override public void getWorkerStatus(StatusVersionPB request, StreamObserver<WorkerStatusPB> response) {
        reply(response, () -> {
            var snapshot = cache.snapshot(request.getLatestCacheVersion(), false);
            String role = config.role().name().toUpperCase(java.util.Locale.ROOT);
            long micros = Math.multiplyExact(System.currentTimeMillis(), 1000);
            long version = statusVersion.updateAndGet(previous -> Math.max(previous + 1, micros));
            return WorkerStatusPB.newBuilder().setRole("RoleType." + role)
                .setRoleType(RoleTypePB.valueOf("ROLE_TYPE_" + role)).setAlive(true)
                .setDpSize(1).setDpRank(0).setTpSize(config.tpSize()).setPrecision("FP16")
                .setStatusVersion(version).setMaxSeqLen(config.maxSeqLen()).setMaxBatchTokensSize(config.maxBatchTokens())
                .setTotalKvCache(tokens(snapshot.totalBlocks())).setAvailableKvCache(tokens(snapshot.availableBlocks()))
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
        catch (RuntimeException error) { observer.onError(Status.INTERNAL.withCause(error).withDescription(error.getMessage()).asRuntimeException()); return; }
        observer.onNext(value); observer.onCompleted();
    }
}

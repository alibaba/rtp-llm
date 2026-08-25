package org.flexlb.it.fixture.engine;

import io.grpc.stub.StreamObserver;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.time.Duration;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Mutable gRPC implementation of an engine's worker and cache status endpoints.
 *
 * <p>Tests change the next snapshot or inject an RPC failure while FlexLB continues to use its
 * production status synchronizers. Cache versions always advance beyond worker status versions so
 * the production cache updater observes every scripted cache-key change.
 */
final class ScriptedWorkerStatusService extends RpcServiceGrpc.RpcServiceImplBase {

    private static final ScheduledExecutorService DELAYED_RESPONSES = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread thread = new Thread(runnable, "scripted-worker-status-delay");
        thread.setDaemon(true);
        return thread;
    });

    private final AtomicReference<EngineRpcService.WorkerStatusPB> workerStatus;
    private final AtomicReference<Throwable> workerStatusFailure = new AtomicReference<>();
    private final AtomicInteger workerStatusCalls = new AtomicInteger();
    private final AtomicReference<WorkerStatusLatency> workerStatusLatency = new AtomicReference<>();
    private final AtomicInteger workerStatusLatencyCalls = new AtomicInteger();
    private final AtomicInteger delayedWorkerStatusResponses = new AtomicInteger();
    private final AtomicReference<Set<Long>> cacheKeys = new AtomicReference<>(Set.of());
    private final AtomicInteger cacheStatusCalls = new AtomicInteger();
    private final AtomicLong cacheStatusVersion = new AtomicLong();

    ScriptedWorkerStatusService(RoleType roleType) {
        this.workerStatus = new AtomicReference<>(healthyStatus(roleType));
    }

    /** Serves the currently scripted worker status or the currently injected failure. */
    @Override
    public void getWorkerStatus(
            EngineRpcService.StatusVersionPB request,
            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
        workerStatusCalls.incrementAndGet();
        Throwable failure = workerStatusFailure.get();
        EngineRpcService.WorkerStatusPB response = workerStatus.get();
        WorkerStatusLatency latency = workerStatusLatency.get();
        if (latency != null && workerStatusLatencyCalls.incrementAndGet() % latency.everyCalls() == 0) {
            delayedWorkerStatusResponses.incrementAndGet();
            DELAYED_RESPONSES.schedule(
                    () -> respondToWorkerStatus(responseObserver, failure, response),
                    latency.delay().toMillis(),
                    TimeUnit.MILLISECONDS);
            return;
        }
        respondToWorkerStatus(responseObserver, failure, response);
    }

    private void respondToWorkerStatus(
            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver,
            Throwable failure,
            EngineRpcService.WorkerStatusPB response) {
        if (failure != null) {
            responseObserver.onError(failure);
            return;
        }
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    /** Serves the currently scripted cache-key set with a monotonic cache version. */
    @Override
    public void getCacheStatus(
            EngineRpcService.CacheVersionPB request,
            StreamObserver<EngineRpcService.CacheStatusPB> responseObserver) {
        cacheStatusCalls.incrementAndGet();
        long statusVersion = workerStatus.get().getStatusVersion();
        long nextCacheVersion = cacheStatusVersion.updateAndGet(current -> Math.max(current + 1, statusVersion + 1));
        Map<Long, Boolean> cacheKeysResponse = cacheKeys.get().stream()
                .collect(java.util.stream.Collectors.toMap(cacheKey -> cacheKey, ignored -> true));
        responseObserver.onNext(EngineRpcService.CacheStatusPB.newBuilder()
                .setAvailableKvCache(1_000_000)
                .setTotalKvCache(1_000_000)
                .setBlockSize(16)
                .setVersion(nextCacheVersion)
                .putAllCacheKeys(cacheKeysResponse)
                .build());
        responseObserver.onCompleted();
    }

    void setWorkerStatus(EngineRpcService.WorkerStatusPB status) {
        workerStatus.set(status);
    }

    void setWorkerStatusFailure(Throwable failure) {
        workerStatusFailure.set(failure);
    }

    void setWorkerStatusLatency(Duration delay, int everyCalls) {
        if (delay == null || delay.isNegative() || delay.isZero()) {
            throw new IllegalArgumentException("Worker-status latency must be positive");
        }
        if (everyCalls <= 0) {
            throw new IllegalArgumentException("Worker-status latency interval must be positive");
        }
        workerStatusLatency.set(new WorkerStatusLatency(delay, everyCalls));
        workerStatusLatencyCalls.set(0);
        delayedWorkerStatusResponses.set(0);
    }

    void clearWorkerStatusLatency() {
        workerStatusLatency.set(null);
        workerStatusLatencyCalls.set(0);
        delayedWorkerStatusResponses.set(0);
    }

    void setCacheKeys(long... values) {
        cacheKeys.set(Arrays.stream(values).boxed().collect(java.util.stream.Collectors.toUnmodifiableSet()));
    }

    int workerStatusCalls() {
        return workerStatusCalls.get();
    }

    int delayedWorkerStatusResponses() {
        return delayedWorkerStatusResponses.get();
    }

    int cacheStatusCalls() {
        return cacheStatusCalls.get();
    }

    static EngineRpcService.WorkerStatusPB healthyStatus(RoleType roleType) {
        return status(roleType, true, 1);
    }

    static EngineRpcService.WorkerStatusPB status(RoleType roleType, boolean alive, long version) {
        return status(roleType, alive, version, 0, 0, 1.0);
    }

    static EngineRpcService.WorkerStatusPB status(
            RoleType roleType,
            boolean alive,
            long version,
            int waitingQueryLen,
            int runningQueryLen,
            double stepLatencyMs) {
        return status(roleType, alive, version, waitingQueryLen, runningQueryLen, stepLatencyMs, null);
    }

    static EngineRpcService.WorkerStatusPB status(
            RoleType roleType,
            boolean alive,
            long version,
            int waitingQueryLen,
            int runningQueryLen,
            double stepLatencyMs,
            EngineRpcService.TaskInfoPB runningTask) {
        return status(roleType, alive, version, waitingQueryLen, runningQueryLen, stepLatencyMs, runningTask, true);
    }

    /** Builds a worker-status response without an embedded cache snapshot. */
    static EngineRpcService.WorkerStatusPB statusWithoutCacheStatus(
            RoleType roleType,
            boolean alive,
            long version,
            int waitingQueryLen,
            int runningQueryLen,
            double stepLatencyMs,
            EngineRpcService.TaskInfoPB runningTask) {
        return status(roleType, alive, version, waitingQueryLen, runningQueryLen, stepLatencyMs, runningTask, false);
    }

    private static EngineRpcService.WorkerStatusPB status(
            RoleType roleType,
            boolean alive,
            long version,
            int waitingQueryLen,
            int runningQueryLen,
            double stepLatencyMs,
            EngineRpcService.TaskInfoPB runningTask,
            boolean includeCacheStatus) {
        EngineRpcService.WorkerStatusPB.Builder builder = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(roleType.getCode())
                .setAlive(alive)
                .setStatusVersion(version)
                .setLatestFinishedVersion(0)
                .setAvailableConcurrency(1)
                .setWaitingQueryLen(waitingQueryLen)
                .setRunningQueryLen(runningQueryLen)
                .setStepLatencyMs(stepLatencyMs)
                .setKvCacheGroupMode(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY);
        if (includeCacheStatus) {
            builder.setAvailableKvCache(1_000_000)
                    .setTotalKvCache(1_000_000)
                    .setBlockSize(16);
        }
        if (runningTask != null) {
            builder.addRunningTaskInfo(runningTask);
        }
        return builder.build();
    }

    /**
     * Builds the long-bucket snapshot used to exercise TTFT with a 1M-token prefill and 16K/32K
     * chunk metadata.
     */
    static EngineRpcService.TaskInfoPB longRunningPrefillTask(String requestId) {
        return EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputLength(1_000_000)
                .setPrefixLength(0)
                .setPrefixLengthValid(true)
                .setIsWaiting(false)
                .setFirstPrefillStepId(1)
                .setLastPrefillStepId(31)
                .setPrefillStepCount(31)
                .setPrefillNonfinalChunkTokensMin(16_384)
                .setPrefillNonfinalChunkTokensMax(32_768)
                .setCompletedPrefillTokens(16_384)
                .setRemainingPrefillTokens(983_616)
                .setLastCompletedPrefillStepId(1)
                .build();
    }

    /** Builds a single-step running prefill task with the requested uncached input length. */
    static EngineRpcService.TaskInfoPB runningPrefillTask(String requestId, long inputLength) {
        return EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputLength(inputLength)
                .setPrefixLength(0)
                .setPrefixLengthValid(true)
                .setIsWaiting(false)
                .setFirstPrefillStepId(1)
                .setLastPrefillStepId(1)
                .setPrefillStepCount(1)
                .setCompletedPrefillTokens(0)
                .setRemainingPrefillTokens(inputLength)
                .setLastCompletedPrefillStepId(0)
                .build();
    }

    private record WorkerStatusLatency(Duration delay, int everyCalls) {}
}

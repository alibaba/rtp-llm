package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.EngineObservation;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
import org.flexlb.dao.master.WorkerStatus.TaskObservation;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * Engine gRPC service for worker and cache status queries.
 */
@Component
public class WorkerStatusRpcClient {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final EngineGrpcClient engineGrpcClient;

    public WorkerStatusRpcClient(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(
            String ip,
            int grpcPort,
            long finishedTaskVersion,
            long requestTimeoutMs,
            RoleType roleType) {
        EngineRpcService.StatusVersionPB request =
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(finishedTaskVersion)
                        .build();

        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalWorkerStatusAsync(
                    ip, grpcPort, request, requestTimeoutMs);
        }
        return engineGrpcClient.getWorkerStatusAsync(ip, grpcPort, request, requestTimeoutMs);
    }

    public CompletableFuture<EngineRpcService.CacheStatusPB> getCacheStatusAsync(
            String ip, int grpcPort, long cacheVersion, long requestTimeoutMs, RoleType roleType) {
        boolean needCacheKeys = roleType.requiresCacheKeys();
        EngineRpcService.CacheVersionPB request =
                EngineRpcService.CacheVersionPB.newBuilder()
                        .setLatestCacheVersion(cacheVersion)
                        .setNeedCacheKeys(needCacheKeys)
                        .build();
        logger.debug(
                "Get cache status Request: {}, cacheVersion: {}, needCacheKeys: {}",
                ip,
                cacheVersion,
                needCacheKeys);

        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalCacheStatusAsync(
                    ip, grpcPort, request, requestTimeoutMs);
        }
        return engineGrpcClient.getCacheStatusAsync(ip, grpcPort, request, requestTimeoutMs);
    }

    /** Convert one protobuf response directly into the immutable status boundary. */
    public static StatusObservation convertToStatusObservation(
            WorkerStatus owner, EngineRpcService.WorkerStatusPB workerStatusPB) {
        Map<String, TaskObservation> runningTasks =
                convertTasks(workerStatusPB.getRunningTaskInfoList());
        Map<String, TaskObservation> finishedTasks =
                convertTasks(workerStatusPB.getFinishedTaskListList());
        EngineObservation engine =
                new EngineObservation(
                        RoleTypeProtoConverter.fromWorkerStatus(workerStatusPB),
                        (long) workerStatusPB.getAvailableConcurrency(),
                        workerStatusPB.getAvailableKvCache(),
                        workerStatusPB.getTotalKvCache(),
                        runningTasks,
                        workerStatusPB.getStepLatencyMs(),
                        workerStatusPB.getIterateCount(),
                        workerStatusPB.getDpSize(),
                        workerStatusPB.getTpSize(),
                        workerStatusPB.getDpRank(),
                        workerStatusPB.getMaxSeqLen(),
                        workerStatusPB.getMaxBatchTokensSize(),
                        workerStatusPB.getRunningQueryLen(),
                        workerStatusPB.getWaitingQueryLen());
        return owner.bindStatusObservation(
                engine,
                workerStatusPB.getAlive(),
                workerStatusPB.getStatusVersion(),
                workerStatusPB.getLatestFinishedVersion(),
                finishedTasks);
    }

    public static CacheStatus convertToCacheStatus(
            EngineRpcService.CacheStatusPB cacheStatusPB, Set<Long> retainedKeys) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(cacheStatusPB.getAvailableKvCache());
        cacheStatus.setTotalKvCache(cacheStatusPB.getTotalKvCache());
        cacheStatus.setBlockSize(cacheStatusPB.getBlockSize());
        cacheStatus.setVersion(cacheStatusPB.getVersion());

        Map<Long, Boolean> cacheKeysMap = cacheStatusPB.getCacheKeysMap();
        // Copy the keySet: protobuf MapField#keySet() returns a VIEW that keeps
        // the whole WorkerStatusPB message graph reachable until the next sync.
        Set<Long> cachedKeysSet =
                retainedKeys != null ? retainedKeys : new HashSet<>(cacheKeysMap.keySet());
        cacheStatus.setCachedKeys(cachedKeysSet);
        cacheStatus.setCacheKeySize(cachedKeysSet.size());
        return cacheStatus;
    }

    /**
     * Convert protobuf task values directly into immutable observations.
     */
    private static Map<String, TaskObservation> convertTasks(
            List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null || taskInfoPBList.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> tasks = new HashMap<>(taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB task : taskInfoPBList) {
            long errorCode = task.hasErrorInfo() ? task.getErrorInfo().getErrorCode() : 0L;
            String errorMessage = errorCode == 0L ? null : task.getErrorInfo().getErrorMessage();
            TaskObservation observation =
                    new TaskObservation(
                            task.getRequestId(),
                            task.getPrefixLength(),
                            0L,
                            task.getInputLength(),
                            task.getWaitingTimeMs(),
                            task.getIterateCount(),
                            task.getEndTimeMs(),
                            task.getDpRank(),
                            errorCode,
                            errorMessage,
                            task.getBatchId(),
                            resolvePhase(task),
                            task.getExecutionTimeMs(),
                            switch (task.getPriorityPreemptionProgress()) {
                                case PRIORITY_PREEMPTION_CANCELING ->
                                        PriorityPreemptionProgress.CANCELING;
                                case PRIORITY_PREEMPTION_CANCELED ->
                                        PriorityPreemptionProgress.CANCELED;
                                case PRIORITY_PREEMPTION_NONE, UNRECOGNIZED ->
                                        PriorityPreemptionProgress.NONE;
                            });
            tasks.put(String.valueOf(task.getRequestId()), observation);
        }
        return Map.copyOf(tasks);
    }

    private static TaskPhase resolvePhase(EngineRpcService.TaskInfoPB task) {
        if (task.getPhase() != EngineRpcService.TaskPhase.TASK_PHASE_PENDING) {
            // phase was added after the legacy is_waiting flag and is the
            // authoritative state whenever it carries a non-default value.
            // In particular, the immediately preceding e0 wire schema did not
            // contain field 9, so its RECEIVED/KV_ALLOCATED payloads are read
            // here with is_waiting=false. Treating that default as an explicit
            // value rejects valid status snapshots during a rolling upgrade.
            return convertPhase(task.getPhase());
        }
        // dsv4 writers only sent is_waiting. Because old proto3 writers omit
        // false on the wire, phase=0/is_waiting=false means legacy RUNNING.
        // New PENDING writers dual-write is_waiting=true.
        return task.getIsWaiting() ? TaskPhase.PENDING : TaskPhase.RUNNING;
    }

    private static TaskPhase convertPhase(EngineRpcService.TaskPhase protoPhase) {
        switch (protoPhase) {
            case TASK_PHASE_RECEIVED:
                return TaskPhase.RECEIVED;
            case TASK_PHASE_KV_ALLOCATED:
                return TaskPhase.KV_ALLOCATED;
            case TASK_PHASE_RUNNING:
                return TaskPhase.RUNNING;
            default:
                return TaskPhase.PENDING;
        }
    }
}

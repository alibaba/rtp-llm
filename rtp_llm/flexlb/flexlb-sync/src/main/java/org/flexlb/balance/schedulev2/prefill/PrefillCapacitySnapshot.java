package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.dao.master.WorkerStatus;

import java.util.Map;
import java.util.Set;

/**
 * 这台 Prefill 的容量判断数据：Worker 上报的活跃请求和 KV，加上本地尚未释放的发布预留。
 */
record PrefillCapacitySnapshot(WorkerStatus.EngineObservation workerStatus,
                               Map<String, DispatchedRequest> localPublishReservations,
                               Set<String> workerActiveRequestIds,
                               long workerStatusUpdatedAtMs) {

    long inflightCount() {
        return countTasks().count();
    }

    private TaskCount countTasks() {
        long count = workerActiveRequestIds.size();
        long pendingKv = 0L;
        for (var entry : localPublishReservations.entrySet()) {
            if (!workerActiveRequestIds.contains(entry.getKey())) {
                count++;
                pendingKv += entry.getValue().kvTokens();
            }
        }
        return new TaskCount(count, pendingKv);
    }

    QueueBlockReason blockReason(long kvTokens, int limit) {
        TaskCount tasks = countTasks();
        if (tasks.count() >= limit) {
            return QueueBlockReason.PREFILL_INFLIGHT_FULL;
        }
        // 空机允许一个大请求，避免它因 KV 需求过大而一直无法调度。
        if (tasks.count() > 0L && workerStatus != null && workerStatus.totalKvCacheTokens() > 0L
                && kvTokens > workerStatus.availableKvCacheTokens() - tasks.pendingKv()) {
            return QueueBlockReason.PREFILL_KV_FULL;
        }
        return null;
    }

    /**
     * 同一不可变快照内计算数量和未被引擎计入的 KV，避免把同一请求计算两次。
     */
    private record TaskCount(long count, long pendingKv) {
    }
}

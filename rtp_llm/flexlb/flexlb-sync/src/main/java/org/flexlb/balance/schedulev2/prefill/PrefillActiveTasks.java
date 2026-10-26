package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.dao.master.WorkerStatus;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 每台 Prefill 的活跃任务：引擎 waiting/running 与本地发布名额按请求 ID 合并。
 * CAS 同时保护引擎快照和发布名额，状态更新不能覆盖并发获取的名额。
 * 等待 Decode 的候选不在这里；临时名额只持有到本轮发布成功或失败。
 */
public final class PrefillActiveTasks {

    private final AtomicReference<Snapshot> snapshot = new AtomicReference<>(new Snapshot(null, Map.of()));

    public void updateEngine(WorkerStatus.EngineObservation engine) {
        snapshot.updateAndGet(old -> new Snapshot(engine, old.claimed()));
    }

    public long inflightCount() {
        return snapshot.get().countTasks().count();
    }

    public QueueBlockReason blockReason(long kvTokens, int limit) {
        return snapshot.get().blockReason(kvTokens, limit);
    }

    public QueueBlockReason tryAcquire(DispatchedRequest record, int limit) {
        while (true) {
            Snapshot old = snapshot.get();
            String id = record.request().requestId();
            if (old.claimed().containsKey(id)) {
                return QueueBlockReason.RESERVATION_CHANGED;
            }
            QueueBlockReason reason = old.blockReason(record.kvTokens(), limit);
            if (reason != null) {
                return reason;
            }
            Map<String, DispatchedRequest> claimed = new HashMap<>(old.claimed());
            claimed.put(id, record);
            if (snapshot.compareAndSet(old, new Snapshot(old.engine(), Map.copyOf(claimed)))) {
                return null;
            }
        }
    }

    public void release(Request request) {
        while (true) {
            Snapshot old = snapshot.get();
            DispatchedRequest record = old.claimed().get(request.requestId());
            if (record == null || record.request() != request) {
                return;
            }
            Map<String, DispatchedRequest> claimed = new HashMap<>(old.claimed());
            claimed.remove(request.requestId());
            if (snapshot.compareAndSet(old, new Snapshot(old.engine(), Map.copyOf(claimed)))) {
                return;
            }
        }
    }

    /**
     * 同一不可变快照内计算数量和未被引擎计入的 KV，避免把同一请求计算两次。
     */
    private record TaskCount(long count, long pendingKv) {
    }

    private record Snapshot(WorkerStatus.EngineObservation engine, Map<String, DispatchedRequest> claimed) {
        private TaskCount countTasks() {
            Set<String> engineIds = new HashSet<>();
            if (engine != null && engine.runningTaskList() != null) {
                for (WorkerStatus.TaskObservation task : engine.runningTaskList().values()) {
                    if (task.requestId() != null) {
                        engineIds.add(task.requestId());
                    }
                }
            }
            long count = engineIds.size();
            long pendingKv = 0L;
            for (var entry : claimed.entrySet()) {
                if (!engineIds.contains(entry.getKey())) {
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
            // 空机允许一个大请求，沿用原有的防饥饿规则。
            if (tasks.count() > 0L && engine != null && engine.totalKvCacheTokens() > 0L
                    && kvTokens > engine.availableKvCacheTokens() - tasks.pendingKv()) {
                return QueueBlockReason.PREFILL_KV_FULL;
            }
            return null;
        }
    }
}

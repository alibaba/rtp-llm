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

    private final AtomicReference<PrefillCapacitySnapshot> capacitySnapshot =
            new AtomicReference<>(new PrefillCapacitySnapshot(null, Map.of(), Set.of(), 0L));

    public void updateWorkerCapacity(WorkerStatus.EngineObservation workerStatus) {
        Set<String> workerActiveRequestIds = new HashSet<>();
        for (WorkerStatus.TaskObservation task : workerStatus.runningTaskList().values()) {
            workerActiveRequestIds.add(task.requestId());
        }
        Set<String> immutableWorkerActiveRequestIds = Set.copyOf(workerActiveRequestIds);
        long nowMs = System.currentTimeMillis();
        capacitySnapshot.updateAndGet(old -> new PrefillCapacitySnapshot(workerStatus, old.localPublishReservations(),
                immutableWorkerActiveRequestIds, nowMs));
    }

    public long inflightCount() {
        return capacitySnapshot.get().inflightCount();
    }

    public QueueBlockReason blockReason(long kvTokens, int limit) {
        return capacitySnapshot.get().blockReason(kvTokens, limit);
    }

    public QueueBlockReason tryAcquire(DispatchedRequest record, int limit) {
        while (true) {
            PrefillCapacitySnapshot old = capacitySnapshot.get();
            String id = record.request().requestId();
            if (old.localPublishReservations().containsKey(id)) {
                return QueueBlockReason.RESERVATION_CHANGED;
            }
            QueueBlockReason reason = old.blockReason(record.kvTokens(), limit);
            if (reason != null) {
                return reason;
            }
            Map<String, DispatchedRequest> localPublishReservations = new HashMap<>(old.localPublishReservations());
            localPublishReservations.put(id, record);
            if (capacitySnapshot.compareAndSet(old,
                    new PrefillCapacitySnapshot(old.workerStatus(), Map.copyOf(localPublishReservations),
                            old.workerActiveRequestIds(), old.workerStatusUpdatedAtMs()))) {
                return null;
            }
        }
    }

    public void release(Request request) {
        DispatchedRequest record = capacitySnapshot.get().localPublishReservations().get(request.requestId());
        if (record != null && record.request() == request) {
            release(record);
        }
    }

    public void release(DispatchedRequest record) {
        release(record, false, 0L, 0L);
    }

    public void releaseExpired(DispatchedRequest record, long nowMs, long statusTimeoutMs) {
        release(record, true, nowMs, statusTimeoutMs);
    }

    private void release(DispatchedRequest record, boolean expired, long nowMs, long statusTimeoutMs) {
        String requestId = record.request().requestId();
        while (true) {
            PrefillCapacitySnapshot old = capacitySnapshot.get();
            if (old.localPublishReservations().get(requestId) != record) {
                return;
            }
            Map<String, DispatchedRequest> localPublishReservations = new HashMap<>(old.localPublishReservations());
            localPublishReservations.remove(requestId);
            Set<String> workerActiveRequestIds = old.workerActiveRequestIds();
            // 本地记录可以过期；并发收到的新引擎快照仍保留实际活跃占用。
            if (expired && nowMs - old.workerStatusUpdatedAtMs() >= statusTimeoutMs) {
                workerActiveRequestIds = new HashSet<>(workerActiveRequestIds);
                workerActiveRequestIds.remove(requestId);
                workerActiveRequestIds = Set.copyOf(workerActiveRequestIds);
            }
            if (capacitySnapshot.compareAndSet(old,
                    new PrefillCapacitySnapshot(old.workerStatus(), Map.copyOf(localPublishReservations),
                            workerActiveRequestIds, old.workerStatusUpdatedAtMs()))) {
                return;
            }
        }
    }

}

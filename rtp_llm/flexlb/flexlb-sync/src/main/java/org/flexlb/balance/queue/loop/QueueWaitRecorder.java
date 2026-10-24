package org.flexlb.balance.queue.loop;

import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.util.PriorityNormalizer;

import java.util.Map;

/**
 * 记录调度线程最近一次让队首等待的原因，以及当时的队列长度和 Prefill 占用，供诊断接口查看。
 * <p>
 * 只保存最近一次。诊断接口读的时候不加队列锁，也不在调度线程上拼装输出。
 */
public final class QueueWaitRecorder {

    private final String key;
    private final QueueState queueState;
    private final PrefillState prefillState;
    /**
     * 最近一次等待的记录；调度线程还没等过时为 null。
     */
    private volatile QueueWaitSnapshot latest;

    public QueueWaitRecorder(String key, QueueState queueState, PrefillState prefillState) {
        this.key = key;
        this.queueState = queueState;
        this.prefillState = prefillState;
    }

    /**
     * 记下队首 {@code head} 等待的原因 {@code reason}，同时在队列锁内读出各优先级的排队数、
     * Prefill 上的请求数和占用的 batch 位置。
     */
    void record(ScheduledRequest head, String reason) {
        int[] requestCountsByPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        long version;
        long prefillRequests;
        int prefillBatchSlots;
        queueState.lock();
        try {
            version = queueState.queueVersion();
            prefillRequests = prefillState.observedRequestCount();
            prefillBatchSlots = prefillState.batchLeasesInUseUnderLock();
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                requestCountsByPriority[priority] = queueState.activeIndex().size(priority);
            }
        } finally {
            queueState.unlock();
        }
        DecodeRoutingView decode = head.decodeEp() == null ? null : head.decodeEp().routingView();
        latest = new QueueWaitSnapshot(key, reason, System.currentTimeMillis(), version, requestCountsByPriority,
                prefillRequests, prefillBatchSlots, decode);
    }

    public Map<String, Object> diagnostics() {
        QueueWaitSnapshot snapshot = latest;
        return snapshot == null ? QueueWaitSnapshot.INITIAL_DIAGNOSTICS : snapshot.diagnostics();
    }
}

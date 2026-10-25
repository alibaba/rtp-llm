package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.RequestOrder;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskPhase;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 一台 Prefill 的队列。同一地址重新上线后是另一个 {@link PrefillEndpoint} 对象，对应另一个队列。
 * <p>
 * 保存两类请求：
 * <ul>
 * <li>等待 dispatch 的请求，按 {@link RequestOrder} 排序；</li>
 * <li>已 dispatch、还没在引擎上结束的请求（{@link DispatchedRequest}）。</li>
 * </ul>
 * dispatch 前检查两件事：已 dispatch 的请求数加上引擎上报里没有对应记录的请求数，要小于每台 Prefill 的在途上限；
 * 引擎剩余 KV 减去已 dispatch 但引擎还没上报的请求要用的 KV，要放得下这个请求。
 * <p>
 * dispatch 和根据引擎上报更新记录只在 Prefill 队列循环线程里做；加入等待请求和移除请求可以在任意线程。
 */
public final class PrefillQueue {

    private final PrefillEndpoint endpoint;
    private final ConcurrentSkipListSet<Request> waiting;
    /**
     * requestId → 已 dispatch 的记录。
     */
    private final ConcurrentHashMap<String, DispatchedRequest> dispatched = new ConcurrentHashMap<>();
    /**
     * 等待请求或已 dispatch 记录每变一次加一，选 Prefill 时用来判断负载视图有没有变。
     */
    private final AtomicLong changeVersion = new AtomicLong();
    /**
     * 引擎上报的任务里，这里没有 dispatch 记录的个数，比如别的 flexlb 发过去的请求。
     */
    private volatile long unknownRequestCount;
    /**
     * 最近一次处理过的引擎上报；还没处理过时为 null。
     */
    private volatile WorkerStatus.EngineObservation latestEngine;
    /**
     * 最近一次处理过的上报版本号，只由循环线程读写。
     */
    private long handledStatusVersion = -1L;

    public PrefillQueue(PrefillEndpoint endpoint, boolean priorityOrdering) {
        this.endpoint = endpoint;
        this.waiting = new ConcurrentSkipListSet<>(RequestOrder.of(priorityOrdering));
    }

    public PrefillEndpoint endpoint() {
        return endpoint;
    }

    public void addWaiting(Request request) {
        if (waiting.add(request)) {
            changeVersion.incrementAndGet();
        }
    }

    public void removeWaiting(Request request) {
        if (waiting.remove(request)) {
            changeVersion.incrementAndGet();
        }
    }

    /**
     * 按 dispatch 顺序遍历等待的请求；遍历期间的增删不会抛异常。
     */
    public Iterator<Request> waitingIterator() {
        return waiting.iterator();
    }

    public int waitingCount() {
        return waiting.size();
    }

    public void addDispatched(DispatchedRequest record) {
        dispatched.put(record.request().requestId(), record);
        changeVersion.incrementAndGet();
    }

    /**
     * 只移除这个请求对象的记录，返回是否移除了。
     */
    public boolean removeDispatched(Request request) {
        DispatchedRequest record = dispatched.get(request.requestId());
        if (record == null || record.request() != request || !dispatched.remove(request.requestId(), record)) {
            return false;
        }
        changeVersion.incrementAndGet();
        return true;
    }

    public int dispatchedCount() {
        return dispatched.size();
    }

    public long unknownRequestCount() {
        return unknownRequestCount;
    }

    /**
     * 有新的引擎上报时更新已 dispatch 记录：引擎上报了的改成 ENGINE_QUEUED 或 ENGINE_RUNNING；
     * 上报过、这次不在任务列表里的说明 Prefill 已结束，移除记录。返回有没有处理新的上报。
     */
    public boolean updateFromEngine() {
        WorkerStatus.CommittedWorkerStatus status = endpoint.getStatus().committedWorkerStatus();
        if (status == null || status.cursor().statusVersion() == handledStatusVersion) {
            return false;
        }
        handledStatusVersion = status.cursor().statusVersion();
        WorkerStatus.EngineObservation engine = status.fields();
        Map<String, WorkerStatus.TaskObservation> tasks =
                engine.runningTaskList() == null ? Map.of() : engine.runningTaskList();
        Set<String> engineRequestIds = new HashSet<>();
        for (WorkerStatus.TaskObservation task : tasks.values()) {
            if (task.requestId() == null) {
                continue;
            }
            engineRequestIds.add(task.requestId());
            DispatchedRequest record = dispatched.get(task.requestId());
            if (record != null) {
                record.setPhase(task.phase() == TaskPhase.RUNNING
                        ? WorkView.Phase.ENGINE_RUNNING
                        : WorkView.Phase.ENGINE_QUEUED);
            }
        }
        long unknown = 0L;
        for (String requestId : engineRequestIds) {
            if (!dispatched.containsKey(requestId)) {
                unknown++;
            }
        }
        for (DispatchedRequest record : dispatched.values()) {
            if (record.seenByEngine() && !engineRequestIds.contains(record.request().requestId())) {
                removeDispatched(record.request());
            }
        }
        unknownRequestCount = unknown;
        latestEngine = engine;
        changeVersion.incrementAndGet();
        return true;
    }

    /**
     * 这个请求现在能不能 dispatch。maxInflight 是每台 Prefill 的在途上限，kvTokens 是这个请求要新占的 KV。
     * 机器上一个在途请求都没有时总是可以，避免大请求一直等不到。
     */
    public boolean canDispatch(long kvTokens, int maxInflight) {
        long inflight = dispatched.size() + unknownRequestCount;
        if (inflight >= maxInflight) {
            return false;
        }
        WorkerStatus.EngineObservation engine = latestEngine;
        if (inflight == 0L || engine == null || engine.totalKvCacheTokens() <= 0L) {
            return true;
        }
        long kvNotSeenByEngine = 0L;
        for (DispatchedRequest record : dispatched.values()) {
            if (!record.seenByEngine()) {
                kvNotSeenByEngine += record.kvTokens();
            }
        }
        return kvTokens <= engine.availableKvCacheTokens() - kvNotSeenByEngine;
    }

    /**
     * 选 Prefill 时用的负载视图：等待的请求按出队顺序排，已 dispatch 的请求带上预估耗时和引擎上的阶段。
     */
    public PrefillTtftEstimate.LoadView loadView(long nowMs,
                                                 Comparator<GroupPlanner.Item> itemOrder,
                                                 GroupPlanner.Constraints constraints) {
        long version = changeVersion.get();
        List<GroupPlanner.Item> items = new ArrayList<>();
        for (Request request : waiting) {
            long seqLen = request.seqLen();
            long hitCache = request.prefillChoice() == null ? 0L : request.prefillChoice().hitCacheTokens();
            items.add(new GroupPlanner.Item(request.requestId(), request.priority(), request.queueSeq(),
                    request.createdAtMs(), request.ctx().getRequestExpiresAtMs(), seqLen, Math.min(hitCache, seqLen)));
        }
        items.sort(itemOrder);
        List<WorkView.RequestWork> works = new ArrayList<>();
        for (DispatchedRequest record : dispatched.values()) {
            works.add(new WorkView.RequestWork(record.request().requestId(), record.phase(), record.predictedMs()));
        }
        works.sort(Comparator.comparing(WorkView.RequestWork::requestId));
        QueueView queue = new QueueView(nowMs, itemOrder, constraints, items, null);
        WorkView work = new WorkView(nowMs, works, List.of(), unknownRequestCount);
        return new PrefillTtftEstimate.LoadView(queue, work, version);
    }
}

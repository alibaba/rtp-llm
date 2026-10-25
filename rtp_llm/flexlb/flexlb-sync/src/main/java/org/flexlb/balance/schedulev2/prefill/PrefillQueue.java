package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.schedulev2.DispatchConfirmTimeout;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestOrder;
import org.flexlb.balance.schedulev2.request.Stage;
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
import java.util.concurrent.ConcurrentLinkedQueue;
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
    private final LoopSignal signal = new LoopSignal();
    private final ConcurrentSkipListSet<Request> waiting;
    /**
     * 主队列线程提交退回请求，Prefill 线程在容量检查之前处理。
     */
    private final ConcurrentLinkedQueue<Request> pendingReturns = new ConcurrentLinkedQueue<>();
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

    public LoopSignal signal() {
        return signal;
    }

    public synchronized void addWaiting(Request request) {
        request.setPrefillQueuedAtMs(System.currentTimeMillis());
        if (waiting.add(request)) {
            changeVersion.incrementAndGet();
        }
        if (request.stage() != Stage.PREFILL_WAITING) {
            removeWaiting(request);
        }
        signal.wake();
    }

    /**
     * 标记最低优先级里最晚入队的请求；退回尚未处理时不重复挑选受害者。
     * 主队列线程只标记，不修改选路结果、名额和等待集合。
     */
    public boolean markLowerPriorityForReturn(int priority) {
        if (!pendingReturns.isEmpty()) {
            return false;
        }
        for (Request request : waiting.descendingSet()) {
            if (request.priority() >= priority) {
                return false;
            }
            if (request.markPrefillReturn(priority)) {
                pendingReturns.add(request);
                signal.wake();
                return true;
            }
        }
        return false;
    }

    public Request pollPendingReturn() {
        return pendingReturns.poll();
    }

    public synchronized void removeWaiting(Request request) {
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

    public synchronized void addDispatched(DispatchedRequest record) {
        dispatched.put(record.request().requestId(), record);
        changeVersion.incrementAndGet();
    }

    /**
     * 只移除这个请求对象的记录，返回是否移除了。
     */
    public synchronized boolean removeDispatched(Request request) {
        DispatchedRequest record = dispatched.get(request.requestId());
        if (record == null || record.request() != request || !dispatched.remove(request.requestId(), record)) {
            return false;
        }
        changeVersion.incrementAndGet();
        signal.wake();
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
    public synchronized boolean updateFromEngine() {
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
     * 移除引擎一直没上报、而请求发布后已经超过 confirmTimeoutMs 的记录，返回移除了几条。
     * <p>
     * 这类记录说明调用方没把请求发给 Prefill，或者请求在两次上报之间就做完了，留着会一直占在途数。
     * 删早了也不会多发：引擎之后再上报这个请求时，会算进 unknownRequestCount，照样占在途数。
     */
    public int removeExpiredDispatched(long nowMs, long confirmTimeoutMs) {
        int removed = 0;
        for (DispatchedRequest record : dispatched.values()) {
            if (record.seenByEngine()) {
                continue;
            }
            if (!DispatchConfirmTimeout.expired(record.request(), nowMs, confirmTimeoutMs)) {
                continue;
            }
            if (removeDispatched(record.request())) {
                removed++;
            }
        }
        return removed;
    }

    /**
     * 这个请求现在能不能 dispatch。maxInflight 是每台 Prefill 的在途上限，kvTokens 是这个请求要新占的 KV。
     * 机器上一个在途请求都没有时总是可以，避免大请求一直等不到。
     */
    public boolean canDispatch(long kvTokens, int maxInflight) {
        long inflight = dispatched.size() + unknownRequestCount;
        if (!hasDispatchSlot(maxInflight)) {
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
     * 在途数量还有空位时才需要继续按请求大小检查 KV；在途已满时后面的请求也不能 dispatch。
     */
    public boolean hasDispatchSlot(int maxInflight) {
        return dispatched.size() + unknownRequestCount < maxInflight;
    }

    /**
     * 用最新的排队与在途记录计算 TTFT，只累加已经预测好的耗时，不调用预测器。
     * <p>
     * 队列线程先加 dispatched 再移除 waiting，过渡期间同一个请求只能算一次。
     * 锁只保护这一台机器的内存记录，避免读到转换了一半的计数。未知引擎任务不虚构耗时，返回 -1。
     */
    public synchronized long estimatedTtftMs(Request incoming, long predictedMs) {
        if (unknownRequestCount > 0L) {
            return -1L;
        }
        long total = predictedMs;
        for (DispatchedRequest record : dispatched.values()) {
            if (!record.request().stage().isEnded()) {
                total = addDuration(total, record.predictedMs());
            }
        }
        long nowMs = System.currentTimeMillis();
        for (Request request : waiting) {
            if (waiting.comparator().compare(request, incoming) >= 0) {
                break;
            }
            if (request.stage() != Stage.PREFILL_WAITING || request.ctx().requestExpired(nowMs)
                    || dispatched.containsKey(request.requestId())) {
                continue;
            }
            total = addDuration(total, request.prefillChoice().predictedMs());
        }
        return total;
    }

    private static long addDuration(long total, long duration) {
        return total > Long.MAX_VALUE - duration ? Long.MAX_VALUE : total + duration;
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

package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestOrder;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;

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
 * <li>已交给 Decode 调度、或已发布但 Prefill 尚未结束的跟踪记录（{@link DispatchedRequest}）。</li>
 * </ul>
 * 出队只检查活跃任务数和 KV，不为等待 Decode 的候选占执行名额。活跃任务按请求 ID 合并引擎任务与本地发布名额；
 * 引擎剩余 KV 减去已占发布名额但引擎还没上报的请求所需 KV，要放得下这个请求。
 * <p>
 * dispatch 在 Prefill 队列线程执行；状态提交线程更新活跃状态、处理完成通知。
 * 清理通过请求 ID 和记录对象的条件删除生效，再归还这条记录的发布名额。
 */
public final class PrefillQueue {

    private final PrefillEndpoint endpoint;
    private final LoopSignal signal = new LoopSignal();
    private final PrefillActiveTasks activeTasks = new PrefillActiveTasks();
    private volatile Runnable capacityChanged = () -> {};
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
     * 只移除这个请求对象的记录，返回是否移除了。条件删除成功后，后续获取名额会发现记录已失效。
     */
    public boolean removeDispatched(Request request) {
        DispatchedRequest record = dispatched.get(request.requestId());
        if (record == null || record.request() != request) {
            return false;
        }
        boolean removedSuccess = removeRequestRecord(record);
        if (!removedSuccess) {
            return false;
        }
        activeTasks.release(record);
        signal.wake();
        capacityChanged.run();
        return true;
    }

    private boolean removeRequestRecord(DispatchedRequest dispatchedRequest) {
        boolean removeSuccess = dispatched.remove(dispatchedRequest.request().requestId(), dispatchedRequest);
        if (!removeSuccess) {
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
     * 节点发布时，从已缓存的 Worker 状态初始化队列容量。
     */
    public void initializeFromCachedStatus() {
        WorkerStatus.CommittedWorkerStatus status = endpoint.getStatus().committedWorkerStatus();
        if (status != null) {
            updateActiveTasks(status.fields());
        }
    }

    private void updateActiveTasks(WorkerStatus.EngineObservation engine) {
        activeTasks.updateWorkerCapacity(engine);
        long nowMs = System.currentTimeMillis();
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
                record.updateStatus(task.phase() == TaskPhase.RUNNING
                        ? WorkView.Phase.ENGINE_RUNNING
                        : WorkView.Phase.ENGINE_QUEUED, nowMs);
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
        capacityChanged.run();
        changeVersion.incrementAndGet();
    }

    /**
     * 状态提交时处理每次上报，避免独立队列轮询错过短暂的 waiting/running 或增量 finished。
     * 活跃列表中消失的已上报请求和 Finished 请求都尝试清理；重复清理不再释放容量。
     */
    public void handlePrefillStatus(WorkerStatus.StatusObservation statusObservation) {
        updateActiveTasks(statusObservation.engine());
        removeFinishedTasks(statusObservation.finishedTasks());
        signal.wake();
    }

    private void removeFinishedTasks(Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            DispatchedRequest record = dispatched.get(task.requestId());
            if (record != null) {
                removeDispatched(record.request());
            }
        }
    }

    /**
     * 移除发布后或最近一次请求状态更新后超过 statusTimeoutMs 的记录，并上报超时指标和日志。
     * <p>
     * 只有该请求的活跃上报续期；扫描判定超时后允许清理，即使随后收到该请求的新状态。
     * 回收后若引擎迟到上报活跃请求，该请求仍计入引擎占用。
     */
    public void removeRequestStatusExpired(long nowMs, long statusTimeoutMs, RequestSchedulerReporter reporter) {
        boolean removed = false;
        for (DispatchedRequest record : dispatched.values()) {
            if (!record.statusExpired(nowMs, statusTimeoutMs)) {
                continue;
            }
            if (removeRequestRecord(record)) {
                activeTasks.releaseExpired(record, nowMs, statusTimeoutMs);
                removed = true;
                Request request = record.request();
                reporter.reportStatusUpdateTimeout(endpoint.getStatus().getRole(), endpoint.ipPort());
                Logger.warn(
                        "Request worker status timeout: requestId={}, role=PREFILL, endpoint={}, publishedAgeMs={}, timeoutMs={}",
                        request.requestId(), endpoint.ipPort(), nowMs - request.publishedAtMs(), statusTimeoutMs);
            }
        }
        if (removed) {
            signal.wake();
            capacityChanged.run();
        }
    }

    /**
     * 准入使用的请求数：引擎活跃请求和本地发布名额按请求 ID 去重。
     */
    public long inflightCount() {
        return activeTasks.inflightCount();
    }

    /**
     * 这个请求能否进入 Decode 候选队列。只检查 Prefill 活跃任务数和 KV，不占名额。
     * 机器上一个在途请求都没有时总是可以，避免大请求一直等不到。
     */
    public boolean canDispatch(long kvTokens, int maxInflight) {
        return activeTasks.blockReason(kvTokens, maxInflight) == null;
    }

    public boolean hasDispatchSlot(int maxInflight) {
        return activeTasks.blockReason(0L, maxInflight) != QueueBlockReason.PREFILL_INFLIGHT_FULL;
    }

    /**
     * 最终调度尝试先占 Prefill 名额，再尝试 Decode；失败时调用 releasePublishSlot。
     */
    public QueueBlockReason tryAcquirePublishSlot(Request request, int maxInflight) {
        DispatchedRequest record = dispatched.get(request.requestId());
        if (record == null || record.request() != request || request.stage().isEnded()) {
            return QueueBlockReason.RESERVATION_CHANGED;
        }
        QueueBlockReason reason = activeTasks.tryAcquire(record, maxInflight);
        if (reason == null && (dispatched.get(request.requestId()) != record || request.stage().isEnded())) {
            activeTasks.release(record);
            return QueueBlockReason.RESERVATION_CHANGED;
        }
        return reason;
    }

    public void releasePublishSlot(Request request) {
        activeTasks.release(request);
    }

    public void onCapacityChanged(Runnable listener) {
        capacityChanged = listener;
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

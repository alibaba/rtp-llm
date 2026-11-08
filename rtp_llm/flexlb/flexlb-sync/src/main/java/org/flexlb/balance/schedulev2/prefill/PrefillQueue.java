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
import org.flexlb.balance.schedulev2.request.SelectedPrefill;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;

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
    private final RequestSchedulerReporter reporter;
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
     * 只保护记录增删和预测工作量更新，不包含状态扫描、容量释放或通知。
     */
    private final Object requestRecordsLock = new Object();
    /**
     * waiting 与 dispatched 的累计预测工作量；所有加减由 requestRecordsLock 保护，volatile 供选路线程无锁读取。
     */
    private volatile long totalPredictedWorkMs;
    /**
     * 请求记录或引擎状态变化时递增，供负载视图标识版本；引擎状态更新在 requestRecordsLock 外，递增必须是原子的。
     */
    private final AtomicLong changeVersion = new AtomicLong();
    /**
     * 引擎上报的任务里，这里没有 dispatch 记录的个数，比如别的 flexlb 发过去的请求。
     */
    private volatile long unknownRequestCount;

    public PrefillQueue(PrefillEndpoint endpoint, boolean priorityOrdering, RequestSchedulerReporter reporter) {
        this.endpoint = endpoint;
        this.reporter = reporter;
        this.waiting = new ConcurrentSkipListSet<>(RequestOrder.of(priorityOrdering));
    }

    public PrefillEndpoint endpoint() {
        return endpoint;
    }

    public LoopSignal signal() {
        return signal;
    }

    public void addWaiting(Request request) {
        SelectedPrefill selectedPrefill = request.selectedPrefill();
        if (selectedPrefill == null || request.stage() != Stage.PREFILL_WAITING) {
            return;
        }
        request.setPrefillQueuedAtMs(System.currentTimeMillis());
        synchronized (requestRecordsLock) {
            if (waiting.add(request)) {
                totalPredictedWorkMs += selectedPrefill.predictedMs();
                changeVersion.incrementAndGet();
            }
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

    public void removeWaiting(Request request) {
        synchronized (requestRecordsLock) {
            if (waiting.remove(request)) {
                if (!dispatched.containsKey(request.requestId())) {
                    totalPredictedWorkMs -= request.selectedPrefill().predictedMs();
                }
                changeVersion.incrementAndGet();
            }
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

    /**
     * 记录从 waiting 交给 Decode 的请求。工作量已在入队时计入，转移不改变总量。
     */
    public void addDispatched(DispatchedRequest record) {
        synchronized (requestRecordsLock) {
            if (record.request().stage() != Stage.PREFILL_WAITING) {
                return;
            }
            DispatchedRequest existingRecord = dispatched.putIfAbsent(record.request().requestId(), record);
            if (existingRecord != null) {
                return;
            }
            changeVersion.incrementAndGet();
        }
    }

    /**
     * 移除请求 ID 对应的记录，返回是否移除了。条件删除成功后，后续获取名额会发现记录已失效。
     */
    public boolean removeDispatched(Request request) {
        DispatchedRequest record = dispatched.get(request.requestId());
        if (record == null) {
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
        synchronized (requestRecordsLock) {
            boolean removeSuccess = dispatched.remove(dispatchedRequest.request().requestId(), dispatchedRequest);
            if (!removeSuccess) {
                return false;
            }
            if (!waiting.contains(dispatchedRequest.request())) {
                totalPredictedWorkMs -= dispatchedRequest.predictedMs();
            }
            changeVersion.incrementAndGet();
            return true;
        }
    }

    public int dispatchedCount() {
        return dispatched.size();
    }

    /**
     * 节点已关闭选路入口，移除它的 dispatch 跟踪并归还本地发布名额；waiting 由队列线程退回主队列。
     */
    public void removeDispatchedOnRetirement() {
        for (DispatchedRequest record : dispatched.values()) {
            removeDispatched(record.request());
        }
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
            updateActiveTasks(status.fields(), Long.MAX_VALUE);
        }
    }

    private void updateActiveTasks(WorkerStatus.EngineObservation engine, long statusTimeoutMs) {
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
        for (DispatchedRequest dispatchedRequest : dispatched.values()) {
            if (dispatchedRequest.statusExpired(nowMs, statusTimeoutMs)) {
                removeDispatched(dispatchedRequest.request());
                reporter.reportStatusUpdateTimeout(endpoint.getStatus().getRole(), endpoint.ipPort());
            }
            if (dispatchedRequest.seenByEngine()
                    && !engineRequestIds.contains(dispatchedRequest.request().requestId())) {
                removeDispatched(dispatchedRequest.request());
            }
        }
        unknownRequestCount = unknown;
        capacityChanged.run();
        changeVersion.incrementAndGet();
    }

    /**
     * 状态提交时处理每次上报，避免独立队列轮询错过短暂的 waiting/running 或增量 finished。
     * 先更新引擎容量并检查请求状态超时，再处理完成通知。
     */
    public void handlePrefillStatus(WorkerStatus.StatusObservation statusObservation, long statusTimeoutMs) {
        updateActiveTasks(statusObservation.engine(), statusTimeoutMs);
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
        if (record == null || request.stage().isEnded()) {
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
     * 当前累计预测工作量加新请求自身耗时，用于选路评分，不承诺实际等待时间。
     * <p>
     * 工作量在记录增删时更新，waiting 与 dispatched 的过渡记录只算一次，不按新请求优先级过滤。
     * 读取不遍历队列，也不获取记录锁。未知引擎任务不虚构耗时，返回 -1。
     */
    public long estimatedWorkMs(long predictedMs) {
        if (unknownRequestCount > 0L) {
            return -1L;
        }
        return totalPredictedWorkMs + predictedMs;
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
            long hitCache = request.selectedPrefill() == null ? 0L : request.selectedPrefill().hitCacheTokens();
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

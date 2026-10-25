package org.flexlb.balance.schedulev2.request;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.PriorityNormalizer;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 新链路里的一个请求。
 * <p>
 * 阶段只通过 CAS 推进，抢到 FAILED 或 CANCELLED 的一方负责完成 future，所以 future 只会完成一次。
 */
public final class Request {

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final long createdAtMs;
    /**
     * 规范化后的优先级（1-100，越大越优先），主队列按它排序。
     */
    private final int priority;
    private final AtomicReference<QueueProgress> progress = new AtomicReference<>(
            new QueueProgress(Stage.MAIN_WAITING, System.nanoTime(), 0L, 0L, 0L, Stage.MAIN_WAITING));
    private final AtomicReference<QueueBlock> lastBlock = new AtomicReference<>();
    /**
     * 子队列名额占用的角色（PREFILL 或 PDFUSION）。主队列出队时设置，
     * 请求发布、失败、取消或退回主队列时取走并归还名额；为 null 表示没占名额。
     */
    private final AtomicReference<RoleType> queueSlotRole = new AtomicReference<>();
    /**
     * 进入主队列的顺序号，同优先级按它先进先出。加入主队列前设置，之后不变。
     */
    private volatile long queueSeq;
    /**
     * 到期后把请求按超时取消的定时任务，请求结束时取消。
     */
    private volatile ScheduledFuture<?> deadlineTask;
    /**
     * 主队列给它选的 Prefill。进入 PREFILL_WAITING 前设置；这台 Prefill 下线、请求退回主队列时清空。
     */
    private volatile PrefillChoice prefillChoice;
    /**
     * 最近一次进入 Prefill 队列的时间，KV 不够时用来判断队头是否等得太久。
     */
    private volatile long prefillQueuedAtMs;
    /**
     * 从 Prefill 队列 dispatch 出去的时间，还没 dispatch 时为 0。
     */
    private volatile long prefillDecidedAtMs;
    /**
     * 进入 PUBLISHED、把结果返回给调用方的时间，还没发布时为 0。调用方从这时起才会把请求发给引擎。
     */
    private volatile long publishedAtMs;
    /**
     * Decode 队列给它 reserve 的 Decode。进入 DECODE_RESERVED 前设置；reserve 被归还时取走。
     */
    private final AtomicReference<DecodeChoice> decodeChoice = new AtomicReference<>();
    private volatile long updatedAtMs;
    private volatile String detail;
    private volatile CancelReason cancelReason;

    public Request(BalanceContext ctx, CompletableFuture<Response> future, long createdAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.createdAtMs = createdAtMs;
        this.updatedAtMs = createdAtMs;
        int contextPriority = ctx.getPriority();
        this.priority =
                PriorityNormalizer.isValid(contextPriority) ? contextPriority : PriorityNormalizer.DEFAULT_PRIORITY;
    }

    public BalanceContext ctx() {
        return ctx;
    }

    public CompletableFuture<Response> future() {
        return future;
    }

    public String requestId() {
        return ctx.getRequestId();
    }

    public Stage stage() {
        return progress.get().stage();
    }

    public QueueProgress queueProgress() {
        return progress.get();
    }

    /**
     * 同一次排队中连续遇到相同节点和原因只计一次阻塞请求；每次尝试仍单独统计。
     */
    public boolean recordBlock(Stage expected, String reason, String endpoint) {
        QueueProgress current = progress.get();
        if (current.stage() != expected) {
            return false;
        }
        QueueBlock next = new QueueBlock(current, reason, endpoint);
        QueueBlock previous = lastBlock.getAndSet(next);
        return !next.equals(previous);
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public int priority() {
        return priority;
    }

    public long queueSeq() {
        return queueSeq;
    }

    public void setQueueSeq(long queueSeq) {
        this.queueSeq = queueSeq;
    }

    public void setDeadlineTask(ScheduledFuture<?> deadlineTask) {
        this.deadlineTask = deadlineTask;
    }

    /**
     * 取消超时定时任务。
     */
    public void cancelDeadlineTask() {
        ScheduledFuture<?> task = deadlineTask;
        if (task != null) {
            task.cancel(false);
        }
    }

    /**
     * 记下这个请求占用了 role 的一个子队列名额。
     */
    public void holdQueueSlot(RoleType role) {
        queueSlotRole.set(role);
    }

    /**
     * 取走占用的名额角色，只有第一个调用方拿到非 null，由它归还名额，所以名额不会重复归还。
     */
    public RoleType takeQueueSlot() {
        return queueSlotRole.getAndSet(null);
    }

    public PrefillChoice prefillChoice() {
        return prefillChoice;
    }

    public void setPrefillChoice(PrefillChoice prefillChoice) {
        this.prefillChoice = prefillChoice;
    }

    public long prefillDecidedAtMs() {
        return prefillDecidedAtMs;
    }

    public long prefillQueuedAtMs() {
        return prefillQueuedAtMs;
    }

    public void setPrefillQueuedAtMs(long prefillQueuedAtMs) {
        this.prefillQueuedAtMs = prefillQueuedAtMs;
    }

    public void setPrefillDecidedAtMs(long prefillDecidedAtMs) {
        this.prefillDecidedAtMs = prefillDecidedAtMs;
    }

    public long publishedAtMs() {
        return publishedAtMs;
    }

    public DecodeChoice decodeChoice() {
        return decodeChoice.get();
    }

    public void setDecodeChoice(DecodeChoice choice) {
        decodeChoice.set(choice);
    }

    /**
     * 取走 Decode 的 reserve，只有第一个调用方拿到非 null，由它归还，所以 reserve 不会重复归还。
     */
    public DecodeChoice takeDecodeChoice() {
        return decodeChoice.getAndSet(null);
    }

    /**
     * 请求的输入 token 数。
     */
    public long seqLen() {
        return ctx.getRequest().getSeqLen();
    }

    /**
     * 从 from 推进到 to。当前阶段已经不是 from，或者不允许这样推进时返回 false。
     */
    public boolean moveTo(Stage from, Stage to) {
        if (!from.canMoveTo(to)) {
            throw new IllegalArgumentException("cannot move from " + from + " to " + to);
        }
        while (true) {
            QueueProgress current = progress.get();
            if (current.stage() != from) {
                return false;
            }
            if (progress.compareAndSet(current, current.next(to, System.nanoTime()))) {
                updatedAtMs = System.currentTimeMillis();
                return true;
            }
        }
    }

    /**
     * 从 from 进入 PUBLISHED，并用选好的机器地址完成 future。当前阶段已经不是 from 时返回 false。
     */
    public boolean publish(Stage from, List<ServerStatus> serverStatuses) {
        if (!moveTo(from, Stage.PUBLISHED)) {
            return false;
        }
        publishedAtMs = System.currentTimeMillis();
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(serverStatuses);
        writeQueueTimes();
        future.complete(response);
        return true;
    }

    /**
     * 发布前失败：进入 FAILED 并用错误完成 future。已经发布或结束时返回 false。
     */
    public boolean fail(StrategyErrorType errorType, String failureDetail) {
        return fail(Response.buildErrorResponse(errorType, failureDetail));
    }

    /**
     * 只在指定排队阶段挤掉请求。和出队竞争同一次 CAS，已经出队的请求不受影响。
     */
    public boolean fail(Stage expected, StrategyErrorType errorType, String failureDetail) {
        if (!moveTo(expected, Stage.FAILED)) {
            return false;
        }
        detail = failureDetail;
        writeQueueTimes();
        future.complete(Response.buildErrorResponse(errorType, failureDetail));
        return true;
    }

    /**
     * 发布前失败：进入 FAILED 并用给定的错误响应完成 future。已经发布或结束时返回 false。
     */
    public boolean fail(Response errorResponse) {
        if (!end(Stage.FAILED, errorResponse.getErrorMessage())) {
            return false;
        }
        writeQueueTimes();
        future.complete(errorResponse);
        return true;
    }

    /**
     * 发布前取消：进入 CANCELLED 并用错误完成 future。已经发布或结束时返回 false。
     */
    public boolean cancel(CancelReason reason) {
        if (!end(Stage.CANCELLED, reason.getMessage())) {
            return false;
        }
        cancelReason = reason;
        StrategyErrorType errorType = reason == CancelReason.DEADLINE_EXCEEDED
                ? StrategyErrorType.BATCH_SLO_EXPIRED
                : StrategyErrorType.REQUEST_CANCELLED;
        writeQueueTimes();
        future.complete(Response.buildErrorResponse(errorType, reason.getMessage()));
        return true;
    }

    private boolean end(Stage target, String endDetail) {
        while (true) {
            Stage current = stage();
            if (!current.canMoveTo(target)) {
                return false;
            }
            if (moveTo(current, target)) {
                detail = endDetail;
                updatedAtMs = System.currentTimeMillis();
                return true;
            }
        }
    }

    /**
     * 完成 future 前写入 PV；subQueueMs 包含 Prefill 和 Decode 等待，并额外提供两者的分项。
     */
    private void writeQueueTimes() {
        QueueProgress times = progress.get();
        ctx.setGlobalQueueMs(times.mainMs());
        ctx.setSubQueueMs(
                java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(times.prefillNanos() + times.decodeNanos()));
        ctx.setPrefillQueueMs(times.prefillMs());
        ctx.setDecodeQueueMs(times.decodeMs());
    }

    /**
     * 转成对外的 RequestState，供查询和取消接口返回。
     */
    public RequestState toRequestState() {
        Stage current = stage();
        RequestState.Phase phase = switch (current) {
            case MAIN_WAITING, PREFILL_WAITING, DECODE_WAITING, DECODE_RESERVED -> RequestState.Phase.QUEUED;
            case PUBLISHED -> RequestState.Phase.ACKNOWLEDGED;
            case FAILED -> RequestState.Phase.FAILED;
            case CANCELLED -> cancelReason == CancelReason.DEADLINE_EXCEEDED
                    ? RequestState.Phase.TIMED_OUT
                    : RequestState.Phase.CANCELLED;
        };
        DeliveryClaimKind claimKind =
                current == Stage.PUBLISHED ? DeliveryClaimKind.ROUTE_DECISION : DeliveryClaimKind.NONE;
        return new RequestState(requestId(), ctx.getNumericRequestId(), phase, claimKind, 0L, createdAtMs, updatedAtMs,
                detail);
    }
}

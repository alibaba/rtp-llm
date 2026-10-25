package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.concurrent.CompletableFuture;
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
    private final AtomicReference<Stage> stage = new AtomicReference<>(Stage.MAIN_WAITING);
    private volatile long updatedAtMs;
    private volatile String detail;
    private volatile CancelReason cancelReason;

    public Request(BalanceContext ctx, CompletableFuture<Response> future, long createdAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.createdAtMs = createdAtMs;
        this.updatedAtMs = createdAtMs;
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
        return stage.get();
    }

    /**
     * 从 from 推进到 to。当前阶段已经不是 from，或者不允许这样推进时返回 false。
     */
    public boolean moveTo(Stage from, Stage to) {
        if (!from.canMoveTo(to)) {
            throw new IllegalArgumentException("cannot move from " + from + " to " + to);
        }
        if (!stage.compareAndSet(from, to)) {
            return false;
        }
        updatedAtMs = System.currentTimeMillis();
        return true;
    }

    /**
     * 发布前失败：进入 FAILED 并用错误完成 future。已经发布或结束时返回 false。
     */
    public boolean fail(StrategyErrorType errorType, String failureDetail) {
        if (!end(Stage.FAILED, failureDetail)) {
            return false;
        }
        future.complete(Response.buildErrorResponse(errorType, failureDetail));
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
        future.complete(Response.buildErrorResponse(errorType, reason.getMessage()));
        return true;
    }

    private boolean end(Stage target, String endDetail) {
        while (true) {
            Stage current = stage.get();
            if (!current.canMoveTo(target)) {
                return false;
            }
            if (stage.compareAndSet(current, target)) {
                detail = endDetail;
                updatedAtMs = System.currentTimeMillis();
                return true;
            }
        }
    }

    /**
     * 转成对外的 RequestState，供查询和取消接口返回。
     */
    public RequestState toRequestState() {
        Stage current = stage.get();
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

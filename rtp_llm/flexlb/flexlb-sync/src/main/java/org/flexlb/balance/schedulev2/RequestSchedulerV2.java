package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * NON_BATCH_V2 链路的入口，只处理 Generation 请求，Encoder 请求仍由 RequestScheduler 处理。
 * <p>
 * 只有 dispatcher.type 是 NON_BATCH_V2 时，RequestScheduler 才会创建它。
 */
@Lazy
@Component
public class RequestSchedulerV2 {

    static final String SHUTTING_DOWN_MESSAGE = "request scheduler is shutting down";

    private final RequestTable requestTable = new RequestTable();
    private final AtomicBoolean closed = new AtomicBoolean();

    public CompletableFuture<Response> submit(BalanceContext context) {
        if (closed.get()) {
            return CompletableFuture.completedFuture(
                    Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE));
        }
        long nowMs = System.currentTimeMillis();
        if (context.requestExpired(nowMs)) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(StrategyErrorType.RESOURCE_EXHAUSTED,
                    "request scheduling deadline has expired before placement"));
        }
        CompletableFuture<Response> future = new CompletableFuture<>();
        Request request = new Request(context, future, nowMs);
        if (!requestTable.add(request)) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(StrategyErrorType.INVALID_REQUEST,
                    "duplicate request_id: " + context.getRequestId()));
        }
        context.setEnqueueTime(nowMs);
        context.setFuture(future);
        future.whenComplete((response, failure) -> handleFutureDone(request, failure));
        if (closed.get()) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
            return future;
        }
        // 主队列接入前，新链路先直接拒绝
        request.fail(StrategyErrorType.DISPATCH_FAILED, "NON_BATCH_V2 pipeline is not ready");
        return future;
    }

    /**
     * future 完成后的收尾：外部取消 future 时把请求标为取消；请求已结束时从请求表移除。
     */
    private void handleFutureDone(Request request, Throwable failure) {
        if (failure instanceof CancellationException) {
            request.cancel(CancelReason.CLIENT_CANCELLED);
        }
        if (request.stage().isEnded()) {
            requestTable.remove(request);
        }
    }

    public RequestState cancelRequest(String requestId, long expectedBatchId, CancelReason reason) {
        Request request = requestTable.get(requestId);
        if (request == null || !request.toRequestState().matchesBatch(expectedBatchId)) {
            return null;
        }
        request.cancel(reason);
        return request.toRequestState();
    }

    public RequestState getRequestState(String requestId, long expectedBatchId) {
        Request request = requestTable.get(requestId);
        if (request == null) {
            return null;
        }
        RequestState state = request.toRequestState();
        return state.matchesBatch(expectedBatchId) ? state : null;
    }

    public int trackedRequestCount() {
        return requestTable.size();
    }

    public int queuedRequestCount() {
        return requestTable.waitingCount();
    }

    public int blockedRequestCount() {
        return 0;
    }

    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> states = new ArrayList<>();
        for (Request request : requestTable.list()) {
            states.add(request.toRequestState());
        }
        states.sort(Comparator.comparingLong(RequestState::createdAtMs).thenComparing(RequestState::requestId));
        return states;
    }

    /**
     * 停止接收新请求，并让所有还在等待的请求失败。
     */
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        for (Request request : requestTable.list()) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
        }
    }
}

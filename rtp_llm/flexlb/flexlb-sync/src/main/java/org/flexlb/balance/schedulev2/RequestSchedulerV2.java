package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.main.MainQueueLoop;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
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

    private final ConfigService configService;
    private final Set<RoleType> deployedRoles;
    /**
     * Prefill 用的角色：PD 分离时是 PREFILL，PD 合一时是 PDFUSION；两者都没部署时为 null。
     */
    private final RoleType prefillRole;
    private final RequestTable requestTable = new RequestTable();
    private final LoopSignal mainLoopSignal = new LoopSignal();
    private final MainQueue mainQueue;
    private final RequestCleanup cleanup;
    private final MainQueueLoop mainQueueLoop;
    private final ScheduledThreadPoolExecutor deadlineTimer;
    private final AtomicBoolean closed = new AtomicBoolean();

    public RequestSchedulerV2(ConfigService configService,
                              QueueCapacity queueCapacity,
                              EndpointRegistry endpointRegistry,
                              PlacementAvailability placementAvailability,
                              DefaultRouter router,
                              CostBasedPrefillStrategy prefillStrategy,
                              ModelMetaConfig modelMetaConfig) {
        this.configService = configService;
        this.deployedRoles = Set.copyOf(modelMetaConfig.requiredRoles());
        this.prefillRole = deployedRoles.contains(RoleType.PREFILL)
                ? RoleType.PREFILL
                : deployedRoles.contains(RoleType.PDFUSION) ? RoleType.PDFUSION : null;
        this.mainQueue =
                new MainQueue(queueCapacity, mainLoopSignal, configService.loadBalanceConfig().isPriorityOrdering());
        this.cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, mainLoopSignal);
        this.mainQueueLoop = new MainQueueLoop(mainQueue, mainLoopSignal, cleanup, queueCapacity, endpointRegistry,
                placementAvailability, configService, new PrefillPicker(router, prefillStrategy), prefillRole,
                this::sendToPrefillQueue);
        this.deadlineTimer = new ScheduledThreadPoolExecutor(1, task -> {
            Thread thread = new Thread(task, "flexlb-v2-deadline");
            thread.setDaemon(true);
            return thread;
        });
        this.deadlineTimer.setRemoveOnCancelPolicy(true);
        mainQueueLoop.start();
    }

    public CompletableFuture<Response> submit(BalanceContext context) {
        if (closed.get()) {
            return CompletableFuture.completedFuture(
                    Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE));
        }
        if (prefillRole == null || !rolesAreDeployed(context)) {
            return CompletableFuture.completedFuture(
                    Response.buildErrorResponse(StrategyErrorType.INVALID_REQUEST, "requested roles are not deployed"));
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
        long delayMs = Math.max(0L, context.getRequestExpiresAtMs() - nowMs);
        try {
            request.setDeadlineTask(deadlineTimer.schedule(() -> request.cancel(CancelReason.DEADLINE_EXCEEDED),
                    delayMs, TimeUnit.MILLISECONDS));
        } catch (RejectedExecutionException closing) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
            return future;
        }
        StrategyErrorType rejection = mainQueue.add(request, configService.loadBalanceConfig());
        if (rejection != null) {
            request.fail(rejection, MainQueue.QUEUE_FULL_MESSAGE);
            return future;
        }
        if (closed.get()) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
        }
        return future;
    }

    /**
     * 请求指定的角色（为空表示全部）都在这个部署里。
     */
    private boolean rolesAreDeployed(BalanceContext context) {
        Set<RoleType> requested = context.getRequestedRoles();
        return requested == null || deployedRoles.containsAll(requested);
    }

    private void sendToPrefillQueue(Request request, SelectedRole selected) {
        // Prefill 队列接入前，选好 Prefill 的请求先直接失败
        selected.close();
        request.fail(StrategyErrorType.DISPATCH_FAILED, "NON_BATCH_V2 prefill queue is not ready");
    }

    /**
     * future 完成后的收尾：外部取消 future 时把请求标为取消；请求失败或取消后归还它占的名额和计数。
     */
    private void handleFutureDone(Request request, Throwable failure) {
        if (failure instanceof CancellationException) {
            request.cancel(CancelReason.CLIENT_CANCELLED);
        }
        if (request.stage().isEnded()) {
            cleanup.releaseEnded(request);
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
        mainQueueLoop.close();
        for (Request request : requestTable.list()) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
        }
        deadlineTimer.shutdownNow();
    }
}

package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.decode.DecodeQueueLoop;
import org.flexlb.balance.schedulev2.decode.DecodeReservations;
import org.flexlb.balance.schedulev2.decode.EmptyDecodeFinder;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.main.MainQueueLoop;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueLoop;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.PrefillInputs;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.V2QueueReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.context.annotation.Lazy;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
    private final LoopSignal decodeLoopSignal = new LoopSignal();
    private final MainQueue mainQueue;
    private final PrefillQueueRegistry prefillQueues;
    private final DecodeQueue decodeQueue;
    private final RequestCleanup cleanup;
    private final PrefillPublishCheck prefillCheck;
    private final MainQueueLoop mainQueueLoop;
    private final DecodeQueueLoop decodeQueueLoop;
    private final ScheduledThreadPoolExecutor deadlineTimer;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final V2QueueReporter queueReporter;
    private final EndpointRegistry endpointRegistry;

    public RequestSchedulerV2(ConfigService configService,
                              QueueCapacity queueCapacity,
                              EndpointRegistry endpointRegistry,
                              PlacementAvailability placementAvailability,
                              DefaultRouter router,
                              CostBasedPrefillStrategy prefillStrategy,
                              DecodeSelector decodeSelector,
                              PrefillQueueRegistry prefillQueues,
                              ModelMetaConfig modelMetaConfig,
                              DecodePreemptionCoordinator preemptionCoordinator,
                              RequestSchedulerReporter reporter,
                              WorkerDirectory workerDirectory,
                              DecodeReservations reservations,
                              V2QueueReporter queueReporter) {
        this.queueReporter = queueReporter;
        this.endpointRegistry = endpointRegistry;
        this.configService = configService;
        this.prefillCheck = new PrefillPublishCheck(endpointRegistry);
        this.deployedRoles = Set.copyOf(modelMetaConfig.requiredRoles());
        this.prefillRole = deployedRoles.contains(RoleType.PREFILL)
                ? RoleType.PREFILL
                : deployedRoles.contains(RoleType.PDFUSION) ? RoleType.PDFUSION : null;
        boolean priorityOrdering = configService.loadBalanceConfig().isPriorityOrdering();
        this.mainQueue = new MainQueue(queueCapacity, mainLoopSignal, priorityOrdering);
        this.prefillQueues = prefillQueues;
        this.decodeQueue = new DecodeQueue(priorityOrdering);
        this.cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, mainLoopSignal, prefillQueues,
                decodeQueue, decodeLoopSignal);
        this.mainQueueLoop = new MainQueueLoop(mainQueue, mainLoopSignal, cleanup, queueCapacity, endpointRegistry,
                placementAvailability, configService, new PrefillPicker(router, prefillStrategy, prefillQueues),
                prefillQueues, prefillRole, this::sendToPrefillQueue, prefillWorkers(), queueReporter);
        this.decodeQueueLoop = new DecodeQueueLoop(decodeQueue, decodeLoopSignal, cleanup, placementAvailability,
                decodeSelector, router, preemptionCoordinator, new EmptyDecodeFinder(workerDirectory), configService,
                reporter, prefillCheck, reservations, queueReporter);
        this.deadlineTimer = new ScheduledThreadPoolExecutor(1, task -> {
            Thread thread = new Thread(task, "flexlb-v2-deadline");
            thread.setDaemon(true);
            return thread;
        });
        this.deadlineTimer.setRemoveOnCancelPolicy(true);
        prefillQueues.start(queue -> new PrefillQueueLoop(queue, prefillQueues, cleanup, mainQueue, endpointRegistry,
                configService, reporter, this::sendToDecodeQueue, queueReporter));
        mainQueueLoop.start();
        decodeQueueLoop.start();
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
        future.whenComplete((response, failure) -> {
            handleFutureDone(request, failure);
            queueReporter.completed(request, response);
        });
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

    private void sendToPrefillQueue(Request request) {
        prefillQueues.addWaiting(request);
    }

    /**
     * 主队列使用线程池查询缓存和预测 Prefill 耗时，线程数沿用原主队列的启动配置。
     */
    private ProcessWorkers<PrefillInputs> prefillWorkers() {
        int count = configService.loadBalanceConfig().getInternalRuntime().getQueuePlannerThreads();
        AtomicInteger workerId = new AtomicInteger();
        ExecutorService executor = Executors.newFixedThreadPool(count, task -> {
            Thread thread = new Thread(task, "flexlb-v2-main-worker-" + workerId.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
        return new ProcessWorkers<>(executor, count, mainLoopSignal,
                configService.loadBalanceConfig().isPriorityOrdering());
    }

    /**
     * Prefill dispatch 出去的请求：要 Decode 时进 Decode 队列，不要 Decode（PD 合一）时直接发布 Prefill 的地址。
     */
    private void sendToDecodeQueue(Request request) {
        if (!needsDecode(request)) {
            try (WorkerEndpoint.EndpointHold hold = prefillCheck.hold(request)) {
                if (hold == null) {
                    cleanup.returnToMain(request, Stage.DECODE_WAITING);
                } else if (request.publish(Stage.DECODE_WAITING, List.of(request.prefillChoice().serverStatus()))) {
                    cleanup.releasePublished(request);
                }
            }
            return;
        }
        decodeQueue.addWaiting(request);
        decodeLoopSignal.wake();
    }

    /**
     * 部署了 Decode，并且请求指定的角色（为空表示全部）里有 Decode。
     */
    private boolean needsDecode(Request request) {
        Set<RoleType> requested = request.ctx().getRequestedRoles();
        return deployedRoles.contains(RoleType.DECODE) && (requested == null || requested.contains(RoleType.DECODE));
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

    @Scheduled(fixedDelay = 1000L)
    public void reportQueues() {
        List<String> endpoints = prefillRole == null
                ? List.of()
                : endpointRegistry.prefillRoutingSnapshot(prefillRole)
                        .stream()
                        .map(entry -> entry.endpoint().ipPort())
                        .toList();
        queueReporter.sample(requestTable, endpoints);
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
        // 先让请求失败再停循环：Decode 队列线程退出前要归还这些请求在 Decode 上的 reserve
        for (Request request : requestTable.list()) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, SHUTTING_DOWN_MESSAGE);
        }
        mainQueueLoop.close();
        prefillQueues.close();
        decodeQueueLoop.close();
        deadlineTimer.shutdownNow();
    }
}

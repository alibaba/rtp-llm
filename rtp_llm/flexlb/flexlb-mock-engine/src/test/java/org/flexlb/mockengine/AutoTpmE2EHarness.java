package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.QueueRouteAdmission;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestSchedulerTestRuntime;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import static org.flexlb.mockengine.MockEngineTestSupport.unary;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Shared E2E harness (task35): a real FlexLB scheduler, eviction manager,
 * endpoint runtime, and batch dispatcher wired to an in-process Java mock
 * engine cluster.
 *
 * <p>The E2E loop: the mocked {@link EngineGrpcClient#batchEnqueueAsync} answer
 * bridges into the real {@link JavaMockEngineCluster.FastRpcService#enqueueBatch}
 * of the target port, so dispatch, fault injection, prefill→decode handoff and
 * CANCELLED completions all run through real mock-engine code. The WorkerStatus
 * pump reads real {@code getWorkerStatus} snapshots and feeds them through the
 * prepared-status transaction, closing the calibrate/settle loop exactly like
 * production polling would.
 *
 * <p>By default, {@code ConfigService}/{@code Router}/{@code EngineGrpcClient}/reporters
 * are Mockito stand-ins — identical to the flexlb-sync unit-test harness pattern.
 * Routing-regression scenarios can opt into the production {@link DefaultRouter}
 * and endpoint-selection strategies while retaining the in-process transport.
 */
final class AutoTpmE2EHarness implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    final FlexlbConfig config = new FlexlbConfig();
    final DecisionPolicyConfig fixedWindowDecision;
    final ConfigService configService = mock(ConfigService.class);
    final DefaultRouter router = mock(DefaultRouter.class);
    final EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
    final BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
    final RequestSchedulerReporter requestReporter = mock(RequestSchedulerReporter.class);

    final Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
    final List<JavaMockEngineCluster.FastRpcService> prefillEngines = new ArrayList<>();
    final List<JavaMockEngineCluster.FastRpcService> decodeEngines = new ArrayList<>();
    final ScheduledExecutorService engineScheduler = Executors.newScheduledThreadPool(8);

    final EndpointRegistry endpointRegistry;
    final RequestScheduler scheduler;
    final DefaultBatchDispatcher dispatcher;
    private final RequestSchedulerTestRuntime schedulerRuntime;

    /** requestIds in the order the mock engines actually received them via enqueueBatch. */
    final List<Long> engineArrivalOrder = new CopyOnWriteArrayList<>();
    /** requestId -> first enqueueBatch arrival time (nanos), for latency measurements. */
    final Map<Long, Long> engineArrivalNanos = new ConcurrentHashMap<>();

    /** Route stand-in, swappable per scenario. Default: capacity-aware, prefill[0]+decode[0]. */
    volatile Function<BalanceContext, Response> routeFn;
    /** Prefill index chosen by the default routeFn, swappable per scenario. */
    volatile Function<BalanceContext, Integer> prefillSelector = ctx -> 0;

    private final Map<Integer, WorkerStatus> statusByPort = new ConcurrentHashMap<>();
    private final Map<Integer, String> ipPortByEnginePort = new ConcurrentHashMap<>();
    private final Map<Integer, Long> pumpCursor = new ConcurrentHashMap<>();
    private final Map<Long, CompletableFuture<Void>> batchAckGates =
            new ConcurrentHashMap<>();
    private final Object pumpLock = new Object();
    private ScheduledExecutorService pumpExecutor;
    private final Path tempDir;

    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel) {
        this(basePort, nPrefill, nDecode, prefillFormulaMs, decodeStepMs,
                realCancelChannel, true, defaultFixedWindowDecision(), false);
    }

    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel, DecisionPolicyConfig decisionPolicy) {
        this(basePort, nPrefill, nDecode, prefillFormulaMs, decodeStepMs,
                realCancelChannel, true, decisionPolicy, false);
    }

    /**
     * @param autoTpm must be decided at construction time: {@code WorkerBatcher}
     *                freezes its queue comparator from the configured ordering
     *                when the endpoint is registered — flipping the switch after
     *                construction does NOT change the batch queue ordering.
     */
    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel, boolean autoTpm) {
        this(basePort, nPrefill, nDecode, prefillFormulaMs, decodeStepMs,
                realCancelChannel, autoTpm, defaultFixedWindowDecision(), false);
    }

    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel, boolean autoTpm,
                      DecisionPolicyConfig decisionPolicy,
                      boolean productionRouting) {
        this.fixedWindowDecision = decisionPolicy.getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW
                ? decisionPolicy : null;
        try {
            tempDir = Files.createTempDirectory("auto-tpm-e2e");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        MockPerformanceModel model = model(prefillFormulaMs, decodeStepMs);
        for (int i = 0; i < nPrefill; i++) {
            int port = basePort + i;
            JavaMockEngineCluster.FastRpcService svc = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, engineScheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, svc);
            prefillEngines.add(svc);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = basePort + nPrefill + i;
            JavaMockEngineCluster.FastRpcService svc = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, engineScheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, svc);
            decodeEngines.add(svc);
        }

        // Conservative defaults; scenarios override before submitting traffic.
        // Priority ordering must be set BEFORE registerEndpoint (WorkerBatcher freezes it).
        if (autoTpm) {
            config.queueScheduler().setOrdering(QueueOrderingConfig.priority());
        }
        config.setDispatcher(new DispatcherConfig());
        config.queueScheduler().setDecision(decisionPolicy);
        // the default fixed_window algorithm reads fixedWaitMs (not windowMs):
        // hold dispatch by default so scenarios can assert stable queue state
        config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(1024);
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(1024);
        when(configService.loadBalanceConfig()).thenReturn(config);

        routeFn = this::defaultRoute;
        when(router.routeForQueue(any(BalanceContext.class)))
                .thenAnswer(inv -> routeResult(inv.getArgument(0)));

        // ---- E2E bridge: mocked gRPC transport → real in-process mock engine ----
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    int port = inv.getArgument(1);
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    JavaMockEngineCluster.FastRpcService svc = services.get(port);
                    if (svc == null) {
                        return CompletableFuture.failedFuture(
                                new RuntimeException("no mock engine on port " + port));
                    }
                    request.getDpSlotsList().stream()
                            .flatMap(slot -> slot.getRequestsList().stream())
                            .forEach(in -> {
                                long rid = in.getInput().getRequestId();
                                engineArrivalOrder.add(rid);
                                engineArrivalNanos.putIfAbsent(rid, System.nanoTime());
                            });
                    CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> future =
                            new CompletableFuture<>();
                    svc.enqueueBatch(request, new StreamObserver<>() {
                        private EngineRpcService.EnqueueBatchResponsePB response;

                        @Override
                        public void onNext(EngineRpcService.EnqueueBatchResponsePB value) {
                            response = value;
                        }

                        @Override
                        public void onError(Throwable t) {
                            future.completeExceptionally(t);
                        }

                        @Override
                        public void onCompleted() {
                            CompletableFuture<?>[] gates = request.getDpSlotsList().stream()
                                    .flatMap(slot -> slot.getRequestsList().stream())
                                    .map(in -> batchAckGates.get(
                                            in.getInput().getRequestId()))
                                    .filter(java.util.Objects::nonNull)
                                    .toArray(CompletableFuture<?>[]::new);
                            CompletableFuture.allOf(gates).whenComplete(
                                    (ignored, gateFailure) -> {
                                        if (gateFailure == null) {
                                            future.complete(response);
                                        } else {
                                            future.completeExceptionally(gateFailure);
                                        }
                                    });
                        }
                    });
                    return future;
                });

        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        EngineCancelChannel cancelChannel = realCancelChannel
                ? new MockEngineCancelChannel(services)
                : new UnsupportedCancelStub();
        CostBasedPrefillStrategy evictionPrefillSelection =
                new CostBasedPrefillStrategy(
                        mock(WorkerDirectory.class),
                        mock(CacheAwareService.class),
                        mock(EngineHealthReporter.class)) {
            @Override
            public SelectedRole select(
                    BalanceContext context, RoleType role, String group) {
                int selectedIndex = prefillSelector.apply(context);
                PrefillEndpoint endpoint = prefillEndpoint(selectedIndex);
                org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin pin =
                        endpoint.tryPinGeneration();
                if (pin == null) {
                    return null;
                }
                return SelectedRole.prefill(
                        pin,
                        prefillServer(
                                selectedIndex,
                                context.getRequestId()),
                        0L);
            }
        };
        schedulerRuntime = new RequestSchedulerTestRuntime(
                configService,
                dispatcher::tryPrepareSubmission,
                reporter,
                requestReporter,
                cancelChannel,
                evictionPrefillSelection);
        endpointRegistry = schedulerRuntime.endpointRegistry();
        scheduler = schedulerRuntime.scheduler();

        for (JavaMockEngineCluster.FastRpcService svc : prefillEngines) {
            registerEndpoint(RoleType.PREFILL, svc);
        }
        for (JavaMockEngineCluster.FastRpcService svc : decodeEngines) {
            registerEndpoint(RoleType.DECODE, svc);
        }
        schedulerRuntime.bindRouter(
                productionRouting ? productionRouter() : router);
    }

    // ==================== endpoint / route wiring ====================

    DecisionPolicyConfig fixedWindowDecision() {
        if (fixedWindowDecision == null) {
            throw new IllegalStateException("FIXED_WINDOW decision is not active");
        }
        return fixedWindowDecision;
    }

    private static DecisionPolicyConfig defaultFixedWindowDecision() {
        DecisionPolicyConfig decision = new DecisionPolicyConfig();
        decision.setMaxRequests(100);
        decision.setMaxCollectionWaitMs(10_000);
        return decision;
    }

    private void registerEndpoint(RoleType role, JavaMockEngineCluster.FastRpcService svc) {
        int grpcPort = svc.getGrpcPort();
        int httpPort = httpPort(grpcPort);
        WorkerStatus ws = publishEndpoint(
                role,
                httpPort,
                grpcPort,
                role == RoleType.DECODE ? 1_000_000L : 0L,
                role == RoleType.DECODE ? 2_000_000L : 0L);
        String ipPort = "127.0.0.1:" + httpPort;
        statusByPort.put(grpcPort, ws);
        ipPortByEnginePort.put(grpcPort, ipPort);
        pumpCursor.put(grpcPort, 0L);
    }

    private static int httpPort(int grpcPort) {
        return grpcPort + 2000;
    }

    private WorkerStatus publishEndpoint(
            RoleType role,
            int httpPort,
            int grpcPort,
            long availableKv,
            long totalKv) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, "g1", "127.0.0.1", httpPort, grpcPort, null);
        WorkerStatusResponse initial = statusResponse(
                role, true, availableKv, totalKv, 1L, 0L);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(initial));
            endpointRegistry.publishPreparedEndpoint(
                    status.getIpPort(), status, prepared);
        } finally {
            status.lock.unlock();
        }
        return status;
    }

    private static WorkerStatusResponse statusResponse(
            RoleType role,
            boolean alive,
            long availableKv,
            long totalKv,
            long statusVersion,
            long latestFinishedVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(alive);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(latestFinishedVersion);
        return response;
    }

    PrefillEndpoint prefillEndpoint(int index) {
        return (PrefillEndpoint) endpointRegistry.get(
                RoleType.PREFILL,
                ipPortByEnginePort.get(prefillEngines.get(index).getGrpcPort()));
    }

    DecodeEndpoint decodeEndpoint(int index) {
        return (DecodeEndpoint) endpointRegistry.get(
                RoleType.DECODE,
                ipPortByEnginePort.get(decodeEngines.get(index).getGrpcPort()));
    }

    void setDecodeKvCapacity(int index, long available, long total) {
        int grpcPort = decodeEngines.get(index).getGrpcPort();
        WorkerStatus status = statusByPort.get(grpcPort);
        WorkerStatusResponse response = statusResponse(
                RoleType.DECODE,
                true,
                available,
                total,
                status.appliedStatusCursor().statusVersion() + 1L,
                status.appliedStatusCursor().latestFinishedTaskVersion());
        schedulerRuntime.applyStatus(status, response);
    }

    /**
     * Let the mock engine accept and execute one request while withholding its
     * EnqueueBatch ACK from the dispatcher until the returned gate is closed.
     */
    AutoCloseable holdBatchAck(long requestId) {
        CompletableFuture<Void> gate = new CompletableFuture<>();
        if (batchAckGates.putIfAbsent(requestId, gate) != null) {
            throw new IllegalStateException(
                    "batch ACK is already gated for request " + requestId);
        }
        return () -> {
            if (batchAckGates.remove(requestId, gate)) {
                gate.complete(null);
            }
        };
    }

    void allowPreemption(VictimStage first, VictimStage... additional) {
        PreemptionConfig preemption = new PreemptionConfig();
        EnumSet<VictimStage> stages = EnumSet.of(first, additional);
        preemption.setAllowedVictimStages(stages);
        if (stages.contains(VictimStage.DECODE_ENGINE_OWNED)) {
            preemption.setEngineCancellation(new EngineCancellationConfig());
        }
        config.priorityOrdering().setPreemption(preemption);
    }

    private ServerStatus prefillServer(int index, long requestId) {
        int grpcPort = prefillEngines.get(index).getGrpcPort();
        return server(
                RoleType.PREFILL, "127.0.0.1", httpPort(grpcPort), grpcPort, requestId);
    }

    private ServerStatus decodeServer(int index, long requestId) {
        int grpcPort = decodeEngines.get(index).getGrpcPort();
        return server(
                RoleType.DECODE, "127.0.0.1", httpPort(grpcPort), grpcPort,
                requestId);
    }

    /** Capacity-aware route stand-in mirroring the production decode hard filter. */
    Response defaultRoute(BalanceContext ctx) {
        DecodeEndpoint decodeEp = decodeEndpoint(0);
        Long decodeConcurrencyLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        if (decodeConcurrencyLimit != null && decodeConcurrencyLimit > 0
                && decodeEp.routingView().engineLoad() + 1 > decodeConcurrencyLimit) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        if (decodeEp.realKvTotal() > 0 && decodeEp.realKvAvailable() < 128) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        int prefillIndex = prefillSelector.apply(ctx);
        ServerStatus prefill = prefillServer(prefillIndex, ctx.getRequestId());
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                prefill,
                decodeServer(0, ctx.getRequestId())));
        return response;
    }

    private PlacementResult<QueueRouteAdmission, PlacementKey> routeResult(BalanceContext context) {
        Response response = routeFn.apply(context);
        if (!response.isSuccess()
                && response.getCode()
                        == StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()) {
            return PlacementResult.blocked(
                    PlacementKey.anyGroup(RoleType.PREFILL));
        }
        if (!response.isSuccess()
                && response.getCode()
                        == StrategyErrorType.NO_DECODE_WORKER.getErrorCode()) {
            return PlacementResult.blocked(
                    PlacementKey.anyGroup(RoleType.DECODE));
        }
        return schedulerRuntime.routeResult(context, response);
    }

    private DefaultRouter productionRouter() {
        WorkerDirectory workers = new WorkerDirectory(endpointRegistry);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of());
        EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);
        ModelMetaConfig modelMeta = mock(ModelMetaConfig.class);
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.DECODE, RoleType.PREFILL));
        return new DefaultRouter(
                new CostBasedPrefillStrategy(
                        workers, cache, healthReporter),
                new CostBasedDecodeStrategy(workers),
                new RandomStrategy(workers),
                configService,
                modelMeta,
                schedulerRuntime.placementAvailability());
    }

    private static ServerStatus server(
            RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }

    // ==================== request construction ====================

    BalanceContext context(long requestId, int priority) {
        return context(requestId, priority, 128, 8);
    }

    BalanceContext context(long requestId, int priority, long seqLen, int maxNewTokens) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(maxNewTokens);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setGenerateInputPb(
                ByteString.copyFrom(generateInputBytes(requestId, (int) seqLen, maxNewTokens)));
        // Mirror production admission with immutable request scheduling metadata.
        // A 30 s request lifetime avoids accidental expiry during eviction tests.
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, ctx.getStartTime() + 30_000));
        return ctx;
    }

    static byte[] generateInputBytes(long requestId, int inputTokens, int maxNewTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(maxNewTokens)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build().toByteArray();
    }

    // ==================== WorkerStatus pump (engine → scheduler feedback) ====================

    /**
     * One pump round: read the real WorkerStatus of every mock engine and feed
     * it into the scheduler + decode endpoint calibrate, advancing the
     * per-engine finished-version cursor exactly like production polling.
     */
    void pumpOnce() {
        synchronized (pumpLock) {
            for (JavaMockEngineCluster.FastRpcService svc : services.values()) {
                pumpEngine(svc);
            }
        }
    }

    void pumpPrefillOnce(int index) {
        synchronized (pumpLock) {
            pumpEngine(prefillEngines.get(index));
        }
    }

    void pumpDecodeOnce(int index) {
        synchronized (pumpLock) {
            pumpEngine(decodeEngines.get(index));
        }
    }

    private void pumpEngine(JavaMockEngineCluster.FastRpcService svc) {
        int port = svc.getGrpcPort();
        EngineRpcService.WorkerStatusPB status = workerStatus(svc, pumpCursor.get(port));

        boolean isDecode = decodeEngines.contains(svc);
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(isDecode ? RoleType.DECODE : RoleType.PREFILL);
        resp.setAlive(status.getAlive());
        resp.setAvailableKvCacheTokens(status.getAvailableKvCache());
        resp.setTotalKvCacheTokens(status.getTotalKvCache());
        resp.setStatusVersion(status.getStatusVersion());
        resp.setLatestFinishedVersion(status.getLatestFinishedVersion());

        Map<String, TaskInfo> running = new HashMap<>();
        for (EngineRpcService.TaskInfoPB task : status.getRunningTaskInfoList()) {
            running.put(String.valueOf(task.getRequestId()), toTaskInfo(task));
        }
        resp.setRunningTaskInfo(running);

        Map<String, TaskInfo> finished = new HashMap<>();
        for (EngineRpcService.TaskInfoPB task : status.getFinishedTaskListList()) {
            finished.put(String.valueOf(task.getRequestId()), toTaskInfo(task));
        }
        resp.setFinishedTaskInfo(finished);

        schedulerRuntime.applyStatus(statusByPort.get(port), resp);
        // Commit the cursor only after every consumer accepted the snapshot.
        // If processing throws, the next pump round must be able to retry the
        // same finished records instead of losing them permanently.
        pumpCursor.put(port, status.getLatestFinishedVersion());
    }

    static TaskInfo toTaskInfo(EngineRpcService.TaskInfoPB task) {
        TaskInfo info = new TaskInfo();
        info.setRequestId(task.getRequestId());
        info.setInputLength(task.getInputLength());
        info.setBatchId(task.getBatchId());
        info.setErrorCode(task.getErrorInfo().getErrorCode());
        info.setErrorMessage(task.getErrorInfo().getErrorMessage());
        info.setEndTimeMs(task.getEndTimeMs());
        if (task.getPriorityPreemptionProgress()
                == EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELING) {
            info.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELING);
        } else if (task.getPriorityPreemptionProgress()
                == EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELED) {
            info.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELED);
        }
        if (task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RUNNING) {
            info.setPhase(TaskPhase.RUNNING);
        } else if (task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED) {
            info.setPhase(TaskPhase.RECEIVED);
        }
        return info;
    }

    static EngineRpcService.WorkerStatusPB workerStatus(
            JavaMockEngineCluster.FastRpcService svc, long sinceVersion) {
        return unary(observer -> svc.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .build(),
                observer));
    }

    void startAutoPump(long intervalMs) {
        if (pumpExecutor != null) {
            return;
        }
        pumpExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "e2e-worker-status-pump");
            t.setDaemon(true);
            return t;
        });
        pumpExecutor.scheduleWithFixedDelay(() -> {
            try {
                pumpOnce();
            } catch (Throwable ignored) {
                // pump must never die silently mid-test; assertions catch stalls
            }
        }, intervalMs, intervalMs, TimeUnit.MILLISECONDS);
    }

    void stopAutoPump() {
        if (pumpExecutor != null) {
            pumpExecutor.shutdownNow();
            pumpExecutor = null;
        }
    }

    // ==================== misc helpers ====================

    static void await(BooleanSupplier condition, long timeoutMs, String message)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(5);
        }
        throw new AssertionError("await timed out: " + message);
    }

    private MockPerformanceModel model(String prefillFormulaMs, double decodeStepMs) {
        try {
            return MockEngineTestSupport.performanceModel(
                    tempDir, prefillFormulaMs, 1.0, decodeStepMs);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() {
        stopAutoPump();
        schedulerRuntime.close();
        dispatcher.shutdown();
        for (JavaMockEngineCluster.FastRpcService svc : services.values()) {
            svc.shutdown();
        }
        engineScheduler.shutdownNow();
    }

    /** Test-local fail-closed cancel transport for non-preemption scenarios. */
    private static final class UnsupportedCancelStub
            implements EngineCancelChannel {
        @Override
        public boolean isSupported(DecodeEndpoint endpoint) {
            return false;
        }

        @Override
        public CompletableFuture<CancelAck> cancel(
                CancelTarget target, long requestId, long timeoutMs) {
            return CompletableFuture.completedFuture(
                    CancelAck.UNSUPPORTED);
        }
    }
}

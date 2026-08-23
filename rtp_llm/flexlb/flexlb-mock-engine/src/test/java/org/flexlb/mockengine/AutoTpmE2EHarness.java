package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.priority.DecodePreemptionCoordinator;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.PlanCommitter;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.balance.scheduler.priority.UnsupportedEngineCancelChannel;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FixedWindowDecisionConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.PriorityOrderingConfig;
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
import org.flexlb.service.monitor.PrioritySchedulerReporter;

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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Shared E2E harness (task35): a REAL FlexLB scheduler stack (PriorityScheduler
 * + PriorityAdmissionScheduler + EndpointRegistry + DefaultBatchDispatcher +
 * AdmissionLease) wired to an in-process Java mock engine cluster.
 *
 * <p>The E2E loop: the mocked {@link EngineGrpcClient#batchEnqueueAsync} answer
 * bridges into the real {@link JavaMockEngineCluster.FastRpcService#enqueueBatch}
 * of the target port, so dispatch, fault injection, prefill→decode handoff and
 * CANCELLED completions all run through real mock-engine code. The WorkerStatus
 * pump reads real {@code getWorkerStatus} snapshots and feeds them back into
 * {@code scheduler.onWorkerStatusUpdate} + {@code DecodeEndpoint.onWorkerStatusUpdate},
 * closing the calibrate/settle loop exactly like production polling would.
 *
 * <p>Only {@code ConfigService}/{@code Router}/{@code EngineGrpcClient}/reporters
 * are Mockito stand-ins — identical to the flexlb-sync unit-test harness pattern
 * (route strategy and transport are out of scope for this campaign).
 */
final class AutoTpmE2EHarness implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    final FlexlbConfig config = new FlexlbConfig();
    final FixedWindowDecisionConfig fixedWindowDecision;
    final ConfigService configService = mock(ConfigService.class);
    final Router router = mock(Router.class);
    final EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
    final BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
    final PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);

    final Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
    final List<JavaMockEngineCluster.FastRpcService> prefillEngines = new ArrayList<>();
    final List<JavaMockEngineCluster.FastRpcService> decodeEngines = new ArrayList<>();
    final ScheduledExecutorService engineScheduler = Executors.newScheduledThreadPool(8);

    final EndpointRegistry endpointRegistry;
    final PriorityScheduler scheduler;
    final PriorityAdmissionScheduler priorityScheduler;

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
    private final Object pumpLock = new Object();
    private ScheduledExecutorService pumpExecutor;
    private final Path tempDir;

    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel) {
        this(basePort, nPrefill, nDecode, prefillFormulaMs, decodeStepMs,
                realCancelChannel, true, defaultFixedWindowDecision());
    }

    AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                      String prefillFormulaMs, double decodeStepMs,
                      boolean realCancelChannel, DecisionPolicyConfig decisionPolicy) {
        this(basePort, nPrefill, nDecode, prefillFormulaMs, decodeStepMs,
                realCancelChannel, true, decisionPolicy);
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
                realCancelChannel, autoTpm, defaultFixedWindowDecision());
    }

    private AutoTpmE2EHarness(int basePort, int nPrefill, int nDecode,
                             String prefillFormulaMs, double decodeStepMs,
                             boolean realCancelChannel, boolean autoTpm,
                             DecisionPolicyConfig decisionPolicy) {
        this.fixedWindowDecision = decisionPolicy instanceof FixedWindowDecisionConfig fixed
                ? fixed : null;
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
            config.queueScheduler().setOrdering(new PriorityOrderingConfig());
        }
        config.setDispatcher(new BatchDispatcherConfig());
        config.queueScheduler().setDecision(decisionPolicy);
        // the default fixed_window algorithm reads fixedWaitMs (not windowMs):
        // hold dispatch by default so scenarios can assert stable queue state
        config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(1024);
        when(configService.loadBalanceConfig()).thenReturn(config);

        routeFn = this::defaultRoute;
        when(router.route(any(BalanceContext.class)))
                .thenAnswer(inv -> routeFn.apply(inv.getArgument(0)));

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
                            future.complete(response);
                        }
                    });
                    return future;
                });

        AtomicReference<PriorityScheduler> schedulerRef = new AtomicReference<>();
        endpointRegistry = new EndpointRegistry(configService, schedulerRef::get, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        EngineCancelChannel cancelChannel = realCancelChannel
                ? new MockEngineCancelChannel(services)
                : new UnsupportedEngineCancelChannel();
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, cancelChannel,
                new DecodePreemptionCoordinator(cancelChannel)) {
            @Override
            protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                                  FlexlbConfig config,
                                                                  String group) {
                return prefillServer(prefillSelector.apply(ctx), ctx.getRequestId());
            }
        };
        scheduler = new PriorityScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null,
                cancelChannel);
        schedulerRef.set(scheduler);

        for (JavaMockEngineCluster.FastRpcService svc : prefillEngines) {
            registerEndpoint(RoleType.PREFILL, svc);
        }
        for (JavaMockEngineCluster.FastRpcService svc : decodeEngines) {
            registerEndpoint(RoleType.DECODE, svc);
        }
    }

    // ==================== endpoint / route wiring ====================

    FixedWindowDecisionConfig fixedWindowDecision() {
        if (fixedWindowDecision == null) {
            throw new IllegalStateException("FIXED_WINDOW decision is not active");
        }
        return fixedWindowDecision;
    }

    private static FixedWindowDecisionConfig defaultFixedWindowDecision() {
        FixedWindowDecisionConfig decision = new FixedWindowDecisionConfig();
        decision.setMaxRequests(100);
        decision.setMaxCollectionWaitMs(10_000);
        return decision;
    }

    private void registerEndpoint(RoleType role, JavaMockEngineCluster.FastRpcService svc) {
        int grpcPort = svc.getGrpcPort();
        int httpPort = httpPort(grpcPort);
        WorkerStatus ws = new WorkerStatus();
        ws.setIp("127.0.0.1");
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        if (role == RoleType.DECODE) {
            ws.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
            ws.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
        }
        String ipPort = "127.0.0.1:" + httpPort;
        endpointRegistry.ensureEndpoint(role, ipPort, ws);
        statusByPort.put(grpcPort, ws);
        ipPortByEnginePort.put(grpcPort, ipPort);
        pumpCursor.put(grpcPort, 0L);
        if (role == RoleType.DECODE) {
            decodeEndpoint(decodeEngines.indexOf(svc))
                    .onWorkerStatusUpdate(ws, new WorkerStatusResponse());
        }
    }

    private static int httpPort(int grpcPort) {
        return grpcPort + 2000;
    }

    PrefillEndpoint prefillEndpoint(int index) {
        return endpointRegistry.getPrefill(
                ipPortByEnginePort.get(prefillEngines.get(index).getGrpcPort()));
    }

    DecodeEndpoint decodeEndpoint(int index) {
        return endpointRegistry.getDecode(
                ipPortByEnginePort.get(decodeEngines.get(index).getGrpcPort()));
    }

    void setDecodeKvCapacity(int index, long available, long total) {
        int grpcPort = decodeEngines.get(index).getGrpcPort();
        WorkerStatus status = statusByPort.get(grpcPort);
        status.getAvailableKvCacheTokens().set(available);
        status.getTotalKvCacheTokens().set(total);
        decodeEndpoint(index).onWorkerStatusUpdate(status, new WorkerStatusResponse());
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

    ServerStatus prefillServer(int index, long requestId) {
        int grpcPort = prefillEngines.get(index).getGrpcPort();
        return server(RoleType.PREFILL, "127.0.0.1", httpPort(grpcPort), grpcPort, requestId);
    }

    ServerStatus decodeServer(int index, long requestId) {
        int grpcPort = decodeEngines.get(index).getGrpcPort();
        return server(RoleType.DECODE, "127.0.0.1", httpPort(grpcPort), grpcPort, requestId);
    }

    /** Capacity-aware route stand-in mirroring the production decode hard filter. */
    Response defaultRoute(BalanceContext ctx) {
        DecodeEndpoint decodeEp = decodeEndpoint(0);
        Long decodeConcurrencyLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        if (decodeConcurrencyLimit != null && decodeConcurrencyLimit > 0
                && decodeEp.getEngineLoad() + 1 > decodeConcurrencyLimit) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        if (decodeEp.realKvTotal() > 0 && decodeEp.realKvAvailable() < 128) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        decodeEp.reserve(ctx.getRequestId(), 128, 136, ctx.getPriority());
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                prefillServer(prefillSelector.apply(ctx), ctx.getRequestId()),
                decodeServer(0, ctx.getRequestId())));
        return response;
    }

    static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
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
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId, (int) seqLen, maxNewTokens));
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

    private void pumpEngine(JavaMockEngineCluster.FastRpcService svc) {
        int port = svc.getGrpcPort();
        EngineRpcService.WorkerStatusPB status = workerStatus(svc, pumpCursor.get(port));

        boolean isDecode = decodeEngines.contains(svc);
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(isDecode ? RoleType.DECODE : RoleType.PREFILL);
        resp.setAlive(status.getAlive());
        resp.setAvailableKvCacheTokens(status.getAvailableKvCache());
        resp.setTotalKvCacheTokens(status.getTotalKvCache());

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

        scheduler.onWorkerStatusUpdate(resp);
        if (isDecode) {
            WorkerStatus ws = statusByPort.get(port);
            ws.getAvailableKvCacheTokens().set(status.getAvailableKvCache());
            endpointRegistry.getDecode(ipPortByEnginePort.get(port))
                    .onWorkerStatusUpdate(ws, resp);
        }
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

    static <T> T unary(java.util.function.Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("unary response timeout");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted waiting for unary response");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        if (response.get() == null) {
            throw new AssertionError("unary response missing");
        }
        return response.get();
    }

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
            Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
            Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
            MAPPER.writeValue(performance.toFile(), Map.of(
                    "block_size", 1024,
                    "sleep_scale", 1.0,
                    "jitter_pct", 0.0,
                    "prefill", Map.of("scale", 1.0),
                    "decode", Map.of("scale", 1.0,
                            "step_ms_by_batch", List.of(List.of(1, decodeStepMs)))));
            MockMasterConfig.writeWithPrefillExpression(master, prefillFormulaMs);
            return MockPerformanceModel.load(performance.toString(), master.toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() {
        stopAutoPump();
        scheduler.shutdown();
        for (JavaMockEngineCluster.FastRpcService svc : services.values()) {
            svc.shutdown();
        }
        engineScheduler.shutdownNow();
    }
}

package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Task34 类别三：基线回归 parity 补强。
 *
 * <ol>
 *   <li>开关全关：priority 字段完全不影响行为——队列保持 arrival FIFO、
 *       priority 调度器零交互（legacy 路径 parity）；</li>
 *   <li>开关全开但全同优先级（全 50）：与关闭态等价——无驱逐、无让位、
 *       admitted 集合一致、排序退化为 deadline/arrival（此前缺失的关键断言）；</li>
 *   <li>混合开关矩阵抽样：任一子开关组合在同优先级负载下均不产生驱逐。</li>
 * </ol>
 */
class AutoTpmBaselineParityTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    @Test
    void decodeAcceptedWorkerStatusClosesLeaseAndReopensAdmissionCapacity() throws Exception {
        Harness h = new Harness(cfg -> {
            Harness.enableAll(cfg);
            cfg.setFlexlbBatchSizeMax(1);
            cfg.setAutoTpmPostSuccessBackpressureLimit(1);
            cfg.setAutoTpmPostSuccessSoftTimeoutMs(60_000);
        });
        try {
            Response first = h.submit(1, 50).get(2, TimeUnit.SECONDS);
            assertTrue(first.isSuccess());
            assertEquals(1, h.activeLeaseCount());

            h.reportDecodePhase(1, TaskPhase.RECEIVED);
            assertEquals(1, h.activeLeaseCount(),
                    "Decode RECEIVED is not Decode ownership");

            h.reportDecodePhase(1, TaskPhase.KV_ALLOCATED);
            assertEquals(0, h.activeLeaseCount());
            // Duplicate/later acceptance observations are idempotent.
            h.reportDecodePhase(1, TaskPhase.RUNNING);

            Response second = h.submit(2, 50).get(2, TimeUnit.SECONDS);
            assertTrue(second.isSuccess(), second.getErrorMessage());
        } finally {
            h.close();
        }
    }

    // ============ ① 开关全关：legacy 路径 parity ============

    @Test
    void switches_off_ignores_priority_and_keeps_arrival_fifo_queue_order() throws Exception {
        Harness h = new Harness(cfg -> { });
        try {
            // 交错优先级提交（p70 最后到达）——关闭态队列必须保持 arrival FIFO
            h.submitSpaced(List.of(
                    req(1, 30), req(2, 70), req(3, 50), req(4, 40), req(5, 70)));

            PrefillQueueSnapshot snapshot = h.prefillQueueSnapshot();
            assertEquals(List.of(1L, 2L, 3L, 4L, 5L), requestIds(snapshot),
                    "switches-off queue must stay arrival FIFO regardless of priority");
            // priority 调度器/上报完全零交互
            verifyNoInteractions(h.priorityReporter);
        } finally {
            h.close();
        }
    }

    // ============ ② 全开 + 全同优先级 = 关闭态等价（关键补强） ============

    @Test
    void all_on_uniform_priority_queue_order_degrades_to_arrival_fifo() throws Exception {
        Harness off = new Harness(cfg -> { });
        Harness on = new Harness(Harness::enableAll);
        try {
            List<long[]> requests = List.of(
                    req(11, 50), req(12, 50), req(13, 50), req(14, 50), req(15, 50));
            off.submitSpaced(requests);
            on.submitSpaced(requests);

            // 全同优先级 + 同 seqLen（deadline = arrival + 同一 SLO）→ 排序退化
            // 为 arrival 序，与关闭态逐项一致
            List<Long> offOrder = requestIds(off.prefillQueueSnapshot());
            List<Long> onOrder = requestIds(on.prefillQueueSnapshot());
            assertEquals(List.of(11L, 12L, 13L, 14L, 15L), offOrder);
            assertEquals(offOrder, onOrder,
                    "uniform-priority auto-tpm queue order must equal switches-off order");
        } finally {
            off.close();
            on.close();
        }
    }

    @Test
    void all_on_uniform_priority_admits_same_set_and_never_evicts_under_capacity_pressure()
            throws Exception {
        Harness off = new Harness(cfg -> cfg.setDecodeConcurrencyLimit(2));
        Harness on = new Harness(cfg -> {
            Harness.enableAll(cfg);
            cfg.setDecodeConcurrencyLimit(2);
        });
        try {
            // 2 个 decode 槽位、5 个全同优先级请求顺序提交：
            // 两种形态 admitted/rejected 集合必须一致
            for (long id = 21; id <= 25; id++) {
                off.submitAndSettle(id, 50);
                on.submitAndSettle(id, 50);
            }

            assertEquals(off.admittedIds(), on.admittedIds(),
                    "uniform-priority auto-tpm must admit exactly the switches-off set");
            assertEquals(off.rejectedIds(), on.rejectedIds(),
                    "uniform-priority auto-tpm must reject exactly the switches-off set");
            // 已占位请求原封未动：无驱逐、无让位
            for (CompletableFuture<Response> admitted : on.admittedFutures()) {
                Response r = admitted.getNow(null);
                assertTrue(r == null || r.isSuccess(),
                        "admitted request must never be evicted under uniform priority");
            }
            on.verifyNoEvictionEverCommitted();
        } finally {
            off.close();
            on.close();
        }
    }

    @Test
    void all_on_uniform_priority_incoming_fails_explicitly_and_victim_is_untouched()
            throws Exception {
        Harness on = new Harness(cfg -> {
            Harness.enableAll(cfg);
            // Keep Decode routable; this case is deliberately a Prefill
            // FIFO/queue-capacity rejection, not a Decode slot rejection.
            cfg.setDecodeConcurrencyLimit(100);
            cfg.setFlexlbBatchQueueMaxSize(1);
        });
        try {
            DecodeEndpoint decodeEp = on.endpointRegistry.getDecode(DECODE_IP_PORT);
            CompletableFuture<Response> holder = on.submit(31, 50);
            await(() -> decodeEp.reservedView().containsKey(31L));
            await(() -> requestIds(on.prefillQueueSnapshot()).contains(31L));
            PrefillQueueSnapshot fullQueue = on.prefillQueueSnapshot();
            assertEquals(1, fullQueue.queueCapacity());
            assertEquals(1, fullQueue.items().size());
            assertEquals(50, fullQueue.items().get(0).priority());
            assertEquals(QueuedRequestSnapshot.PREFILL_QUEUED,
                    fullQueue.items().get(0).state());
            assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                    AdmissionFailureClassifier.classifyPrefill(
                            new PriorityRequestEnvelope(32, 50, 128, 8,
                                    System.currentTimeMillis(), 0, 0, 128, 136),
                            fullQueue).reason());

            // Prefill 队列已被先到的同优先级请求占满：Master 必须从
            // 这一份队列快照判定 SAME_PRIORITY_AHEAD，不能按 QoS 阈值猜原因。
            for (long id = 32; id <= 34; id++) {
                Response r = on.submit(id, 50).get(2, TimeUnit.SECONDS);
                assertFalse(r.isSuccess());
                assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                        r.getCode(), r.getErrorMessage());
                assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                        r.getAdmissionRejectReason());
            }
            assertTrue(decodeEp.reservedView().containsKey(31L), "victim reservation untouched");
            assertFalse(holder.isCompletedExceptionally());
            on.verifyNoEvictionEverCommitted();
        } finally {
            on.close();
        }
    }

    // ============ ③ 混合开关矩阵抽样 ============

    @Test
    void mixed_switch_matrix_uniform_priority_stays_equivalent() throws Exception {
        List<Consumer<FlexlbConfig>> matrix = List.of(
                cfg -> cfg.setAutoTpmPrefillQueueEvictEnabled(true),
                cfg -> cfg.setAutoTpmDecodeReservedEvictEnabled(true),
                cfg -> {
                    cfg.setAutoTpmPrefillQueueEvictEnabled(true);
                    cfg.setAutoTpmDecodeReservedEvictEnabled(true);
                });
        for (Consumer<FlexlbConfig> combo : matrix) {
            Harness h = new Harness(cfg -> {
                cfg.setAutoTpmEnabled(true);
                combo.accept(cfg);
                cfg.setDecodeConcurrencyLimit(1);
            });
            try {
                DecodeEndpoint decodeEp = h.endpointRegistry.getDecode(DECODE_IP_PORT);
                h.submit(41, 50);
                await(() -> decodeEp.reservedView().containsKey(41L));

                Response r = h.submit(42, 50).get(2, TimeUnit.SECONDS);

                // 任一子开关组合：同优先级下无驱逐、无让位、失败明确
                assertFalse(r.isSuccess(), "incoming must fail explicitly");
                assertTrue(decodeEp.reservedView().containsKey(41L),
                        "equal-priority reservation must never yield under any switch combo");
                h.verifyNoEvictionEverCommitted();
            } finally {
                h.close();
            }
        }
    }

    // ==================== per-case harness ====================

    private static final class Harness {
        final ConfigService configService = mock(ConfigService.class);
        final Router router = mock(Router.class);
        final PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);
        final FlexlbConfig config = new FlexlbConfig();
        final EndpointRegistry endpointRegistry;
        final FlexlbBatchScheduler scheduler;
        final DefaultBatchDispatcher dispatcher;
        final PriorityAdmissionScheduler priorityScheduler;
        private final List<Long> submittedIds = new ArrayList<>();
        private final List<CompletableFuture<Response>> submittedFutures = new ArrayList<>();

        Harness(Consumer<FlexlbConfig> customize) {
            EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);

            config.setScheduleWorkerSize(1);
            // park 模式：队列驻留不派发，便于确定性观测队列序
            config.setFlexlbBatchSizeMax(100);
            config.setFlexlbBatchFixedWaitMs(10_000);
            config.setFlexlbBatchWindowMs(10_000);
            config.setCostSloMs(50_000L);
            config.setCostSloRiskMarginMs(50L);
            config.setDecodeConcurrencyLimit(100);
            customize.accept(config);
            when(configService.loadBalanceConfig()).thenReturn(config);

            when(router.route(any(BalanceContext.class)))
                    .thenAnswer(inv -> routeAnswer(inv.getArgument(0)));
            when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                    any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                    .thenAnswer(inv -> {
                        EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                        .setBatchId(request.getBatchId());
                        for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
                            response.addSuccesses(
                                    EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                            .setRequestId(input.getRequestId()));
                        }
                        return CompletableFuture.completedFuture(response.build());
                    });

            endpointRegistry = new EndpointRegistry(configService, this::getScheduler, reporter);
            dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
            priorityScheduler = new PriorityAdmissionScheduler(
                    configService, router, endpointRegistry, new PlanCommitter(),
                    new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                            PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                    priorityReporter, reporter, new UnsupportedEngineCancelChannel());
            scheduler = new FlexlbBatchScheduler(configService, router,
                    endpointRegistry, dispatcher, reporter, priorityScheduler, null);

            WorkerStatus prefillWs = new WorkerStatus();
            prefillWs.setIp("10.0.0.1");
            prefillWs.setPort(8080);
            prefillWs.setGrpcPort(8081);
            endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

            WorkerStatus decodeWs = new WorkerStatus();
            decodeWs.setIp("10.0.0.2");
            decodeWs.setPort(8081);
            decodeWs.setGrpcPort(8082);
            decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
            decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
            endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .onWorkerStatusUpdate(decodeWs, new WorkerStatusResponse());
        }

        static void enableAll(FlexlbConfig cfg) {
            cfg.setAutoTpmEnabled(true);
            cfg.setAutoTpmPrefillQueueEvictEnabled(true);
            cfg.setAutoTpmDecodeReservedEvictEnabled(true);
            // PR-D: rescue removed — orTimeout + AdmissionLease handle stuck/deadline requests
        }

        private FlexlbBatchScheduler getScheduler() {
            return scheduler;
        }

        private Response routeAnswer(BalanceContext ctx) {
            DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
            if (decodeEp.getTotalLoad() + 1 > config.getDecodeConcurrencyLimit()) {
                return Response.error(StrategyErrorType.NO_DECODE_WORKER);
            }
            decodeEp.reserve(ctx.getRequestId(), 128, 136, ctx.getPriority(), ctx.getDeadlineMs());
            return successRoute(ctx.getRequestId());
        }

        /** 顺序提交并等待入队（arrival 单调递增，避免同 ms 并列）。 */
        void submitSpaced(List<long[]> requests) throws Exception {
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(PREFILL_IP_PORT);
            for (long[] r : requests) {
                int before = prefillEp.getBatcher().queueSize();
                submitAndTrack(r[0], (int) r[1]);
                await(() -> prefillEp.getBatcher().queueSize() == before + 1);
                TimeUnit.MILLISECONDS.sleep(3);
            }
        }

        void submitAndSettle(long requestId, int priority) throws Exception {
            CompletableFuture<Response> future = submitAndTrack(requestId, priority);
            DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
            await(() -> future.isDone() || decodeEp.reservedView().containsKey(requestId));
        }

        private CompletableFuture<Response> submitAndTrack(long requestId, int priority) {
            CompletableFuture<Response> future = submit(requestId, priority);
            submittedIds.add(requestId);
            submittedFutures.add(future);
            return future;
        }

        private CompletableFuture<Response> submit(long requestId, int priority) {
            BalanceContext ctx = context(requestId, priority, 128);
            if (config.isAutoTpmEnabled()) {
                long now = System.currentTimeMillis();
                ctx.setBudget(ScheduleBudget.forDeadline(priority, now, now + 30_000));
            }
            return scheduler.submit(ctx);
        }

        void reportDecodePhase(long requestId, TaskPhase phase) {
            reportPhase(RoleType.DECODE, requestId, phase);
        }

        void reportPhase(RoleType role, long requestId, TaskPhase phase) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(phase);
            task.setInputLength(128);
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setRole(role);
            response.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
            if (role == RoleType.DECODE) {
                endpointRegistry.getDecode(DECODE_IP_PORT)
                        .onWorkerStatusUpdate(new WorkerStatus(), response);
            } else if (role == RoleType.PREFILL) {
                endpointRegistry.getPrefill(PREFILL_IP_PORT)
                        .onWorkerStatusUpdate(new WorkerStatus(), response);
            }
            scheduler.onWorkerStatusUpdate(response);
        }

        int activeLeaseCount() {
            return priorityScheduler.activeLeaseCount();
        }

        /** admitted = 拿到 decode 占位（future 未失败）。 */
        List<Long> admittedIds() {
            List<Long> ids = new ArrayList<>();
            for (int i = 0; i < submittedIds.size(); i++) {
                if (!isRejected(submittedFutures.get(i))) {
                    ids.add(submittedIds.get(i));
                }
            }
            return ids;
        }

        List<Long> rejectedIds() {
            List<Long> ids = new ArrayList<>();
            for (int i = 0; i < submittedIds.size(); i++) {
                if (isRejected(submittedFutures.get(i))) {
                    ids.add(submittedIds.get(i));
                }
            }
            return ids;
        }

        List<CompletableFuture<Response>> admittedFutures() {
            List<CompletableFuture<Response>> list = new ArrayList<>();
            for (CompletableFuture<Response> f : submittedFutures) {
                if (!isRejected(f)) {
                    list.add(f);
                }
            }
            return list;
        }

        private static boolean isRejected(CompletableFuture<Response> future) {
            Response r = future.getNow(null);
            return r != null && !r.isSuccess();
        }

        PrefillQueueSnapshot prefillQueueSnapshot() {
            return endpointRegistry.getPrefill(PREFILL_IP_PORT)
                    .getBatcher().queueManager().snapshot();
        }

        void verifyNoEvictionEverCommitted() {
            verify(priorityReporter, never())
                    .reportVictim(anyInt(), anyInt(), anyString(), anyString());
            verify(priorityReporter, never())
                    .reportEvictionCommit(anyInt(), anyString(), org.mockito.ArgumentMatchers.eq("success"));
        }

        void close() {
            scheduler.shutdown();
            dispatcher.shutdown();
        }
    }

    // ==================== helpers ====================

    private static long[] req(long requestId, int priority) {
        return new long[]{requestId, priority};
    }

    private static List<Long> requestIds(PrefillQueueSnapshot snapshot) {
        return snapshot.items().stream().map(QueuedRequestSnapshot::requestId).toList();
    }

    private static void await(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (!condition.getAsBoolean()) {
            if (System.currentTimeMillis() > deadline) {
                throw new AssertionError("condition not met within 2s");
            }
            TimeUnit.MILLISECONDS.sleep(5);
        }
    }

    private static BalanceContext context(long requestId, int priority, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static List<EngineRpcService.GenerateInputPB> batchInputs(
            EngineRpcService.EnqueueBatchRequestPB request) {
        List<EngineRpcService.GenerateInputPB> inputs = new ArrayList<>();
        for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
            for (EngineRpcService.EnqueueBatchExternalInputPB item : slot.getRequestsList()) {
                inputs.add(item.getInput());
            }
        }
        return inputs;
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort,
                                       long requestId) {
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
}

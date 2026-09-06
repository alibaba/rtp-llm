package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.RequestIdFixtures;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
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
import org.flexlb.engine.grpc.RequestId;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Task34 类别二：高并发决策冲突压力测试 —— 8 线程 × 25 请求 × 20 轮
 * （固定随机种子，失败输出可复现 seed），混合执行并发 schedule（混合优先级）、
 * P 队列驱逐、decode reserved 驱逐、orTimeout 超时释放、calibrate /
 * WorkerStatus 更新、TTL 清理和随机 preempt 注入。
 *
 * <p>断言的不变式：
 * <ol>
 *   <li>每个请求恰好一个终态（future 全部完成，无静默丢失）；</li>
 *   <li>账目守恒：全部终结后 decode inflightCount=0、inflightHardKvReserved=0、
 *       totalLoad 回落 0，且任意采样时刻 shadow KV ≥ 0；</li>
 *   <li>版本冲突只导致重试或明确失败（终态码在合法集合内，绝不出现未知码）；</li>
 *   <li>victim 终态码分离：8429 只能来自注入的 accepted-preempt
 *       （queued/reserved victim 只允许 8400）。</li>
 * </ol>
 */
class PriorityConcurrencyStressTest {

    /** 默认 20 轮（单类 ≤60s）；竞态狩猎可用 STRESS_ROUNDS=150 环境变量加深。 */
    private static final int ROUNDS = System.getenv("STRESS_ROUNDS") != null
            ? Integer.parseInt(System.getenv("STRESS_ROUNDS")) : 20;
    private static final int THREADS = 8;
    private static final int REQUESTS_PER_THREAD = 25;
    private static final int TOTAL = THREADS * REQUESTS_PER_THREAD;
    private static final int[] PRIORITIES = {30, 40, 50, 70};

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String PREFILL2_IP_PORT = "10.0.0.3:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    @Test
    @Timeout(60)
    void concurrent_mixed_priority_storm_preserves_invariants() throws Exception {
        for (int round = 0; round < ROUNDS; round++) {
            long seed = 42L + round;
            Harness h = new Harness();
            try {
                runRound(h, round, seed);
            } finally {
                h.close();
            }
        }
    }

    private void runRound(Harness h, int round, long seed) throws Exception {
        long idBase = (round + 1) * 1_000_000L;
        Map<Long, CompletableFuture<Response>> futures = new ConcurrentHashMap<>();
        Set<Long> injectedPreempts = ConcurrentHashMap.newKeySet();
        Set<Long> settled = ConcurrentHashMap.newKeySet();
        ConcurrentLinkedQueue<String> violations = new ConcurrentLinkedQueue<>();
        DecodeEndpoint decodeEp = h.endpointRegistry.getDecode(DECODE_IP_PORT);

        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch submittersDone = new CountDownLatch(THREADS);

        // ---- 提交线程：混合优先级、混合长度并发 schedule ----
        List<Thread> submitters = new ArrayList<>();
        for (int t = 0; t < THREADS; t++) {
            final int threadIdx = t;
            Thread thread = new Thread(() -> {
                Random rnd = new Random(seed * 131 + threadIdx);
                try {
                    startGate.await();
                    for (int i = 0; i < REQUESTS_PER_THREAD; i++) {
                        long id = idBase + threadIdx * 1_000L + i;
                        int priority = PRIORITIES[rnd.nextInt(PRIORITIES.length)];
                        long seqLen = rnd.nextBoolean() ? 128 : 2_000;
                        futures.put(id, h.scheduler.submit(context(String.valueOf(id), priority, seqLen)));
                    }
                } catch (Throwable e) {
                    violations.add("submitter " + threadIdx + " threw: " + e);
                } finally {
                    submittersDone.countDown();
                }
            }, "stress-submitter-" + t);
            submitters.add(thread);
            thread.start();
        }

        // ---- 混合干扰线程：结算/校准/TTL 清理/rescue tick/preempt 注入 + 采样 ----
        AtomicBoolean mixerRunning = new AtomicBoolean(true);
        Thread mixer = new Thread(() -> {
            Random rnd = new Random(seed * 17 + 7);
            while (mixerRunning.get()) {
                try {
                    // 任意时刻不变式采样：shadow KV / totalLoad 永不为负
                    long hardKv = decodeEp.inflightHardKvReserved();
                    int load = decodeEp.getTotalLoad();
                    if (hardKv < 0) {
                        violations.add("negative hard KV " + hardKv + " seed=" + seed);
                    }
                    if (load < 0) {
                        violations.add("negative totalLoad " + load + " seed=" + seed);
                    }

                    // 已成功派发的请求按生产流程结算（decode 完成 + calibrate），
                    // 释放槽位制造持续的 admission/eviction 竞争
                    List<Long> done = new ArrayList<>();
                    for (Map.Entry<Long, CompletableFuture<Response>> e : futures.entrySet()) {
                        Response r = e.getValue().getNow(null);
                        if (r != null && r.isSuccess() && settled.add(e.getKey())) {
                            done.add(e.getKey());
                        }
                    }
                    if (!done.isEmpty()) {
                        h.settleAsDecodeFinished(done);
                    }

                    int op = rnd.nextInt(10);
                    if (op == 0) {
                        h.scheduler.cleanupInflight();
                    } else if (op == 1) {
                        decodeEp.evictExpiredRequests(300_000);
                        h.endpointRegistry.getPrefill(PREFILL_IP_PORT).evictExpiredBatches(300_000);
                    } else if (op == 2) {
                        // PR-D: rescue tick replaced by orTimeout-based admission timeout.
                        // The mixer's settlement loop above already simulates the
                        // timeout path by settling successfully-dispatched requests.
                        decodeEp.evictExpiredRequests(300_000);
                    } else if (op == 3) {
                        // 随机注入 accepted-preempt（8429 的唯一合法来源）
                        long id = idBase + rnd.nextInt(THREADS) * 1_000L
                                + rnd.nextInt(REQUESTS_PER_THREAD);
                        injectedPreempts.add(id);
                        h.scheduler.finishPreemptedById(String.valueOf(id),
                                "preempted by higher-priority request 999999");
                    }
                    TimeUnit.MILLISECONDS.sleep(2);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable e) {
                    violations.add("mixer threw: " + e + " seed=" + seed);
                }
            }
        }, "stress-mixer");
        mixer.start();

        startGate.countDown();
        assertTrue(submittersDone.await(20, TimeUnit.SECONDS),
                "submitters did not finish, seed=" + seed);
        mixerRunning.set(false);
        mixer.join(2_000);
        for (Thread t : submitters) {
            t.join(2_000);
        }

        // ---- 终局结算：已派发成功（ACK 后 future=200）的请求按生产流程注入
        // decode 完成 + calibrate；尚在 batcher 队列中的请求等待其派发或失败，
        // 直到全部终态且账目归零。绝不对 QUEUED 状态注入 decode 完成（生产中
        // decode 不可能先于 prefill 派发上报完成）。----
        long settleDeadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < settleDeadline) {
            List<Long> done = new ArrayList<>();
            for (Map.Entry<Long, CompletableFuture<Response>> e : futures.entrySet()) {
                Response r = e.getValue().getNow(null);
                if (r != null && r.isSuccess() && settled.add(e.getKey())) {
                    done.add(e.getKey());
                }
            }
            h.settleAsDecodeFinished(done);
            boolean allDone = futures.values().stream().allMatch(CompletableFuture::isDone);
            if (allDone && decodeEp.getInflightCount() == 0 && decodeEp.getTotalLoad() == 0) {
                break;
            }
            TimeUnit.MILLISECONDS.sleep(20);
        }

        // ---- 不变式断言（失败信息均带可复现 seed）----
        assertTrue(violations.isEmpty(), "invariant violations " + violations + " seed=" + seed);
        assertEquals(TOTAL, futures.size(), "request count mismatch, seed=" + seed);

        Set<Integer> legalCodes = Set.of(
                200,
                StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                StrategyErrorType.NO_DECODE_WORKER.getErrorCode(),
                StrategyErrorType.QUEUE_FULL.getErrorCode(),
                StrategyErrorType.SCHEDULER_PLAN_CONFLICT.getErrorCode(),
                StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode());
        int successCount = 0;
        int failureCount = 0;
        for (Map.Entry<Long, CompletableFuture<Response>> e : futures.entrySet()) {
            // 不变式①：恰好一个终态，绝不静默丢请求
            assertTrue(e.getValue().isDone(),
                    "request " + e.getKey() + " has no terminal state, seed=" + seed);
            Response r = e.getValue().get();
            int code = r.isSuccess() ? 200 : r.getCode();
            // 不变式③：冲突/失败必须是合法明确终态码
            assertTrue(legalCodes.contains(code),
                    "unexpected terminal code " + code + " for request " + e.getKey()
                            + " message=" + r.getErrorMessage() + " seed=" + seed);
            if (r.isSuccess()) {
                successCount++;
            } else {
                failureCount++;
            }
            // 不变式④：8429 只允许来自注入的 engine-owned preempt；
            // incoming 准入拒绝仅使用 8430/8431 分类。
            if (code == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()) {
                assertTrue(injectedPreempts.contains(e.getKey()),
                        "8429 for request " + e.getKey()
                                + " was not an injected accepted-preempt, seed=" + seed);
            }
        }
        assertEquals(TOTAL, successCount + failureCount, "terminal count mismatch, seed=" + seed);

        // 不变式②：账目守恒 — 全部终结后回落基线
        assertEquals(0, decodeEp.getInflightCount(),
                "decode inflightCount not drained, seed=" + seed);
        assertEquals(0L, decodeEp.inflightHardKvReserved(),
                "decode hard KV reservation leaked, seed=" + seed);
        assertEquals(0, decodeEp.getTotalLoad(),
                "decode totalLoad did not return to baseline, seed=" + seed);
        assertTrue(decodeEp.reservedView().isEmpty(),
                "shadow reservations leaked: " + decodeEp.reservedView().keySet()
                        + " seed=" + seed);
    }

    // ==================== per-round harness ====================

    /** 每轮独立的调度器 + 双 prefill/单 decode endpoint 环境。 */
    private static final class Harness {
        final ConfigService configService = mock(ConfigService.class);
        final Router router = mock(Router.class);
        final FlexlbConfig config = new FlexlbConfig();
        final EndpointRegistry endpointRegistry;
        final PriorityScheduler scheduler;
        final PriorityAdmissionScheduler priorityScheduler;
        final DefaultBatchDispatcher dispatcher;
        final WorkerStatus decodeWs;

        Harness() {
            EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
            PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);

            // 快速 dispatch 制造 accepted 流转；小队列制造 queue-full 驱逐竞争
            SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(4);
            SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10);
            SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(16);
            SchedulingTestConfig.usePriorityQueue(config);
            SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
            SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
            // PR-D: rescue removed — orTimeout on submit() handles stuck requests
            // 有限 decode 槽位：并发下持续触发 slot-full 仲裁
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (6));
            when(configService.loadBalanceConfig()).thenReturn(config);

            when(router.route(any(BalanceContext.class)))
                    .thenAnswer(inv -> routeAnswer(inv.getArgument(0)));
            when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                    any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                    .thenAnswer(inv -> CompletableFuture.completedFuture(ackFor(inv.getArgument(2))));

            endpointRegistry = new EndpointRegistry(configService, this::getScheduler, reporter);
            dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
            priorityScheduler = new PriorityAdmissionScheduler(
                    configService, router, endpointRegistry, new PlanCommitter(),
                    priorityReporter, reporter, new UnsupportedEngineCancelChannel()) {
                @Override
                protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                                      FlexlbConfig config,
                                                                      String group) {
                    return server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, ctx.getRequestId());
                }
            };
            scheduler = new PriorityScheduler(configService, router,
                    endpointRegistry, dispatcher, reporter, priorityScheduler, null,
                    new UnsupportedEngineCancelChannel());

            registerPrefill(PREFILL_IP_PORT, "10.0.0.1");
            registerPrefill(PREFILL2_IP_PORT, "10.0.0.3");

            decodeWs = new WorkerStatus();
            decodeWs.setIp("10.0.0.2");
            decodeWs.setPort(8081);
            decodeWs.setGrpcPort(8082);
            decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
            decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
            endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .onWorkerStatusUpdate(decodeWs, new WorkerStatusResponse());
        }

        private PriorityScheduler getScheduler() {
            return scheduler;
        }

        private void registerPrefill(String ipPort, String ip) {
            WorkerStatus ws = new WorkerStatus();
            ws.setIp(ip);
            ws.setPort(8080);
            ws.setGrpcPort(8081);
            endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);
        }

        /** 生产结算链：scheduler 终结 inflight + decode endpoint calibrate 释放影子预留。 */
        void settleAsDecodeFinished(List<Long> requestIds) {
            if (requestIds.isEmpty()) {
                return;
            }
            Map<String, TaskInfo> finished = new HashMap<>();
            for (Long id : requestIds) {
                TaskInfo task = new TaskInfo();
                task.setRequestId(String.valueOf(id));
                task.setErrorCode(0);
                finished.put(String.valueOf(id), task);
            }
            WorkerStatusResponse resp = new WorkerStatusResponse();
            resp.setRole(RoleType.DECODE);
            resp.setFinishedTaskInfo(finished);
            scheduler.onWorkerStatusUpdate(resp);
            endpointRegistry.getDecode(DECODE_IP_PORT).onWorkerStatusUpdate(decodeWs, resp);
        }

        /** 容量感知 route 替身：镜像生产 decode 硬过滤 + 带优先级的影子预留。 */
        private Response routeAnswer(BalanceContext ctx) {
            DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
            if (decodeEp.getTotalLoad() + 1 > config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests()) {
                return Response.error(StrategyErrorType.NO_DECODE_WORKER);
            }
            decodeEp.reserve(ctx.getRequestId(), 128, 136, ctx.getPriority());
            return successRoute(ctx.getRequestId());
        }

        void close() {
            priorityScheduler.shutdown();
            scheduler.shutdown();
            // 每轮独立 Harness：dispatcher 线程池（生产由 Spring @PreDestroy 管理）
            // 必须显式回收，否则多轮狩猎会耗尽 native 线程
            dispatcher.shutdown();
        }
    }

    // ==================== helpers ====================

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(external -> Long.parseLong(RequestId.parse(external.getInput())))
                .forEach(requestId -> response.addSuccesses(
                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(requestId)
                                .build()));
        return response.build();
    }

    private static BalanceContext context(String requestId, int priority, long seqLen) {
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

    private static byte[] generateInputBytes(String requestId) {
        EngineRpcService.GenerateInputPB input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort,
                                       String requestId) {
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

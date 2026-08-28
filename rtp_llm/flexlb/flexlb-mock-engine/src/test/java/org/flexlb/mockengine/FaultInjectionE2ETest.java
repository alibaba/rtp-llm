package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Task35 场景 C：11 种故障注入逐一验证 —— 明确错误传播到调度器终态，
 * post-send 不确定性进入安全 fence，调度器不崩溃且同集群其余引擎不受影响。
 *
 * <p>覆盖 {@link FaultInjectionConfig} 全部字段：failOnEnqueue、enqueueErrorCode、
 * enqueueErrorMessage、enqueueDelayMs、generateDelayMs、generateError、fetchError、
 * noRespond、kvPressureTokens、queueDepthLimit、crashAfterNRequests。
 *
 * <p>控制面（enqueueBatch）故障走真实调度器栈：明确拒绝断言 8510 终态，
 * missing ACK 则断言保守保留记账；数据面（generate_stream/fetch_response）
 * 故障不经过 LB 控制面，用直连 RPC 断言 mock 行为，同时验证调度器路径不受影响。
 *
 * <p>已知缺陷（只报不修，见任务报告）：{@code enqueueErrorCode} 字段从未被
 * {@link JavaMockEngineCluster} 读取 —— 错误响应只携带 message 不携带自定义
 * 错误码，c01 用断言固化该现状。
 */
class FaultInjectionE2ETest {

    private static final int BASE_PORT = 62900;

    /** 通用布防：单请求批次 + 快派发，每个请求恰好一次 enqueueBatch。 */
    private static void arm(AutoTpmE2EHarness h) {
        h.startAutoPump(10);
    }

    private static Response submitTo(AutoTpmE2EHarness h, int prefillIndex,
                                     long requestId) throws Exception {
        h.prefillSelector = ctx -> prefillIndex;
        CompletableFuture<Response> future = h.scheduler.submit(h.context(requestId, 50));
        return future.get(10, TimeUnit.SECONDS);
    }

    // ==================== C1 failOnEnqueue + enqueueErrorMessage + enqueueErrorCode ====================

    @Test
    @Timeout(30)
    void c01_fail_on_enqueue_propagates_8510_message_and_isolates_healthy_engine() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT, 2, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            h.prefillEngines.get(0).setFaultConfig(FaultInjectionConfig.builder()
                    .failOnEnqueue(true)
                    .enqueueErrorMessage("injected-boom")
                    .enqueueErrorCode(9999)
                    .build());

            Response failed = submitTo(h, 0, 9101);
            assertFalse(failed.isSuccess());
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), failed.getCode(),
                    "engine-side enqueue rejection must propagate as 8510: " + failed.getErrorMessage());
            assertTrue(failed.getErrorMessage().contains("injected-boom"),
                    "custom enqueueErrorMessage must reach the caller: " + failed.getErrorMessage());
            // 缺陷固化：enqueueErrorCode 从未被 mock 引擎写入 wire（死配置）
            assertFalse(failed.getErrorMessage().contains("9999"),
                    "KNOWN DEFECT: enqueueErrorCode is never consumed by JavaMockEngineCluster");

            // 故障引擎无泄漏（请求从未被接受）
            assertEquals(0, h.prefillEngines.get(0).getRunningCount());
            assertEquals(0, h.prefillEngines.get(0).getAcceptedCount());

            // 健康引擎完全不受影响
            Response healthy = submitTo(h, 1, 9102);
            assertTrue(healthy.isSuccess(),
                    "healthy engine must be unaffected: " + healthy.getErrorMessage());
        }
    }

    // ==================== C2 enqueueDelayMs ====================

    @Test
    @Timeout(30)
    void c02_enqueue_delay_defers_ack_but_request_succeeds() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 10, 1, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            h.prefillEngines.get(0).setFaultConfig(FaultInjectionConfig.builder()
                    .enqueueDelayMs(300)
                    .build());

            long start = System.nanoTime();
            Response response = submitTo(h, 0, 9201);
            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

            assertTrue(response.isSuccess(),
                    "delayed ack must still succeed: " + response.getErrorMessage());
            assertTrue(elapsedMs >= 300,
                    "terminal must wait for the delayed engine ack, took " + elapsedMs + "ms");
            assertEquals(1, h.prefillEngines.get(0).getAcceptedCount());
        }
    }

    // ==================== C3 generateDelayMs ====================

    @Test
    @Timeout(30)
    void c03_generate_delay_slows_prefill_execution_without_breaking_completion() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 20, 1, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            JavaMockEngineCluster.FastRpcService prefill = h.prefillEngines.get(0);
            prefill.setFaultConfig(FaultInjectionConfig.builder()
                    .generateDelayMs(300)
                    .build());

            long start = System.nanoTime();
            Response response = submitTo(h, 0, 9301);
            assertTrue(response.isSuccess(), "ack is not delayed by generateDelayMs");

            // prefill 执行被拉长 300ms，但最终完整排空、无泄漏
            AutoTpmE2EHarness.await(() -> prefill.getRunningCount() == 0, 5_000,
                    "slowed prefill must still drain");
            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            assertTrue(elapsedMs >= 300,
                    "prefill execution must include the injected delay, took " + elapsedMs + "ms");
            assertEquals(1, prefill.getAcceptedCount());
            assertFalse(prefill.isLeakDetected());
        }
    }

    // ==================== C4 generateError（数据面直连） ====================

    @Test
    @Timeout(30)
    void c04_generate_error_fails_direct_stream_and_scheduler_path_unaffected() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 30, 2, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            h.prefillEngines.get(0).setFaultConfig(FaultInjectionConfig.builder()
                    .generateError(true)
                    .build());

            Throwable error = callGenerateStream(h.prefillEngines.get(0), 9401).error();
            assertNotNull(error, "generate_stream must fail on the faulted engine");
            assertTrue(error.getMessage().contains("injected generate_error"), error.getMessage());
            assertEquals(0, h.prefillEngines.get(0).getRunningCount(), "no leak on rejected stream");

            Response healthy = submitTo(h, 1, 9402);
            assertTrue(healthy.isSuccess(),
                    "generate_error is data-plane only, scheduler path stays healthy");
        }
    }

    // ==================== C5 fetchError（数据面直连） ====================

    @Test
    @Timeout(30)
    void c05_fetch_error_fails_fetch_response_and_scheduler_path_unaffected() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 40, 2, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            h.prefillEngines.get(0).setFaultConfig(FaultInjectionConfig.builder()
                    .fetchError(true)
                    .build());

            StreamResult result = callFetchResponse(h.prefillEngines.get(0), 9501);
            assertNotNull(result.error(), "fetch_response must fail on the faulted engine");
            assertTrue(result.error().getMessage().contains("injected fetch_error"),
                    result.error().getMessage());

            Response healthy = submitTo(h, 1, 9502);
            assertTrue(healthy.isSuccess(),
                    "fetch_error is data-plane only, scheduler path stays healthy");
        }
    }

    // ==================== C6 noRespond ====================

    @Test
    @Timeout(30)
    void c06_no_respond_hangs_stream_but_dispatch_ack_succeeds_and_engine_drains() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 50, 1, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            JavaMockEngineCluster.FastRpcService prefill = h.prefillEngines.get(0);
            prefill.setFaultConfig(FaultInjectionConfig.builder()
                    .noRespond(true)
                    .build());

            // 控制面语义：enqueue ack 不受 noRespond 影响 → 调度器视角成功
            Response response = submitTo(h, 0, 9601);
            assertTrue(response.isSuccess(),
                    "noRespond does not affect the enqueue ack: " + response.getErrorMessage());

            // 数据面语义：流上永远没有任何事件（不完成、不报错）
            StreamResult silent = callGenerateStream(prefill, 9602);
            assertFalse(silent.completed(), "noRespond stream must never complete");
            assertNull(silent.error(), "noRespond stream must never error");

            // 引擎内部状态照常排空、不泄漏、不影响后续恢复
            AutoTpmE2EHarness.await(() -> prefill.getRunningCount() == 0, 5_000,
                    "noRespond engine still settles its internal accounting");
            prefill.clearFaultConfig();
            Response recovered = submitTo(h, 0, 9603);
            assertTrue(recovered.isSuccess(), "engine recovers after clearing the fault");
        }
    }

    // ==================== C7 kvPressureTokens ====================

    @Test
    @Timeout(30)
    void c07_kv_pressure_propagates_through_worker_status_and_clears() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 60, 1, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            JavaMockEngineCluster.FastRpcService decode = h.decodeEngines.get(0);
            long totalKv = decode.getTotalKvTokens();
            // The scheduler fixture is already published at status version 1.
            // Consume the engine's matching baseline before asserting a newer snapshot.
            h.pumpOnce();
            decode.setFaultConfig(FaultInjectionConfig.builder()
                    .kvPressureTokens(totalKv)
                    .build());

            h.pumpOnce();
            assertEquals(0L, h.decodeEndpoint(0).getStatus().getAvailableKvCacheTokens(),
                    "full KV pressure must surface as zero available tokens in WorkerStatus");

            decode.clearFaultConfig();
            h.pumpOnce();
            assertEquals(totalKv, h.decodeEndpoint(0).getStatus().getAvailableKvCacheTokens(),
                    "clearing the pressure must restore the full capacity view");
        }
    }

    // ==================== C8 queueDepthLimit ====================

    @Test
    @Timeout(30)
    void c08_queue_depth_limit_rejects_overflow_and_isolates_healthy_engine() throws Exception {
        // 长 prefill（800ms）让第一个请求稳定占住 pending 名额
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 70, 2, 1, "800", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            JavaMockEngineCluster.FastRpcService prefill = h.prefillEngines.get(0);

            Response first = submitTo(h, 0, 9801);
            assertTrue(first.isSuccess());
            assertTrue(prefill.getRunningCount() >= 1, "first request holds the queue slot");

            prefill.setFaultConfig(FaultInjectionConfig.builder()
                    .queueDepthLimit(1)
                    .build());
            Response rejected = submitTo(h, 0, 9802);
            assertFalse(rejected.isSuccess());
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), rejected.getCode());
            assertTrue(rejected.getErrorMessage().contains("queue depth limit exceeded"),
                    rejected.getErrorMessage());

            Response healthy = submitTo(h, 1, 9803);
            assertTrue(healthy.isSuccess(), "healthy engine keeps accepting");
        }
    }

    // ==================== C9 crashAfterNRequests ====================

    @Test
    @Timeout(30)
    void c09_crash_after_n_requests_fences_missing_ack_and_isolates_healthy_engine() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT + 80, 2, 1, "5", 1.0, false, DecisionPolicyConfig.single())) {
            arm(h);
            JavaMockEngineCluster.FastRpcService prefill = h.prefillEngines.get(0);
            PrefillEndpoint prefillEndpoint = h.prefillEndpoint(0);
            prefill.setFaultConfig(FaultInjectionConfig.builder()
                    .crashAfterNRequests(1)
                    .build());

            // 首个 EnqueueBatch 触发 crash 并返回空 ACK。请求可能已经越过发送边界，
            // 因此 Master 必须保留 Future 与端点记账，直到 Engine 给出权威终态。
            h.prefillSelector = ctx -> 0;
            CompletableFuture<Response> crashed = h.scheduler.submit(h.context(9902, 50));
            CountDownLatch crashedTerminal = new CountDownLatch(1);
            crashed.whenComplete((ignored, error) -> crashedTerminal.countDown());

            AutoTpmE2EHarness.await(prefill::isStopped, 2_000,
                    "the first enqueue must trigger the configured engine crash");
            AutoTpmE2EHarness.await(() -> h.scheduler.getInflightSize() == 1
                            && prefillEndpoint.getInflightBatchCount() == 1
                            && prefillEndpoint.getLocallyOwnedRequestCount() == 1,
                    2_000, "missing ACK must retain scheduler and Prefill accounting");

            assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount(),
                    "batch delivery must not consume the route-request ledger");
            assertFalse(crashedTerminal.await(250, TimeUnit.MILLISECONDS),
                    "missing ACK without an authoritative Engine terminal must stay fenced");
            assertFalse(crashed.isDone(), "the fenced request must remain incomplete");

            // 一个请求处于不确定性 fence 时，不得阻塞同集群的健康 Prefill。
            Response healthy = submitTo(h, 1, 9904);
            assertTrue(healthy.isSuccess(), "the crash never spreads to the healthy engine");
            assertFalse(crashed.isDone(), "healthy delivery must not settle the unrelated fence");
            assertEquals(1, prefillEndpoint.getInflightBatchCount());
            assertEquals(1, prefillEndpoint.getLocallyOwnedRequestCount());
        }
    }

    // ==================== 数据面直连 helpers ====================

    private record StreamResult(Throwable error, boolean completed) {
    }

    /** 直连 generate_stream，等 500ms 观察流事件（fault 场景内即时返回或保持沉默）。 */
    private static StreamResult callGenerateStream(JavaMockEngineCluster.FastRpcService svc,
                                                   long requestId) throws InterruptedException {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < 8; token++) {
            input.addTokenIds(token);
        }
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch terminal = new CountDownLatch(1);
        svc.generateStreamCall(input.build(), new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
            }

            @Override
            public void onError(Throwable t) {
                error.set(t);
                terminal.countDown();
            }

            @Override
            public void onCompleted() {
                terminal.countDown();
            }
        });
        boolean finished = terminal.await(500, TimeUnit.MILLISECONDS);
        return new StreamResult(error.get(), finished && error.get() == null);
    }

    /** 直连 fetch_response，等 500ms 观察流事件。 */
    private static StreamResult callFetchResponse(JavaMockEngineCluster.FastRpcService svc,
                                                  long requestId) throws InterruptedException {
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch terminal = new CountDownLatch(1);
        svc.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder()
                .setRequestId(requestId)
                .build(), new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
            }

            @Override
            public void onError(Throwable t) {
                error.set(t);
                terminal.countDown();
            }

            @Override
            public void onCompleted() {
                terminal.countDown();
            }
        });
        boolean finished = terminal.await(500, TimeUnit.MILLISECONDS);
        return new StreamResult(error.get(), finished && error.get() == null);
    }
}

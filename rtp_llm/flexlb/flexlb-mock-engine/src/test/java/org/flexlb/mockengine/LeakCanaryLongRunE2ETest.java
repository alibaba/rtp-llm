package org.flexlb.mockengine;

import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Task35 场景 D：泄漏金丝雀长跑（≥60s）—— 混合优先级队列流量 +
 * reducer deadline 超时收敛全程生效，中途注入两轮瞬态故障（enqueue 延迟 / enqueue
 * 拒绝），结束后强断言：
 * <ul>
 *   <li>全部请求到达明确终态，且终态码只落在已知集合内；</li>
 *   <li>引擎侧零泄漏（leak canary：pending/running/activeDecode 全部归零）；</li>
 *   <li>调度器侧账目归零（decode 影子层、prefill 批队列全部清空）。</li>
 * </ul>
 */
class LeakCanaryLongRunE2ETest {

    private static final int BASE_PORT = 63000;
    private static final long TRAFFIC_MILLIS = 62_000;
    private static final long SUBMIT_INTERVAL_MS = 15;
    private static final int[] PRIORITIES = {30, 50, 70};

    @Test
    @Timeout(15)
    void prefillRejectionRetainsDecodeUntilInactivityExpiry() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT + 10, 1, 1, "5", 1.0, true)) {
            h.config.getRequestLifecycle().getRequest().setTimeoutMs(2_000L);
            h.fixedWindowDecision().setMaxRequests(1);
            h.prefillEngines.get(0).setFaultConfig(FaultInjectionConfig.builder()
                    .failOnEnqueue(true).enqueueErrorMessage("reject before admission").build());
            h.startAutoPump(10);
            Response rejected = h.scheduler.submit(h.context(99_999L, 50)).get(5, TimeUnit.SECONDS);
            assertEquals(8510, rejected.getCode());
            assertEquals(1, h.decodeEndpoint(0).getInflightCount(),
                    "Prefill rejection alone cannot prove that Decode is safe to release");
            assertTrue(h.decodeEndpoint(0).routingView().inflightHardKv() > 0);
            AutoTpmE2EHarness.await(() -> h.decodeEndpoint(0).getInflightCount() == 0,
                    7_000, "an unobserved rejected request must expire without a Decode terminal report");
            assertEquals(0L, h.decodeEndpoint(0).routingView().inflightHardKv());
            assertEquals(0, h.prefillEndpoint(0).queuedRequestCount());
            assertEquals(0, h.decodeEngines.get(0).getAcceptedCount(),
                    "this scenario must not manufacture a Decode completion to reclaim ownership");
        }
    }

    @Test
    @Timeout(115)
    void d_long_run_mixed_traffic_with_transient_faults_leaks_nothing() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 2, 1, "5", 1.0, true)) {
            // Prefill rejection is not Decode terminal evidence. Requests that
            // never reach Decode must settle through bounded inactivity cleanup.
            h.config.getRequestLifecycle().getRequest().setTimeoutMs(5_000L);
            // Priority ordering and exact terminal cleanup remain active without preemption.
            // 小批次与快速派发维持持续流量。
            h.fixedWindowDecision().setMaxRequests(4);
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.prefillSelector = ctx -> (int) (ctx.getRequestId() % 2);
            h.startAutoPump(10);

            JavaMockEngineCluster.FastRpcService faultTarget = h.prefillEngines.get(0);
            List<CompletableFuture<Response>> futures = new ArrayList<>(5_000);
            long start = System.currentTimeMillis();
            long rid = 100_000;
            int faultPhase = 0; // 0=clean 1=delay-burst 2=clean 3=reject-burst 4=clean
            while (System.currentTimeMillis() - start < TRAFFIC_MILLIS) {
                long elapsed = System.currentTimeMillis() - start;
                // 瞬态故障窗口 1：20s~25s enqueue 延迟 100ms
                // 瞬态故障窗口 2：40s~43s enqueue 直接拒绝
                int wantedPhase = elapsed >= 43_000 ? 4
                        : elapsed >= 40_000 ? 3
                        : elapsed >= 25_000 ? 2
                        : elapsed >= 20_000 ? 1 : 0;
                if (wantedPhase != faultPhase) {
                    faultPhase = wantedPhase;
                    switch (faultPhase) {
                        case 1 -> faultTarget.setFaultConfig(FaultInjectionConfig.builder()
                                .enqueueDelayMs(100).build());
                        case 3 -> faultTarget.setFaultConfig(FaultInjectionConfig.builder()
                                .failOnEnqueue(true)
                                .enqueueErrorMessage("transient burst rejection").build());
                        default -> faultTarget.clearFaultConfig();
                    }
                }

                int priority = PRIORITIES[(int) (rid % PRIORITIES.length)];
                futures.add(h.scheduler.submit(h.context(rid++, priority)));
                Thread.sleep(SUBMIT_INTERVAL_MS);
            }
            long trafficElapsed = System.currentTimeMillis() - start;
            assertTrue(trafficElapsed >= TRAFFIC_MILLIS,
                    "canary must sustain traffic for at least 60s, ran " + trafficElapsed + "ms");

            // 全部终态
            AutoTpmE2EHarness.await(
                    () -> futures.stream().allMatch(CompletableFuture::isDone), 20_000,
                    "all " + futures.size() + " requests must reach a terminal state");

            // 终态码只允许落在已知集合内
            Map<Integer, Integer> codeTally = new TreeMap<>();
            for (int i = 0; i < futures.size(); i++) {
                Response response = futures.get(i).get(1, TimeUnit.SECONDS);
                int code = response.isSuccess() ? 200 : response.getCode();
                codeTally.merge(code, 1, Integer::sum);
                assertTrue(code == 200 || code == 8400 || code == 8429
                                || code == 8430 || code == 8431
                                || code == 8510 || code == 8511 || code == 8515,
                        "unexpected terminal code " + code + ": " + response.getErrorMessage());
            }

            // 排空：引擎全部归零后再过泄漏金丝雀
            for (JavaMockEngineCluster.FastRpcService svc : h.services.values()) {
                AutoTpmE2EHarness.await(() -> svc.getRunningCount() == 0, 15_000,
                        "engine " + svc.getGrpcPort() + " must drain to zero running");
            }
            Thread.sleep(2_500);
            for (JavaMockEngineCluster.FastRpcService svc : h.services.values()) {
                svc.checkLeakDrain(TimeUnit.SECONDS.toNanos(2));
                assertFalse(svc.isLeakDetected(),
                        "LEAK on engine " + svc.getGrpcPort());
            }

            // Successful engine work settles through finished reports. Rejected
            // work has no Decode terminal and expires after its observation window;
            // the final rejection burst ended over 19 seconds before this check.
            assertEquals(0, h.decodeEndpoint(0).getInflightCount(),
                    "decode shadow inflight must settle to zero");
            assertEquals(0L, h.decodeEndpoint(0).routingView().inflightHardKv(),
                    "no orphaned hard-KV reservation");
            assertEquals(0, h.decodeEndpoint(0).resourceSnapshot().acceptedCount());
            assertEquals(0, h.prefillEndpoint(0).queuedRequestCount());
            assertEquals(0, h.prefillEndpoint(1).queuedRequestCount());

            long accepted = h.services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getAcceptedCount).sum();
            long completed = h.services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount).sum();
            System.out.printf(
                    "[task35-D] %d requests over %.1fs, terminal codes=%s, "
                            + "engines accepted=%d completed=%d, zero leaks%n",
                    futures.size(), trafficElapsed / 1000.0, codeTally, accepted, completed);
            assertTrue(codeTally.getOrDefault(200, 0) > 0, "the run must contain successes");
            assertTrue(codeTally.getOrDefault(8510, 0) > 0,
                    "the rejection burst must surface as 8510 terminals");
        }
    }
}

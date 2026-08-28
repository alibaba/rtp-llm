package org.flexlb.mockengine;

import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Task35 场景 B：饱和集群下高优先级提前执行的量化验证。
 *
 * <p>P70/P50/P30 各 50 个请求轮转提交到单 prefill 集群；Auto-TPM 的优先级
 * 批队列（priority desc → arrival asc FIFO）应使高优先级请求显著更早到达引擎。
 * 双重断言：平均引擎到达位次 + 平均调度延迟（submit → 引擎 enqueue 到达）
 * 均满足 P70 < P50 < P30，并输出量化数值；全部请求必须到达成功终态。
 *
 * <p>时序设计（hold-then-flip）：先用大批次上限 + 长 fixedWait 停住派发，
 * 把 150 个请求全部无冲突地压进优先级队列（消除提交/派发并发导致的
 * queueVersion 冲突 8515），确认队列饱和后一次性翻小批次参数放行 ——
 * fixed_window 派发线程每 ~1ms 重读 config，翻转即时生效。
 */
class PriorityLatencyE2ETest {

    private static final int BASE_PORT = 62800;
    private static final int PER_PRIORITY = 50;
    private static final int[] PRIORITIES = {30, 50, 70};

    @Test
    @Timeout(90)
    void b_high_priority_dispatches_earlier_under_saturation() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "5", 1.0, false)) {
            // hold：批次上限大于总量 + 长 fixedWait → 零派发，队列稳定吸收提交
            h.fixedWindowDecision().setMaxRequests(200);
            h.fixedWindowDecision().setMaxCollectionWaitMs(10_000);
            h.config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(1024);
            // This case measures Prefill priority ordering, not Decode KV
            // admission. Keep the independent Decode expected-KV gate out of
            // the fixture so every request reaches the queue under test.
            h.config.getRouter().getRoles().getDecode().getAvailability()
                    .setMaxKvUsagePercent(0);

            Map<Long, Long> submitNanos = new HashMap<>();
            Map<Long, Integer> priorityByRid = new HashMap<>();
            List<CompletableFuture<Response>> futures = new ArrayList<>();
            long rid = 1000;
            for (int i = 0; i < PER_PRIORITY; i++) {
                // 轮转顺序 30→50→70，天然对 P70 不利（每轮最后提交），
                // 断言仍成立说明提前量来自优先级而非提交顺序
                for (int priority : PRIORITIES) {
                    long requestId = rid++;
                    submitNanos.put(requestId, System.nanoTime());
                    priorityByRid.put(requestId, priority);
                    futures.add(h.scheduler.submit(h.context(requestId, priority)));
                }
            }
            int total = PER_PRIORITY * PRIORITIES.length;
            assertEquals(total, futures.size());
            assertEquals(total, h.prefillEndpoint(0).queuedRequestCount(),
                    "all requests must be committed into the priority queue before release");

            // flip：小批次 + 短 fixedWait 放行派发，持续饱和下由优先级序主导
            h.fixedWindowDecision().setMaxRequests(2);
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.startAutoPump(10);

            AutoTpmE2EHarness.await(
                    () -> futures.stream().allMatch(CompletableFuture::isDone), 60_000,
                    "all " + total + " requests must reach a terminal state");

            for (CompletableFuture<Response> future : futures) {
                Response response = future.get(1, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(),
                        "no eviction switches on — every request must succeed, got "
                                + response.getCode() + ": " + response.getErrorMessage());
            }
            assertEquals(total, h.engineArrivalOrder.size(),
                    "every request must have reached the engine exactly once");

            // 量化：平均到达位次 + 平均调度延迟（ms）
            Map<Integer, Double> avgRank = new HashMap<>();
            Map<Integer, Double> avgLatencyMs = new HashMap<>();
            for (int priority : PRIORITIES) {
                long rankSum = 0;
                long latencySum = 0;
                int count = 0;
                for (int index = 0; index < h.engineArrivalOrder.size(); index++) {
                    long requestId = h.engineArrivalOrder.get(index);
                    if (priorityByRid.get(requestId) == priority) {
                        rankSum += index;
                        latencySum += h.engineArrivalNanos.get(requestId)
                                - submitNanos.get(requestId);
                        count++;
                    }
                }
                assertEquals(PER_PRIORITY, count);
                avgRank.put(priority, rankSum / (double) count);
                avgLatencyMs.put(priority, latencySum / (double) count / 1_000_000.0);
            }
            System.out.printf(
                    "[task35-B] avg schedule latency ms: P70=%.2f P50=%.2f P30=%.2f | "
                            + "avg arrival rank: P70=%.1f P50=%.1f P30=%.1f (n=%d each)%n",
                    avgLatencyMs.get(70), avgLatencyMs.get(50), avgLatencyMs.get(30),
                    avgRank.get(70), avgRank.get(50), avgRank.get(30), PER_PRIORITY);

            assertTrue(avgLatencyMs.get(70) < avgLatencyMs.get(50),
                    "P70 avg latency must beat P50: " + avgLatencyMs);
            assertTrue(avgLatencyMs.get(50) < avgLatencyMs.get(30),
                    "P50 avg latency must beat P30: " + avgLatencyMs);
            assertTrue(avgRank.get(70) < avgRank.get(50),
                    "P70 avg arrival rank must beat P50: " + avgRank);
            assertTrue(avgRank.get(50) < avgRank.get(30),
                    "P50 avg arrival rank must beat P30: " + avgRank);
        }
    }
}

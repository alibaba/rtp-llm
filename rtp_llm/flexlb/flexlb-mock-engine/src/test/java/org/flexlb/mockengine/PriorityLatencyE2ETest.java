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
 * Task35 场景 B：饱和集群下队列派发顺序的量化验证。
 *
 * <p>P70/P50/P30 各 50 个请求轮转提交到单 prefill 集群；Auto-TPM 的全局队列
 * 按到达顺序派发（FIFO），不因优先级让后到的高优先级请求插到先到的请求前面。
 * 队列持续饱和（提交远快于派发）时，各优先级的平均引擎到达位次必须接近相等；
 * 多个小批次并发派发、异步 RPC 完成带来的到达顺序抖动在个位数位次以内，
 * 不构成系统性的优先级提前量。优先级只在引擎子队列共享槽位占满、需要挑选
 * 驱逐对象时才参与比较，由 {@link PreemptionPhasesE2ETest} 覆盖。
 *
 * <p>时序设计（hold-then-flip）：先用大批次上限 + 长 fixedWait 停住派发，
 * 把 150 个请求全部无冲突地压进队列（消除提交/派发并发导致的
 * queueVersion 冲突 8515），确认队列饱和后一次性翻小批次参数放行 ——
 * fixed_window 派发线程每 ~1ms 重读 config，翻转即时生效。
 */
class PriorityLatencyE2ETest {

    private static final int BASE_PORT = 62800;
    private static final int PER_PRIORITY = 50;
    private static final int[] PRIORITIES = {30, 50, 70};
    /**
     * Concurrent small-batch dispatch and async RPC completion jitter the
     * measured arrival rank by a few positions even under strict FIFO
     * (observed consistently around 1-2 positions across repeated runs);
     * this bound stays well clear of that noise while still catching a
     * systematic priority-driven jump of the kind this test used to require.
     */
    private static final double MAX_AVG_RANK_SPREAD = 15.0;

    @Test
    @Timeout(90)
    void b_queueDispatchFollowsEnqueueOrderRegardlessOfPriorityUnderSaturation() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "5", 1.0, false)) {
            // hold：批次上限大于总量 + 长 fixedWait → 零派发，队列稳定吸收提交
            h.fixedWindowDecision().setMaxRequests(200);
            h.fixedWindowDecision().setMaxCollectionWaitMs(10_000);
            // This case measures Prefill dispatch ordering, not Decode KV
            // admission. Use the large fixture KV pool without a percentage
            // reserve so every request reaches the queue under test.
            h.config.getRouter().getRoles().getDecode().getAvailability()
                    .setMaxKvUsagePercent(100);

            Map<Long, Integer> priorityByRid = new HashMap<>();
            List<CompletableFuture<Response>> futures = new ArrayList<>();
            long rid = 1000;
            for (int i = 0; i < PER_PRIORITY; i++) {
                // 轮转顺序 30→50→70，刻意让高优先级总是每轮最后提交，
                // 断言不随优先级倾斜说明入队顺序而非提交巧合决定了到达顺序
                for (int priority : PRIORITIES) {
                    long requestId = rid++;
                    priorityByRid.put(requestId, priority);
                    futures.add(h.scheduler.submit(h.context(String.valueOf(requestId), priority)));
                }
            }
            int total = PER_PRIORITY * PRIORITIES.length;
            assertEquals(total, futures.size());
            AutoTpmE2EHarness.await(
                    () -> h.scheduler.getQueuedRequestCount() == total,
                    10_000,
                    "all requests must be committed into the scheduler queue");
            assertEquals(total, h.scheduler.getQueuedRequestCount(),
                    "all requests must be committed into the global queue before release");

            // flip：小批次 + 短 fixedWait 放行派发，持续饱和下由入队顺序主导
            h.fixedWindowDecision().setMaxRequests(2);
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.startAutoPump(10);

            AutoTpmE2EHarness.await(
                    () -> {
                        h.assertAutoPumpHealthy();
                        return futures.stream().allMatch(CompletableFuture::isDone);
                    }, 60_000,
                    "all " + total + " requests must reach a terminal state");

            for (CompletableFuture<Response> future : futures) {
                Response response = future.get(1, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(),
                        "no eviction switches on — every request must succeed, got "
                                + response.getCode() + ": " + response.getErrorMessage()
                                + ", arrivals=" + h.engineArrivalOrder.size()
                                + ", queued=" + h.scheduler.getQueuedRequestCount()
                                + ", prefillWork=" + h.prefillEndpoint(0).captureRouteProjectionInputs().work()
                                + ", decode=" + h.decodeEndpoint(0).resourceSnapshot());
            }
            assertEquals(total, h.engineArrivalOrder.size(),
                    "every request must have reached the engine exactly once");

            // 量化：平均到达位次（engineArrivalOrder 中的下标）
            Map<Integer, Double> avgRank = new HashMap<>();
            for (int priority : PRIORITIES) {
                long rankSum = 0;
                int count = 0;
                for (int index = 0; index < h.engineArrivalOrder.size(); index++) {
                    long requestId = h.engineArrivalOrder.get(index);
                    if (priorityByRid.get(requestId) == priority) {
                        rankSum += index;
                        count++;
                    }
                }
                assertEquals(PER_PRIORITY, count);
                avgRank.put(priority, rankSum / (double) count);
            }
            System.out.printf(
                    "[task35-B] avg arrival rank: P70=%.1f P50=%.1f P30=%.1f (n=%d each)%n",
                    avgRank.get(70), avgRank.get(50), avgRank.get(30), PER_PRIORITY);

            double spread = avgRank.values().stream().mapToDouble(Double::doubleValue).max().orElseThrow()
                    - avgRank.values().stream().mapToDouble(Double::doubleValue).min().orElseThrow();
            assertTrue(spread < MAX_AVG_RANK_SPREAD,
                    "no priority should gain a systematic arrival-rank advantage once the queue "
                            + "is never actually full: avgRank=" + avgRank + ", spread=" + spread);
        }
    }
}

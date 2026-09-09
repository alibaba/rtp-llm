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
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/**
 * Task35 场景 E：FIFO 基线，priority 不影响排队次序，且没有抢占/victim。
 * 与场景 B 同流量（P70/P50/P30 各 50，轮转提交）。
 *
 * <p>每个 Prefill 仅允许一个在途批次，以引擎到达顺序验证 FIFO 排队顺序。
 */
class BaselineParityE2ETest {

    private static final int BASE_PORT = 63100;
    private static final int PER_PRIORITY = 50;
    private static final int[] PRIORITIES = {30, 50, 70};

    @Test
    @Timeout(90)
    void e_switches_off_priority_has_no_effect_and_dispatch_is_fifo() throws Exception {
        // autoTpm=false：构造时选择 FIFO 排队，不启用抢占。
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "5", 1.0, false, false)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.getDispatcher().setMaxInflightPerPrefillWorker(1);
            h.startAutoPump(10);

            // 预热请求不计入顺序和延迟断言。
            h.scheduler.submit(h.context(1999, 50)).get(10, TimeUnit.SECONDS);
            AutoTpmE2EHarness.await(() -> !h.engineArrivalOrder.isEmpty(), 5_000,
                    "warm-up request must reach the engine");
            h.engineArrivalOrder.clear();
            h.engineArrivalNanos.clear();

            List<Long> submissionOrder = new ArrayList<>();
            Map<Long, Integer> priorityByRid = new HashMap<>();
            Map<Long, Long> submitNanos = new HashMap<>();
            List<CompletableFuture<Response>> futures = new ArrayList<>();
            long rid = 2000;
            long lastSubmitMs = 0;
            for (int i = 0; i < PER_PRIORITY; i++) {
                for (int priority : PRIORITIES) {
                    // 各优先级按相同的 2ms 间隔轮转提交。
                    while (System.currentTimeMillis() - lastSubmitMs < 2) {
                        Thread.onSpinWait();
                    }
                    lastSubmitMs = System.currentTimeMillis();
                    long requestId = rid++;
                    submissionOrder.add(requestId);
                    priorityByRid.put(requestId, priority);
                    submitNanos.put(requestId, System.nanoTime());
                    futures.add(h.scheduler.submit(h.context(requestId, priority)));
                }
            }
            int total = PER_PRIORITY * PRIORITIES.length;

            AutoTpmE2EHarness.await(
                    () -> futures.stream().allMatch(CompletableFuture::isDone), 60_000,
                    "all " + total + " baseline requests must reach a terminal state");
            for (CompletableFuture<Response> future : futures) {
                Response response = future.get(1, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(),
                        "every FIFO baseline request must succeed, got "
                                + response.getCode() + ": " + response.getErrorMessage());
            }

            // 单批次并发下，引擎到达顺序与 FIFO 提交顺序逐位相同。
            assertEquals(submissionOrder, new ArrayList<>(h.engineArrivalOrder),
                    "with one in-flight batch, engine arrivals must preserve FIFO regardless of priority");

            // 无任何抢占痕迹
            verify(h.requestReporter, never()).reportVictim(anyInt(), anyInt(),
                    anyString(), anyString());
            verify(h.requestReporter, never()).reportPriorityPreempt(anyString());

            // 对照数据：三档平均调度延迟应该同量级（仅输出，不做脆断言）
            Map<Integer, Double> avgLatencyMs = new HashMap<>();
            for (int priority : PRIORITIES) {
                long latencySum = 0;
                int count = 0;
                for (long requestId : submissionOrder) {
                    if (priorityByRid.get(requestId) == priority) {
                        latencySum += h.engineArrivalNanos.get(requestId)
                                - submitNanos.get(requestId);
                        count++;
                    }
                }
                avgLatencyMs.put(priority, latencySum / (double) count / 1_000_000.0);
            }
            System.out.printf(
                    "[task35-E] baseline avg schedule latency ms (FIFO, priority ignored): "
                            + "P70=%.2f P50=%.2f P30=%.2f (n=%d each)%n",
                    avgLatencyMs.get(70), avgLatencyMs.get(50), avgLatencyMs.get(30),
                    PER_PRIORITY);
        }
    }
}

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
 * Task35 场景 E：基线对照 —— Auto-TPM 开关全部关闭（默认值）时行为与旧逻辑
 * 完全一致：priority 对派发次序无任何影响（严格 FIFO），无任何抢占/victim，
 * 与场景 B 同流量（P70/P50/P30 各 50，轮转提交）。
 *
 * <p>注意：旧逻辑 fixed_window 的排序键是毫秒级 enqueuedAtMs，同毫秒并列时
 * 队列不保证稳定序（这是旧逻辑真实行为，非缺陷）。为使严格 FIFO 断言
 * 良定义，提交时保证每个请求拿到唯一的 enqueuedAtMs（间隔 ≥2ms）。
 */
class BaselineParityE2ETest {

    private static final int BASE_PORT = 63100;
    private static final int PER_PRIORITY = 50;
    private static final int[] PRIORITIES = {30, 50, 70};

    @Test
    @Timeout(90)
    void e_switches_off_priority_has_no_effect_and_dispatch_is_fifo() throws Exception {
        // autoTpm=false：批队列用 LEGACY 序（构造时冻结），全部开关保持默认关闭
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "5", 1.0, false, false)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(1024);
            h.startAutoPump(10);

            // 预热：首笔请求走冷 gRPC 通道 + JIT，异步发送可能被后续批次超越
            // （传输层竞态，非被测行为）。先发一笔不计入断言的请求压热链路。
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
                    // 唯一 enqueuedAtMs：等待时钟前进 ≥2ms 再提交
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
                        "baseline must behave like legacy — every request succeeds, got "
                                + response.getCode() + ": " + response.getErrorMessage());
            }

            // 旧逻辑 = 严格 FIFO：引擎到达顺序与提交顺序逐位相同，priority 无影响
            assertEquals(submissionOrder, new ArrayList<>(h.engineArrivalOrder),
                    "with all switches off the dispatch order must be exactly FIFO");

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

package org.flexlb.mockengine;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchRequestPB;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/**
 * Task35 场景 E：基线对照 —— Auto-TPM 开关全部关闭（默认值）时行为与旧逻辑
 * 完全一致：priority 对派发次序无任何影响（严格 FIFO），无任何抢占/victim，
 * 与场景 B 同流量（P70/P50/P30 各 50，轮转提交）。
 *
 * <p>FIFO 使用唯一递增的 enqueueSeq，同毫秒提交也保持顺序。这里验证单个
 * Prefill 的批次决策顺序；异步派发线程进入引擎调用的先后不属于 FIFO 保证。
 */
class BaselineParityE2ETest {

    private static final int BASE_PORT = 63100;
    private static final int PER_PRIORITY = 50;
    private static final int[] PRIORITIES = {30, 50, 70};

    @Test
    @Timeout(90)
    void e_switches_off_priority_has_no_effect_and_dispatch_is_fifo() throws Exception {
        // autoTpm=false：批队列用 FIFO 序（构造时冻结），全部开关保持默认关闭
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "5", 1.0, false, false)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.startAutoPump(10);

            // 预热首笔调度与引擎调用，并从测量记录中排除。
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
            for (int i = 0; i < PER_PRIORITY; i++) {
                for (int priority : PRIORITIES) {
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

            List<Long> arrivals = new ArrayList<>(h.engineArrivalOrder);
            assertEquals(total, arrivals.size(), "every measured request must reach the engine exactly once");
            assertEquals(new HashSet<>(submissionOrder), new HashSet<>(arrivals),
                    "engine arrivals must contain every measured request and no unexpected request");

            // RequestSchedulerTestRuntime supplies incrementing batch IDs. This
            // fixture has one Prefill and one WorkerBatcher, so BatchDeliveryStrategy
            // assigns them in decision order before committing and submitting to
            // the asynchronous dispatcher. Preserve request order within each batch;
            // ordering RPC batches by ID removes only dispatch-thread arrival races.
            ArgumentCaptor<EnqueueBatchRequestPB> batches = ArgumentCaptor.forClass(EnqueueBatchRequestPB.class);
            verify(h.grpcClient, atLeastOnce()).batchEnqueueAsync(anyString(), anyInt(), batches.capture());
            List<Long> decisionOrder = batches.getAllValues().stream()
                    .sorted(Comparator.comparingLong(EnqueueBatchRequestPB::getBatchId))
                    .flatMap(batch -> batch.getDpSlotsList().stream())
                    .flatMap(slot -> slot.getRequestsList().stream())
                    .map(request -> request.getInput().getRequestId())
                    .filter(priorityByRid::containsKey)
                    .toList();
            assertEquals(submissionOrder, decisionOrder,
                    "with all switches off batch decisions and their request order must be exactly FIFO");

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

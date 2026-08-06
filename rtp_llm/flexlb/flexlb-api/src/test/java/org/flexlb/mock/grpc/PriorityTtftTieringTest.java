package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.StabilityMonitor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage 4 E2E — the priority-tiering TTFT benefit is observable: with the
 * yield queue enabled, later-arriving P70 requests are dispatched ahead of
 * an earlier-arriving P30 request, so their queue wait (submit → future
 * completion, the mock TTFT proxy) is strictly lower. The P30 request is
 * NOT starved — the deadline stays relaxed, so it is eventually served
 * successfully once the high tier drains (README §4: 低优最终完成).
 *
 * <p>Gates asserted:
 * <ul>
 *   <li>dispatch order: every P70 enqueue reaches the engine before the
 *       P30 enqueue, although the P30 was submitted first</li>
 *   <li>TTFT tiering: max TTFT over the P70 population &lt; TTFT of the
 *       P30 request</li>
 *   <li>no request fails: tiering only re-orders, it never rejects</li>
 *   <li>all accounting layers drain to zero</li>
 * </ul>
 *
 * <p>Determinism: a primer holds the single backpressure slot so the whole
 * population queues up while parked; releases are driven one finished
 * report at a time, so dispatch order equals queue (priority) order.
 */
class PriorityTtftTieringTest extends FlexLBMockTestBase {

    private static final int LOW_PRIORITY = 30;
    private static final int HIGH_PRIORITY = 70;

    private static final long PRIMER_ID = 9500;
    private static final long LOW_ID = 9301;
    private static final long[] HIGH_IDS = {9701, 9702, 9703};

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        // Single backpressure slot: the primer parks dispatch so the whole
        // population queues before any tiering-relevant dispatch happens.
        cfg.setFlexlbBatchFixedMaxInflightBatches(1);
        cfg.setFlexlbBatchEnqueueDeadlineMs(60_000L);
        return cfg;
    }

    @Test
    void laterHighPriority_isServedFirst_ttftTieringObservable_lowStillServed() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        Map<Long, Long> submittedAt = new ConcurrentHashMap<>();
        Map<Long, Long> completedAt = new ConcurrentHashMap<>();

        // Primer takes the only backpressure slot.
        CompletableFuture<Response> primer =
                monitor.track(submitTimed(PRIMER_ID, HIGH_PRIORITY, submittedAt, completedAt));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "primer must reach the engine before the scenario starts");

        // The low-priority request arrives FIRST...
        CompletableFuture<Response> low =
                monitor.track(submitTimed(LOW_ID, LOW_PRIORITY, submittedAt, completedAt));
        Thread.sleep(200);

        // ...the high-priority population arrives strictly later.
        List<CompletableFuture<Response>> highs = new ArrayList<>();
        for (long id : HIGH_IDS) {
            highs.add(monitor.track(submitTimed(id, HIGH_PRIORITY, submittedAt, completedAt)));
        }

        // Everything queued while parked; release one slot per pump so the
        // dispatch order is exactly the priority-sorted queue order.
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "population must be fully queued before the drain starts");
        long deadline = System.currentTimeMillis() + 15_000;
        while (System.currentTimeMillis() < deadline
                && !(low.isDone() && highs.stream().allMatch(CompletableFuture::isDone))) {
            simulatePrefillFinishedReport();
            Thread.sleep(20);
        }

        // Gate: tiering never rejects — every request completed successfully.
        assertTrue(primer.get(1, TimeUnit.SECONDS).isSuccess(), "primer must succeed");
        assertTrue(low.get(1, TimeUnit.SECONDS).isSuccess(),
                "low-priority request must eventually be SERVED (not starved, not rejected)");
        for (CompletableFuture<Response> high : highs) {
            assertTrue(high.get(1, TimeUnit.SECONDS).isSuccess(), "high tier must succeed");
        }

        // Gate: dispatch order — every later-arriving P70 hits the engine
        // before the earlier-arriving P30.
        List<Long> arrivalOrder = enqueueArrivalOrder();
        int lowIndex = arrivalOrder.indexOf(LOW_ID);
        assertTrue(lowIndex >= 0, "low request must be dispatched eventually");
        for (long id : HIGH_IDS) {
            int highIndex = arrivalOrder.indexOf(id);
            assertTrue(highIndex >= 0 && highIndex < lowIndex,
                    "P70 " + id + " must be dispatched before P30 " + LOW_ID
                            + ", arrival order: " + arrivalOrder);
        }

        // Gate: TTFT tiering benefit is observable — the slowest P70 still
        // beats the P30 (both waited in the same parked window).
        long lowTtft = completedAt.get(LOW_ID) - submittedAt.get(LOW_ID);
        for (long id : HIGH_IDS) {
            long highTtft = completedAt.get(id) - submittedAt.get(id);
            assertTrue(highTtft < lowTtft,
                    "TTFT tiering: P70 " + id + " ttft=" + highTtft
                            + "ms must beat P30 ttft=" + lowTtft + "ms");
        }

        // Gate: zero leak across every accounting layer.
        monitor.assertQuiescent(5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitTimed(long requestId, int priority,
                                                    Map<Long, Long> submittedAt,
                                                    Map<Long, Long> completedAt) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        submittedAt.put(requestId, System.currentTimeMillis());
        CompletableFuture<Response> future = scheduler.submit(ctx);
        future.whenComplete((r, t) -> completedAt.put(requestId, System.currentTimeMillis()));
        return future;
    }

    /** Flattened requestId arrival order across all EnqueueBatch calls. */
    private List<Long> enqueueArrivalOrder() {
        List<Long> order = new ArrayList<>();
        for (EngineRpcService.EnqueueBatchRequestPB batch
                : mockPrefillWorker.getRpcService().getEnqueuedRequests()) {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    order.add(ext.getInput().getRequestId());
                }
            }
        }
        return order;
    }

    private static void awaitTrue(java.util.function.BooleanSupplier condition,
                                  long timeoutMs, String message) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(20);
        }
        assertTrue(condition.getAsBoolean(), message);
    }
}

package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.StabilityMonitor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage 4 E2E — yield-without-starvation over the mock-worker harness with
 * {@code autoTpmEnabled + autoTpmQueueYieldEnabled} on.
 *
 * <p>Scenario: the engine is saturated (backpressure holds dispatch parked),
 * a low-priority (P30) request is queued, and sustained high-priority (P70)
 * traffic keeps strictly higher-priority items ahead of it. The
 * priority-sorted queue means the low item never reaches the queue head, so
 * only the yielded-deadline eviction can clear it. Gates asserted:
 *
 * <ul>
 *   <li>the yielded low-priority future MUST complete — no permanent
 *       starvation — settled as {@code NO_AVAILABLE_WORKER} (8400) with the
 *       yield attribution message (P0 contract: queue-layer victims are
 *       always 8400, never 429x)</li>
 *   <li>the low request is never dispatched to the engine and no Engine
 *       Cancel is ever issued for a queue yield (铁律: 让位绝不调 cancel)</li>
 *   <li>high-priority traffic is unaffected: every high future completes
 *       successfully once capacity frees up</li>
 *   <li>every accounting layer drains to zero (StabilityMonitor)</li>
 * </ul>
 *
 * <p>Determinism: the queue deadline starts effectively infinite and is
 * lowered mid-test to a value strictly between the low item's elapsed wait
 * and the high items' elapsed wait (guard-asserted 400ms+ separation), so
 * exactly the low item expires — no wall-clock race with the head items.
 * The config object is read live by the batcher on every decide cycle.
 */
class YieldNoStarvationTest extends FlexLBMockTestBase {

    private static final long RELAXED_DEADLINE_MS = 60_000L;
    private static final int LOW_PRIORITY = 30;
    private static final int HIGH_PRIORITY = 70;

    private static final long PRIMER_ID = 9100;
    private static final long LOW_ID = 9300;
    private static final long[] HIGH_IDS = {9101, 9102, 9103};

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        // One batch in flight at a time: the primer request saturates the
        // engine and parks dispatch until the finished report is pumped.
        cfg.setFlexlbBatchFixedMaxInflightBatches(1);
        cfg.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        return cfg;
    }

    @Test
    void sustainedHighTraffic_yieldedLowIsClearedBy8400_neverStarves() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // Primer saturates the engine: dispatched immediately, its batch
        // entry holds the backpressure gate until the finished report.
        CompletableFuture<Response> primer = monitor.track(submitWithPriority(PRIMER_ID, HIGH_PRIORITY));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "primer must reach the engine before the scenario starts");

        // Low-priority request enters the parked queue first...
        long lowEnqueuedAt = System.currentTimeMillis();
        CompletableFuture<Response> low = monitor.track(submitWithPriority(LOW_ID, LOW_PRIORITY));

        // ...then sustained high-priority traffic sorts ahead of it.
        Thread.sleep(1_000);
        long highEnqueuedAt = System.currentTimeMillis();
        List<CompletableFuture<Response>> highs = new ArrayList<>();
        for (long id : HIGH_IDS) {
            highs.add(monitor.track(submitWithPriority(id, HIGH_PRIORITY)));
        }
        Thread.sleep(1_000);

        // Everything is still parked: nothing beyond the primer dispatched,
        // the low item is held in queue (yielded, not terminal).
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "backpressure must keep the queue parked during the pressure phase");
        assertFalse(low.isDone(), "yielded low item must stay queued, not terminal");

        // Lower the deadline to a midpoint that only the low item exceeds.
        long now = System.currentTimeMillis();
        long lowElapsed = now - lowEnqueuedAt;
        long highElapsed = now - highEnqueuedAt;
        assertTrue(lowElapsed - highElapsed >= 400,
                "timing guard: low/high elapsed separation collapsed (machine stall?)");
        config.setFlexlbBatchEnqueueDeadlineMs((lowElapsed + highElapsed) / 2);

        // Gate 1: the yielded low item is cleared — future completes with
        // 8400 and the yield attribution; the highs stay queued.
        Response lowResponse = low.get(3, TimeUnit.SECONDS);
        assertFalse(lowResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), lowResponse.getCode(),
                "queue-layer yield victim must be 8400, never 429x");
        assertNotNull(lowResponse.getErrorMessage());
        assertTrue(lowResponse.getErrorMessage().contains("auto_tpm: yielded for priority=" + HIGH_PRIORITY),
                "8400 message must carry the yield reason, got: " + lowResponse.getErrorMessage());

        // Gate 2: queue yield never touches the engine — the low request was
        // never enqueued and no Cancel RPC was issued anywhere.
        assertFalse(enqueuedRequestIds().contains(LOW_ID),
                "yielded low request must never reach the engine");
        assertEquals(0, mockPrefillWorker.getCancelCount(), "queue yield must never call Engine cancel");
        assertEquals(0, mockDecodeWorker.getCancelCount(), "queue yield must never call Engine cancel");

        // Gate 3: relax the deadline again and release the engine — every
        // high-priority request is served (yield had no effect on them).
        config.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        assertTrue(primer.get(5, TimeUnit.SECONDS).isSuccess(), "primer must succeed");
        for (CompletableFuture<Response> high : highs) {
            // Quiescence pumping below drains the backpressure; poll here with
            // an explicit pump so each high's dispatch is driven forward.
            Response highResponse = pumpUntilDone(high, 10_000);
            assertTrue(highResponse.isSuccess(), "high-priority traffic must be unaffected by yield");
        }
        Set<Long> served = enqueuedRequestIds();
        for (long id : HIGH_IDS) {
            assertTrue(served.contains(id), "high request " + id + " must be dispatched");
        }

        // Gate 4: zero leak across every accounting layer.
        monitor.assertQuiescent(5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        return scheduler.submit(ctx);
    }

    private Set<Long> enqueuedRequestIds() {
        Set<Long> ids = new HashSet<>();
        for (EngineRpcService.EnqueueBatchRequestPB batch
                : mockPrefillWorker.getRpcService().getEnqueuedRequests()) {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    ids.add(ext.getInput().getRequestId());
                }
            }
        }
        return ids;
    }

    /** Pump finished reports until the future settles (drives backpressure release). */
    private Response pumpUntilDone(CompletableFuture<Response> future, long timeoutMs) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (!future.isDone() && System.currentTimeMillis() < deadline) {
            simulatePrefillFinishedReport();
            Thread.sleep(20);
        }
        return future.get(1, TimeUnit.SECONDS);
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

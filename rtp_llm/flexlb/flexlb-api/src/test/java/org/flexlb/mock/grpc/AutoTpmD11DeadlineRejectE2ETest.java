package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * D11 (task40, scope widened per owner sign-off) guard — SLO-deadline
 * admission rejection at the scheduling entry point:
 *
 * <ul>
 *   <li>AUTO_TPM on: ANY request — prioritized or the 0 sentinel — whose
 *       deadline ({@code startTime + resolveSloMs(seqLen)}) is already ≤ now
 *       is rejected 8400 at {@code BatchScheduler.submit} — before
 *       InflightStore registration, before routing, and without ever reaching
 *       the engine; the reason message carries "slo deadline exceeded" with
 *       the deadline/now values</li>
 *   <li>AUTO_TPM off: the same expired request keeps the pre-D11 baseline
 *       behavior (scheduled and completed normally) — off-state parity, for
 *       both prioritized and no-priority requests</li>
 * </ul>
 */
class AutoTpmD11DeadlineRejectE2ETest extends FlexLBMockTestBase {

    private static final int P50 = 50;
    /** Comfortably past the base config's costSloMs=50_000. */
    private static final long EXPIRED_BY_MS = 60_000L;

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        return cfg;
    }

    // ---- on: expired deadline is rejected 8400 at the entry point ----

    @Test
    void expiredDeadline_switchOn_rejected8400AtEntry_neverReachesEngine() throws Exception {
        CompletableFuture<Response> future = submitExpired(8201, P50);

        // Rejection happens synchronously at the admission gate.
        assertTrue(future.isDone(), "D11 rejection must settle at submit time");
        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(),
                "expired-deadline admission rejection must carry 8400");
        assertTrue(response.getErrorMessage().contains("slo deadline exceeded"),
                "reason message must name the slo deadline: " + response.getErrorMessage());
        assertTrue(response.getErrorMessage().contains("deadline_ms=")
                        && response.getErrorMessage().contains("now_ms="),
                "reason message must carry deadline/now values: " + response.getErrorMessage());

        // Rejected before registration and before dispatch.
        assertEquals(0, inflightStore.activeCount(),
                "the rejected request must never enter the inflight store");
        Thread.sleep(200);
        assertEquals(0, mockPrefillWorker.getEnqueueCount(),
                "the rejected request must never reach the engine");
    }

    // ---- off: parity — the same expired request is scheduled normally ----

    @Test
    void expiredDeadline_switchOff_keepsBaselineBehavior() throws Exception {
        config.setAutoTpmEnabled(false);

        CompletableFuture<Response> future = submitExpired(8202, P50);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess(),
                "with AUTO_TPM off an expired deadline must not be rejected (parity)");
        assertEquals(1, mockPrefillWorker.getEnqueueCount());

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- widened scope: the 0 sentinel is deadline-rejected too ----

    @Test
    void expiredDeadline_noPrioritySentinel_switchOn_rejected8400AtEntry() throws Exception {
        BalanceContext ctx = createBalanceContext(8203);
        ctx.setStartTime(System.currentTimeMillis() - EXPIRED_BY_MS);

        CompletableFuture<Response> future = scheduler.submit(ctx);
        assertTrue(future.isDone(), "D11 rejection must settle at submit time");
        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode(),
                "the expired 0 sentinel must be rejected 8400 at the entry too");
        assertTrue(response.getErrorMessage().contains("slo deadline exceeded"),
                "reason message must name the slo deadline: " + response.getErrorMessage());

        // Rejected before registration and before dispatch.
        assertEquals(0, inflightStore.activeCount(),
                "the rejected sentinel must never enter the inflight store");
        Thread.sleep(200);
        assertEquals(0, mockPrefillWorker.getEnqueueCount(),
                "the rejected sentinel must never reach the engine");
    }

    @Test
    void expiredDeadline_noPrioritySentinel_switchOff_keepsBaselineBehavior() throws Exception {
        config.setAutoTpmEnabled(false);

        BalanceContext ctx = createBalanceContext(8204);
        ctx.setStartTime(System.currentTimeMillis() - EXPIRED_BY_MS);

        CompletableFuture<Response> future = scheduler.submit(ctx);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess(),
                "with AUTO_TPM off an expired 0 sentinel must not be rejected (parity)");
        assertEquals(1, mockPrefillWorker.getEnqueueCount());

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    /** Submit a prioritized request whose SLO deadline already passed. */
    private CompletableFuture<Response> submitExpired(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        ctx.getRequest().setPriority(priority);
        ctx.setStartTime(System.currentTimeMillis() - EXPIRED_BY_MS);
        return scheduler.submit(ctx);
    }
}

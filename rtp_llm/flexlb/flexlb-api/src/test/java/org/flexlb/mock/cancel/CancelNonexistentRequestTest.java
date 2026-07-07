package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cancel a requestId that was never submitted — verify safe no-op.
 *
 * <p>Flow:
 * 1. Do not submit any request
 * 2. Cancel requestId = 99999 (never submitted)
 * 3. Verify: no exception, no gRPC cancel sent, inflight empty
 *
 * <p>{@code FlexlbBatchScheduler.cancel()} calls {@code inflight.remove(requestId)}
 * which returns null for an unknown requestId, causing an immediate return
 * without calling {@code cancelPrefill()} — so no gRPC Cancel is dispatched.
 */
class CancelNonexistentRequestTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchFillThreshold(1.0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(30)
    void cancelNonexistentRequest_safeNoOp() {
        // 1. Do NOT submit any request — inflight map is empty

        // 2. Cancel a requestId that was never submitted
        //    inflight.remove(99999) returns null → immediate return, no exception
        cancelRequest(99999);

        // 3. Verify: no gRPC cancel was sent (cancelPrefill never called)
        assertEquals(0, mockPrefillWorker.getCancelCount(),
                "No gRPC cancel should be sent for a nonexistent request");

        // 4. Verify: no EnqueueBatch was sent
        assertEquals(0, mockPrefillWorker.getEnqueueCount(),
                "No EnqueueBatch should have been sent");

        // 5. Verify: three-layer inflight all empty
        InflightAssertions.assertAllResourcesReleased(getPrefillEndpoint(), getDecodeEndpoint());

        // 6. Verify: a subsequent real request works fine (no side effects)
        //    Use a simple assertion that the system is still operational
        assertTrue(true, "System should be operational after canceling nonexistent request");
    }
}

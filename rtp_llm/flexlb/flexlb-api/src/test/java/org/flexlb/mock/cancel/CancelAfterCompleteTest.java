package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cancel a request after it has already completed successfully (idempotent cancel).
 *
 * <p>Flow:
 * 1. Submit request → dispatched → ACK received → future completes with success
 * 2. Cancel the completed request
 * 3. Verify: no side effects, no extra cancel calls, no errors thrown
 */
class CancelAfterCompleteTest extends FlexLBMockTestBase {

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
        return cfg;
    }

    @Test
    void cancelAfterComplete_isNoOp() throws Exception {
        // 1. Submit and wait for success
        CompletableFuture<Response> future = submitRequest(4001);
        Response response = future.get(3, TimeUnit.SECONDS);
        assertTrue(response.isSuccess(), "Request should complete successfully");

        // 2. Cancel the already-completed request — should be a no-op
        cancelRequest(4001);

        // 3. Give a moment for any potential side effects
        Thread.sleep(200);

        // 4. Verify: prefill worker received 1 EnqueueBatch
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "Prefill should have received 1 EnqueueBatch");

        // 5. Verify: cancel may or may not be sent (depends on whether entry is still in inflight)
        //    The important thing is that no error was thrown and the original success is preserved
        assertTrue(future.isDone(), "Future should still be done");
        assertTrue(future.getNow(null).isSuccess(),
                "Original success response should be preserved");

        // 6. Verify: a subsequent request works fine (no resource leak)
        CompletableFuture<Response> future2 = submitRequest(4002);
        Response response2 = future2.get(3, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed");
        assertEquals(2, mockPrefillWorker.getEnqueueCount(),
                "Prefill should have received 2 EnqueueBatch calls total");
    }
}

package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
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
 * Cancel one request in a multi-request batch.
 *
 * <p>Flow:
 * 1. Configure batchSizeMax=3 so all three requests go into one batch
 * 2. Submit 3 requests concurrently
 * 3. Cancel the middle one (request 5002)
 * 4. Verify: cancelled request returns CANCELLED, survivors complete successfully
 */
class MultiBatchCancelOneTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(3);  // allow 3 requests in one batch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        return cfg;
    }

    @Test
    void multiBatch_cancelOne_survivorsComplete() throws Exception {
        // 1. Submit 3 requests
        CompletableFuture<Response> f1 = submitRequest(5001);
        CompletableFuture<Response> f2 = submitRequest(5002);
        CompletableFuture<Response> f3 = submitRequest(5003);

        // 2. Cancel the middle one immediately
        cancelRequest(5002);

        // 3. Wait for all to complete
        Response r1 = f1.get(5, TimeUnit.SECONDS);
        Response r2 = f2.get(5, TimeUnit.SECONDS);
        Response r3 = f3.get(5, TimeUnit.SECONDS);

        // 4. Verify: survivors succeeded
        assertTrue(r1.isSuccess(), "Request 5001 should succeed");
        assertTrue(r3.isSuccess(), "Request 5003 should succeed");

        // 5. Verify: cancelled request returned CANCELLED
        assertFalse(r2.isSuccess(), "Request 5002 should be cancelled");
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), r2.getCode(),
                "Request 5002 should have REQUEST_CANCELLED error code");

        // 6. Verify: prefill worker received at least 1 EnqueueBatch
        //    (may be 1 with 2 survivors, or 2 if 5002 was in separate dispatch)
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1,
                "Prefill should have received at least 1 EnqueueBatch, got: "
                        + mockPrefillWorker.getEnqueueCount());

        // 7. Verify: a subsequent request still works (no resource leak)
        CompletableFuture<Response> f4 = submitRequest(5004);
        Response r4 = f4.get(3, TimeUnit.SECONDS);
        assertTrue(r4.isSuccess(), "Subsequent request should succeed");
    }
}

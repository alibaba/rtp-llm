package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Dispatch failure: EnqueueBatch returns error response.
 *
 * <p>Flow:
 * 1. Configure prefill worker with failOnEnqueue=true
 * 2. Submit request → dispatched → EnqueueBatch returns error
 * 3. Verify: request fails with BATCH_DISPATCH_FAILED, subsequent request recovers
 *
 * <p>Note: failAck() now calls repackPrefillBatch() to remove the failed request
 * from PrefillEndpoint.inflightBatches, preventing inflight leak.
 */
class DispatchFailureTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .failOnEnqueue(true)  // Return error for EnqueueBatch
                .enqueueErrorMessage("mock engine overloaded")
                .enqueueErrorCode(13)
                .build();
    }

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
    void dispatchFailure_requestFailsAndRecovers() throws Exception {
        // 1. Submit request — will fail because EnqueueBatch returns error
        CompletableFuture<Response> future = submitRequest(7001);
        Response response = future.get(5, TimeUnit.SECONDS);

        // 2. Verify: request failed with BATCH_DISPATCH_FAILED
        assertFalse(response.isSuccess(), "Request should fail when EnqueueBatch returns error");
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), response.getCode(),
                "Request should have BATCH_DISPATCH_FAILED error code");

        // 3. Verify: prefill worker received the EnqueueBatch
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "Prefill should have received 1 EnqueueBatch (which failed)");

        // 4. Verify: decode worker never received any request
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");

        // 5. Verify: PrefillEndpoint.inflightBatches cleaned up by failAck()
        InflightAssertions.assertPrefillInflightEmpty(getPrefillEndpoint());

        // 6. Verify: a subsequent request works fine (worker recovered)
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());

        CompletableFuture<Response> future2 = submitRequest(7002);
        Response response2 = future2.get(5, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed after recovery");
        assertEquals(2, mockPrefillWorker.getEnqueueCount(),
                "Prefill should have received 2 EnqueueBatch calls total");
    }
}

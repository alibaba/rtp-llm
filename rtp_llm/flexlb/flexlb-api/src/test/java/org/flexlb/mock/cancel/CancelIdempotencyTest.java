package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cancel idempotency: canceling the same requestId twice should be a no-op
 * the second time.
 *
 * <p>Flow:
 * 1. Configure prefill with long delay (2s) so ACK hasn't arrived when we cancel
 * 2. Submit request → dispatched to prefill worker via gRPC
 * 3. Cancel once → future completes while the scheduler waits for the enqueue ACK
 * 4. Cancel again → no-op while cancellation reconciliation is pending
 * 5. ACK arrives → exactly one gRPC cancel is sent, inflight is cleaned
 *
 * <p>The idempotency is guaranteed by {@code FlexlbBatchScheduler.cancel()}:
 * the CANCEL_REQUESTED state is idempotent until the enqueue ACK arrives.
 */
class CancelIdempotencyTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(2000)  // long delay: cancel before ACK arrives
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // single request triggers immediate dispatch
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
    void cancelIdempotency_secondCancelIsNoOp() throws Exception {
        // 1. Submit request — dispatches immediately (batchSizeMax=1)
        CompletableFuture<Response> future = submitRequest(8001);

        // 2. Wait for dispatch to reach the prefill worker (ensures batchId is set
        //    so repackPrefillBatch can clean up PrefillEndpoint.inflightBatches)
        long pollDeadline = System.currentTimeMillis() + 3000;
        while (mockPrefillWorker.getEnqueueCount() < 1 && System.currentTimeMillis() < pollDeadline) {
            Thread.sleep(10);
        }
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "Prefill worker should have received 1 EnqueueBatch before cancel");

        // 3. First cancel completes the client future, but engine Cancel must wait
        //    until EnqueueBatch returns its ACK.
        cancelRequest(8001);

        // 4. Verify response is CANCELLED
        Response response = future.get(5, TimeUnit.SECONDS);
        assertFalse(response.isSuccess(), "First cancel should make request return CANCELLED");
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode(),
                "Response should have REQUEST_CANCELLED error code");

        // 5. Second cancel is idempotent while CANCEL_REQUESTED is pending.
        cancelRequest(8001);

        assertEquals(0, mockPrefillWorker.getCancelCount(),
                "Engine cancel must not be sent before the enqueue ACK");
        long cancelDeadline = System.currentTimeMillis() + 3000;
        while (mockPrefillWorker.getCancelCount() < 1 && System.currentTimeMillis() < cancelDeadline) {
            Thread.sleep(10);
        }

        // 6. The ACK triggers exactly one engine Cancel.
        assertEquals(1, mockPrefillWorker.getCancelCount(),
                "Second cancel should not send another gRPC cancel");

        // 7. Verify: three-layer inflight cleanup
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 3000);

        // 8. Verify: subsequent request works fine (no resource leak)
        CompletableFuture<Response> future2 = submitRequest(8002);
        Response response2 = future2.get(5, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed");
    }
}

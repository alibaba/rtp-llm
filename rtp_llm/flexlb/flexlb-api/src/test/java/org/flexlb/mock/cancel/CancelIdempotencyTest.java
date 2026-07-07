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
 * 3. Cancel once → future completes with CANCELLED, gRPC cancel sent
 * 4. Cancel again → no-op (entry already removed from inflight by first cancel)
 * 5. Verify: only 1 gRPC cancel sent, inflight clean, subsequent request works
 *
 * <p>The idempotency is guaranteed by {@code FlexlbBatchScheduler.cancel()}:
 * {@code inflight.remove(requestId)} returns null on the second call, causing
 * an immediate return without sending another gRPC Cancel.
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

        // 3. First cancel — should complete future with CANCELLED
        //    (ackFinished=false because worker has 2s delay, ACK hasn't arrived)
        cancelRequest(8001);

        // 4. Verify response is CANCELLED
        Response response = future.get(5, TimeUnit.SECONDS);
        assertFalse(response.isSuccess(), "First cancel should make request return CANCELLED");
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode(),
                "Response should have REQUEST_CANCELLED error code");

        // 5. Second cancel — should be a no-op (entry already removed from inflight)
        //    inflight.remove(requestId) returns null → immediate return, no gRPC cancel
        cancelRequest(8001);

        // 6. Verify: only 1 gRPC cancel was sent (second cancel is no-op)
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

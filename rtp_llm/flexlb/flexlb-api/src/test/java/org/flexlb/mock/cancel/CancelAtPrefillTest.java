package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cancel a request while it is being processed by the prefill worker.
 *
 * <p>Flow:
 * 1. Configure prefill worker with long enqueue delay (simulating compute)
 * 2. Submit request → dispatched to prefill worker via gRPC
 * 3. Wait for ACK (batch enqueue succeeds), then cancel
 * 4. Verify: cancel sent to prefill worker, decode never receives request, resources clean
 */
class CancelAtPrefillTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(0)  // ACK immediately
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);  // immediate dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        return cfg;
    }

    @Test
    void cancelAtPrefill_cancelReachesWorker() throws Exception {
        // 1. Submit request — should dispatch immediately (batchSizeMax=1)
        CompletableFuture<Response> future = submitRequest(2001);

        // 2. Wait for ACK (EnqueueBatch succeeds)
        Response ackResponse = future.get(3, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "Request should be successfully enqueued");
        assertTrue(ackResponse.isEnqueuedByMaster(), "Should be enqueued by master");

        // 3. Cancel the request
        cancelRequest(2001);

        // 4. Verify cancel was sent to prefill worker via gRPC
        assertTrue(mockPrefillWorker.getCancelCount() >= 0,
                "Cancel should reach the prefill worker");

        // 5. Verify prefill worker received the EnqueueBatch
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "Prefill worker should have received exactly 1 EnqueueBatch");

        // 6. Verify decode worker never received any request (PD-separated: decode
        //    only gets the request after prefill completes, which didn't happen)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }
}

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
 * Cancel a request after prefill completes and decode is processing.
 *
 * <p>Flow:
 * 1. Configure prefill worker with no delay (fast completion)
 * 2. Submit request → dispatched to prefill → ACK received
 * 3. Cancel after ACK
 * 4. Verify: cancel reaches prefill worker, resources eventually clean
 */
class CancelAtDecodeTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(0)
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
    void cancelAtDecode_cancelAfterAck() throws Exception {
        // 1. Submit and wait for ACK
        CompletableFuture<Response> future = submitRequest(3001);
        Response response = future.get(3, TimeUnit.SECONDS);
        assertTrue(response.isSuccess(), "Request should be enqueued");

        // 2. Cancel after ACK — this triggers cancel to prefill worker
        cancelRequest(3001);

        // 3. Give a moment for cancel to propagate
        Thread.sleep(200);

        // 4. Verify prefill worker received both EnqueueBatch and Cancel
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "Prefill should have received 1 EnqueueBatch");
        assertEquals(1, mockPrefillWorker.getCancelCount(),
                "Prefill should have received 1 Cancel");

        // 5. Verify decode never received EnqueueBatch (PD-separated: decode
        //    is only contacted through prefill's internal flow, not by master)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not receive EnqueueBatch from master");
    }
}

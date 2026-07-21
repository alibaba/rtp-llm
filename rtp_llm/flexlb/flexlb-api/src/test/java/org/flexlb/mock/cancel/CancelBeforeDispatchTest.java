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

/**
 * Cancel a request while it is still in the batcher queue (before dispatch).
 *
 * <p>Flow:
 * 1. Configure long batch window so single request stays queued
 * 2. Submit request → enters batcher queue but doesn't dispatch yet
 * 3. Cancel immediately → request removed from scheduler inflight
 * 4. Verify: mock prefill worker never received EnqueueBatch, resources clean
 */
class CancelBeforeDispatchTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(2);         // need 2 requests to fill batch
        cfg.setFlexlbBatchWindowMs(10_000);   // long window
        cfg.setCostSloMs(50_000L);            // very long SLO → no urgent dispatch
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        return cfg;
    }

    @Test
    void cancelBeforeDispatch_requestNeverReachesWorker() throws Exception {
        // 1. Submit request — it enters the batcher queue
        CompletableFuture<Response> future = submitRequest(1001);
        assertFalse(future.isDone(), "Request should still be in queue");

        // 2. Cancel immediately
        cancelRequest(1001);

        // 3. Verify response — should be CANCELLED
        Response response = future.get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());

        // 4. Verify mock prefill worker never received EnqueueBatch
        assertEquals(0, mockPrefillWorker.getEnqueueCount(),
                "Prefill worker should never have received EnqueueBatch");

        // 5. Verify mock decode worker never received anything
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should never have received any requests");

        // 6. QUEUED is authoritative: no Enqueue happened, so no engine Cancel is needed.
        assertEquals(0, mockPrefillWorker.getCancelCount(),
                "Queued cancellation should not call the prefill worker");
    }
}

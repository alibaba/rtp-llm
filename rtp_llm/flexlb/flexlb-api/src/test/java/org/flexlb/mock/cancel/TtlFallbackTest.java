package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.mock.InflightAssertions;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * TTL fallback: prefill worker ignores cancel, inflight entries cleaned by TTL.
 *
 * <p>Flow:
 * 1. Configure prefill worker with ignoreCancel=true (worker never acks Cancel)
 * 2. Configure very short inflightTtlMs (500ms)
 * 3. Submit request → dispatched → ACK received
 * 4. Cancel request → gRPC Cancel sent but worker ignores it
 * 5. Wait for TTL and trigger cleanup manually
 * 6. Verify: resources eventually cleaned by TTL eviction
 */
class TtlFallbackTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(0)
                .ignoreCancel(true)  // Worker ignores cancel — never sends response
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
        cfg.setFlexlbBatchEnqueueDeadlineMs(2_000L);  // shorter deadline for cancel timeout
        cfg.setFlexlbInflightTtlMs(500L);  // very short TTL
        return cfg;
    }

    @Test
    void ttlFallback_resourcesEventuallyCleaned() throws Exception {
        // 1. Submit and wait for ACK
        CompletableFuture<Response> future = submitRequest(6001);
        Response response = future.get(3, TimeUnit.SECONDS);
        assertTrue(response.isSuccess(), "Request should be enqueued");

        // 2. Cancel — worker will ignore it (no response)
        cancelRequest(6001);

        // 3. Verify cancel was attempted (sent to worker but ignored)
        assertEquals(1, mockPrefillWorker.getCancelCount(),
                "Cancel should have been sent to prefill worker");

        // 4. Wait for TTL to expire
        Thread.sleep(600);

        // 5. Trigger TTL cleanup manually (simulates @Scheduled)
        triggerTtlCleanup();

        // 6. Verify: resources cleaned by TTL
        InflightAssertions.assertResourcesReleasedWithin(getPrefillEndpoint(), getDecodeEndpoint(), 2000);

        // 7. Verify: a subsequent request works fine (no resource leak after TTL)
        CompletableFuture<Response> future2 = submitRequest(6002);
        Response response2 = future2.get(3, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed after TTL cleanup");
    }
}

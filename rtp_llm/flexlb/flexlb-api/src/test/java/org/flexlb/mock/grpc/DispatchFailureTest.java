package org.flexlb.mock.grpc;

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

/** Verifies cleanup and recovery when EnqueueBatch returns an error response. */
class DispatchFailureTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .failOnEnqueue(true)
                .enqueueErrorMessage("mock engine overloaded")
                .enqueueErrorCode(13)
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        return super.createConfig();
    }

    @Test
    void dispatchFailure_requestFailsAndRecovers() throws Exception {
        CompletableFuture<Response> future = submitRequest("7001");
        Response response = future.get(5, TimeUnit.SECONDS);

        assertFalse(response.isSuccess(), "Request should fail when EnqueueBatch returns error");
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(1, mockPrefillWorker.getEnqueueCount(), response.getErrorMessage());
        assertEquals(0, mockDecodeWorker.getEnqueueCount());
        InflightAssertions.assertPrefillInflightEmpty(getPrefillEndpoint());

        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        Response recovered = submitRequest("7002").get(5, TimeUnit.SECONDS);
        assertTrue(recovered.isSuccess(), "Subsequent request should succeed after recovery");
        assertEquals(2, mockPrefillWorker.getEnqueueCount());
    }
}

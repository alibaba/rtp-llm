package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Lost EnqueueBatch ACKs remain pending until request inactivity expires their local accounting. */
class GrpcTimeoutTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(3000)  // 3s: far exceeds the 500ms deadline
                .build();
    }

    @Override
    protected long enqueueTimeoutMillis() {
        return 500L;
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig config = super.createConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(1_800L);
        return config;
    }

    @Test
    @Timeout(15)
    void grpcTimeout_requestExpiresWithoutEngineEvidenceAndRecovers() throws Exception {
        CompletableFuture<Response> future = submitRequest(10001);

        // The RPC expires before the request lease: uncertainty remains pending briefly.
        assertThrows(TimeoutException.class,
                () -> future.get(750, TimeUnit.MILLISECONDS));
        assertFalse(future.isDone());
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1,
                "the worker records EnqueueBatch before delaying its ACK");
        assertEquals(1, getPrefillEndpoint().ownershipStats().batchCount());
        assertEquals(1, getPrefillEndpoint().ownershipStats().locallyOwnedRequests());
        assertEquals(1, getDecodeEndpoint().resourceSnapshot().reservedCount());
        assertEquals(0, mockDecodeWorker.getEnqueueCount());

        Response expired = future.get(5, TimeUnit.SECONDS);
        assertFalse(expired.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), expired.getCode());
        assertTrue(expired.getErrorMessage().contains("REQUEST_INACTIVE"));
        org.flexlb.mock.InflightAssertions.assertResourcesReleasedWithin(getPrefillEndpoint(), getDecodeEndpoint(), 3_000L);
        assertEquals(0, requestRegistry().liveRequestCount());
        assertEquals(0, getPrefillEndpoint().ownershipStats().batchCount());
        assertEquals(0, getPrefillEndpoint().ownershipStats().locallyOwnedRequests());
        assertEquals(0, getDecodeEndpoint().resourceSnapshot().reservedCount());

        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        Response recovered = submitRequest(10002).get(5, TimeUnit.SECONDS);
        assertTrue(recovered.isSuccess(), "the same gRPC channel remains usable after expiry");
        assertFalse(future.join().isSuccess(), "later traffic cannot reopen the expired generation");
    }
}

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
    protected FlexlbConfig createConfig() {
        FlexlbConfig config = super.createConfig();
        config.getDispatcher().setEnqueueRpcTimeoutMs(500);
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(1_800L);
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
        assertEquals(1, getPrefillEndpoint().getInflightBatchCount());
        assertEquals(1, getPrefillEndpoint().getLocallyOwnedRequestCount());
        assertEquals(1, getDecodeEndpoint().getInflightCount());
        assertEquals(0, mockDecodeWorker.getEnqueueCount());

        Response expired = future.get(5, TimeUnit.SECONDS);
        assertFalse(expired.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), expired.getCode());
        assertTrue(expired.getErrorMessage().contains("REQUEST_INACTIVE"));
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, getPrefillEndpoint().getInflightBatchCount());
        assertEquals(0, getPrefillEndpoint().getLocallyOwnedRequestCount());
        assertEquals(0, getDecodeEndpoint().getInflightCount());

        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        Response recovered = submitRequest(10002).get(5, TimeUnit.SECONDS);
        assertTrue(recovered.isSuccess(), "the same gRPC channel remains usable after expiry");
        assertFalse(future.join().isSuccess(), "later traffic cannot reopen the expired generation");
    }
}

package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Missing remote cleanup proof keeps accounting visible and prevents a successful shutdown. */
class WorkerOfflineTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig config = super.createConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(1_800L);
        return config;
    }

    @Test
    @Timeout(20)
    void workerOffline_uncertainDispatchExpiresWithoutAuthoritativeStatus() throws Exception {
        CompletableFuture<Response> first = submitRequest(20001);
        Response acknowledged = first.get(5, TimeUnit.SECONDS);
        assertTrue(acknowledged.isSuccess());
        assertTrue(acknowledged.isEnqueuedByMaster());

        mockPrefillWorker.stop();
        Thread.sleep(500); // Let the gRPC channel process the worker's GOAWAY.

        CompletableFuture<Response> offline = submitRequest(20002);
        assertThrows(TimeoutException.class,
                () -> offline.get(600, TimeUnit.MILLISECONDS));
        assertFalse(offline.isDone(), "post-send uncertainty stays pending before request TTL");
        assertTrue(getPrefillEndpoint().ownershipStats().batchCount() >= 1);
        assertTrue(getDecodeEndpoint().resourceSnapshot().reservedCount() >= 1);
        assertEquals(0, mockDecodeWorker.getEnqueueCount());

        Response expired = offline.get(5, TimeUnit.SECONDS);
        assertFalse(expired.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), expired.getCode());
        assertTrue(expired.getErrorMessage().contains("REQUEST_INACTIVE"));
        assertTrue(requestRegistry().liveRequestCount() > 0);
        assertTrue(getPrefillEndpoint().ownershipStats().locallyOwnedRequests() > 0);
        assertTrue(getDecodeEndpoint().resourceSnapshot().reservedCount() > 0);
        assertThrows(IllegalStateException.class, schedulerRuntime::close);
        assertTrue(first.join().isSuccess(), "cleanup does not replace an already published ACK");
    }
}

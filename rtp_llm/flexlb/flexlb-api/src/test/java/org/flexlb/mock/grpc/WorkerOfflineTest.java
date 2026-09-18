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

/**
 * Worker offline: stop the mock prefill worker's gRPC server while requests
 * are in-flight, verifying that the master detects the connection failure and
 * retains resources behind an Engine ownership fence.
 *
 * <p>Flow:
 * 1. Start mock prefill worker (normal config)
 * 2. Submit request → ACK succeeds (proves the gRPC link works)
 * 3. Delay the next ACK and wait until the worker receives the request
 * 4. Stop the worker before its ACK is delivered
 * 6. Verify: the post-send outcome remains pending and inflight ownership is retained
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>After {@code server.shutdown()}, the TCP port is no longer listening</li>
 *   <li>A request received before shutdown has an ambiguous outcome without an ACK</li>
 *   <li>A later connection refusal proves a new request was not sent</li>
 *   <li>{@link org.flexlb.engine.grpc.EngineGrpcClient} completes the asynchronous
 *       EnqueueBatch call exceptionally and deliberately does not replay an
 *       invocation whose acceptance is ambiguous.</li>
 *   <li>The received invocation is ambiguous without its ACK, so the scheduler
 *       cannot safely publish failure or release ownership without Engine proof</li>
 * </ul>
 *
 * <p>Note: {@code MockWorker.stop()} already supports graceful gRPC server shutdown
 * (up to 5 seconds wait). The test calls it explicitly mid-test; the base class
 * {@code @AfterEach} calls it again, which is safe (no-op on an already-terminated server).
 */
class WorkerOfflineTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        return super.createConfig();
    }

    @Test
    @Timeout(20)
    void workerOffline_uncertainDispatchRetainsFenceUntilAuthoritativeStatus() throws Exception {
        // 1. Submit request with normal worker — should succeed
        CompletableFuture<Response> future1 = submitRequest(20001);
        Response ackResponse = future1.get(5, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "First request should succeed while worker is online");
        assertTrue(ackResponse.isEnqueuedByMaster(), "Should be enqueued by master");
        int existingBatches = getPrefillEndpoint().getInflightBatchCount();

        mockPrefillWorker.setBehavior(mockPrefillWorker.getBehavior().toBuilder()
                .enqueueDelayMs(10_000).build());
        CompletableFuture<Response> future2 = submitRequest(20002);
        long receivedDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (mockPrefillWorker.getEnqueueCount() < 2 && System.nanoTime() < receivedDeadline) {
            Thread.sleep(10);
        }
        assertEquals(2, mockPrefillWorker.getEnqueueCount(),
                "Worker must receive the request before shutdown makes its ACK ambiguous");
        mockPrefillWorker.stop();
        assertThrows(TimeoutException.class,
                () -> future2.get(2, TimeUnit.SECONDS));
        assertFalse(future2.isDone(),
                "offline post-send ambiguity must wait for authoritative Engine status");

        // 6. The uncertain request remains charged; releasing it here could double-admit.
        assertTrue(getPrefillEndpoint().getInflightBatchCount() >= existingBatches + 1);

        // 7. Verify: decode worker never received any enqueue request (PD-separated)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }

    @Test
    @Timeout(20)
    void workerOffline_unsentDispatchReleasesOwnership() throws Exception {
        Response first = submitRequest(20011).get(5, TimeUnit.SECONDS);
        assertTrue(first.isSuccess());
        assertTrue(first.isEnqueuedByMaster(), "Should be enqueued by master");
        int existingBatches = getPrefillEndpoint().getInflightBatchCount();
        mockPrefillWorker.stop();
        Thread.sleep(500);

        Response rejected = submitRequest(20012).get(5, TimeUnit.SECONDS);

        assertFalse(rejected.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), rejected.getCode());
        String errMsg = rejected.getErrorMessage();
        assertTrue(errMsg != null && !errMsg.isEmpty(), "Error message should not be empty");
        assertEquals(existingBatches, getPrefillEndpoint().getInflightBatchCount());
        assertEquals(1, mockPrefillWorker.getEnqueueCount(), "Unsent request must never reach the worker");
        assertEquals(0, mockDecodeWorker.getEnqueueCount());
    }
}

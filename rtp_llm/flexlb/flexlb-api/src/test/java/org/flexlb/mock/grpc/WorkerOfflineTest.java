package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
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
 * 4. Stop the mock prefill worker's gRPC server (simulates worker crash)
 * 5. Submit a new request → gRPC call fails (connection refused / channel broken)
 * 6. Verify: the post-send outcome remains pending and inflight ownership is retained
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>After {@code server.shutdown()}, the TCP port is no longer listening</li>
 *   <li>The gRPC client channel may still be "open" from the client's perspective,
 *       but the next call will fail because:</li>
 *   <li>The server sends a GOAWAY frame during graceful shutdown, and/or</li>
 *   <li>The TCP connection attempt fails with "Connection refused" (20ms timeout)</li>
 *   <li>{@link org.flexlb.engine.grpc.EngineGrpcClient} completes the asynchronous
 *       EnqueueBatch call exceptionally and deliberately does not replay an
 *       invocation whose acceptance is ambiguous.</li>
 *   <li>The asynchronous invocation is ambiguous after it starts, so the scheduler
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

        // 2. Stop the mock prefill worker's gRPC server (simulates worker crash)
        mockPrefillWorker.stop();

        // 3. Brief pause to let the gRPC client detect the connection loss
        //    (GOAWAY processing / keepalive detection is async)
        Thread.sleep(500);

        // 4. Submit a new request — gRPC call should fail (connection refused)
        CompletableFuture<Response> future2 = submitRequest(20002);
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
}

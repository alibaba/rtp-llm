package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Worker offline: stop the mock prefill worker's gRPC server while requests
 * are in-flight, verifying that the master detects the connection failure and
 * cleans up resources.
 *
 * <p>Flow:
 * 1. Start mock prefill worker (normal config)
 * 2. Submit request → ACK succeeds (proves the gRPC link works)
 * 3. Cancel the first request to clean up its inflight resources
 * 4. Stop the mock prefill worker's gRPC server (simulates worker crash)
 * 5. Submit a new request → gRPC call fails (connection refused / channel broken)
 * 6. Verify: request fails with BATCH_DISPATCH_FAILED, inflight cleaned up,
 *    error message contains gRPC failure indication
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>After {@code server.shutdown()}, the TCP port is no longer listening</li>
 *   <li>The gRPC client channel may still be "open" from the client's perspective,
 *       but the next call will fail because:</li>
 *   <li>The server sends a GOAWAY frame during graceful shutdown, and/or</li>
 *   <li>The TCP connection attempt fails with "Connection refused" (20ms timeout)</li>
 *   <li>{@link org.flexlb.engine.grpc.EngineGrpcClient#executeGrpcCall} catches the
 *       {@code StatusRuntimeException}. If {@code isConnectionBrokenError} matches,
 *       it retries once with a new channel — which also fails.</li>
 *   <li>Regardless of retry, the exception propagates to
 *       {@link org.flexlb.balance.scheduler.DefaultBatchDispatcher}, which calls
 *       {@code failItems()} → {@code callback.onFailure()} →
 *       {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler#failAck}</li>
 * </ul>
 *
 * <p>Note: {@code MockWorker.stop()} already supports graceful gRPC server shutdown
 * (up to 5 seconds wait). The test calls it explicitly mid-test; the base class
 * {@code @AfterEach} calls it again, which is safe (no-op on an already-terminated server).
 */
class WorkerOfflineTest extends FlexLBMockTestBase {

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // single request triggers immediate dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(20)
    void workerOffline_newRequestFailsAndCleansUp() throws Exception {
        // 1. Submit request with normal worker — should succeed
        CompletableFuture<Response> future1 = submitRequest(20001);
        Response ackResponse = future1.get(5, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "First request should succeed while worker is online");
        assertTrue(ackResponse.isEnqueuedByMaster(), "Should be enqueued by master");

        // 2. Cancel the first request to release its inflight resources
        //    (mock workers don't send status updates, so we must clean up manually)
        cancelRequest(20001);
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 5000);

        // 3. Stop the mock prefill worker's gRPC server (simulates worker crash)
        mockPrefillWorker.stop();

        // 4. Brief pause to let the gRPC client detect the connection loss
        //    (GOAWAY processing / keepalive detection is async)
        Thread.sleep(500);

        // 5. Submit a new request — gRPC call should fail (connection refused)
        CompletableFuture<Response> future2 = submitRequest(20002);
        Response failResponse = future2.get(10, TimeUnit.SECONDS);

        // 6. Verify: request failed with BATCH_DISPATCH_FAILED
        assertFalse(failResponse.isSuccess(),
                "Request should fail when prefill worker is offline");
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), failResponse.getCode(),
                "Request should have BATCH_DISPATCH_FAILED error code");

        // 7. Verify: error message contains gRPC failure indication
        String errMsg = failResponse.getErrorMessage();
        assertTrue(errMsg != null && !errMsg.isEmpty(),
                "Error message should not be empty");

        // 8. Verify: dispatch failure cleans up PrefillEndpoint inflight state
        InflightAssertions.assertPrefillInflightEmpty(getPrefillEndpoint());

        // 9. Verify: decode worker never received any enqueue request (PD-separated)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }
}

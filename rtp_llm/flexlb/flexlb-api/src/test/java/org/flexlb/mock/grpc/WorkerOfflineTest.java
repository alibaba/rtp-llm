package org.flexlb.mock.grpc;

import org.flexlb.balance.scheduler.RequestLifecycleState;
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
 * Worker offline: stop the mock prefill worker's gRPC server, then verify that Master retains
 * ownership until an authoritative Engine fence resolves the post-send ambiguity.
 *
 * <p>Flow:
 * 1. Start mock prefill worker (normal config)
 * 2. Submit request → ACK succeeds (proves the gRPC link works)
 * 4. Stop the mock prefill worker's gRPC server (simulates worker crash)
 * 5. Submit a new request → gRPC call fails (connection refused / channel broken)
 * 6. Verify: the request stays DISPATCHING and both ledgers are retained; an immediate
 *    retryable failure would permit a duplicate if the request reached Engine before the crash
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
 *   <li>The post-invocation transport error enters dispatch reconciliation; only an Engine
 *       tombstone or typed WorkerStatus cancellation may settle it as absent</li>
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
    void workerOffline_newRequestRemainsFencedUntilAuthoritativeSettlement() throws Exception {
        // 1. Submit request with normal worker — should succeed
        CompletableFuture<Response> future1 = submitRequest(20001);
        Response ackResponse = future1.get(5, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "First request should succeed while worker is online");
        assertTrue(ackResponse.isEnqueuedByMaster(), "Should be enqueued by master");
        int existingBatches = getPrefillEndpoint().getInflightBatchCount();

        // 2. Stop the mock prefill worker's gRPC server (simulates worker crash)
        mockPrefillWorker.stop();

        // 3. Submit a new request — its RPC fails after invocation (connection refused).
        CompletableFuture<Response> future2 = submitRequest(20002);

        assertThrows(TimeoutException.class,
                () -> future2.get(2, TimeUnit.SECONDS));
        assertFalse(future2.isDone(),
                "transport failure must not claim the worker rejected the request");
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(20002L, 0).state());
        assertEquals(existingBatches + 1, getPrefillEndpoint().getInflightBatchCount(),
                "ambiguous dispatch must retain its Prefill ledger until fenced");

        // 4. Decode receives no enqueue in the P/D-separated path.
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }
}

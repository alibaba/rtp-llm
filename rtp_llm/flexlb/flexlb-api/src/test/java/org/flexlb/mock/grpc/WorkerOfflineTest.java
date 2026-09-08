package org.flexlb.mock.grpc;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestLifecycleState;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

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

    private final CompletableFuture<EngineCancelChannel.CancelOutcome> dispatchFence =
            new CompletableFuture<>();
    private final CountDownLatch cancelInvoked = new CountDownLatch(1);
    private final AtomicLong canceledRequestId = new AtomicLong(-1);

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

    @Override
    protected EngineCancelChannel createEngineCancelChannel() {
        return new EngineCancelChannel() {
            @Override
            public boolean isSupported(DecodeEndpoint endpoint) {
                return true;
            }

            @Override
            public CompletableFuture<CancelOutcome> cancel(
                    CancelTarget target, long requestId, long timeoutMs) {
                canceledRequestId.set(requestId);
                cancelInvoked.countDown();
                return dispatchFence;
            }
        };
    }

    @Test
    @Timeout(20)
    void workerOffline_newRequestRemainsFencedUntilAuthoritativeSettlement() throws Exception {
        // 1. Submit request with normal worker — should succeed
        CompletableFuture<Response> future1 = submitRequest(20001);
        Response ackResponse = future1.get(5, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "First request should succeed while worker is online");
        assertTrue(ackResponse.isEnqueuedByMaster(), "Should be enqueued by master");
        reportSuccessfulCompletion(20001L);
        InflightAssertions.assertSchedulerInflightEmptyWithin(scheduler, 5_000);
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 5_000);

        // 2. Stop the mock prefill worker's gRPC server (simulates worker crash)
        mockPrefillWorker.stop();

        // 3. Submit a new request — its RPC fails after invocation (connection refused).
        CompletableFuture<Response> future2 = submitRequest(20002);

        assertThrows(TimeoutException.class,
                () -> future2.get(2, TimeUnit.SECONDS));
        assertFalse(future2.isDone(),
                "transport failure must not claim the worker rejected the request");
        assertTrue(cancelInvoked.await(5, TimeUnit.SECONDS),
                "an ambiguous transport failure must invoke the Engine ownership fence");
        assertEquals(20002L, canceledRequestId.get());
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(20002L, 0).state());
        assertEquals(1, getPrefillEndpoint().getInflightBatchCount(),
                "ambiguous dispatch must retain its Prefill ledger until fenced");
        assertEquals(1, getDecodeEndpoint().getInflightCount(),
                "ambiguous dispatch must retain its Decode reservation until fenced");

        // 4. An Engine tombstone proves non-ownership and atomically settles both ledgers.
        dispatchFence.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        Response fenced = future2.get(5, TimeUnit.SECONDS);
        assertFalse(fenced.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), fenced.getCode());
        InflightAssertions.assertSchedulerInflightEmptyWithin(scheduler, 5_000);
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 5_000);

        // 5. Decode receives no enqueue in the P/D-separated path.
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }

    private void reportSuccessfulCompletion(long requestId) {
        RequestLifecycleSnapshot state = scheduler.getRequestState(requestId, 0);
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(state.batchId());

        WorkerStatusResponse prefillFinished = new WorkerStatusResponse();
        prefillFinished.setRole(RoleType.PREFILL);
        prefillFinished.setFinishedTaskInfo(Map.of(Long.toString(requestId), task));
        getPrefillEndpoint().onWorkerStatusUpdate(
                getPrefillEndpoint().getStatus(), prefillFinished);
        scheduler.onWorkerStatusUpdate(prefillFinished);

        WorkerStatusResponse decodeFinished = new WorkerStatusResponse();
        decodeFinished.setRole(RoleType.DECODE);
        decodeFinished.setFinishedTaskInfo(Map.of(Long.toString(requestId), task));
        getDecodeEndpoint().onWorkerStatusUpdate(
                getDecodeEndpoint().getStatus(), decodeFinished);
        scheduler.onWorkerStatusUpdate(decodeFinished);
    }
}

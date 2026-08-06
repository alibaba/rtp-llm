package org.flexlb.mock.grpc;

import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.mock.StabilityMonitor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Template for mid-flight cancel scenarios: a request is cancelled while its
 * EnqueueBatch RPC is still in flight, the engine-side Cancel RPC is delivered
 * through the real {@code EngineGrpcClient#cancelAsync}, and all inflight
 * accounting drains back to zero afterwards (leak canary).
 */
class CancelMidFlightTest extends FlexLBMockTestBase {

    private static final long ENQUEUE_DELAY_MS = 1_500L;

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        // Slow enqueue keeps the request in flight long enough to cancel it.
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(ENQUEUE_DELAY_MS)
                .build();
    }

    @Test
    void cancelWhileEnqueueInFlight_futureCancelsAndAccountingDrains() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint());

        CompletableFuture<Response> future = monitor.track(submitRequest(9001));

        // Wait until the EnqueueBatch RPC has reached the mock worker.
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "EnqueueBatch should reach the mock worker");
        InflightItem item = inflightStore.get("9001");
        assertNotNull(item, "request should be tracked while in flight");

        // Cancel mid-flight: CAS on the item, scheduler hook, engine-side RPC.
        assertTrue(item.cancel(), "cancel should win the CAS while in flight");
        item.fireOnCancel();
        EngineRpcService.EmptyPB ack = grpcClient.cancelAsync(
                prefillIp, prefillGrpcPort,
                EngineRpcService.CancelRequestPB.newBuilder().setRequestId(9001).build(),
                2_000L).get(3, TimeUnit.SECONDS);
        assertNotNull(ack);

        assertThrows(CancellationException.class, () -> future.get(1, TimeUnit.SECONDS));
        assertEquals(1, mockPrefillWorker.getCancelCount());
        assertEquals(9001, mockPrefillWorker.getRpcService()
                .getCancelledRequests().get(0).getRequestId());

        // Leak canary: after the delayed enqueue response lands, every layer
        // of accounting must drain to zero despite the mid-flight cancel.
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 4_000);
    }

    @Test
    void cancelUnknownRequest_isIdempotentNoOp() throws Exception {
        assertNull(inflightStore.get("424242"), "unknown request must not be tracked");

        // Engine-side cancel for an unknown id is an acknowledged no-op.
        for (int i = 0; i < 2; i++) {
            grpcClient.cancelAsync(
                    prefillIp, prefillGrpcPort,
                    EngineRpcService.CancelRequestPB.newBuilder().setRequestId(424242).build(),
                    2_000L).get(3, TimeUnit.SECONDS);
        }
        assertEquals(2, mockPrefillWorker.getCancelCount());
        assertEquals(0, inflightStore.activeCount());
    }

    @Test
    void cancelAfterCompletion_losesTheCas() throws Exception {
        // Fast worker for this scenario.
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());

        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint());
        CompletableFuture<Response> future = monitor.track(submitRequest(9002));
        Response response = future.get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());

        InflightItem item = inflightStore.get("9002");
        assertNotNull(item, "terminal item remains as tombstone until TTL");
        assertFalse(item.cancel(), "cancel after completion must lose the CAS");

        // Successful batches are released by the engine's finished report
        // (status-sync → calibrate); the harness has no sync runner, so
        // simulate one report before asserting quiescence.
        simulatePrefillFinishedReport();
        monitor.assertQuiescent(3_000);
    }

    private static void awaitTrue(java.util.function.BooleanSupplier condition,
                                  long timeoutMs, String message) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(20);
        }
        assertTrue(condition.getAsBoolean(), message);
    }
}

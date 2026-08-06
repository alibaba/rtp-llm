package org.flexlb.mock.grpc;

import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.mock.StabilityMonitor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Leak-canary template: mixes successful, failed, and mid-flight-cancelled
 * requests, then asserts via {@link StabilityMonitor} that every layer of
 * inflight accounting drains back to zero — no leaked store entries, no
 * leaked endpoint reservations, no hanging futures.
 */
class InflightLeakTest extends FlexLBMockTestBase {

    private static final int PHASE_SIZE = 3;

    @Test
    void mixedOutcomes_leaveNoInflightLeak() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                // Successful batches are released only by the engine's finished
                // report (status-sync → calibrate); the harness has no sync
                // runner, so pump a simulated report on every canary poll.
                .pump(this::simulatePrefillFinishedReport);

        // Phase 1: successful requests.
        for (int i = 0; i < PHASE_SIZE; i++) {
            Response response = monitor.track(submitRequest(8100 + i)).get(5, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), "phase-1 request should succeed");
        }

        // Phase 2: enqueue failures.
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .failOnEnqueue(true)
                .enqueueErrorMessage("mock engine overloaded")
                .enqueueErrorCode(13)
                .build());
        for (int i = 0; i < PHASE_SIZE; i++) {
            Response response = monitor.track(submitRequest(8200 + i)).get(5, TimeUnit.SECONDS);
            assertFalse(response.isSuccess(), "phase-2 request should fail");
        }

        // Phase 3: mid-flight cancels while the enqueue RPC is delayed.
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .enqueueDelayMs(800L)
                .build());
        List<CompletableFuture<Response>> cancelled = new ArrayList<>();
        for (int i = 0; i < PHASE_SIZE; i++) {
            long requestId = 8300 + i;
            cancelled.add(monitor.track(submitRequest(requestId)));
            awaitTracked(String.valueOf(requestId), 3_000);
            InflightItem item = inflightStore.get(String.valueOf(requestId));
            assertTrue(item.cancel(), "mid-flight cancel should win the CAS");
            item.fireOnCancel();
            grpcClient.cancelAsync(
                    prefillIp, prefillGrpcPort,
                    EngineRpcService.CancelRequestPB.newBuilder().setRequestId(requestId).build(),
                    2_000L).get(3, TimeUnit.SECONDS);
        }
        for (CompletableFuture<Response> future : cancelled) {
            assertTrue(future.isDone(), "cancelled future must be settled");
        }
        assertEquals(PHASE_SIZE, mockPrefillWorker.getCancelCount());

        // Canary: all three phases drained, only tombstones remain.
        monitor.assertQuiescent(6_000);
        assertEquals(0, inflightStore.activeCount());
        assertEquals(3 * PHASE_SIZE, inflightStore.totalSize(),
                "terminal items remain as tombstones until TTL");
    }

    private void awaitTracked(String requestId, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (inflightStore.get(requestId) != null) {
                return;
            }
            Thread.sleep(10);
        }
        assertNotNull(inflightStore.get(requestId), "request should be tracked in the store");
    }
}

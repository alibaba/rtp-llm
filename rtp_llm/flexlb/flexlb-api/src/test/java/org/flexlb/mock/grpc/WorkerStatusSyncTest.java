package org.flexlb.mock.grpc;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Worker status sync: verify that the master correctly retrieves and applies
 * worker status changes via the gRPC GetWorkerStatus call.
 *
 * <p>Flow:
 * 1. Start mock prefill with available_concurrency = 10
 * 2. Submit request → succeed (normal operation)
 * 3. Call {@code grpcClient.getWorkerStatus()} → verify response has concurrency = 10
 * 4. Apply status update to WorkerStatus and PrefillEndpoint (simulates EngineSyncRunner)
 * 5. Verify WorkerStatus reflects concurrency = 10
 * 6. Change mock prefill behavior: available_concurrency = 0 (worker full)
 * 7. Call {@code grpcClient.getWorkerStatus()} again → verify concurrency = 0
 * 8. Apply status update
 * 9. Verify WorkerStatus reflects concurrency = 0 (master perceives worker is full)
 * 10. Clean up
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>The gRPC client calls {@code getWorkerStatus} with a {@code StatusVersionPB}
 *       request containing the latest finished task version</li>
 *   <li>The mock returns a {@code WorkerStatusPB} with configurable
 *       {@code available_concurrency}, {@code available_kv_cache}, {@code total_kv_cache},
 *       {@code alive}, {@code status_version}, etc.</li>
 *   <li>{@link EngineStatusConverter#convertToWorkerStatusResponse} converts the PB
 *       to a {@link WorkerStatusResponse}</li>
 *   <li>{@link WorkerStatus#updateFromResponse} updates the in-memory status</li>
 *   <li>{@link PrefillEndpoint#onWorkerStatusUpdate} replaces the endpoint's status
 *       reference and calls {@code calibrate()} for inflight reconciliation</li>
 * </ul>
 *
 * <p>Note: In production, {@link org.flexlb.sync.runner.GrpcWorkerStatusRunner} performs
 * this flow periodically. The test calls the gRPC method directly to test the link
 * without requiring the full sync runner infrastructure (WorkerAddressService,
 * statusCheckExecutor, EngineHealthReporter, etc.).
 */
class WorkerStatusSyncTest extends FlexLBMockTestBase {

    private static final long SYNC_TIMEOUT_MS = 5000;

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .availableConcurrency(10)
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchFillThreshold(1.0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(20)
    void workerStatusSync_masterPerceivesConcurrencyChange() throws Exception {
        // 1. Submit request — normal operation with concurrency=10
        CompletableFuture<org.flexlb.dao.loadbalance.Response> future = submitRequest(30001);
        org.flexlb.dao.loadbalance.Response ackResponse = future.get(5, TimeUnit.SECONDS);
        assertTrue(ackResponse.isSuccess(), "Request should succeed with concurrency=10");

        // 2. Trigger status sync via gRPC — first call (concurrency=10)
        long statusCallCountBefore = mockPrefillWorker.getWorkerStatusCallCount();
        EngineRpcService.WorkerStatusPB pb1 = fetchWorkerStatus();

        // 3. Verify gRPC response has correct values
        assertTrue(pb1.getAlive(), "Worker should be alive");
        assertEquals(10, pb1.getAvailableConcurrency(),
                "Initial available_concurrency should be 10");

        // 4. Apply status update (simulates GrpcWorkerStatusRunner flow)
        applyWorkerStatus(pb1);

        // 5. Verify WorkerStatus and PrefillEndpoint reflect the update
        PrefillEndpoint prefillEp = getPrefillEndpoint();
        WorkerStatus ws = prefillEp.getStatus();
        assertNotNull(ws, "PrefillEndpoint should have a WorkerStatus");
        assertEquals(10L, ws.getAvailableConcurrency(),
                "WorkerStatus should reflect concurrency=10 after sync");
        assertTrue(ws.isAlive(), "Worker should be alive after sync");

        // 6. Verify mock received the GetWorkerStatus gRPC call
        assertEquals(statusCallCountBefore + 1, mockPrefillWorker.getWorkerStatusCallCount(),
                "Mock should have received 1 additional GetWorkerStatus call");

        // 7. Cancel the request to clean up inflight
        cancelRequest(30001);
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 5000);

        // 8. Change mock behavior: available_concurrency = 0 (worker full)
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .availableConcurrency(0)
                .build());

        // 9. Trigger second status sync via gRPC — concurrency=0
        long statusCallCountBefore2 = mockPrefillWorker.getWorkerStatusCallCount();
        EngineRpcService.WorkerStatusPB pb2 = fetchWorkerStatus();

        // 10. Verify gRPC response reflects the change
        assertEquals(0, pb2.getAvailableConcurrency(),
                "Updated available_concurrency should be 0");

        // 11. Apply status update
        applyWorkerStatus(pb2);

        // 12. Verify master perceives worker is full
        WorkerStatus ws2 = getPrefillEndpoint().getStatus();
        assertEquals(0L, ws2.getAvailableConcurrency(),
                "WorkerStatus should reflect concurrency=0 after sync — "
                        + "master should perceive worker is full");

        // 13. Verify mock received the second GetWorkerStatus call
        assertEquals(statusCallCountBefore2 + 1, mockPrefillWorker.getWorkerStatusCallCount(),
                "Mock should have received 1 more GetWorkerStatus call");
    }

    // ==================== Helper: simulate GrpcWorkerStatusRunner flow ====================

    /**
     * Call {@code grpcClient.getWorkerStatus()} to fetch the worker status from the
     * mock prefill worker via gRPC.
     */
    private EngineRpcService.WorkerStatusPB fetchWorkerStatus() {
        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestFinishedVersion(0)
                .build();
        return grpcClient.getWorkerStatus(prefillIp, prefillGrpcPort, request, SYNC_TIMEOUT_MS);
    }

    /**
     * Apply the gRPC worker status response to the in-memory WorkerStatus and
     * PrefillEndpoint, simulating what {@link org.flexlb.sync.runner.GrpcWorkerStatusRunner}
     * does in production:
     * <ol>
     *   <li>Convert {@code WorkerStatusPB} → {@code WorkerStatusResponse}</li>
     *   <li>Update {@code WorkerStatus} via {@code updateFromResponse}</li>
     *   <li>Notify {@code PrefillEndpoint} via {@code onWorkerStatusUpdate}</li>
     * </ol>
     */
    private void applyWorkerStatus(EngineRpcService.WorkerStatusPB pb) {
        WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(pb);
        // Ensure correct role (mock returns the role from MockWorkerBehavior, which
        // MockPrefillWorker sets to ROLE_TYPE_PREFILL)
        response.setRole(RoleType.PREFILL);

        WorkerStatus ws = getPrefillEndpoint().getStatus();
        ws.updateFromResponse(response);
        getPrefillEndpoint().onWorkerStatusUpdate(ws, response);
    }
}

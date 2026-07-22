package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockPrefillWorker;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.when;

/**
 * Multiple workers: start two mock prefill workers and verify that requests
 * are distributed across both, not all sent to a single worker.
 *
 * <p>Flow:
 * 1. Worker A is started by the base class (default behavior)
 * 2. Worker B is started via {@link #addPrefillWorker} (registered in EndpointRegistry)
 * 3. Reconfigure the mock Router to alternate between A and B
 * 4. Submit 4 requests
 * 5. Verify: both workers received at least 1 EnqueueBatch call
 * 6. Verify: total EnqueueBatch count matches the number of submitted requests
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>The base class starts one prefill worker (worker A) and registers it in
 *       {@code EndpointRegistry} and {@code EngineWorkerStatus}</li>
 *   <li>{@link #addPrefillWorker} starts an additional worker B, creates its
 *       {@code WorkerStatus} and {@code PrefillEndpoint}, and registers both</li>
 *   <li>The mock {@code Router} is reset and reconfigured to return routing
 *       responses that alternate between A and B</li>
 *   <li>Each routing response contains {@code ServerStatus} entries for the
 *       selected prefill worker and the shared decode worker</li>
 *   <li>The scheduler looks up the prefill endpoint by {@code ip:httpPort}
 *       and offers the request to that worker's {@code WorkerBatcher}</li>
 * </ul>
 */
class MultipleWorkersTest extends FlexLBMockTestBase {

    private MockPrefillWorker workerB;
    private int workerBGrpcPort;
    private int workerBHttpPort;
    private String workerBIpPort;

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // each request dispatches independently
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(30)
    void multipleWorkers_requestsDistributedAcrossBoth() throws Exception {
        // 1. Start an additional prefill worker B
        workerB = addPrefillWorker(MockWorkerBehavior.builder().build());
        workerBGrpcPort = workerB.getPort();
        workerBHttpPort = workerB.getHttpPort();
        workerBIpPort = workerIpPort(workerB);

        // 2. Reconfigure the Router to alternate between worker A and B
        AtomicInteger routeCounter = new AtomicInteger(0);
        reset(router);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            boolean useB = routeCounter.getAndIncrement() % 2 == 1;
            return buildRouteResponse(ctx.getRequestId(), useB);
        });

        // 3. Submit 4 requests — should alternate A, B, A, B
        long[] requestIds = {40001, 40002, 40003, 40004};
        CompletableFuture<Response>[] futures = new CompletableFuture[requestIds.length];
        for (int i = 0; i < requestIds.length; i++) {
            futures[i] = submitRequest(requestIds[i]);
        }

        // 4. Wait for all requests to complete (ACK)
        for (int i = 0; i < requestIds.length; i++) {
            Response resp = futures[i].get(10, TimeUnit.SECONDS);
            assertTrue(resp.isSuccess(),
                    "Request " + requestIds[i] + " should succeed, got code=" + resp.getCode());
        }

        // 5. Verify: worker A received at least 1 EnqueueBatch
        int countA = mockPrefillWorker.getEnqueueCount();
        assertTrue(countA >= 1,
                "Worker A should have received at least 1 EnqueueBatch, got " + countA);

        // 6. Verify: worker B received at least 1 EnqueueBatch
        int countB = workerB.getEnqueueCount();
        assertTrue(countB >= 1,
                "Worker B should have received at least 1 EnqueueBatch, got " + countB);

        // 7. Verify: total EnqueueBatch count matches number of submitted requests
        int totalCount = countA + countB;
        assertTrue(totalCount >= requestIds.length,
                "Total EnqueueBatch count (" + totalCount + ") should be >= "
                        + requestIds.length + " submitted requests");

        // 8. Verify: decode worker received no enqueue (PD-separated — decode
        //    only gets requests after prefill completes, which mock doesn't simulate)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any enqueue request");

    }

    // ==================== Helper: build routing response ====================

    /**
     * Build a routing response that points to either worker A or worker B
     * for the prefill role, and always to the shared decode worker.
     */
    private Response buildRouteResponse(long requestId, boolean useWorkerB) {
        Response response = new Response();
        response.setSuccess(true);

        // Prefill: worker A or B
        ServerStatus prefill = new ServerStatus();
        prefill.setSuccess(true);
        prefill.setRole(RoleType.PREFILL);
        if (useWorkerB) {
            prefill.setServerIp("127.0.0.1");
            prefill.setHttpPort(workerBHttpPort);
            prefill.setGrpcPort(workerBGrpcPort);
        } else {
            prefill.setServerIp(prefillIp);
            prefill.setHttpPort(prefillHttpPort);
            prefill.setGrpcPort(prefillGrpcPort);
        }
        prefill.setDpRank(0);
        prefill.setGroup("test-group");
        prefill.setRequestId(requestId);

        // Decode: shared decode worker (always worker A's decode)
        ServerStatus decode = new ServerStatus();
        decode.setSuccess(true);
        decode.setRole(RoleType.DECODE);
        decode.setServerIp(decodeIp);
        decode.setHttpPort(decodeHttpPort);
        decode.setGrpcPort(decodeGrpcPort);
        decode.setDpRank(0);
        decode.setGroup("test-group");
        decode.setRequestId(requestId);

        response.setServerStatus(List.of(prefill, decode));
        return response;
    }
}

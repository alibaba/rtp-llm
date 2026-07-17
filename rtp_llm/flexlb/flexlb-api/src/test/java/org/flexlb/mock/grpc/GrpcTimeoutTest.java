package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * gRPC batchEnqueue timeout: mock prefill worker delays response beyond the
 * configured gRPC deadline, verifying that master correctly handles the
 * timeout (DEADLINE_EXCEEDED) and cleans up inflight resources.
 *
 * <p>Flow:
 * 1. Configure mock prefill with enqueueDelayMs=3000 (3s) and master deadline=500ms
 * 2. Submit request → dispatched → gRPC batchEnqueue times out at 500ms
 * 3. Verify: request transitions to TIMED_OUT, inflight is cleaned up,
 *    mock prefill received EnqueueBatch (but couldn't respond in time)
 * 4. Recover: change behavior to delay=0, submit new request → succeeds
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>gRPC client sets {@code withDeadlineAfter(deadlineMs)} on the blocking stub</li>
 *   <li>When the deadline fires, the blocking call throws {@code StatusRuntimeException}
 *       with status DEADLINE_EXCEEDED</li>
 *   <li>{@link org.flexlb.balance.scheduler.DefaultBatchDispatcher} catches this in its
 *       {@code catch (Throwable)} block and calls {@code onFailure()}</li>
 *   <li>The scheduler returns BATCH_SLO_EXPIRED</li>
 * </ul>
 *
 * <p>Note: The mock's {@code enqueueBatch} records the request <em>before</em> sleeping,
 * so the test can verify the mock received the call even though the client timed out.
 * The server thread continues sleeping after the client gives up — this is harmless
 * because gRPC Java's default server executor is a cached thread pool that allocates
 * a new thread for each concurrent request.
 */
class GrpcTimeoutTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(3000)  // 3s: far exceeds the 500ms deadline
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // single request triggers immediate dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchFillThreshold(1.0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(500);  // 500ms deadline — will time out
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(15)
    void grpcTimeout_requestFailsAndRecovers() throws Exception {
        // 1. Submit request — gRPC deadline fires at 500ms, mock is still sleeping
        CompletableFuture<Response> future = submitRequest(10001);
        Response response = future.get(5, TimeUnit.SECONDS);

        // 2. Verify: request failed with an explicit timeout result
        assertFalse(response.isSuccess(), "Request should fail when EnqueueBatch times out");
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode(),
                "Request should have BATCH_SLO_EXPIRED error code");

        // 3. Verify: mock prefill received the EnqueueBatch call (recorded before sleep)
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1,
                "Prefill worker should have received at least 1 EnqueueBatch call");

        // 4. Verify: timeout handling cleans up PrefillEndpoint inflight state
        InflightAssertions.assertPrefillInflightEmpty(getPrefillEndpoint());

        // 5. Verify: decode worker never received any request (PD-separated)
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");

        // 6. Recover: change behavior to normal delay
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());

        // 7. Submit a new request — should succeed on the same gRPC channel
        //    (deadline exceeded only cancels the specific call, not the channel)
        CompletableFuture<Response> future2 = submitRequest(10002);
        Response response2 = future2.get(5, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed after recovery");

        // 8. Cleanup: cancel the successful request to release inflight resources
        //    (mock workers don't send status updates, so we must clean up manually)
        cancelRequest(10002);

        // 9. Verify: all inflight resources released
        InflightAssertions.assertResourcesReleasedWithin(
                getPrefillEndpoint(), getDecodeEndpoint(), 5000);
    }
}

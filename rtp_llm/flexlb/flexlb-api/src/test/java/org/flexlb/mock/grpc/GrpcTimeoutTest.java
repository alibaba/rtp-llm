package org.flexlb.mock.grpc;

import org.flexlb.balance.scheduler.RequestLifecycleState;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.flexlb.mock.InflightAssertions.assertResourcesReleasedWithin;

/**
 * gRPC batchEnqueue timeout: mock prefill worker delays response beyond the
 * configured gRPC deadline, verifying that Master does not publish a false
 * failure after the request may already have reached the worker.
 *
 * <p>Flow:
 * 1. Configure mock prefill with enqueueDelayMs=3000 (3s) and master deadline=500ms
 * 2. Submit request → dispatched → gRPC batchEnqueue times out at 500ms
 * 3. Verify: the future stays pending in DISPATCHING while the Engine fence is unresolved,
 *    and the mock prefill did receive EnqueueBatch
 * 4. Recover: change behavior to delay=0, submit a different request → succeeds
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>gRPC client sets {@code withDeadlineAfter(deadlineMs)} on the blocking stub</li>
 *   <li>When the deadline fires, the blocking call throws {@code StatusRuntimeException}
 *       with status DEADLINE_EXCEEDED</li>
 *   <li>{@link org.flexlb.balance.scheduler.DefaultBatchDispatcher} reports the post-invocation
 *       failure as dispatch-uncertain</li>
 *   <li>The scheduler retains both ledgers until Cancel/WorkerStatus proves ownership</li>
 * </ul>
 *
 * <p>Note: The mock's {@code enqueueBatch} records the request <em>before</em> sleeping,
 * so the test can verify the mock received the call even though the client timed out.
 * The server thread continues sleeping after the client gives up — this is harmless
 * because gRPC Java's default server executor is a cached thread pool that allocates
 * a new thread for each concurrent request.
 */
class GrpcTimeoutTest extends FlexLBMockTestBase {

    private final CompletableFuture<EngineCancelChannel.CancelOutcome> dispatchFence =
            new CompletableFuture<>();

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(3000)  // 3s: far exceeds the 500ms deadline
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchSizeMax(1);        // single request triggers immediate dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(500);  // 500ms deadline — will time out
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Override
    protected EngineCancelChannel createEngineCancelChannel() {
        return new EngineCancelChannel() {
            @Override
            public boolean isSupported(org.flexlb.balance.endpoint.DecodeEndpoint endpoint) {
                return true;
            }

            @Override
            public CompletableFuture<CancelOutcome> cancel(
                    CancelTarget target, long requestId, long timeoutMs) {
                return dispatchFence;
            }
        };
    }

    @Test
    @Timeout(15)
    void grpcTimeout_retainsOwnershipUntilFenceAndChannelRecovers() throws Exception {
        // 1. Submit request — gRPC deadline fires at 500ms, mock is still sleeping.
        CompletableFuture<Response> future = submitRequest(10001);

        // A transport deadline cannot prove non-acceptance. Publishing an immediate retryable
        // failure here could duplicate a request that the worker already owns.
        assertThrows(TimeoutException.class,
                () -> future.get(1500, TimeUnit.MILLISECONDS));
        assertFalse(future.isDone(), "dispatch must remain fenced while ownership is unresolved");
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(10001L, 0).state());
        assertEquals(1, getPrefillEndpoint().getInflightBatchCount(),
                "ambiguous dispatch must retain its Prefill batch ledger");
        assertEquals(1, getDecodeEndpoint().getInflightCount(),
                "ambiguous dispatch must retain its Decode reservation");

        // 2. The mock records the request before sleeping, proving the ambiguity is real.
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1,
                "Prefill worker should have received at least 1 EnqueueBatch call");

        // 3. A controlled engine tombstone proves non-ownership and settles both ledgers together.
        dispatchFence.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        Response timedOut = future.get(5, TimeUnit.SECONDS);
        assertFalse(timedOut.isSuccess());
        assertResourcesReleasedWithin(getPrefillEndpoint(), getDecodeEndpoint(), 5_000);

        // 4. A separate request can still use the same channel after the per-call deadline.
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        CompletableFuture<Response> future2 = submitRequest(10002);
        Response response2 = future2.get(5, TimeUnit.SECONDS);
        assertTrue(response2.isSuccess(), "Subsequent request should succeed after recovery");
        assertEquals(0, mockDecodeWorker.getEnqueueCount(),
                "Decode worker should not have received any request");
    }
}

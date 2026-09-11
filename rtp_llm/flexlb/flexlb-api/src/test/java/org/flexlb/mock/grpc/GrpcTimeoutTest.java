package org.flexlb.mock.grpc;

import org.flexlb.config.DispatcherConfig;
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
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * gRPC batchEnqueue timeout: mock prefill worker delays response beyond the
 * configured gRPC deadline, verifying that master correctly handles the
 * timeout (DEADLINE_EXCEEDED) and retains an Engine ownership fence.
 *
 * <p>Flow:
 * 1. Configure mock prefill with enqueueDelayMs=3000 (3s) and master deadline=500ms
 * 2. Submit request → dispatched → gRPC batchEnqueue times out at 500ms
 * 3. Verify: the frontend future remains pending and inflight ownership is retained,
 *    because the mock prefill received EnqueueBatch but its ACK was lost
 * 4. Recover: change behavior to delay=0, submit new request → succeeds
 *
 * <p>Key mechanism:
 * <ul>
 *   <li>gRPC client sets {@code withDeadlineAfter(deadlineMs)} on the blocking stub</li>
 *   <li>When the deadline fires, the blocking call throws {@code StatusRuntimeException}
 *       with status DEADLINE_EXCEEDED</li>
 *   <li>{@link org.flexlb.balance.scheduler.DefaultBatchDispatcher} catches this in its
 *       {@code catch (Throwable)} block and calls {@code onTimeout()}</li>
 *   <li>The scheduler cannot classify a post-send timeout as a definite rejection;
 *       it retains the request-scoped Engine fence until authoritative status arrives</li>
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
        FlexlbConfig cfg = super.createConfig();
        DispatcherConfig dispatcher = assertInstanceOf(
                DispatcherConfig.class, cfg.getDispatcher());
        dispatcher.setEnqueueRpcTimeoutMs(500); // will time out
        return cfg;
    }

    @Test
    @Timeout(15)
    void grpcTimeout_requestFailsAndRecovers() throws Exception {
        // 1. Submit request — gRPC deadline fires at 500ms, mock is still sleeping
        CompletableFuture<Response> future = submitRequest(10001);

        // 2. A post-send timeout is ambiguous: no terminal response may be published
        // until an authoritative Engine status settles ownership.
        assertThrows(TimeoutException.class,
                () -> future.get(1, TimeUnit.SECONDS));
        assertFalse(future.isDone(), "lost ACK must retain the Engine ownership fence");

        // 3. Verify: mock prefill received the EnqueueBatch call (recorded before sleep)
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1,
                "Prefill worker should have received at least 1 EnqueueBatch call");

        // 4. The uncertain request remains charged instead of being unsafely rolled back.
        assertEquals(1, getPrefillEndpoint().getInflightBatchCount());

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
        assertFalse(future.isDone(), "recovery traffic cannot settle the earlier lost ACK");

    }
}

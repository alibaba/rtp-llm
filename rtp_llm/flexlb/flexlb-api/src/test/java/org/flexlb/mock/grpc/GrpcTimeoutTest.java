package org.flexlb.mock.grpc;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.GrpcEngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.DispatcherConfig;
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

    private volatile boolean engineCancelSupported;

    @Override
    protected EngineCancelChannel createEngineCancelChannel() {
        EngineCancelChannel realChannel = new GrpcEngineCancelChannel(grpcClient);
        return new EngineCancelChannel() {
            @Override
            public boolean isSupported(DecodeEndpoint endpoint) {
                return engineCancelSupported;
            }

            @Override
            public CompletableFuture<CancelAck> cancel(
                    CancelTarget target, long requestId, long timeoutMs) {
                return engineCancelSupported
                        ? realChannel.cancel(target, requestId, timeoutMs)
                        : CompletableFuture.completedFuture(CancelAck.UNSUPPORTED);
            }
        };
    }

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
    void grpcTimeout_withoutCancelProofRetainsOwnershipAndAllowsRecovery() throws Exception {
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
        assertFalse(mockPrefillWorker.getRpcService().isCancelled(10001L));
        assertTrue(scheduler.ownsRequestGeneration(10001L));

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

    @Test
    @Timeout(15)
    void grpcTimeout_requestFailsAndRecovers() throws Exception {
        engineCancelSupported = true;
        Response response = submitRequest(10001).get(5, TimeUnit.SECONDS);

        assertFalse(response.isSuccess(), "Request should fail when EnqueueBatch times out");
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(RequestState.Phase.TIMED_OUT,
                scheduler.getRequestState(10001L, 0).state());
        assertTrue(mockPrefillWorker.getEnqueueCount() >= 1);
        assertTrue(mockPrefillWorker.getRpcService().isCancelled(10001L),
                "The engine must fence late enqueue before timeout cleanup");
        InflightAssertions.assertPrefillInflightEmpty(getPrefillEndpoint());
        assertEquals(0, mockDecodeWorker.getEnqueueCount());

        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        assertTrue(submitRequest(10002).get(5, TimeUnit.SECONDS).isSuccess(),
                "Subsequent request should succeed after recovery");
    }
}

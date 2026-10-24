package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Real request owner and Decode ledger: releasing an unsent permit cannot expose a committed route to withdrawal. */
@Timeout(15)
class RejectedDeliveryWithdrawalRaceTest {
    @Test
    void rejectedCommittedBatchTerminalizesOwnerBeforeCapacityListenerCanClaimWithdrawal() throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(TimeUnit.HOURS.toMillis(1));
        var service = SchedulingTestConfig.configService(config);
        var reporter = mock(DeliveryMetricsReporter.class);
        var owner = SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        var grpc = mock(EngineGrpcClient.class);
        var dispatcher = new DeliveryDispatcher(grpc, service, reporter);
        ReflectionTestUtils.setField(owner.runtime, "dispatcher", dispatcher);
        var execution = org.mockito.Mockito.spy(owner.runtime.deliveries());
        ReflectionTestUtils.setField(owner.runtime, "deliveries", execution);
        doReturn(701L).when(execution).nextBatchId();
        var ledger = new DeliverySettlementTestSupport();
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.reserveBatch(any(), anyLong(), anyInt())).thenAnswer(call ->
                ledger.reserveBatch(call.getArgument(0), call.getArgument(1), call.getArgument(2)));
        when(prefill.release(org.mockito.ArgumentMatchers.isNull(), any())).thenAnswer(call -> EndpointTestSupport.releaseRequest(ledger.prefill, call.getArgument(1)));
        when(prefill.commitLocked(any())).thenAnswer(call -> ledger.prefill.commitLocked(call.getArgument(0), null));
        doAnswer(call -> {
            var released = ledger.prefill.release(call.getArgument(0), null);
            if (released.generationHandoff() != null) { ((AutoCloseable) released.generationHandoff()).close(); }
            return released.released();
        }).when(prefill).release(any(org.flexlb.balance.endpoint.PrefillState.Reservation.class), isNull());
        DecodeEndpoint decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8080, 8081, null), SchedulerTestSupport.repository(owner));
        var worker = new WorkerBatcher("rejected-withdrawal", prefill, QueueExecutionSettings.capture(config),
                dispatcher, ledger.prefill);
        ReflectionTestUtils.setField(prefill, "batcher", worker);
        ReflectionTestUtils.setField(prefill, "ownership", ledger.prefill);

        var context = RequestProtocolTestSupport.context(config, 1L);
        context.data().setGenerateInputPb(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1L)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()).build().toByteString());
        var future = RequestProtocolTestSupport.register(owner, context);
        DecodeResources.ReservationHandle handle;
        try (var pin = decode.tryPinGeneration()) {
            handle = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 1L, 1L, 2L, 50);
        }
        assertNotNull(handle);
        DeliverySettlementTestSupport.queueDecode(decode, handle);
        var status = new ServerStatus();
        status.setRole(RoleType.PREFILL);
        status.setServerIp("127.0.0.1");
        status.setHttpPort(8080);
        status.setGrpcPort(8090);
        context.data().setFuture(future);
        var item = SchedulingTestConfig.createRoute(context, new Response(), status, null,
                prefill, decode, handle, System.currentTimeMillis());
        RequestProtocolTestSupport.bind(owner, new RequestProtocolTestSupport.Registered(item, future));
        ledger.enqueue(item);

        var attempts = new AtomicInteger();
        var invalidWithdrawal = new AtomicReference<AdmissionHandle>();
        Runnable preemptionContender = () -> {
            attempts.incrementAndGet();
            // This is synchronous on the permit/resource release path, so no timing or sleeps are needed.
            var withdrawal = ((QueuedRequestScheduler) owner).claimQueuedRoute(decode, handle, Integer.MAX_VALUE);
            if (withdrawal != null) { invalidWithdrawal.compareAndSet(null, withdrawal); }
        };
        Runnable stopWatching = decode.watchEngineDispatchCapacity(preemptionContender);
        var actualExecutor = (ThreadPoolExecutor) ReflectionTestUtils.getField(owner.runtime.deliveries(), "dispatchExecutor");
        var rejectingExecutor = mock(ThreadPoolExecutor.class);
        var rejection = new RejectedExecutionException("executor rejected for withdrawal race");
        doThrow(rejection).when(rejectingExecutor).execute(any());
        try {
            ReflectionTestUtils.setField(owner.runtime.deliveries(), "dispatchExecutor", rejectingExecutor);
            try {
                assertNull(worker.processSelection(List.of(item), "withdrawal-race", new FormulaPredictor("100"),
                        OptionalLong.empty()));
            } finally { ReflectionTestUtils.setField(owner.runtime.deliveries(), "dispatchExecutor", actualExecutor); }

            assertTrue(attempts.get() > 0, "the real Decode capacity listener must exercise the competing claim");
            assertNull(invalidWithdrawal.get(), "committed unsent route became withdrawable before owner failure was recorded");
            var response = future.get(5, TimeUnit.SECONDS);
            assertFalse(response.isSuccess());
            assertTrue(response.getErrorMessage().contains("executor rejected for withdrawal race"));
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            assertEquals(0, ledger.prefill.stats().batchCount());
            assertEquals(0, ledger.prefill.stats().locallyOwnedRequests());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, decode.routingView().engineCapacityUsed());
            assertEquals(0, decode.routingView().inflightHardKv());
            assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            verify(owner, times(1)).onDeliveryEvidence(org.mockito.ArgumentMatchers.isNull(),
                    org.mockito.ArgumentMatchers.argThat(evidence -> evidence instanceof ContextResult.PreparationFailure failure
                            && failure.route() == item && failure.detail().contains(rejection.getMessage())));
            verify(owner, never()).claimDelivery(any(), any(), anyLong(), any());
            verifyNoInteractions(grpc);
        } finally {
            ReflectionTestUtils.setField(owner.runtime.deliveries(), "dispatchExecutor", actualExecutor);
            stopWatching.run();
            var outstanding = invalidWithdrawal.get();
            if (outstanding != null) { outstanding.finish(); }
            owner.runtime.shutdown();
            decode.close();
            decode.awaitRetirement();
        }
    }
}

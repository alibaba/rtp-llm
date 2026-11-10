package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.TheoryCacheHitReporter;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Real request owner and Decode ledger: releasing an unsent permit cannot expose a committed route to withdrawal.
 */
@Timeout(15)
class RejectedDeliveryWithdrawalRaceTest {
    @Test
    void rejectedCommittedBatchTerminalizesOwnerBeforeCapacityListenerCanClaimWithdrawal()
            throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(TimeUnit.HOURS.toMillis(1));
        var service = SchedulingTestConfig.configService(config);
        var reporter = mock(DeliveryMetricsReporter.class);
        var owner =
                SchedulerTestSupport.create(
                        service,
                        reporter,
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var grpc = mock(EngineGrpcClient.class);
        var dispatcher =
                org.mockito.Mockito.spy(
                        new DeliveryCoordinator(
                                new org.flexlb.balance.scheduler.RequestRepository(),
                                grpc,
                                service,
                                reporter,
                                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                        .decodeAdmissionCapacity(service),
                                org.mockito.Mockito.mock(
                                        org.flexlb.balance.scheduler.DeliveryExecution.class)));
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner),
                "dispatcher",
                dispatcher);
        var execution =
                org.mockito.Mockito.spy(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner)
                                .deliveries());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner),
                "deliveries",
                execution);
        doReturn(701L).when(execution).nextBatchId();
        var ledger = new DeliverySettlementTestSupport();
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.acceptsNewRequests()).thenReturn(true);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.getHttpPort()).thenReturn(8080);
        when(prefill.batchCapacityAvailable(anyInt()))
                .thenAnswer(call -> ledger.prefill.batchCapacityAvailable(call.getArgument(0)));
        when(prefill.releaseRequest(any()))
                .thenAnswer(
                        call ->
                                EndpointTestSupport.releaseRequest(
                                        ledger.prefill, call.getArgument(0)));
        when(prefill.dequeueForDelivery(any()))
                .thenAnswer(call -> ledger.prefill.dequeueForDelivery(call.getArgument(0)));

        DecodeEndpoint decode =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        SchedulerTestSupport.repository(owner));
        var worker =
                org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                        org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                "rejected-withdrawal",
                                prefill,
                                QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                config)),
                                dispatcher,
                                ledger.prefill,
                                SchedulerTestSupport.queueDependencies(
                                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                                owner))));
        org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(prefill, worker);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, ledger.prefill);

        var context = RequestProtocolTestSupport.context(config, 1L);
        context.data()
                .setGenerateInputPb(
                        EngineRpcService.GenerateInputPB.newBuilder()
                                .setRequestId(1L)
                                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder())
                                .build()
                                .toByteString());
        var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
        DecodeResources.RequestReservation handle;
        {
            var selectedEndpoint = decode;

            handle = EndpointTestSupport.registerDecode(decode, 1L, 1L, 2L, 50);
        }
        assertNotNull(handle);
        DeliverySettlementTestSupport.queueDecode(decode, handle);
        context.data().setFuture(future);
        var item =
                RequestExecutionTestSupport.registerPrefillRoute(
                        context, prefill, decode, handle, 0L, System.currentTimeMillis());
        RequestProtocolTestSupport.bind(
                owner, new RequestProtocolTestSupport.Registered(item, future));
        ledger.enqueue(item);

        var attempts = new AtomicInteger();
        var invalidWithdrawal = new AtomicReference<RoutingContext>();
        Runnable preemptionContender =
                () -> {
                    attempts.incrementAndGet();
                    // This is synchronous on the permit/resource release path, so no timing or
                    // sleeps are needed.
                    var withdrawal =
                            SchedulerTestSupport.routePreemption(owner)
                                    .beginRoutePreemption(decode, handle, Integer.MAX_VALUE);
                    if (withdrawal != null) {
                        invalidWithdrawal.compareAndSet(null, withdrawal);
                    }
                };
        Runnable stopWatching = decode.watchEngineDispatchCapacity(preemptionContender);
        var actualExecutor =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner)
                                        .deliveries(),
                                "dispatchExecutor");
        var rejectingExecutor = mock(ThreadPoolExecutor.class);
        var rejection = new RejectedExecutionException("executor rejected for withdrawal race");
        doThrow(rejection).when(rejectingExecutor).execute(any());
        try {
            ReflectionTestUtils.setField(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner).deliveries(),
                    "dispatchExecutor",
                    rejectingExecutor);
            try {
                assertNull(
                        worker.processSelection(
                                List.of(item),
                                "withdrawal-race",
                                new FormulaPredictor("100"),
                                OptionalLong.empty()));
            } finally {
                ReflectionTestUtils.setField(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner)
                                .deliveries(),
                        "dispatchExecutor",
                        actualExecutor);
            }

            assertTrue(
                    attempts.get() > 0,
                    "the real Decode capacity listener must exercise the competing claim");
            assertNull(
                    invalidWithdrawal.get(),
                    "committed unsent route became withdrawable before owner failure was recorded");
            var response = future.get(5, TimeUnit.SECONDS);
            assertFalse(response.isSuccess());
            assertTrue(
                    response.getErrorMessage().contains("executor rejected for withdrawal race"));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
            assertEquals(0, ledger.prefill.stats().batchCount());
            assertEquals(0, ledger.prefill.stats().locallyOwnedRequests());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, decode.routingView().engineCapacityUsed());
            assertEquals(0, decode.routingView().inflightHardKv());
            assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            verify(owner, times(1))
                    .recordPreparationFailure(
                            org.mockito.ArgumentMatchers.argThat(
                                    evidence ->
                                            evidence
                                                            instanceof
                                                            ContextResult.PreparationFailure failure
                                                    && failure.route() == item
                                                    && failure.detail()
                                                            .contains(rejection.getMessage())));
            verify(
                            org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                    owner.dispatcher),
                            never())
                    .commit(
                            any(Delivery.Member.class),
                            any(),
                            anyLong(),
                            org.mockito.ArgumentMatchers.isNull());
            verifyNoInteractions(grpc);
        } finally {
            ReflectionTestUtils.setField(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner).deliveries(),
                    "dispatchExecutor",
                    actualExecutor);
            stopWatching.run();
            var outstanding = invalidWithdrawal.get();
            if (outstanding != null) {
                AbstractRequestScheduler.finishRouting(outstanding);
            }
            SchedulerTestSupport.shutdown(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
            decode.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(decode);
        }
    }
}

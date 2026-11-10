package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks that remote-stop waits cannot starve local work and that shutdown drains accepted stop continuations.
 */
@Timeout(15)
class StopExecutorSafetyTest {

    @Test
    void thirtyOutstandingRpcsDoNotOccupyEightStopWorkers() throws Exception {
        var runtime = runtime();
        var cleanup =
                new DeliveryCoordinator(
                        new org.flexlb.balance.scheduler.RequestRepository(),
                        runtime.engineClient,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new org.flexlb.service.monitor.DeliveryMetricsReporter(null),
                        new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                .decodeAdmissionCapacity(
                                        DeliveryTestSupport.service(
                                                SchedulingTestConfig.batchConfig())),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.scheduler.DeliveryExecution.class));
        var client = runtime.engineClient;
        var rpcReplies =
                new java.util.concurrent.CopyOnWriteArrayList<
                        CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>>();
        var entered = new CountDownLatch(30);
        var answers = new java.util.ArrayList<CompletableFuture<?>>();
        when(client.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenAnswer(
                        call -> {
                            var reply =
                                    new CompletableFuture<
                                            org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>();
                            rpcReplies.add(reply);
                            entered.countDown();
                            return reply;
                        });
        try {
            for (int i = 0; i < 30; i++) {
                var context = context();
                org.flexlb.balance.scheduler.SchedulerTestServices.bind(
                        context.data().getRequestScheduler(), runtime);
                when(context.snapshot().delivery()).thenReturn(null);
                when(context.snapshot().cancellation()).thenReturn(null);
                var claim =
                        new PreemptionRegistration(
                                context,
                                i,
                                new org.flexlb.balance.endpoint.DecodeResources.RequestReservation(
                                        1, String.valueOf(i + 1), i + 1),
                                new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
                answers.add(
                        RequestProtocolTestSupport.cancelForPreemption(
                                cleanup, claim, null, "priority preemption", 500L));
            }
            assertTrue(
                    entered.await(2, TimeUnit.SECONDS),
                    "unanswered RPCs must not retain worker threads");
            assertEquals(30, rpcReplies.size());
        } finally {
            rpcReplies.forEach(
                    reply ->
                            reply.complete(
                                    org.flexlb.engine.grpc.EngineGrpcClient.CancelReply
                                            .REQUEST_CLEANED));
            CompletableFuture.allOf(answers.toArray(CompletableFuture[]::new))
                    .get(3, TimeUnit.SECONDS);
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    private RequestContext context() {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        when(context.data().getRequestId()).thenReturn("41");
        return context;
    }

    private SchedulerTestServices runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerTestServices(
                new RequestRepository(),
                mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                new DeliveryCoordinator(
                        new org.flexlb.balance.scheduler.RequestRepository(),
                        null,
                        config,
                        new org.flexlb.service.monitor.DeliveryMetricsReporter(null),
                        new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                .decodeAdmissionCapacity(config),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.scheduler.DeliveryExecution.class)),
                config,
                mock(org.flexlb.engine.grpc.EngineGrpcClient.class));
    }
}

package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Timeout(15)
class StopExecutorSafetyTest {
    @Test void saturatedStopPoolDoesNotOccupyNormalRequestWorkers() throws Exception {
        var runtime = runtime();
        var release = new CountDownLatch(1);
        int size = ((ThreadPoolExecutor) ReflectionTestUtils.getField(runtime, "stopWorkers")).getMaximumPoolSize();
        var entered = new CountDownLatch(size);
        var normal = new CountDownLatch(1);
        try {
            for (int i = 0; i < size; i++) {
                runtime.executeStopWork(context(), () -> {
                    entered.countDown();
                    RequestProtocolTestSupport.await(release);
                });
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            runtime.executeRequestWork(context(), normal::countDown);
            assertTrue(normal.await(1, TimeUnit.SECONDS));
        } finally { release.countDown(); runtime.shutdown(); }
    }

    @Test void normalPoolSaturationDoesNotPreventStopProgress() throws Exception {
        var runtime = runtime();
        var release = new CountDownLatch(1);
        var workers = (ThreadPoolExecutor) SchedulerTestSupport.requestWorkers(runtime);
        var entered = new CountDownLatch(workers.getMaximumPoolSize());
        var stop = new CountDownLatch(1);
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++) {
                runtime.executeRequestWork(context(), () -> {
                    entered.countDown(); RequestProtocolTestSupport.await(release);
                });
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            runtime.executeStopWork(context(), stop::countDown);
            assertTrue(stop.await(1, TimeUnit.SECONDS));
        } finally { release.countDown(); runtime.shutdown(); }
    }

    @Test void shutdownWaitsForStopToRequestToStopHandoffWithoutClosingEitherPoolEarly() throws Exception {
        var runtime = runtime();
        var context = context();
        var entered = new CountDownLatch(1);
        var handoff = new CountDownLatch(1);
        var finalEntered = new CountDownLatch(1);
        var releaseFinal = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        var closer = new Thread(() -> {
            try { SchedulerTestSupport.closeRequestWork(runtime); }
            catch (Throwable e) { failure.set(e); }
        });
        try {
            runtime.executeStopWork(context, () -> {
                entered.countDown(); RequestProtocolTestSupport.await(handoff);
                runtime.executeRequestWork(context, () -> runtime.executeStopWork(context, () -> {
                    finalEntered.countDown(); RequestProtocolTestSupport.await(releaseFinal);
                }));
            });
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            closer.start();
            RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
            handoff.countDown();
            assertTrue(finalEntered.await(3, TimeUnit.SECONDS));
            assertTrue(closer.isAlive());
            releaseFinal.countDown(); closer.join(3000);
            assertFalse(closer.isAlive()); assertNull(failure.get());
        } finally {
            handoff.countDown(); releaseFinal.countDown(); closer.join(3000); runtime.shutdown();
        }
    }

    @Test void stopCallbackCannotSynchronouslyShutDownItsOwnRuntime() throws Exception {
        var runtime = runtime();
        var result = new CompletableFuture<Boolean>();
        try {
            runtime.executeStopWork(context(), () -> {
                try { runtime.shutdown(); result.complete(false); }
                catch (IllegalStateException expected) { result.complete(runtime.isAccepting()); }
            });
            assertTrue(result.get(3, TimeUnit.SECONDS));
        } finally { runtime.shutdown(); }
    }

    @Test void rejectedStopSubmissionRecoversOffCallerAndRemainsTracked() throws Exception {
        var runtime = runtime();
        var caller = Thread.currentThread();
        var worker = new CompletableFuture<Thread>();
        ((ExecutorService) ReflectionTestUtils.getField(runtime, "stopWorkers")).shutdown();
        try {
            runtime.executeStopWork(context(), () -> worker.complete(Thread.currentThread()));
            assertNotSame(caller, worker.get(3, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(runtime);
        } finally { runtime.shutdown(); }
    }

    @Test void thirtyOutstandingRpcsDoNotOccupyEightStopWorkers() throws Exception {
        var runtime = runtime();
        var cleanup = new DeliveryCleanup();
        var client = mock(org.flexlb.engine.grpc.EngineGrpcClient.class);
        var rpcReplies = new java.util.concurrent.CopyOnWriteArrayList<CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>>();
        var entered = new CountDownLatch(30);
        var answers = new java.util.ArrayList<CompletableFuture<?>>();
        when(client.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> {
            var reply = new CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>();
            rpcReplies.add(reply); entered.countDown(); return reply;
        });
        try {
            for (int i = 0; i < 30; i++) {
                var context = context();
                when(context.snapshot().delivery()).thenReturn(null);
                when(context.snapshot().cancellation()).thenReturn(null);
                var claim = new PreemptionRegistration(context, i,
                        new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(1, i + 1, i + 1),
                        new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
                answers.add(cleanup.preempt(claim, client, runtime, 500L));
            }
            assertTrue(entered.await(2, TimeUnit.SECONDS), "unanswered RPCs must not retain worker threads");
            assertEquals(30, rpcReplies.size());
        } finally {
            rpcReplies.forEach(reply -> reply.complete(org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            CompletableFuture.allOf(answers.toArray(CompletableFuture[]::new)).get(3, TimeUnit.SECONDS);
            runtime.shutdown();
        }
    }

    private RequestContext context() { return mock(RequestContext.class, RETURNS_DEEP_STUBS); }
    private SchedulerRuntime runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(new RequestRepository(), mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                new DeliveryDispatcher(null, config, new org.flexlb.service.monitor.DeliveryMetricsReporter(null)),
                config, mock(org.flexlb.engine.grpc.EngineGrpcClient.class));
    }
}

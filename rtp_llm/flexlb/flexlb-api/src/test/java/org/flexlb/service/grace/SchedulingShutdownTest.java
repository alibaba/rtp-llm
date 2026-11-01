package org.flexlb.service.grace;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.*;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.httpserver.FlexlbGrpcServer;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.mock.env.MockEnvironment;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.*;

/** Exercises the production application shutdown entry, rather than a test runtime's teardown. */
class SchedulingShutdownTest {
    static void registerComponents(AnnotationConfigApplicationContext context) {
        var scheduler = mock(RequestScheduler.class);
        when(scheduler.lifecycle()).thenReturn(mock(RequestLifecycle.class));
        when(scheduler.awaitSchedulingCompletion(anyLong())).thenReturn(true);
        context.registerBean(RequestScheduler.class, () -> scheduler);
        context.registerBean(RequestRepository.class, RequestRepository::new);
        context.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
        context.registerBean(DeliveryExecution.class, () -> mock(DeliveryExecution.class));
        context.registerBean(RequestTaskPool.class, () -> mock(RequestTaskPool.class));
        context.registerBean(ExpirationTimer.class, () -> mock(ExpirationTimer.class));
        context.registerBean(
                ResponseCompletionExecutor.class, () -> mock(ResponseCompletionExecutor.class));
        context.registerBean(ScheduledTaskTimer.class, () -> mock(ScheduledTaskTimer.class));
        context.registerBean(DeliveryDispatcher.class, () -> mock(DeliveryDispatcher.class));
    }

    @Test
    void stopsSchedulingBeforeSettlementAndClosesAllExecutorsAfterSettlement() {
        var f = new Fixture(mock(RequestTaskPool.class));
        assertTrue(f.application.offline());
        assertTrue(f.requests.isClosed());
        var order =
                inOrder(
                        f.grpc,
                        f.scheduler,
                        f.owner,
                        f.deliveries,
                        f.endpoints,
                        f.deadlines,
                        f.tasks,
                        f.dispatcher,
                        f.completions,
                        f.timer);
        order.verify(f.grpc).drain();
        order.verify(f.scheduler).awaitSchedulingCompletion(anyLong());
        order.verify(f.scheduler).close();
        order.verify(f.owner).stopAdmissionsAndAwaitCompletion();
        order.verify(f.deliveries).awaitDeliveries();
        order.verify(f.owner).awaitDeliveryCleanup(true);
        order.verify(f.endpoints).close();
        order.verify(f.owner).awaitDeliveryCleanup(false);
        order.verify(f.deadlines).close();
        order.verify(f.tasks).awaitIdle();
        order.verify(f.owner).closeOutstandingAndTerminalize();
        order.verify(f.dispatcher).awaitStops();
        order.verify(f.tasks).close();
        order.verify(f.completions).close();
        order.verify(f.timer).close();
        order.verify(f.tasks).throwIfFailed();
        assertTrue(f.application.offline());
        verify(f.scheduler).close();
    }

    @Test
    void cleanupFailureStillClosesExecutorsAndIsReported() {
        var f = new Fixture(mock(RequestTaskPool.class));
        var first = new IllegalStateException("endpoint release failed");
        var second = new IllegalArgumentException("completion close failed");
        doThrow(first).when(f.endpoints).close();
        doThrow(second).when(f.completions).close();
        assertSame(first, assertThrows(IllegalStateException.class, f.application::offline));
        assertArrayEquals(new Throwable[] {second}, first.getSuppressed());
        verify(f.tasks).close();
        verify(f.timer).close();
        verify(f.tasks).throwIfFailed();
    }

    @Test
    void waitsForRealRequestWorkBeforeFinalSettlement() throws Exception {
        var tasks = new RequestTaskPool();
        var f = new Fixture(tasks);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        tasks.execute(
                mock(RequestContext.class),
                () -> {
                    entered.countDown();
                    try {
                        release.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
        assertTrue(entered.await(2, TimeUnit.SECONDS));
        var shutdown = CompletableFuture.supplyAsync(f.application::offline);
        try {
            assertThrows(TimeoutException.class, () -> shutdown.get(100, TimeUnit.MILLISECONDS));
            verify(f.owner, never()).closeOutstandingAndTerminalize();
            verify(f.completions, never()).close();
        } finally {
            release.countDown();
            assertTrue(shutdown.get(3, TimeUnit.SECONDS));
            tasks.close();
        }
        verify(f.owner).closeOutstandingAndTerminalize();
        verify(f.completions).close();
    }

    @Test
    void rejectsShutdownFromAnOwnedTaskBeforeClosingAdmission() throws Exception {
        var tasks = new RequestTaskPool();
        var f = new Fixture(tasks);
        var result = new CompletableFuture<Throwable>();
        try {
            tasks.execute(
                    mock(RequestContext.class),
                    () -> {
                        try {
                            f.application.offline();
                            result.complete(null);
                        } catch (Throwable failure) {
                            result.complete(failure);
                        }
                    });
            assertInstanceOf(IllegalStateException.class, result.get(2, TimeUnit.SECONDS));
            assertFalse(f.requests.isStopping());
            verify(f.scheduler, never()).close();
            assertTrue(f.application.offline());
        } finally {
            tasks.close();
        }
    }

    private static class Fixture {
        final RequestRepository requests = new RequestRepository();
        final RequestScheduler scheduler = mock(RequestScheduler.class);
        final RequestLifecycle owner = mock(RequestLifecycle.class);
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);
        final DeliveryExecution deliveries = mock(DeliveryExecution.class);
        final RequestTaskPool tasks;
        final ExpirationTimer deadlines = mock(ExpirationTimer.class);
        final ResponseCompletionExecutor completions = mock(ResponseCompletionExecutor.class);
        final ScheduledTaskTimer timer = mock(ScheduledTaskTimer.class);
        final DeliveryDispatcher dispatcher = mock(DeliveryDispatcher.class);
        final FlexlbGrpcServer grpc = mock(FlexlbGrpcServer.class);
        final ApplicationLifecycle application;

        Fixture(RequestTaskPool tasks) {
            this.tasks = tasks;
            when(scheduler.lifecycle()).thenReturn(owner);
            when(scheduler.awaitSchedulingCompletion(anyLong())).thenReturn(true);
            application =
                    new ApplicationLifecycle(
                            mock(MasterStatusService.class),
                            grpc,
                            mock(GracefulLifecycleReporter.class),
                            new MockEnvironment(),
                            0L);
            application.configureScheduling(
                    scheduler,
                    requests,
                    endpoints,
                    deliveries,
                    tasks,
                    deadlines,
                    completions,
                    timer,
                    dispatcher);
        }
    }
}

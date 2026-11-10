package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/**
 * Verifies shared scheduling shutdown, including cleanup failures and real in-flight work.
 */
class SchedulingLifecycleTest {
    @Test
    void stopsSchedulingBeforeSettlementAndClosesAllExecutorsAfterSettlement() {
        var f = new Fixture(mock(RequestTaskPool.class));
        f.lifecycle.shutdown(System.nanoTime());
        assertTrue(f.requests.isClosed());
        var order =
                inOrder(
                        f.scheduler,
                        f.endpoints,
                        f.deadlines,
                        f.tasks,
                        f.dispatcher,
                        f.completions);
        order.verify(f.scheduler).close();
        order.verify(f.dispatcher).shutdown(org.mockito.Mockito.anyList());
        order.verify(f.tasks).awaitIdle();
        order.verify(f.endpoints).close();
        order.verify(f.deadlines).close();
        order.verify(f.tasks).awaitIdle();
        order.verify(f.scheduler).closeOutstandingAndTerminalize();
        order.verify(f.dispatcher).awaitStops();
        order.verify(f.tasks).close();
        order.verify(f.completions).close();
        order.verify(f.tasks).throwIfFailed();
        f.lifecycle.shutdown(System.nanoTime());
        verify(f.scheduler).close();
    }

    @Test
    void cleanupFailureStillClosesExecutorsAndIsReported() {
        var f = new Fixture(mock(RequestTaskPool.class));
        var first = new IllegalStateException("endpoint release failed");
        var second = new IllegalArgumentException("completion close failed");
        doThrow(first).when(f.endpoints).close();
        doThrow(second).when(f.completions).close();
        assertSame(
                first,
                assertThrows(
                        IllegalStateException.class,
                        () -> f.lifecycle.shutdown(System.nanoTime())));
        assertArrayEquals(new Throwable[] {second}, first.getSuppressed());
        verify(f.tasks).close();
        verify(f.tasks).throwIfFailed();
    }

    @Test
    void waitsForRealRequestWorkBeforeClosingEndpoints() throws Exception {
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
        var shutdown = CompletableFuture.runAsync(() -> f.lifecycle.shutdown(System.nanoTime()));
        try {
            assertThrows(TimeoutException.class, () -> shutdown.get(100, TimeUnit.MILLISECONDS));
            verify(f.endpoints, never()).close();
            verify(f.scheduler, never()).closeOutstandingAndTerminalize();
            verify(f.completions, never()).close();
        } finally {
            release.countDown();
            shutdown.get(3, TimeUnit.SECONDS);
            tasks.close();
        }
        verify(f.scheduler).closeOutstandingAndTerminalize();
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
                            f.lifecycle.shutdown(System.nanoTime());
                            result.complete(null);
                        } catch (Throwable failure) {
                            result.complete(failure);
                        }
                    });
            assertInstanceOf(IllegalStateException.class, result.get(2, TimeUnit.SECONDS));
            assertFalse(f.requests.isStopping());
            verify(f.scheduler, never()).close();
            f.lifecycle.shutdown(System.nanoTime());
        } finally {
            tasks.close();
        }
    }

    /**
     * Combines a real request task pool with observed shutdown collaborators to check ordering and failure cleanup.
     */
    private static class Fixture {
        final RequestRepository requests = new RequestRepository();
        AbstractRequestScheduler scheduler;
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);
        final RequestTaskPool tasks;
        final ExpirationTimer deadlines = mock(ExpirationTimer.class);
        final ResponseCompletionExecutor completions = mock(ResponseCompletionExecutor.class);
        final DeliveryDispatcher dispatcher = mock(DeliveryDispatcher.class);
        final AbstractRequestScheduler lifecycle;

        Fixture(RequestTaskPool tasks) {
            this.tasks = tasks;
            scheduler =
                    org.mockito.Mockito.spy(
                            new SchedulerTestServices.PassiveScheduler(
                                    requests,
                                    tasks,
                                    deadlines,
                                    dispatcher,
                                    completions,
                                    mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                                    mock(org.flexlb.service.TheoryCacheHitReporter.class),
                                    endpoints));
            lifecycle = scheduler;
        }
    }
}

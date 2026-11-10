package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SchedulerExecutionTest {
    @Test
    void releasedRouteRestoresOneQueueEntryWithoutChangingItsOriginalSequence() {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = (OrderedRequestQueue) ReflectionTestUtils.getField(scheduler, "orderedQueue");
        var first = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 99010L);
        var second = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 99011L);
        try {
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, first);
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, second);
            scheduler.onRequest(first);
            scheduler.onRequest(second);
            long originalSequence = first.data().getPlacementSequence();
            assertEquals(2, queue.size());
            scheduler.onRequest(first);
            assertEquals(2, queue.size());
            assertEquals(originalSequence, first.data().getPlacementSequence());
            assertEquals(
                    List.of(first, second),
                    queue.scanForPlanningCandidates(2, 2, entry -> true).stream()
                            .map(GlobalQueueEntry::context)
                            .toList());
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    @Timeout(10)
    void changeDuringSchedulingRunsAgainWithoutAnotherExternalNotification() throws Exception {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = mock(PlacementQueue.class);
        when(queue.scanForPlanningCandidates(anyInt(), anyInt(), any())).thenReturn(List.of());
        when(queue.drain()).thenReturn(List.of());
        ReflectionTestUtils.setField(scheduler, "orderedQueue", queue);
        var endpoint = endpoint();
        var checks = new AtomicInteger();
        var resumed = new CountDownLatch(1);
        var runningThread = new AtomicReference<Thread>();
        when(queue.hasUnscannedRequests())
                .thenAnswer(
                        call -> {
                            runningThread.set(Thread.currentThread());
                            if (checks.incrementAndGet() == 1)
                                scheduler.onEndpointChanged(endpoint);
                            else resumed.countDown();
                            return false;
                        });
        try {
            scheduler.start();
            assertTrue(resumed.await(2, TimeUnit.SECONDS));
            assertTrue(runningThread.get().getName().equals("flexlb-scheduling"));
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle));
        }
        assertTrue(scheduler.awaitStopped(System.nanoTime()));
    }

    @Test
    @Timeout(10)
    void stoppedSchedulerDoesNotAcceptLateWakeupsOrRestart() throws Exception {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = mock(PlacementQueue.class);
        var calls = new AtomicInteger();
        var ran = new CountDownLatch(1);
        when(queue.scanForPlanningCandidates(anyInt(), anyInt(), any()))
                .thenAnswer(
                        call -> {
                            calls.incrementAndGet();
                            ran.countDown();
                            return List.of();
                        });
        when(queue.drain()).thenReturn(List.of());
        ReflectionTestUtils.setField(scheduler, "orderedQueue", queue);
        try {
            scheduler.start();
            assertTrue(ran.await(2, TimeUnit.SECONDS));
            scheduler.stop();
            assertTrue(scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(2)));
            int completed = calls.get();
            scheduler.onEndpointChanged(endpoint());
            scheduler.start();
            scheduler.schedule();
            assertEquals(completed, calls.get());
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    @Timeout(10)
    void failedSchedulingClosesWithoutWaitingForItsOwnThread() {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = mock(PlacementQueue.class);
        var failure = new IllegalStateException("queue scan failed");
        when(queue.scanForPlanningCandidates(anyInt(), anyInt(), any())).thenThrow(failure);
        when(queue.drain()).thenReturn(List.of());
        ReflectionTestUtils.setField(scheduler, "orderedQueue", queue);
        scheduler.start();
        assertTrue(scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(2)));
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalStateException.class,
                () -> SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle)));
        org.mockito.Mockito.verify(queue, org.mockito.Mockito.atLeastOnce()).drain();
    }

    private static AbstractRequestScheduler owner() {
        return SchedulerTestSupport.create(
                SchedulingTestConfig.configService(SchedulingTestConfig.batchConfig()),
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(TheoryCacheHitReporter.class));
    }

    private static WorkerEndpoint endpoint() {
        var endpoint = mock(WorkerEndpoint.class);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        return endpoint;
    }
}

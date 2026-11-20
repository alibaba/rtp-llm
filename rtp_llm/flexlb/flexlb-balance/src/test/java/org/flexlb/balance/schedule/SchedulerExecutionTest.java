package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.CapacityWaitQueue;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.WorkerAdmissionCapacity;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks how the scheduler selects tasks, restores requests and handles wakeups and shutdown.
 */
class SchedulerExecutionTest {
    @Test
    void releasedRouteRestoresOneQueueEntryWithoutChangingItsOriginalSequence() {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = (OrderedRequestQueue) ReflectionTestUtils.getField(scheduler, "orderedQueue");
        var first = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 99010L);
        var second = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 99011L);
        try {
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, first);
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, second);
            scheduler.scheduleRequest(first);
            scheduler.scheduleRequest(second);
            long originalSequence = first.getPlacementSequence();
            assertEquals(2, queue.size());
            scheduler.scheduleRequest(first);
            assertEquals(2, queue.size());
            assertEquals(originalSequence, first.getPlacementSequence());
            assertEquals(
                    List.of(first, second), queue.scanForPlanningCandidates(2, 2, entry -> true));
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    void capacityWaiterCannotUseAnOldSlotWhenItWasNotScheduled() {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        ReflectionTestUtils.setField(scheduler, "plannerCount", 1);
        var waiting =
                (CapacityWaitQueue) ReflectionTestUtils.getField(scheduler, "waitingRequests");
        var executing = (Map<?, ?>) ReflectionTestUtils.getField(scheduler, "executing");
        var checksToSchedule = new ArrayList<RequestPreemption.Work>();
        var first = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 99020L);
        var waiter = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 99021L);
        var worker = CapacityKey.exact(RoleType.PREFILL, "g", "worker-a");
        var endpoint = endpoint();
        var notifications =
                (CapacityNotifier) ReflectionTestUtils.getField(scheduler, "capacityNotifier");
        var router = mock(RequestRouteSelector.class);
        ReflectionTestUtils.setField(scheduler, "router", router);
        when(router.requestCapacity(endpoint))
                .thenReturn(
                        new WorkerAdmissionCapacity(worker, 1L, 1L, 1L),
                        new WorkerAdmissionCapacity(worker, 1L, 2L, 0L),
                        new WorkerAdmissionCapacity(worker, 1L, 3L, 1L));
        try {
            // Drive the planner directly without a periodic probe changing the capacity.
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, first);
            RequestProtocolTestSupport.submitForManualPlacement(lifecycle, waiter);
            scheduler.scheduleRequest(first);
            scheduler.scheduleRequest(waiter);
            waiting.park(waiter, worker, Set.of());
            notifications.notifyChanged(endpoint);

            ReflectionTestUtils.invokeMethod(scheduler, "chooseTasks", false, checksToSchedule);
            assertEquals(Set.of(first), executing.keySet(), "the FIFO head takes the only planner");

            notifications.notifyChanged(endpoint);
            scheduler.removeRequest(first);
            executing.remove(first);
            List<?> next =
                    ReflectionTestUtils.invokeMethod(
                            scheduler, "chooseTasks", false, checksToSchedule);
            assertEquals(List.of(), next, "an unused retry slot must not survive its withdrawal");

            notifications.notifyChanged(endpoint);
            ReflectionTestUtils.invokeMethod(scheduler, "chooseTasks", false, checksToSchedule);
            assertEquals(Set.of(waiter), executing.keySet(), "fresh capacity lets the waiter run");
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    @Timeout(10)
    void changeDuringSchedulingRunsAgainWithoutAnotherExternalNotification() throws Exception {
        var lifecycle = owner();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle);
        var queue = mock(SchedulingQueue.class);
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
                                ((org.flexlb.worker.endpoint.CapacityNotifier)
                                                ReflectionTestUtils.getField(
                                                        scheduler, "capacityNotifier"))
                                        .notifyChanged(endpoint);
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
        var queue = mock(SchedulingQueue.class);
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
            ((org.flexlb.worker.endpoint.CapacityNotifier)
                            ReflectionTestUtils.getField(scheduler, "capacityNotifier"))
                    .notifyChanged(endpoint());
            scheduler.start();
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
        var queue = mock(SchedulingQueue.class);
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
                SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(TheoryCacheHitReporter.class));
    }

    private static WorkerEndpoint endpoint() {
        var endpoint = mock(WorkerEndpoint.class);
        var status = createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        return endpoint;
    }
}

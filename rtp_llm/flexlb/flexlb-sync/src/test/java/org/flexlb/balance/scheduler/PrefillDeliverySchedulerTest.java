package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Holds executor tasks until explicitly run to exercise notification and task handoffs without sleeps.
 */
class PrefillDeliverySchedulerTest {
    @Test
    void repeatedNotificationsBeforeClaimProduceOneTask() throws Exception {
        var f = new Fixture();
        f.notifyChange();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(f.tasks.isEmpty());
        verify(f.queue).scheduleOnce();
    }

    @Test
    void notificationsAfterClaimBeforeExecutionAreRetainedAndMerged() throws Exception {
        var f = new Fixture();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.notifyChange();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size(), "no second task while the first is claimed");
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(f.tasks.isEmpty());
        verify(f.queue, times(2)).scheduleOnce();
    }

    @Test
    void notificationDuringFinalCapacityCheckSurvivesTaskCompletion() throws Exception {
        var f = new Fixture();
        when(f.queue.needsImmediateScheduling(any(), anyLong()))
                .thenAnswer(
                        call -> {
                            f.notifyChange();
                            return false;
                        })
                .thenReturn(false);
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(f.tasks.isEmpty());
    }

    @Test
    void notificationAfterCompletionStartsANewTask() throws Exception {
        var f = new Fixture();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size());
    }

    @Test
    void remainingWorkSchedulesAnotherTurnWithoutNotification() throws Exception {
        var f = new Fixture();
        when(f.queue.needsImmediateScheduling(any(), anyLong())).thenReturn(true, false);
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(f.tasks.isEmpty());
        verify(f.queue, times(2)).scheduleOnce();
    }

    @Test
    void timerAndNotificationShareOnePendingTask() throws Exception {
        var f = new Fixture();
        var head = mock(RouteResult.class);
        when(head.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        when(f.queue.scheduleOnce())
                .thenReturn(
                        PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                                head, System.currentTimeMillis() + 60000, "collection"));
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        long deadline = f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(deadline != Long.MAX_VALUE);
        assertTrue(f.tasks.isEmpty());
        f.scheduler.scheduleAt(deadline);
        f.notifyChange();
        f.scheduler.scheduleAt(deadline);
        assertEquals(1, f.tasks.size());
        f.runTask();
        f.scheduler.scheduleAt(System.nanoTime());
        assertEquals(1, f.tasks.size(), "notification after timer claim requires one more turn");
    }

    @Test
    void notificationReplacesTimerWithoutLeavingDuplicateDeadline() throws Exception {
        var f = new Fixture();
        var head = mock(RouteResult.class);
        when(head.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        when(f.queue.scheduleOnce())
                .thenReturn(
                        PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                                head, System.currentTimeMillis() + 60000, "collection"),
                        PrefillQueueScheduler.Decision.NO_WAIT);
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        long deadline = f.scheduler.scheduleAt(System.nanoTime());
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        assertEquals(Long.MAX_VALUE, f.scheduler.scheduleAt(deadline));
        assertTrue(f.tasks.isEmpty());
    }

    @Test
    void rejectedSubmissionFailsQueueInsteadOfLeavingClaimedTask() throws Exception {
        var f = new Fixture();
        var failure = new RejectedExecutionException("rejected");
        doThrow(failure).when(f.executor).execute(any());
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        verify(f.queue).failQueuedRequests(failure);
        assertTrue(f.failures.contains(failure));
        assertEquals(0, ReflectionTestUtils.getField(f.scheduler, "queueSchedulingInFlight"));
        assertTrue(f.records().isEmpty());
    }

    @Test
    void taskFailureDiscardsPendingNotificationAndCleansQueue() throws Exception {
        var f = new Fixture();
        var failure = new IllegalStateException("failed");
        when(f.queue.scheduleOnce())
                .thenAnswer(
                        call -> {
                            f.notifyChange();
                            throw failure;
                        });
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        verify(f.queue).failQueuedRequests(failure);
        f.scheduler.scheduleAt(System.nanoTime());
        assertTrue(f.tasks.isEmpty());
        assertTrue(f.records().isEmpty());
    }

    @Test
    void stopPreventsClaimedTaskFromSchedulingAndDiscardsPendingChecks() throws Exception {
        var f = new Fixture();
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.notifyChange();
        f.scheduler.stop();
        f.runTask();
        f.scheduler.clearAfterTasksStopped();
        f.notifyChange();
        assertEquals(Long.MAX_VALUE, f.scheduler.scheduleAt(System.nanoTime()));
        assertTrue(f.tasks.isEmpty());
        verify(f.queue, times(0)).scheduleOnce();
        verify(f.queue).observeWait(null);
        assertTrue(f.records().isEmpty());
    }

    @Test
    void subscriptionPrecedesCapacityRecheck() throws Exception {
        var f = new Fixture();
        var head = mock(RouteResult.class);
        when(head.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        var boundary = CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR);
        var wait = PrefillQueueScheduler.Decision.capacityBlocked(head, boundary);
        when(f.queue.scheduleOnce()).thenReturn(wait);
        f.notifyChange();
        f.scheduler.scheduleAt(System.nanoTime());
        f.runTask();
        var order = org.mockito.Mockito.inOrder(f.queue);
        order.verify(f.queue).observeWait(wait);
        order.verify(f.queue).needsImmediateScheduling(any(), anyLong());
    }

    @Test
    void immediateCheckUsesCurrentHeadAndCurrentCapacity() {
        var endpoint = mock(PrefillEndpoint.class);
        var dispatcher = mock(DeliveryCoordinator.class);
        var queue =
                new PrefillQueueScheduler(
                        "test", endpoint, mock(QueueExecutionSettings.class), dispatcher);
        var head = mock(RouteResult.class);
        when(endpoint.captureQueue(1))
                .thenReturn(new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(head), head));
        var boundary = CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR);
        var wait = PrefillQueueScheduler.Decision.capacityBlocked(head, boundary);
        assertFalse(queue.needsImmediateScheduling(wait, 100L));
        when(dispatcher.capacityAvailable(head, boundary)).thenReturn(true);
        assertTrue(queue.needsImmediateScheduling(wait, 100L));
        assertTrue(queue.needsImmediateScheduling(PrefillQueueScheduler.Decision.NO_WAIT, 100L));
        when(endpoint.captureQueue(1))
                .thenReturn(new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(), null));
        assertFalse(queue.needsImmediateScheduling(wait, 100L));
    }

    /**
     * Replaces only the queue decision collaborator; the real scheduler owns task and timer transitions.
     */
    private static final class Fixture {
        final PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        final PrefillQueueScheduler queue = mock(PrefillQueueScheduler.class);
        final ExecutorService executor = mock(ExecutorService.class);
        final ArrayDeque<Runnable> tasks = new ArrayDeque<>();
        final List<Throwable> failures = new ArrayList<>();
        final PrefillDeliveryScheduler scheduler =
                new PrefillDeliveryScheduler(
                        mock(QueueExecutionSettings.class),
                        mock(DeliveryCoordinator.class),
                        executor,
                        2,
                        () -> {},
                        failures::add);

        Fixture() throws Exception {
            when(queue.scheduleOnce()).thenReturn(PrefillQueueScheduler.Decision.NO_WAIT);
            doAnswer(
                            call -> {
                                tasks.addLast(call.getArgument(0));
                                return null;
                            })
                    .when(executor)
                    .execute(any());
            var type = Class.forName(PrefillDeliveryScheduler.class.getName() + "$QueueTask");
            var constructor =
                    type.getDeclaredConstructor(PrefillEndpoint.class, PrefillQueueScheduler.class);
            constructor.setAccessible(true);
            records().put(endpoint, constructor.newInstance(endpoint, queue));
        }

        void notifyChange() {
            scheduler.endpointChanged(endpoint);
        }

        void runTask() {
            tasks.removeFirst().run();
        }

        @SuppressWarnings("unchecked")
        Map<PrefillEndpoint, Object> records() {
            return (Map<PrefillEndpoint, Object>)
                    ReflectionTestUtils.getField(scheduler, "prefillQueueSchedulers");
        }
    }
}

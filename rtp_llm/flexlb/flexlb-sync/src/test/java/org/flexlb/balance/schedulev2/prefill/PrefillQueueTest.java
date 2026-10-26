package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;

class PrefillQueueTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final PrefillQueue queue = endpoint.prefillQueue();
    private final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);

    @Test
    void cleanupDoesNotWaitForQueueMonitor() throws Exception {
        Request normal = request("normal");
        Request expired = request("expired");
        publish(expired);
        queue.addDispatched(new DispatchedRequest(normal, 10L, 5L));
        queue.addDispatched(new DispatchedRequest(expired, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(normal, 2));
        assertEquals(null, queue.tryAcquirePublishSlot(expired, 2));
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            synchronized (queue) {
                assertTrue(executor.submit(() -> queue.removeDispatched(normal))
                        .get(2, java.util.concurrent.TimeUnit.SECONDS));
                executor.submit(
                        () -> queue.removeRequestStatusExpired(System.currentTimeMillis() + 10_000L, 1000L, reporter))
                        .get(2, java.util.concurrent.TimeUnit.SECONDS);
                assertEquals(0, queue.dispatchedCount());
            }
        }
        assertEquals(0L, queue.inflightCount());
    }

    @Test
    void concurrentDeleteReleasesAndNotifiesOnce() throws Exception {
        Request request = request("delete-once");
        queue.addDispatched(new DispatchedRequest(request, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(request, 1));
        var notifications = new java.util.concurrent.atomic.AtomicInteger();
        queue.onCapacityChanged(notifications::incrementAndGet);
        var start = new java.util.concurrent.CountDownLatch(1);
        try (var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> {
                start.await();
                return queue.removeDispatched(request);
            });
            var second = executor.submit(() -> {
                start.await();
                return queue.removeDispatched(request);
            });
            start.countDown();
            assertNotEquals(first.get(2, java.util.concurrent.TimeUnit.SECONDS),
                    second.get(2, java.util.concurrent.TimeUnit.SECONDS));
        }
        assertEquals(1, notifications.get());
        assertEquals(0L, queue.inflightCount());
        assertEquals(org.flexlb.balance.schedulev2.QueueBlockReason.RESERVATION_CHANGED,
                queue.tryAcquirePublishSlot(request, 1));
    }

    @Test
    void expiryKeepsEngineStatusThatArrivesAfterTimeoutDecision() {
        Request request = request("late-status");
        publish(request);
        var record = org.mockito.Mockito.spy(new DispatchedRequest(request, 10L, 5L));
        queue.addDispatched(record);
        assertEquals(null, queue.tryAcquirePublishSlot(request, 1));
        org.mockito.Mockito.doAnswer(call -> {
            V2TestSupport.reportTasks(queue, Map.of(request.requestId(), TaskPhase.RUNNING), Map.of());
            return true;
        }).when(record).statusExpired(org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.anyLong());
        queue.removeRequestStatusExpired(System.currentTimeMillis(), 1000L, reporter);
        assertEquals(0, queue.dispatchedCount());
        assertEquals(1L, queue.inflightCount());
        assertFalse(queue.removeDispatched(request));
    }

    @Test
    void delayedReleaseDoesNotRemoveReplacementClaim() {
        var tasks = new PrefillActiveTasks();
        Request request = request("replacement");
        var old = new DispatchedRequest(request, 10L, 5L);
        var replacement = new DispatchedRequest(request, 10L, 5L);
        assertEquals(null, tasks.tryAcquire(old, 1));
        tasks.release(old);
        assertEquals(null, tasks.tryAcquire(replacement, 1));
        tasks.release(old);
        tasks.releaseExpired(old, System.currentTimeMillis(), 1000L);
        assertEquals(1L, tasks.inflightCount());
        tasks.release(replacement);
        assertEquals(0L, tasks.inflightCount());
    }

    @Test
    void finishedOnlyReportReleasesClaimWithoutConfirmationTimeout() {
        Request request = request("finished-only");
        publish(request);
        queue.addDispatched(new DispatchedRequest(request, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(request, 1));
        V2TestSupport.reportTasks(queue, Map.of(), Map.of(request.requestId(), TaskPhase.RUNNING));

        assertEquals(0, queue.dispatchedCount());
        assertEquals(0L, queue.inflightCount());
        queue.removeRequestStatusExpired(request.publishedAtMs() + 1_000L, 1_000L, reporter);
        assertEquals(0, queue.dispatchedCount());
        verifyNoInteractions(reporter);
        V2TestSupport.reportTasks(queue, Map.of(), Map.of(request.requestId(), TaskPhase.RUNNING));
        assertEquals(0L, queue.inflightCount());
    }

    @Test
    void waitingRequestExpiresWhenItsStatusStopsUpdating() {
        Request request = request("waiting");
        publish(request);
        queue.addDispatched(new DispatchedRequest(request, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(request, 1));
        V2TestSupport.reportTasks(queue, Map.of("waiting", TaskPhase.PENDING), Map.of());
        queue.removeRequestStatusExpired(System.currentTimeMillis(), 1000L, reporter);
        assertEquals(1, queue.dispatchedCount());
        verifyNoInteractions(reporter);
        assertEquals(1L, queue.inflightCount());
        queue.removeRequestStatusExpired(System.currentTimeMillis() + 10_000L, 1000L, reporter);
        assertEquals(0, queue.dispatchedCount());
        assertEquals(0L, queue.inflightCount());
        verify(reporter).reportStatusUpdateTimeout(RoleType.PREFILL, endpoint.ipPort());
        queue.removeRequestStatusExpired(System.currentTimeMillis() + 20_000L, 1000L, reporter);
        verifyNoMoreInteractions(reporter);
    }

    @Test
    void inflightLimitCountsDispatchedAndUnknownEngineRequests() {
        Request published = request("1");
        queue.addDispatched(new DispatchedRequest(published, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(published, 4));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("other", TaskPhase.RUNNING));

        assertEquals(2L, queue.inflightCount());
        assertEquals(1L, queue.unknownRequestCount());
        assertTrue(queue.canDispatch(10L, 3));
        assertFalse(queue.canDispatch(10L, 2));
    }

    @Test
    void kvCheckSubtractsOnlyRequestsTheEngineHasNotReported() {
        Request published = request("1");
        queue.addDispatched(new DispatchedRequest(published, 60L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(published, 4));
        V2TestSupport.report(endpoint, 1L, 100L, 1_000L, Map.of());

        assertTrue(queue.canDispatch(40L, 10));
        assertFalse(queue.canDispatch(41L, 10));

        V2TestSupport.report(endpoint, 2L, 100L, 1_000L, Map.of("1", TaskPhase.PENDING));

        assertEquals(1L, queue.inflightCount());
        assertTrue(queue.canDispatch(100L, 10));
    }

    @Test
    void emptyMachineAcceptsRequestLargerThanAvailableKv() {
        V2TestSupport.report(endpoint, 1L, 10L, 1_000L, Map.of());

        assertTrue(queue.canDispatch(500L, 10));
    }

    @Test
    void missingActiveReportReleasesClaimAndLateFinishedDoesNotReleaseAgain() {
        Request reported = request("1");
        Request notReported = request("2");
        queue.addDispatched(new DispatchedRequest(reported, 10L, 5L));
        queue.addDispatched(new DispatchedRequest(notReported, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(reported, 1));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("1", TaskPhase.RUNNING));

        V2TestSupport.report(endpoint, 2L, 1_000L, 1_000L, Map.of());

        assertEquals(1, queue.dispatchedCount());
        assertEquals(0L, queue.inflightCount());
        assertFalse(queue.removeDispatched(reported));
        V2TestSupport.reportTasks(queue, Map.of(), Map.of("1", TaskPhase.RUNNING));
        assertEquals(1, queue.dispatchedCount());
        assertEquals(0L, queue.inflightCount());
        queue.removeRequestStatusExpired(System.currentTimeMillis() + 10_000L, 1000L, reporter);
        verifyNoInteractions(reporter);
        assertTrue(queue.removeDispatched(notReported));
    }

    @Test
    void publishedRecordNotReportedByEngineIsRemovedAfterConfirmTimeout() {
        Request unpublished = request("1");
        Request reported = request("2");
        Request notReported = request("3");
        publish(reported);
        publish(notReported);
        queue.addDispatched(new DispatchedRequest(unpublished, 10L, 5L));
        DispatchedRequest record = new DispatchedRequest(reported, 10L, 5L);
        queue.addDispatched(record);
        queue.addDispatched(new DispatchedRequest(notReported, 10L, 5L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("2", TaskPhase.RUNNING));
        long publishedAtMs = notReported.publishedAtMs();
        record.updateStatus(WorkView.Phase.ENGINE_RUNNING, publishedAtMs + 50L);

        queue.removeRequestStatusExpired(publishedAtMs + 79L, 80L, reporter);
        assertEquals(3, queue.dispatchedCount());

        queue.removeRequestStatusExpired(publishedAtMs + 80L, 80L, reporter);
        assertEquals(2, queue.dispatchedCount());
        assertFalse(queue.removeDispatched(notReported));
        queue.removeRequestStatusExpired(publishedAtMs + 129L, 80L, reporter);
        assertEquals(2, queue.dispatchedCount());
        queue.removeRequestStatusExpired(publishedAtMs + 130L, 80L, reporter);
        assertEquals(1, queue.dispatchedCount());
        assertFalse(queue.removeDispatched(reported));
        assertTrue(queue.removeDispatched(unpublished));
    }

    @Test
    void decodeWaitingCandidatesDoNotConsumeActiveTaskLimit() {
        for (int i = 0; i < 20; i++) {
            queue.addDispatched(new DispatchedRequest(request("candidate-" + i), 10L, 5L));
        }
        assertTrue(queue.hasDispatchSlot(4));
        Request first = request("first");
        queue.addDispatched(new DispatchedRequest(first, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(first, 1));
        assertFalse(queue.hasDispatchSlot(1));
        queue.releasePublishSlot(first);
        assertTrue(queue.hasDispatchSlot(1));
    }

    @Test
    void engineAcknowledgementDoesNotDoubleCountPublishedTask() {
        Request first = request("first");
        queue.addDispatched(new DispatchedRequest(first, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(first, 4));
        V2TestSupport.report(endpoint, 1L, 1000L, 1000L,
                Map.of("first", TaskPhase.RUNNING, "external", TaskPhase.PENDING));
        assertTrue(queue.hasDispatchSlot(3));
        assertFalse(queue.hasDispatchSlot(2));
        V2TestSupport.report(endpoint, 2L, 1000L, 1000L, Map.of());
        assertTrue(queue.hasDispatchSlot(1));
        V2TestSupport.reportTasks(queue, Map.of(), Map.of("first", TaskPhase.RUNNING));
        assertTrue(queue.hasDispatchSlot(1));
    }

    @Test
    void concurrentAcquisitionsCannotExceedFourActiveTasks() throws Exception {
        var workers = java.util.concurrent.Executors.newFixedThreadPool(16);
        var start = new java.util.concurrent.CountDownLatch(1);
        var successes = new java.util.concurrent.atomic.AtomicInteger();
        var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
        try {
            for (int i = 0; i < 16; i++) {
                Request request = request("concurrent-" + i);
                queue.addDispatched(new DispatchedRequest(request, 10L, 5L));
                futures.add(workers.submit(() -> {
                    start.await();
                    if (queue.tryAcquirePublishSlot(request, 4) == null) {
                        successes.incrementAndGet();
                    }
                    return null;
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get(5, java.util.concurrent.TimeUnit.SECONDS);
            }
            assertEquals(4, successes.get());
            assertFalse(queue.hasDispatchSlot(4));
        } finally {
            workers.shutdownNow();
        }
    }

    @Test
    void loadViewHasWaitingItemsDispatchedWorkAndUnknownCount() {
        Request waiting = request("w");
        V2TestSupport.choose(waiting, endpoint, 4L, 20L);
        waiting.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        queue.addWaiting(waiting);
        Request dispatched = request("d");
        queue.addDispatched(new DispatchedRequest(dispatched, 10L, 30L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("d", TaskPhase.RUNNING, "x", TaskPhase.PENDING));
        GroupPlanner.Constraints constraints = new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);
        Comparator<GroupPlanner.Item> order = Comparator.comparingLong(GroupPlanner.Item::enqueueSeq);

        PrefillTtftEstimate.LoadView view = queue.loadView(1_000L, order, constraints);

        assertEquals(List.of("w"), view.queue().activeItems().stream().map(GroupPlanner.Item::requestId).toList());
        assertEquals(4L, view.queue().activeItems().getFirst().hitCache());
        assertEquals(List.of(new WorkView.RequestWork("d", WorkView.Phase.ENGINE_RUNNING, 30L)),
                view.work().requests());
        assertEquals(1L, view.work().unknownRequestCount());

        queue.removeWaiting(waiting);
        assertNotEquals(view.ownershipVersion(), queue.loadView(1_000L, order, constraints).ownershipVersion());
    }

    private static void publish(Request request) {
        request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
        request.publish(Stage.DECODE_WAITING, List.of());
    }

    @Test
    void ttftCountsRequestOnceWhileItMovesFromWaitingToDispatched() {
        Request earlier = request("1");
        earlier.setQueueSeq(1L);
        V2TestSupport.choose(earlier, endpoint, 0L, 10L);
        earlier.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        queue.addWaiting(earlier);
        Request incoming = request("2");
        incoming.setQueueSeq(2L);
        assertEquals(25L, queue.estimatedTtftMs(incoming, 15L));

        queue.addDispatched(new DispatchedRequest(earlier, 16L, 10L));
        assertEquals(25L, queue.estimatedTtftMs(incoming, 15L));
        earlier.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
        queue.removeWaiting(earlier);
        assertEquals(25L, queue.estimatedTtftMs(incoming, 15L));
    }

    @Test
    void ttftDoesNotInventDurationForUnknownEngineWork() {
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("other", TaskPhase.RUNNING));
        assertEquals(-1L, queue.estimatedTtftMs(request("1"), 15L));
    }

    @Test
    void highPriorityTtftExcludesLowerPriorityWaitingWork() {
        PrefillQueue priorityQueue = new PrefillQueue(endpoint, true);
        Request low = V2TestSupport.request(config, "low", 10, 16L);
        low.setQueueSeq(1L);
        V2TestSupport.choose(low, endpoint, 0L, 100L);
        low.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        priorityQueue.addWaiting(low);
        Request high = V2TestSupport.request(config, "high", 80, 16L);
        high.setQueueSeq(2L);

        assertEquals(15L, priorityQueue.estimatedTtftMs(high, 15L));
        priorityQueue.addDispatched(new DispatchedRequest(low, 16L, 100L));
        assertEquals(115L, priorityQueue.estimatedTtftMs(high, 15L));
    }

    private Request request(String requestId) {
        return V2TestSupport.request(config, requestId, 50, 16L);
    }
}

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
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillQueueTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final PrefillQueue queue = new PrefillQueue(endpoint, false);

    @Test
    void inflightLimitCountsDispatchedAndUnknownEngineRequests() {
        Request published = request("1");
        queue.addDispatched(new DispatchedRequest(published, 10L, 5L));
        assertEquals(null, queue.tryAcquirePublishSlot(published, 4));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("other", TaskPhase.RUNNING));
        queue.updateFromEngine();

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
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(40L, 10));
        assertFalse(queue.canDispatch(41L, 10));

        V2TestSupport.report(endpoint, 2L, 100L, 1_000L, Map.of("1", TaskPhase.PENDING));
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(100L, 10));
    }

    @Test
    void emptyMachineAcceptsRequestLargerThanAvailableKv() {
        V2TestSupport.report(endpoint, 1L, 10L, 1_000L, Map.of());
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(500L, 10));
    }

    @Test
    void recordIsRemovedAfterEngineReportedItAndThenDropsIt() {
        Request reported = request("1");
        Request notReported = request("2");
        queue.addDispatched(new DispatchedRequest(reported, 10L, 5L));
        queue.addDispatched(new DispatchedRequest(notReported, 10L, 5L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("1", TaskPhase.RUNNING));
        queue.updateFromEngine();

        V2TestSupport.report(endpoint, 2L, 1_000L, 1_000L, Map.of());
        assertTrue(queue.updateFromEngine());

        assertEquals(1, queue.dispatchedCount());
        assertFalse(queue.removeDispatched(reported));
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
        queue.addDispatched(new DispatchedRequest(reported, 10L, 5L));
        queue.addDispatched(new DispatchedRequest(notReported, 10L, 5L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("2", TaskPhase.RUNNING));
        queue.updateFromEngine();
        long publishedAtMs = notReported.publishedAtMs();

        assertEquals(0, queue.removeExpiredDispatched(publishedAtMs + 79L, 80L));
        assertEquals(3, queue.dispatchedCount());

        assertEquals(1, queue.removeExpiredDispatched(publishedAtMs + 80L, 80L));
        assertEquals(2, queue.dispatchedCount());
        assertFalse(queue.removeDispatched(notReported));
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
        queue.updateFromEngine();
        assertTrue(queue.hasDispatchSlot(3));
        assertFalse(queue.hasDispatchSlot(2));
        V2TestSupport.report(endpoint, 2L, 1000L, 1000L, Map.of());
        queue.updateFromEngine();
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
    void sameStatusVersionIsHandledOnce() {
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of());

        assertTrue(queue.updateFromEngine());
        assertFalse(queue.updateFromEngine());
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
        queue.updateFromEngine();
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
        queue.updateFromEngine();
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

package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks priority/FIFO queue order, exact unlinking and retry progress through blocked entries.
 */
class OrderedRequestQueueTest {
    private static final org.flexlb.config.FlexlbConfig ENTRY_CONFIG =
            SchedulingTestConfig.newConfig();

    @Test
    void capacityRetryStartsAtTheAllowedRequestWithoutRecheckingTheHead() {
        for (boolean priority : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var first = entry(50);
            var second = entry(50);
            var third = entry(50);
            queue.add(first);
            queue.add(second);
            queue.add(third);
            queue.scanForPlanningCandidates(3, 3, request -> true);
            queue.retry(third);
            var examined = new java.util.ArrayList<RequestContext>();
            assertEquals(
                    List.of(third),
                    queue.scanForPlanningCandidates(
                            3,
                            3,
                            request -> {
                                examined.add(request);
                                return true;
                            }));
            assertEquals(List.of(third), examined);
            assertEquals(List.of(first, second, third), queue.drain());
        }
    }

    @Test
    void newCapacityScanPreservesPriorityAndFifoOrder() {
        for (boolean priority : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var first = entry(20);
            var second = entry(90);
            var third = entry(90);
            queue.add(first);
            queue.add(second);
            queue.add(third);
            var expected = priority ? List.of(second, third, first) : List.of(first, second, third);
            assertEquals(expected, queue.scanForPlanningCandidates(3, 3, item -> true));
            assertFalse(queue.hasUnscannedRequests());
            queue.beginScan();
            assertEquals(expected, queue.scanForPlanningCandidates(3, 3, item -> true));
            assertEquals(3, queue.size());
        }
    }

    @Test
    void withdrawnRouteReturnsAheadOfNewerSamePriorityRequests() {
        for (boolean priority : new boolean[] {false, true}) {
            OrderedRequestQueue queue = new OrderedRequestQueue(priority);
            RequestContext first = entry(50);
            RequestContext second = entry(50);
            RequestContext third = entry(50);
            queue.add(first);
            queue.add(second);
            queue.add(third);
            int[] capturedCounts = queue.priorityCounts();
            assertEquals(3, capturedCounts[50]);
            long sequence = second.getPlacementSequence();
            queue.scanForPlanningCandidates(3, 3, candidate -> true);
            queue.remove(second);
            assertEquals(2, queue.priorityCounts()[50]);
            assertEquals(3, capturedCounts[50]);
            queue.restore(second);
            assertEquals(3, queue.priorityCounts()[50]);
            assertEquals(sequence, second.getPlacementSequence());
            assertEquals(
                    List.of(second, third),
                    queue.scanForPlanningCandidates(3, 3, candidate -> true));
            assertEquals(List.of(first, second, third), queue.drain());
            assertEquals(0, queue.priorityCounts()[50]);
        }
    }

    @Test
    void fifoUnlinksCompletedSuffixWithoutScanningFromHead() {
        OrderedRequestQueue queue = new OrderedRequestQueue(false);
        RequestContext first = entry(50);
        RequestContext middle = entry(50);
        RequestContext last = entry(50);
        queue.add(first);
        queue.add(middle);
        queue.add(last);

        assertTrue(queue.remove(middle));

        assertEquals(2, queue.size());
        assertFalse(queue.contains(middle));
        assertEquals(
                List.of(first, last), queue.scanForPlanningCandidates(10, 20, candidate -> true));
    }

    @Test
    void priorityKeepsFifoInsideBucketAfterArbitraryUnlink() {
        OrderedRequestQueue queue = new OrderedRequestQueue(true);
        RequestContext low = entry(10);
        RequestContext highFirst = entry(90);
        RequestContext highSecond = entry(90);
        queue.add(low);
        queue.add(highFirst);
        queue.add(highSecond);

        assertTrue(queue.remove(highFirst));

        assertEquals(
                List.of(highSecond, low),
                queue.scanForPlanningCandidates(10, 20, candidate -> true));
    }

    @Test
    void drainMarksAndDetachesEveryLiveEntryAtomically() {
        OrderedRequestQueue queue = new OrderedRequestQueue(false);
        RequestContext first = entry(50);
        RequestContext second = entry(50);
        queue.add(first);
        queue.add(second);

        assertEquals(List.of(first, second), queue.drain());

        assertEquals(0, queue.size());
        assertFalse(queue.contains(first));
        assertFalse(queue.contains(second));
    }

    @Test
    void budgetCountsBlockedEntriesAndContinuesPastTheBlockedPrefix() {
        for (boolean priority : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priority);
            for (int i = 0; i < 100; i++) {
                queue.add(entry(50));
            }
            var healthy = entry(50);
            queue.add(healthy);
            var checks = new AtomicInteger();
            for (int round = 0; round < 3; round++) {
                assertTrue(
                        queue.scanForPlanningCandidates(
                                        15,
                                        30,
                                        candidate -> {
                                            checks.incrementAndGet();
                                            return candidate == healthy;
                                        })
                                .isEmpty());
                assertEquals((round + 1) * 30, checks.get());
                assertTrue(queue.hasUnscannedRequests());
            }
            assertEquals(
                    List.of(healthy),
                    queue.scanForPlanningCandidates(
                            15,
                            30,
                            candidate -> {
                                checks.incrementAndGet();
                                return candidate == healthy;
                            }));
            assertEquals(101, checks.get());
            assertFalse(queue.hasUnscannedRequests());
            assertTrue(
                    queue.scanForPlanningCandidates(
                                    15,
                                    30,
                                    candidate -> {
                                        throw new AssertionError(
                                                "exhausted scan must not start over");
                                    })
                            .isEmpty());
        }
    }

    @Test
    void newHighPriorityWorkInterruptsAnUnfinishedLowPriorityScan() {
        var queue = new OrderedRequestQueue(true);
        var first = entry(50);
        var next = entry(50);
        queue.add(first);
        queue.add(next);
        assertTrue(queue.scanForPlanningCandidates(1, 1, candidate -> false).isEmpty());
        var high = entry(90);
        queue.add(high);
        assertEquals(List.of(high), queue.scanForPlanningCandidates(1, 1, candidate -> true));
        queue.remove(high);
        assertEquals(List.of(next), queue.scanForPlanningCandidates(1, 1, candidate -> true));
    }

    private static RequestContext entry(int priority) {
        var context = new RequestContext(ENTRY_CONFIG);
        context.setRequest(org.flexlb.worker.endpoint.Request.builder().build());
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.explicit(priority, Long.MAX_VALUE));

        return context;
    }
}

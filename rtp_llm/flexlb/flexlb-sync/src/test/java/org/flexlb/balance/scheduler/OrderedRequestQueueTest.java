package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class OrderedRequestQueueTest {

    @Test
    void fifoUnlinksCompletedSuffixWithoutScanningFromHead() {
        OrderedRequestQueue queue = new OrderedRequestQueue(false);
        GlobalQueueEntry first = entry(50);
        GlobalQueueEntry middle = entry(50);
        GlobalQueueEntry last = entry(50);
        queue.add(first);
        queue.add(middle);
        queue.add(last);

        assertTrue(queue.remove(middle));

        assertEquals(2, queue.size());
        assertFalse(middle.linked);
        assertEquals(List.of(first, last), queue.scanForPlanningCandidates(
                10, 20, candidate -> true));
    }

    @Test
    void priorityKeepsFifoInsideBucketAfterArbitraryUnlink() {
        OrderedRequestQueue queue = new OrderedRequestQueue(true);
        GlobalQueueEntry low = entry(10);
        GlobalQueueEntry highFirst = entry(90);
        GlobalQueueEntry highSecond = entry(90);
        queue.add(low);
        queue.add(highFirst);
        queue.add(highSecond);

        assertTrue(queue.remove(highFirst));

        assertSame(highSecond, queue.peekHead());
        assertEquals(List.of(highSecond, low), queue.scanForPlanningCandidates(
                10, 20, candidate -> true));
    }

    @Test
    void drainMarksAndDetachesEveryLiveEntryAtomically() {
        OrderedRequestQueue queue = new OrderedRequestQueue(false);
        GlobalQueueEntry first = entry(50);
        GlobalQueueEntry second = entry(50);
        queue.add(first);
        queue.add(second);

        assertEquals(List.of(first, second), queue.drain());

        assertEquals(0, queue.size());
        assertTrue(first.removed);
        assertTrue(second.removed);
        assertFalse(first.linked);
        assertFalse(second.linked);
    }

    @Test
    void budgetCountsBlockedEntriesAndContinuesPastTheBlockedPrefix() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            for (int i = 0; i < 100; i++) {
                queue.add(entry(50));
            }
            var healthy = entry(50);
            queue.add(healthy);
            var checks = new AtomicInteger();
            for (int round = 0; round < 3; round++) {
                assertTrue(queue.scanForPlanningCandidates(15, 30, candidate -> {
                    checks.incrementAndGet();
                    return candidate == healthy;
                }).isEmpty());
                assertEquals((round + 1) * 30, checks.get());
                assertTrue(queue.hasUnscannedRequests());
            }
            assertEquals(List.of(healthy), queue.scanForPlanningCandidates(15, 30, candidate -> {
                checks.incrementAndGet();
                return candidate == healthy;
            }));
            assertEquals(101, checks.get());
            assertFalse(queue.hasUnscannedRequests());
            assertTrue(queue.scanForPlanningCandidates(15, 30, candidate -> {
                throw new AssertionError("exhausted scan must not start over");
            }).isEmpty());
        }
    }

    @Test
    void readyRetryBypassesTheBlockedBacklogWithinOneScanBudget() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var first = new GlobalQueueEntry(null, new CompletableFuture<>(), 50);
            queue.add(first);
            for (int i = 1; i < 250_000; i++) {
                queue.add(new GlobalQueueEntry(null, new CompletableFuture<>(), 50));
            }
            queue.scanForPlanningCandidates(15, 30, candidate -> false);
            queue.markRequestReadyForRetry(first);
            var checked = new AtomicInteger();
            assertEquals(List.of(first), queue.scanForPlanningCandidates(15, 30, candidate -> {
                checked.incrementAndGet();
                return candidate == first;
            }));
            assertTrue(checked.get() <= 30);
        }
    }

    @Test
    void readyRetriesKeepTheirOrderAcrossPlanningFrontiers() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var requests = new ArrayList<GlobalQueueEntry>();
            for (int i = 0; i < 16; i++) {
                var request = entry(50);
                queue.add(request);
                requests.add(request);
            }
            queue.scanForPlanningCandidates(16, 32, candidate -> true);
            requests.forEach(queue::markRequestReadyForRetry);
            for (int i = 0; i < 16; i++) {
                var request = entry(50);
                queue.add(request);
                requests.add(request);
            }
            assertEquals(requests.subList(0, 16),
                    queue.scanForPlanningCandidates(16, 32, candidate -> true));
            assertEquals(requests.subList(16, 32),
                    queue.scanForPlanningCandidates(16, 32, candidate -> true));
            assertFalse(queue.hasUnscannedRequests());
        }
    }

    @Test
    void blockedRetriesDoNotPreventTheForwardScanFromProgressing() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var first = entry(50);
            queue.add(first);
            for (int i = 0; i < 40; i++) {
                queue.add(entry(50));
            }
            var last = entry(50);
            queue.add(last);
            queue.scanForPlanningCandidates(1, 1, candidate -> false);
            boolean reachedLast = false;
            for (int attempt = 0; attempt < 50 && !reachedLast; attempt++) {
                queue.markRequestReadyForRetry(first);
                var candidates = queue.scanForPlanningCandidates(1, 2,
                        candidate -> candidate == last);
                reachedLast = candidates.contains(last);
            }
            assertTrue(reachedLast, "blocked retries must not rewind the forward scan");
        }
    }

    @Test
    void retryAheadOfTheCursorIsExaminedOnlyOnceWithATightBudget() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var first = entry(50);
            var second = entry(50);
            queue.add(first);
            queue.add(second);
            queue.markRequestReadyForRetry(second);
            assertEquals(List.of(first, second),
                    queue.scanForPlanningCandidates(2, 2, candidate -> true));
            assertFalse(queue.hasUnscannedRequests());
        }
    }

    @Test
    void forwardAndRetryIndexesNeverReturnTheSameRequestTwice() {
        var queue = new OrderedRequestQueue(false);
        var first = entry(50);
        var second = entry(50);
        queue.add(first);
        queue.add(second);
        queue.markRequestReadyForRetry(first);
        queue.markRequestReadyForRetry(first);
        queue.markRequestReadyForRetry(second);
        assertEquals(List.of(first, second),
                queue.scanForPlanningCandidates(15, 30, candidate -> true));
        queue.remove(first);
        queue.remove(second);
        assertFalse(queue.hasUnscannedRequests());
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

    @Test
    void cancellationRepairsBothForwardAndRetryPositions() {
        var queue = new OrderedRequestQueue(true);
        var first = entry(50);
        var next = entry(50);
        var last = entry(50);
        queue.add(first);
        queue.add(next);
        queue.add(last);
        queue.scanForPlanningCandidates(1, 1, candidate -> false);
        queue.markRequestReadyForRetry(first);
        queue.remove(first);
        queue.remove(next);
        assertEquals(List.of(last), queue.scanForPlanningCandidates(1, 2, candidate -> true));
        queue.remove(last);
        assertFalse(queue.hasUnscannedRequests());
        var newcomer = entry(50);
        queue.add(newcomer);
        assertEquals(List.of(newcomer), queue.scanForPlanningCandidates(1, 2, candidate -> true));
    }

    @Test
    void cleanupWakeDuringCapturePreservesPriorityWithinTheFrontier() {
        var queue = new OrderedRequestQueue(true);
        var high = entry(90);
        var low = entry(50);
        var cancelled = entry(50);
        queue.add(high);
        queue.scanForPlanningCandidates(1, 1, candidate -> false);
        queue.add(low);
        queue.add(cancelled);
        var captured = queue.scanForPlanningCandidates(3, 6, candidate -> {
            if (candidate == cancelled) {
                queue.remove(cancelled);
                queue.markRequestReadyForRetry(high);
                return false;
            }
            return true;
        });
        assertEquals(List.of(high, low), captured);
    }

    @Test
    void mixedArrivalsCancellationsAndWakesDoNotLoseReadyRequests() {
        for (boolean priority : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priority);
            var entries = new ArrayList<GlobalQueueEntry>();
            var ready = new HashSet<GlobalQueueEntry>();
            var random = new Random(73);
            for (int step = 0; step < 2_000; step++) {
                int operation = entries.isEmpty() ? 0 : random.nextInt(4);
                if (operation == 0) {
                    var entry = entry(random.nextInt(5) * 20);
                    entries.add(entry);
                    ready.add(entry);
                    queue.add(entry);
                } else if (operation == 1) {
                    var entry = entries.get(random.nextInt(entries.size()));
                    ready.add(entry);
                    queue.markRequestReadyForRetry(entry);
                } else if (operation == 2) {
                    var entry = entries.remove(random.nextInt(entries.size()));
                    ready.remove(entry);
                    queue.remove(entry);
                } else {
                    var checked = new AtomicInteger();
                    var captured = queue.scanForPlanningCandidates(3, 6, entry -> {
                        checked.incrementAndGet();
                        return ready.contains(entry);
                    });
                    assertTrue(checked.get() <= 6);
                    assertEquals(captured.size(), new HashSet<>(captured).size());
                    Comparator<GlobalQueueEntry> order = Comparator.comparingLong(entry -> entry.sequence);
                    if (priority) {
                        order = Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority)
                                .reversed().thenComparing(order);
                    }
                    assertEquals(ready.stream().sorted(order).limit(captured.size()).toList(), captured,
                            "each frontier must be a prefix of the ready requests in queue order");
                    for (var entry : captured) {
                        ready.remove(entry);
                        if (random.nextBoolean()) {
                            entries.remove(entry);
                            queue.remove(entry);
                        }
                    }
                }
            }
            int passes = 0;
            while (queue.hasUnscannedRequests()) {
                assertTrue(++passes <= 2_000);
                for (var entry : queue.scanForPlanningCandidates(3, 6, ready::contains)) {
                    ready.remove(entry);
                    queue.remove(entry);
                }
            }
            assertTrue(ready.isEmpty(), "every awakened or newly added request must be visited");
        }
    }

    private static GlobalQueueEntry entry(int priority) {
        return new GlobalQueueEntry(
                mock(BalanceContext.class),
                new CompletableFuture<Response>(),
                priority);
    }
}

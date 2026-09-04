package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;

import java.util.List;
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

        assertTrue(queue.markCompleted(middle));

        assertEquals(2, queue.size());
        assertFalse(middle.linked);
        assertEquals(List.of(first, last), queue.snapshotPrefix(
                10, candidate -> true, null));
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
        assertEquals(List.of(highSecond, low), queue.snapshotPrefix(
                10, candidate -> true, null));
        assertTrue(queue.hasHigherPriorityEntry(
                low, candidate -> true));
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

    private static GlobalQueueEntry entry(int priority) {
        return new GlobalQueueEntry(
                mock(BalanceContext.class),
                new CompletableFuture<Response>(),
                priority);
    }
}

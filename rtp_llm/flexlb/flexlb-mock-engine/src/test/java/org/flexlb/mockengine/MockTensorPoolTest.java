package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockTensorPoolTest {
    @Test void alignedFirstFitCoalescesOnlyAdjacentFreeSpans() {
        var pool = new MockTensorPool(32);
        var a = pool.reserve(8); var b = pool.reserve(8); var c = pool.reserve(8); var d = pool.reserve(8);
        pool.release(b, List.of()); pool.release(d, List.of());
        assertEquals(16, pool.stats().free()); assertEquals(8, pool.stats().largest());
        assertThrows(MockTensorPool.CapacityException.class, () -> pool.reserve(16));
        pool.release(a, List.of());
        assertEquals(0, pool.reserve(16).offset);
        assertEquals(24, pool.stats().used());
        pool.release(c, List.of()); assertEquals(16, pool.stats().largest());
        var odd = new MockTensorPool(16);
        assertEquals(0, odd.reserve(1).offset); assertEquals(8, odd.reserve(1).offset);
        assertEquals(new MockTensorPool.Stats(16, 2, 0, 14, 0), odd.stats());
    }
    @Test void pendingEventsRetainCapacityUntilCompletionWithoutHoldingPoolLock() throws Exception {
        var pool = new MockTensorPool(16);
        var block = pool.reserve(16); var event = new CompletableFuture<Void>();
        pool.release(block, List.of(event)); pool.release(block, List.of());
        assertEquals(16, pool.stats().pending()); assertEquals(0, pool.stats().free());
        var entered = new java.util.concurrent.CountDownLatch(1);
        var allocate = CompletableFuture.supplyAsync(() -> { entered.countDown(); return pool.reserve(16); });
        assertTrue(entered.await(1, TimeUnit.SECONDS));
        assertFalse(allocate.isDone());
        assertEquals(16, pool.stats().used(), "event wait must leave stats available");
        event.complete(null);
        assertEquals(0, allocate.get(2, TimeUnit.SECONDS).offset);
        assertEquals(new MockTensorPool.Stats(16, 16, 0, 0, 0), pool.stats());
    }
    @Test void completedEventsReclaimOnStatsAndZeroCapacityIsExplicit() {
        var pool = new MockTensorPool(8); var block = pool.reserve(8);
        var event = new CompletableFuture<Void>(); pool.release(block, List.of(event)); event.complete(null);
        assertEquals(new MockTensorPool.Stats(8, 0, 0, 8, 8), pool.stats());
        assertNull(new MockTensorPool(0).reserve(0));
        assertThrows(MockTensorPool.CapacityException.class, () -> new MockTensorPool(0).reserve(1));
    }
}

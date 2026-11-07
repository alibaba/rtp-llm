package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PlacementOperationTest {
    @Test
    void competingCompletionsClaimCleanupOnceAndKeepPlacementVisible() throws Exception {
        var context = context();
        var operation = context.beginPlacement(null);
        var cleanupCount = new AtomicInteger();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(4)) {
            var pending = new ArrayList<Future<?>>();
            for (int i = 0; i < 8; i++) {
                pending.add(
                        workers.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    if (context.beginPlacementCompletion(operation)) {
                                        assertFalse(Thread.holdsLock(context));
                                        cleanupCount.incrementAndGet();
                                    }
                                    return null;
                                }));
            }
            start.countDown();
            for (var task : pending) task.get(5, TimeUnit.SECONDS);
        }
        assertEquals(1, cleanupCount.get());
        assertSame(operation, context.snapshot().placement().operation());
        assertTrue(context.snapshot().placement().completing());
        assertNull(context.beginPlacement(null), "cleanup must block a replacement placement");
        context.completePlacement(operation, ContextResult.PublicationResult.PUBLISHED, 1L);
        assertNull(context.snapshot().placement());
        assertFalse(context.beginPlacementCompletion(operation));
        assertDoesNotThrow(operation::close);
    }

    @Test
    void equalValuesAndOldOperationsCannotCompleteTheCurrentAttempt() {
        var context = context();
        var first = context.beginPlacement(null);
        assertThrows(IllegalStateException.class, first::close);
        var equalValue = new PlacementOperation(context, null);
        assertEquals(first, equalValue);
        assertFalse(context.beginPlacementCompletion(equalValue));
        assertTrue(context.beginPlacementCompletion(first));
        context.completePlacement(first, ContextResult.PublicationResult.PUBLISHED, 1L);
        var replacement = context.beginPlacement(null);
        assertNotNull(replacement);
        assertFalse(context.beginPlacementCompletion(first));
        assertNull(context.completePlacement(first, ContextResult.PublicationResult.FAILED, 2L));
        assertSame(replacement, context.snapshot().placement().operation());
        assertDoesNotThrow(first::close);
        assertTrue(context.beginPlacementCompletion(replacement));
        context.completePlacement(replacement, ContextResult.PublicationResult.PUBLISHED, 3L);
    }

    private static RequestContext context() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4401L);
        context.register("4401", System.currentTimeMillis(), Long.MAX_VALUE, 60_000L);
        return context;
    }
}

package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks exact-attempt completion and ensures competing finishers cannot both claim placement cleanup.
 */
class RoutingContextTest {
    @Test
    void finishingAnUnpublishedSelectionDoesNotClaimAnAssignedRoute() {
        var context = context();
        var operation = context.beginRouteSelection();
        assertEquals(ContextResult.RoutingPhase.SELECTING, context.snapshot().routingPhase());
        assertTrue(context.beginRoutingCompletion(operation));
        assertEquals(
                ContextResult.RoutingPhase.FINISHING_SELECTION, context.snapshot().routingPhase());
        context.completeRouting(operation, ContextResult.RoutingCompletion.FINISHED, 1L);
        assertEquals(ContextResult.RoutingPhase.NONE, context.snapshot().routingPhase());
        assertFalse(context.snapshot().preparedDelivery());
        assertNull(context.snapshot().route());
    }

    @Test
    void unpublishedCleanupRouteCannotBecomeDispatchable() {
        var context = context();
        var operation = context.beginRouteSelection();
        var route = org.mockito.Mockito.mock(RouteResult.class);
        context.retainRouteForCleanup(operation, route);
        assertSame(route, context.snapshot().route());
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.snapshot().preparedDelivery());
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () ->
                        context.completeRouting(
                                operation, ContextResult.RoutingCompletion.WITHDRAWN, 1L));
        assertSame(operation, context.snapshot().routingContext());
        assertTrue(context.beginRoutingCompletion(operation));
        context.completeRouting(operation, ContextResult.RoutingCompletion.FAILED, 2L);
        assertSame(route, context.snapshot().route());
        assertFalse(
                context.snapshot().preparedDelivery(),
                "retained cleanup cannot be sent after routing ends");
        context.beginClose(null, 3L);
        assertFalse(context.snapshot().preparedDelivery(), "ending request cannot be sent");
    }

    @Test
    void competingCompletionsClaimCleanupOnceAndKeepPlacementVisible() throws Exception {
        var context = context();
        var operation = context.beginRouteSelection();
        var cleanupCount = new AtomicInteger();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(4)) {
            var pending = new ArrayList<Future<?>>();
            for (int i = 0; i < 8; i++) {
                pending.add(
                        workers.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    if (context.beginRoutingCompletion(operation)) {
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
        assertSame(operation, context.snapshot().routingContext());
        assertFalse(context.beginRoutingCompletion(operation), "cleanup is already claimed");
        assertNull(context.beginRouteSelection(), "cleanup must block a replacement placement");
        context.completeRouting(operation, ContextResult.RoutingCompletion.FINISHED, 1L);
        assertNull(context.snapshot().routingContext());
        assertFalse(context.beginRoutingCompletion(operation));
    }

    @Test
    void equalValuesAndOldOperationsCannotCompleteTheCurrentAttempt() {
        var context = context();
        var first = context.beginRouteSelection();
        var equalValue = new RoutingContext(context, null);
        assertEquals(first, equalValue);
        assertFalse(context.beginRoutingCompletion(equalValue));
        assertTrue(context.beginRoutingCompletion(first));
        context.completeRouting(first, ContextResult.RoutingCompletion.FINISHED, 1L);
        var replacement = context.beginRouteSelection();
        assertNotNull(replacement);
        assertFalse(context.beginRoutingCompletion(first));
        assertNull(context.completeRouting(first, ContextResult.RoutingCompletion.FAILED, 2L));
        assertSame(replacement, context.snapshot().routingContext());
        assertTrue(context.beginRoutingCompletion(replacement));
        context.completeRouting(replacement, ContextResult.RoutingCompletion.FINISHED, 3L);
    }

    private static RequestContext context() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4401L);
        context.register(
                "4401",
                System.currentTimeMillis(),
                Long.MAX_VALUE,
                60_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        return context;
    }
}

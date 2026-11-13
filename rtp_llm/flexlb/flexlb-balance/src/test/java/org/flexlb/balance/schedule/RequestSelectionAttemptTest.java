package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks that one completion ends a selection attempt and stale attempts cannot finish a replacement.
 */
class RequestSelectionAttemptTest {

    @Test
    void competingCompletionsFinishSelectionOnce() throws Exception {
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
                                    if (context.finishSelectionWithoutCommit(operation, null, 1L)
                                            != null) {
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
        assertNull(context.currentSelectionAttempt());
        assertNull(context.finishSelectionWithoutCommit(operation, null, 1L));
    }

    @Test
    void equalValuesAndOldOperationsCannotCompleteTheCurrentAttempt() {
        var context = context();
        var first = context.beginRouteSelection();
        var equalValue = new SelectionAttempt(context, first.id() + 1);
        org.junit.jupiter.api.Assertions.assertNotEquals(first, equalValue);
        assertNull(context.finishSelectionWithoutCommit(equalValue, null, 1L));
        context.finishSelectionWithoutCommit(first, null, 1L);
        var replacement = context.beginRouteSelection();
        assertNotNull(replacement);
        assertNull(context.finishSelectionWithoutCommit(first, null, 2L));
        assertNull(
                context.finishSelectionWithoutCommit(
                        first,
                        org.flexlb.worker.endpoint.Response.buildErrorResponse(
                                org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                                "selection failed"),
                        2L));
        assertEquals(replacement, context.currentSelectionAttempt());
        context.finishSelectionWithoutCommit(replacement, null, 3L);
    }

    private static RequestContext context() {
        var context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.setRequest(org.flexlb.worker.endpoint.Request.builder().requestId("4401").build());
        RequestProtocolTestSupport.register(
                context,
                "4401",
                System.currentTimeMillis(),
                Long.MAX_VALUE,
                60_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        return context;
    }
}

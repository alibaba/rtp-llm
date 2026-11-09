package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that terminal facts cannot archive a request before notifications and physical resource release complete.
 */
class RequestCleanupArchiveRaceTest {
    @RepeatedTest(16)
    @Timeout(15)
    void terminalFactsRequireBothNotificationAndPhysicalReleaseProof() throws Exception {
        var context = RequestCleanupPassTest.openContext(7501L);
        var route = context.snapshot().route();
        RequestCleanupPassTest.ended(
                context.beginClose(
                        new EndEvidence(
                                context.snapshot().route(),
                                EndKind.FAILURE,
                                null,
                                false,
                                0L,
                                StrategyErrorType.DISPATCH_FAILED,
                                "cleanup required",
                                false,
                                false),
                        1001L));
        assertNotNull(context.releaseReason());
        var ready = new CountDownLatch(2);
        var start = new CountDownLatch(1);
        var workers = Executors.newFixedThreadPool(2);
        try {
            var resources =
                    workers.submit(
                            () -> {
                                ready.countDown();
                                assertTrue(start.await(5, TimeUnit.SECONDS));
                                synchronized (context) {
                                    context.recordResourcesReleased(route, true, true, 1002L);
                                    return context.snapshot();
                                }
                            });
            var effects =
                    workers.submit(
                            () -> {
                                ready.countDown();
                                assertTrue(start.await(5, TimeUnit.SECONDS));
                                synchronized (context) {
                                    context.recordCloseNotifications(true, true, 1002L);
                                    return context.snapshot();
                                }
                            });
            assertTrue(ready.await(5, TimeUnit.SECONDS));
            start.countDown();
            var released = resources.get(5, TimeUnit.SECONDS);
            var notified = effects.get(5, TimeUnit.SECONDS);
            var current = context.snapshot();
            assertNotNull(current.terminalRecord());
            assertTrue(
                    released.terminalRecord() != null || notified.terminalRecord() != null,
                    "the last real obligation must close atomically");
            assertEquals(Phase.CLOSED, current.phase());
            assertNull(current.route());
            assertNull(context.releaseReason());
            context.recordResourcesReleased(route, true, true, 1003L);
            assertSame(current, context.snapshot());
        } finally {
            start.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}

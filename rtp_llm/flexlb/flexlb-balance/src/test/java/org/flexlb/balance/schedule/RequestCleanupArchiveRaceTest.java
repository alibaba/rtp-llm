package org.flexlb.balance.schedule;

import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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
                context.endRequest(
                        new RequestEndEvent(
                                context.snapshot().route(),
                                EndKind.FAILURE,
                                false,
                                0L,
                                StrategyErrorType.DISPATCH_FAILED,
                                "cleanup required"),
                        1001L,
                        null));
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
                                    context.recordCleanupProgress(route, true, true, false, 1002L);
                                    return context.snapshot();
                                }
                            });
            var effects =
                    workers.submit(
                            () -> {
                                ready.countDown();
                                assertTrue(start.await(5, TimeUnit.SECONDS));
                                synchronized (context) {
                                    context.recordCleanupProgress(
                                            context.snapshot().route(), false, false, true, 1002L);
                                    return context.snapshot();
                                }
                            });
            assertTrue(ready.await(5, TimeUnit.SECONDS));
            start.countDown();
            var released = resources.get(5, TimeUnit.SECONDS);
            var notified = effects.get(5, TimeUnit.SECONDS);
            var current = context.snapshot();
            assertNotNull(current.requestRecordForArchive());
            assertTrue(
                    released.requestRecordForArchive() != null
                            || notified.requestRecordForArchive() != null,
                    "the last real obligation must close atomically");
            assertEquals(Phase.CLOSED, current.phase());
            assertNull(current.route());
            assertNull(context.releaseReason());
            context.recordCleanupProgress(route, true, true, false, 1003L);
            assertEquals(current, context.snapshot());
        } finally {
            start.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}

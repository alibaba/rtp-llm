package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;

class RequestCleanupArchiveRaceTest {
    @RepeatedTest(16)
    @Timeout(15)
    void terminalEffectsCannotCloseWhileAnExternalReleaseIsStillRunning() throws Exception {
        var context = RequestCleanupPassTest.openContext(7501L);
        var terminal = RequestCleanupPassTest.granted(context.beginClose(new EndEvidence(context.snapshot().route(),
                EndKind.FAILURE, null, false, 0L, StrategyErrorType.DISPATCH_FAILED, "cleanup required", false, false), 1001L));
        var pass = RequestCleanupPassTest.granted(context.beginSettlement());
        var ready = new CountDownLatch(2);
        var start = new CountDownLatch(1);
        var workers = Executors.newFixedThreadPool(2);
        try {
            var resources = workers.submit(() -> {
                ready.countDown();
                assertTrue(start.await(5, TimeUnit.SECONDS));
                return context.completeClose(pass, RequestCleanupPassTest.released(true, true), 1002L);
            });
            var effects = workers.submit(() -> {
                ready.countDown();
                assertTrue(start.await(5, TimeUnit.SECONDS));
                return context.completeClose(terminal,
                        new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, true, true), 1002L);
            });
            assertTrue(ready.await(5, TimeUnit.SECONDS));
            start.countDown();
            var update = resources.get(5, TimeUnit.SECONDS);
            var notified = effects.get(5, TimeUnit.SECONDS);
            var current = context.snapshot();
            assertNotNull(current.terminalRecord());
            assertTrue(update.after().terminalRecord() != null || notified.after().terminalRecord() != null,
                    "the last real obligation must close atomically");
            assertEquals(Phase.CLOSED, current.phase());
            assertNull(current.route());
            assertInstanceOf(Rejected.class, context.beginSettlement());
            var duplicate = context.completeClose(pass, RequestCleanupPassTest.released(true, true), 1003L);
            assertEquals(UpdateKind.STALE, duplicate.kind());
            assertSame(duplicate.before(), duplicate.after());
        } finally {
            start.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}

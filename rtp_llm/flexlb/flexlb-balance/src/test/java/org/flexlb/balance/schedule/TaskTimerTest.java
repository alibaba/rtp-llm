package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies independent task cancellation and rejection after timer shutdown.
 */
class TaskTimerTest {
    @Test
    void cancellingOneTaskLeavesOtherTasksRunnable() throws Exception {
        var timer = SchedulerConfiguration.newTimerExecutor(2, "timer-test-");
        var cancelledRan = new AtomicBoolean();
        try {
            var cancelled =
                    timer.schedule(
                            () -> cancelledRan.set(true),
                            60_000L,
                            java.util.concurrent.TimeUnit.MILLISECONDS);
            cancelled.cancel(false);
            cancelled.cancel(false);
            var completed = new CompletableFuture<Void>();
            timer.schedule(
                    () -> completed.complete(null), 0L, java.util.concurrent.TimeUnit.MILLISECONDS);
            completed.get(2, TimeUnit.SECONDS);
        } finally {
            timer.close();
        }
        assertFalse(cancelledRan.get());
    }

    @Test
    void closeDiscardsPendingTimeoutWithoutRunningItsCallback() {
        var timer = SchedulerConfiguration.newTimerExecutor(2, "timer-test-");
        var ran = new AtomicBoolean();
        timer.schedule(() -> ran.set(true), 60_000L, java.util.concurrent.TimeUnit.MILLISECONDS);
        timer.close();
        assertFalse(ran.get());
        assertThrows(
                java.util.concurrent.RejectedExecutionException.class,
                () -> timer.schedule(() -> {}, 0L, java.util.concurrent.TimeUnit.MILLISECONDS));
    }
}

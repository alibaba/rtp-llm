package org.flexlb.balance.scheduler;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TaskTimerTest {
    @Test
    void cancellingOneTaskLeavesOtherTasksRunnable() throws Exception {
        var timer = new ScheduledTaskTimer();
        var cancelledRan = new AtomicBoolean();
        try {
            var cancelled = timer.schedule(() -> cancelledRan.set(true), 60_000L);
            cancelled.cancel();
            cancelled.cancel();
            var completed = new CompletableFuture<Void>();
            timer.schedule(() -> completed.complete(null), 0L);
            completed.get(2, TimeUnit.SECONDS);
        } finally {
            timer.close();
        }
        assertFalse(cancelledRan.get());
    }

    @Test
    void closeDiscardsPendingTimeoutWithoutRunningItsCallback() {
        var timer = new ScheduledTaskTimer();
        var ran = new AtomicBoolean();
        timer.schedule(() -> ran.set(true), 60_000L);
        timer.close();
        assertFalse(ran.get());
        assertThrows(
                java.util.concurrent.RejectedExecutionException.class,
                () -> timer.schedule(() -> {}, 0L));
    }
}

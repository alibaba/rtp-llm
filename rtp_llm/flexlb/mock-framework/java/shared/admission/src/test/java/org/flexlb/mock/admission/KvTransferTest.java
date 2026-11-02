package org.flexlb.mock.admission;

import org.flexlb.mock.engine.RequestState;

import static org.junit.jupiter.api.Assertions.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class KvTransferTest {
    @Test void completionAndCancellationReleaseEachUseExactlyOnce() throws Exception {
        var released = new AtomicInteger();
        try (var transfer = new KvTransfer()) {
            for (int i = 0; i < 100; i++) {
                var execution = transfer.submit(released::incrementAndGet);
                boolean cancelled = execution.cancel();
                var outcome = execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
                assertEquals(cancelled ? KvTransfer.State.CANCELLED : KvTransfer.State.COMPLETED, outcome.state());
                assertFalse(execution.cancel());
            }
        }
        assertEquals(100, released.get());
    }
    @Test void cleanupFailureIsReturnedInsteadOfPublishingSuccess() throws Exception {
        try (var transfer = new KvTransfer()) {
            var error = new IllegalStateException("use failed");
            var execution = transfer.submit(() -> { throw error; });
            var outcome = execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(KvTransfer.State.FAILED, outcome.state());
            assertSame(error, outcome.error());
        }
    }
    @Test void rejectedSubmitLeavesOwnershipWithCaller() {
        var transfer = new KvTransfer(); transfer.close();
        var released = new AtomicInteger();
        RequestState.Use use = released::incrementAndGet;
        assertThrows(IllegalStateException.class, () -> transfer.submit(use));
        assertEquals(0, released.get());
        use.close(); assertEquals(1, released.get());
    }
}

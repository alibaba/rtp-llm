package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class WorkerStatusProgressTest {
    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
    }

    @Test
    void discoveryDoesNotEstablishHealth() {
        var worker = worker();
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(0L, worker.pollHealth().lastSuccessfulPollUs());
    }

    @Test
    void staleFinishCannotClearNextApplication() {
        var worker = worker();
        var first = worker.tryBeginStatusPoll(1L);
        worker.recordStatusRpcResult(first, true, true, null);
        assertTrue(worker.tryBeginStatusApplication(first));
        first.close();
        worker.finishStatusApplication(first);
        var second = worker.tryBeginStatusPoll(2L);
        assertTrue(worker.tryBeginStatusApplication(second));
        second.close();
        worker.finishStatusApplication(first);
        try (var third = worker.tryBeginStatusPoll(3L)) {
            assertFalse(worker.tryBeginStatusApplication(third));
        } finally { worker.finishStatusApplication(second); }
    }

    @Test
    @Timeout(5)
    void wireContactDoesNotWaitForApplicationLock() throws Exception {
        var worker = worker();
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            worker.lock.lock();
            try {
                CompletableFuture.runAsync(() -> worker.recordStatusRpcResult(lease, true, true, null))
                        .get(1, TimeUnit.SECONDS);
                assertTrue(worker.pollHealth().healthy());
            } finally { worker.lock.unlock(); }
        }
    }
}

package org.flexlb.cache;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Checks that cache worker setup records each supplied health observation once.
 */
class WorkerStatusTestSupportTest {
    @Test
    void oneNotAliveSnapshotIsOneWorkerFailure() {
        var worker =
                WorkerStatusTestSupport.workerStatus(
                        "127.0.0.1", 8000, RoleType.PREFILL, false, null);
        assertEquals(1L, worker.pollHealth().consecutiveWorkerFailures());
        assertFalse(worker.pollHealth().reportedAlive());
    }
}

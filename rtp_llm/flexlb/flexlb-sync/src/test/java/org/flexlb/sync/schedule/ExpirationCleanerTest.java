package org.flexlb.sync.schedule;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExpirationCleanerTest {

    private static final long WORKER_TIMEOUT_US = 3_000_000L;
    private static final long VIT_WORKER_TIMEOUT_US = 10_000_000L;

    @Test
    void should_useLongerExpirationWindowForVitWorkers() {
        ExpirationCleaner cleaner = new ExpirationCleaner(
                Mockito.mock(FlexMonitor.class), WORKER_TIMEOUT_US,
                WORKER_TIMEOUT_US, VIT_WORKER_TIMEOUT_US);
        long fiveSecondsAgo = System.nanoTime() / 1000 - 5_000_000L;

        Map<String, WorkerStatus> vitWorkers = workerMapWithLastUpdate(fiveSecondsAgo);
        Map<String, WorkerStatus> prefillWorkers = workerMapWithLastUpdate(fiveSecondsAgo);

        cleaner.doClean(vitWorkers, RoleType.VIT);
        cleaner.doClean(prefillWorkers, RoleType.PREFILL);

        assertTrue(vitWorkers.containsKey("127.0.0.1:8080"));
        assertFalse(prefillWorkers.containsKey("127.0.0.1:8080"));
    }

    @Test
    void should_removeVitWorkerAfterVitExpirationWindow() {
        ExpirationCleaner cleaner = new ExpirationCleaner(
                Mockito.mock(FlexMonitor.class), WORKER_TIMEOUT_US,
                WORKER_TIMEOUT_US, VIT_WORKER_TIMEOUT_US);
        long elevenSecondsAgo = System.nanoTime() / 1000 - 11_000_000L;
        Map<String, WorkerStatus> vitWorkers = workerMapWithLastUpdate(elevenSecondsAgo);

        cleaner.doClean(vitWorkers, RoleType.VIT);

        assertFalse(vitWorkers.containsKey("127.0.0.1:8080"));
    }

    private static Map<String, WorkerStatus> workerMapWithLastUpdate(long lastUpdateTimeUs) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.getStatusLastUpdateTime().set(lastUpdateTimeUs);
        Map<String, WorkerStatus> workerStatusMap = new ConcurrentHashMap<>();
        workerStatusMap.put("127.0.0.1:8080", workerStatus);
        return workerStatusMap;
    }
}

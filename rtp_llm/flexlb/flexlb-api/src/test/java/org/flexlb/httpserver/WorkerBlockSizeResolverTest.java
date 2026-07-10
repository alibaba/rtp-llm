package org.flexlb.httpserver;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class WorkerBlockSizeResolverTest {

    private final ModelWorkerStatus modelWorkerStatus =
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
    private WorkerBlockSizeResolver resolver;

    @BeforeEach
    void setUp() {
        clearWorkerStatuses();
        resolver = new WorkerBlockSizeResolver();
    }

    @AfterEach
    void tearDown() {
        resolver.shutdown();
        clearWorkerStatuses();
    }

    private void clearWorkerStatuses() {
        modelWorkerStatus.getPrefillStatusMap().clear();
        modelWorkerStatus.getDecodeStatusMap().clear();
        modelWorkerStatus.getPdFusionStatusMap().clear();
        modelWorkerStatus.getVitStatusMap().clear();
    }

    @Test
    void resolvesAndCachesBlockSizeFromHealthyWorker() {
        modelWorkerStatus.getPrefillStatusMap().put("10.0.0.1:8000", worker(64));

        assertEquals(64L, resolver.resolve());

        modelWorkerStatus.getPrefillStatusMap().clear();
        assertEquals(64L, resolver.resolve());
    }

    @Test
    void refreshesBlockSizeWhenWorkersChangeConsistently() {
        modelWorkerStatus.getPrefillStatusMap().put("10.0.0.1:8000", worker(64));
        resolver.refresh();

        modelWorkerStatus.getPrefillStatusMap().clear();
        modelWorkerStatus.getPrefillStatusMap().put("10.0.0.2:8000", worker(16));
        resolver.refresh();

        assertEquals(16L, resolver.resolve());
    }

    @Test
    void keepsLastValidBlockSizeWhenWorkersAreInconsistent() {
        modelWorkerStatus.getPrefillStatusMap().put("10.0.0.1:8000", worker(64));
        resolver.refresh();
        modelWorkerStatus.getDecodeStatusMap().put("10.0.0.2:8000", worker(16));
        resolver.refresh();

        assertEquals(64L, resolver.resolve());
    }

    @Test
    void failsWhenNoHealthyWorkerProvidesBlockSize() {
        assertThrows(IllegalStateException.class, resolver::resolve);
    }

    private WorkerStatus worker(long blockSize) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(blockSize);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        workerStatus.setCacheStatus(cacheStatus);
        return workerStatus;
    }
}

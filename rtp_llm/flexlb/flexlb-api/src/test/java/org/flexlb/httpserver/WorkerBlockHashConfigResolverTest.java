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

class WorkerBlockHashConfigResolverTest {

    private final ModelWorkerStatus modelWorkerStatus =
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
    private WorkerBlockHashConfigResolver resolver;

    @BeforeEach
    void setUp() {
        clearWorkerStatuses();
        resolver = new WorkerBlockHashConfigResolver();
    }

    @AfterEach
    void tearDown() {
        resolver.shutdown();
        clearWorkerStatuses();
    }

    @Test
    void resolvesAndCachesConfigFromHealthyPrefillWorker() {
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.1:8000", worker(64, 1));

        assertEquals(config(64, 1), resolver.resolve());

        modelWorkerStatus.getPrefillStatusMap().clear();
        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void refreshesConfigWhenPrefillWorkersChangeConsistently() {
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.1:8000", worker(64, 1));
        resolver.refresh();

        modelWorkerStatus.getPrefillStatusMap().clear();
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.2:8000", worker(16, 0));
        resolver.refresh();

        assertEquals(config(16, 0), resolver.resolve());
    }

    @Test
    void keepsLastValidConfigWhenPrefillWorkersAreInconsistent() {
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.1:8000", worker(64, 1));
        resolver.refresh();
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.2:8000", worker(64, 0));
        resolver.refresh();

        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void ignoresPdFusionConfigWhenPrefillConfigIsAvailable() {
        modelWorkerStatus.getPrefillStatusMap().put(
                "10.0.0.1:8000", worker(64, 1));
        modelWorkerStatus.getPdFusionStatusMap().put(
                "10.0.0.2:8000", worker(64, 0));

        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void fallsBackToPdFusionWhenNoPrefillConfigIsAvailable() {
        modelWorkerStatus.getPdFusionStatusMap().put(
                "10.0.0.2:8000", worker(64, 0));

        assertEquals(config(64, 0), resolver.resolve());
    }

    @Test
    void failsWhenNoHealthyWorkerProvidesConfig() {
        assertThrows(IllegalStateException.class, resolver::resolve);
    }

    private void clearWorkerStatuses() {
        modelWorkerStatus.getPrefillStatusMap().clear();
        modelWorkerStatus.getDecodeStatusMap().clear();
        modelWorkerStatus.getPdFusionStatusMap().clear();
        modelWorkerStatus.getVitStatusMap().clear();
    }

    private WorkerBlockHashConfigResolver.BlockHashConfig config(
            long blockSize,
            int lookaheadTokens) {
        return new WorkerBlockHashConfigResolver.BlockHashConfig(
                blockSize, lookaheadTokens);
    }

    private WorkerStatus worker(long blockSize, int lookaheadTokens) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(blockSize);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        workerStatus.setCacheStatus(cacheStatus);
        workerStatus.setBlockHashLookaheadTokens(lookaheadTokens);
        return workerStatus;
    }
}

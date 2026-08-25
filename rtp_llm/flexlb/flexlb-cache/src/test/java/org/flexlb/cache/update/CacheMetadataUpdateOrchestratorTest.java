package org.flexlb.cache.update;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheMetadataUpdateOrchestrator;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CacheMetadataUpdateOrchestratorTest {

    private final CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
    private final LocalSyncCacheMatchProvider localSyncProvider =
            mock(LocalSyncCacheMatchProvider.class);

    @Test
    void routesWorkerStatusUpdatesToLocalSync() {
        when(configuration.isLocalSyncEnabled()).thenReturn(true);
        WorkerStatus workerStatus = new WorkerStatus();
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(localSyncProvider.updateFromWorkerStatus(workerStatus)).thenReturn(expected);

        WorkerCacheUpdateResult actual = orchestrator().updateFromWorkerStatus(workerStatus);

        assertSame(expected, actual);
        verify(localSyncProvider).updateFromWorkerStatus(workerStatus);
    }

    @Test
    void rejectsSnapshotUpdatesWhenKvcmIsEnabled() {
        when(configuration.isLocalSyncEnabled()).thenReturn(false);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        WorkerCacheUpdateResult result = orchestrator().updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("127.0.0.1:8080", result.getEngineIpPort());
        verifyNoInteractions(localSyncProvider);
    }

    private CacheMetadataUpdateOrchestrator orchestrator() {
        return new CacheMetadataUpdateOrchestrator(configuration, localSyncProvider);
    }
}

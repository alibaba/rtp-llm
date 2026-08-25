package org.flexlb.cache.update;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheMetadataUpdateOrchestrator;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CacheMetadataUpdateOrchestratorTest {

    private final LocalSyncCacheMatchProvider localSyncProvider =
            mock(LocalSyncCacheMatchProvider.class);
    private final LocalStandbyCacheMatchProvider localStandbyProvider =
            mock(LocalStandbyCacheMatchProvider.class);

    @Test
    void routesWorkerStatusUpdatesToLocalSync() {
        WorkerStatus workerStatus = new WorkerStatus();
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(localSyncProvider.updateFromWorkerStatus(workerStatus))
                .thenReturn(expected);

        WorkerCacheUpdateResult actual =
                orchestrator(false).updateFromWorkerStatus(workerStatus);

        assertSame(expected, actual);
        verify(localSyncProvider).updateFromWorkerStatus(workerStatus);
    }

    @Test
    void rejectsWorkerStatusUpdatesWhenKvcmIsEnabled() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        WorkerCacheUpdateResult result =
                orchestrator(true).updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("127.0.0.1:8080", result.getEngineIpPort());
        assertEquals(
                "Local Sync cache metadata updates are disabled when KVCM is enabled",
                result.getErrorMessage());
        verifyNoInteractions(localSyncProvider);
    }

    @Test
    void routesRoutedRequestUpdatesToLocalStandby() {
        Request request = new Request();
        List<ServerStatus> selectedWorkers = List.of(new ServerStatus());

        orchestrator(true).updateFromRoutedRequest(request, selectedWorkers);

        verify(localStandbyProvider).updateFromRoutedRequest(request, selectedWorkers);
    }

    @Test
    void ignoresRoutedRequestUpdatesInLocalSyncMode() {
        orchestrator(false)
                .updateFromRoutedRequest(new Request(), List.of(new ServerStatus()));

        verifyNoInteractions(localStandbyProvider);
    }

    private CacheMetadataUpdateOrchestrator orchestrator(boolean kvcmEnabled) {
        return new CacheMetadataUpdateOrchestrator(
                new CacheMatchConfiguration(
                        modelMetaConfig(kvcmEnabled)),
                localSyncProvider,
                localStandbyProvider);
    }

    private ModelMetaConfig modelMetaConfig(boolean kvcmEnabled) {
        KvcmConfig kvcmConfig = new KvcmConfig();
        kvcmConfig.setEnabled(kvcmEnabled);

        ServiceRoute serviceRoute = new ServiceRoute();
        serviceRoute.setServiceId("test-service");
        serviceRoute.setKvcm(kvcmConfig);

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        return modelMetaConfig;
    }
}

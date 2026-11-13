package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.worker.cache.WorkerCacheUpdateResult;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Routes cache metadata updates to the source that owns the update contract.
 *
 * <p>Worker-status metadata updates and removals target Local Sync only when enabled.
 * Routed-request updates target Local Standby when enabled. These two inputs intentionally
 * maintain different cache views.
 */
@Slf4j
@Component
public class CacheMetadataUpdateOrchestrator {

    private final LocalSyncCacheMatchProvider localSyncProvider;
    private final LocalStandbyCacheMatchProvider localStandbyProvider;
    private final CacheMatchConfiguration configuration;

    public CacheMetadataUpdateOrchestrator(
            CacheMatchConfiguration configuration,
            LocalSyncCacheMatchProvider localSyncProvider,
            LocalStandbyCacheMatchProvider localStandbyProvider) {
        this.configuration = configuration;
        this.localSyncProvider = localSyncProvider;
        this.localStandbyProvider = localStandbyProvider;
        log.info(
                "Cache metadata update orchestrator initialized: "
                        + "localSyncEnabled={}, localStandbyEnabled={}",
                configuration.isLocalSyncEnabled(),
                configuration.isLocalStandbyEnabled());
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus) {
        if (configuration.isLocalSyncEnabled()) {
            return localSyncProvider.updateFromWorkerStatus(workerStatus);
        }

        String logicalIpPort = workerStatus == null ? null : workerStatus.getLogicalIpPort();
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .logicalIpPort(logicalIpPort)
                .errorMessage("Local Sync cache metadata updates are disabled when KVCM is enabled")
                .build();
    }

    public void removeEngineBlockCache(String engineIpPort) {
        if (configuration.isLocalSyncEnabled()) {
            localSyncProvider.removeEngineBlockCache(engineIpPort);
        }
    }

    public void updateFromRoutedRequest(Request request, List<ServerStatus> selectedWorkers) {
        if (configuration.isLocalStandbyEnabled()) {
            localStandbyProvider.updateFromRoutedRequest(request, selectedWorkers);
        }
    }
}

package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.List;

/** Routes cache metadata updates to the source that owns the update contract. */
@Slf4j
@Component
public class CacheMetadataUpdateOrchestrator {

    private final LocalSyncCacheMatchProvider localSyncProvider;
    private final CacheMatchConfiguration configuration;

    public CacheMetadataUpdateOrchestrator(
            CacheMatchConfiguration configuration,
            LocalSyncCacheMatchProvider localSyncProvider) {
        this.configuration = configuration;
        this.localSyncProvider = localSyncProvider;
        log.info("Cache metadata update orchestrator initialized: localSyncEnabled={}",
                configuration.isLocalSyncEnabled());
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus) {
        if (configuration.isLocalSyncEnabled()) {
            return localSyncProvider.updateFromWorkerStatus(workerStatus);
        }

        String engineIpPort = workerStatus == null ? null : workerStatus.getIpPort();
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .engineIpPort(engineIpPort)
                .errorMessage("Local Sync cache metadata updates are disabled when KVCM is enabled")
                .build();
    }

    /** KVCM owns its metadata; request-side standby updates are added with standby support. */
    public void updateFromRoutedRequest(
            Request request, List<ServerStatus> selectedWorkers) {
        // No local metadata update is required in this feature stage.
    }
}

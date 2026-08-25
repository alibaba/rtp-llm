package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Service;

import java.util.List;

/** Unified cache matching and metadata update service. */
@Slf4j
@Service
public class CacheAwareService {

    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchQueryOrchestrator queryOrchestrator;
    private final CacheMetadataUpdateOrchestrator updateOrchestrator;

    public CacheAwareService(
            CacheMetricsReporter cacheMetricsReporter,
            CacheMatchQueryOrchestrator queryOrchestrator,
            CacheMetadataUpdateOrchestrator updateOrchestrator) {
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.queryOrchestrator = queryOrchestrator;
        this.updateOrchestrator = updateOrchestrator;
    }

    public CacheMatchResult findMatchingEngines(CacheMatchQuery query) {
        long startTimeUs = System.nanoTime() / 1_000;
        try {
            CacheMatchResult result = queryOrchestrator.findMatchingEngines(query);
            cacheMetricsReporter.reportFindMatchingEnginesRT(
                    query.roleType(), startTimeUs, "0");
            return result;
        } catch (RuntimeException error) {
            CacheMatchSource source = queryOrchestrator.effectiveSource();
            long queryTimeUs = System.nanoTime() / 1_000 - startTimeUs;
            cacheMetricsReporter.reportFindMatchingEnginesRT(
                    query.roleType(), startTimeUs, "1");
            log.error("Error finding matching engines, requestId={}, role={}",
                    query.requestId(), query.roleType(), error);
            return CacheMatchResult.failed(source, queryTimeUs);
        }
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus) {
        return updateOrchestrator.updateFromWorkerStatus(workerStatus);
    }

    /** Reserved for metadata sources, such as local standby, introduced later. */
    public void updateFromRoutedRequest(
            Request request, List<ServerStatus> selectedWorkers) {
        updateOrchestrator.updateFromRoutedRequest(request, selectedWorkers);
    }
}

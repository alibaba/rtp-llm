package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Unified cache matching and metadata update service.
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class CacheAwareService {

    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchQueryOrchestrator queryOrchestrator;
    private final LocalStandbyComparisonService comparisonService;
    private final CacheMetadataUpdateOrchestrator updateOrchestrator;

    public CacheAwareService(CacheMetricsReporter cacheMetricsReporter,
                             CacheMatchQueryOrchestrator queryOrchestrator,
                             LocalStandbyComparisonService comparisonService,
                             CacheMetadataUpdateOrchestrator updateOrchestrator) {
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.queryOrchestrator = queryOrchestrator;
        this.comparisonService = comparisonService;
        this.updateOrchestrator = updateOrchestrator;
    }

    public Mono<CacheMatchResult> findMatchingEngines(CacheMatchQuery query) {
        return Mono.defer(() -> {
            long startTime = System.nanoTime();
            return queryOrchestrator.findMatchingEngines(query)
                    .doOnSuccess(ignored -> cacheMetricsReporter.reportFindMatchingEnginesRT(
                            query.roleType(), startTime / 1_000, "0"))
                    .onErrorResume(error -> {
                        CacheMatchSource source = queryOrchestrator.effectiveSource();
                        long queryTimeUs = (System.nanoTime() - startTime) / 1_000;
                        cacheMetricsReporter.reportFindMatchingEnginesRT(query.roleType(), startTime / 1_000, "1");
                        log.error("Error finding matching engines, requestId={}, role={}",
                                query.requestId(), query.roleType(), error);
                        return Mono.just(CacheMatchResult.failed(source, queryTimeUs));
                    });
        });
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus) {
        return updateOrchestrator.updateFromWorkerStatus(workerStatus);
    }

    public void updateFromRoutedRequest(Request request, List<ServerStatus> selectedWorkers) {
        updateOrchestrator.updateFromRoutedRequest(request, selectedWorkers);
    }

    public CompletableFuture<CacheHitComparisonResult> buildCacheHitComparison(CacheHitFeedback feedback) {
        return comparisonService.buildCacheHitComparison(feedback);
    }
}

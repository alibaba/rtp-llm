package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.hash.RequestBlockHashService;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/** Unified cache matching and metadata update service. */
@Slf4j
@Service
public class CacheAwareService {

    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchQueryOrchestrator queryOrchestrator;
    private final CacheHitFeedbackTracker feedbackTracker;
    private final CacheMetadataUpdateOrchestrator updateOrchestrator;
    private final RequestBlockHashService requestBlockHashService;

    public CacheAwareService(CacheMetricsReporter cacheMetricsReporter,
                             CacheMatchQueryOrchestrator queryOrchestrator,
                             LocalStandbyComparisonService comparisonService,
                             CacheMetadataUpdateOrchestrator updateOrchestrator,
                             RequestBlockHashService requestBlockHashService) {
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.queryOrchestrator = queryOrchestrator;
        this.feedbackTracker = new CacheHitFeedbackTracker(comparisonService);
        this.updateOrchestrator = updateOrchestrator;
        this.requestBlockHashService = requestBlockHashService;
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

    public void removeEngineBlockCache(String engineIpPort) {
        updateOrchestrator.removeEngineBlockCache(engineIpPort);
    }

    public CompletableFuture<Void> prepareBlockCacheKeys(BalanceContext context) {
        return requestBlockHashService.prepareBlockCacheKeys(context).toFuture();
    }

    public void updateFromRoutedRequest(
            Request request, List<ServerStatus> selectedWorkers) {
        updateOrchestrator.updateFromRoutedRequest(request, selectedWorkers);
    }

    public void trackRoutingPrediction(String requestId,
                                       RoleType role,
                                       String group,
                                       WorkerStatus worker,
                                       long inputTokens,
                                       long predictedHitTokens,
                                       CacheMatchResult result) {
        try {
            feedbackTracker.track(requestId, role, group, worker, inputTokens, predictedHitTokens, result);
        } catch (RuntimeException error) {
            log.warn("Cache prediction telemetry failed, requestId={}", requestId, error);
        }
    }

    public List<CompletableFuture<CacheHitComparisonResult>> observeCacheHitFeedback(WorkerStatus worker,
                                                                                     WorkerStatus.StatusObservation status) {
        return feedbackTracker.observe(worker, status);
    }
}

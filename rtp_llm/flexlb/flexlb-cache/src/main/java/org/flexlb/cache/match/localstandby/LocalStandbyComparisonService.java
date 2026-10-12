package org.flexlb.cache.match.localstandby;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Compares local standby predictions with KVCM and eventual engine results off the routing path.
 */
@Slf4j
@Component
public class LocalStandbyComparisonService {

    private final boolean enabled;
    private final LocalStandbyCacheMatchProvider localStandbyProvider;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final Cache<LocalStandbyPredictionKey, PendingPrediction> pendingLocalStandbyPredictions;

    public LocalStandbyComparisonService(CacheMatchConfiguration configuration,
                                         LocalStandbyCacheMatchProvider localStandbyProvider,
                                         CacheMetricsReporter cacheMetricsReporter) {
        LocalStandbyConfig config = configuration.getLocalStandbyConfig();
        this.enabled = configuration.isLocalStandbyEnabled();
        this.localStandbyProvider = localStandbyProvider;
        this.cacheMetricsReporter = cacheMetricsReporter;
        int queueCapacity = enabled
                ? config.getAsyncQueueCapacity()
                : LocalStandbyConfig.DEFAULT_ASYNC_QUEUE_CAPACITY;
        this.pendingLocalStandbyPredictions = Caffeine.newBuilder()
                .maximumSize(queueCapacity)
                .expireAfterWrite(enabled ? config.getTtlMs() : LocalStandbyConfig.DEFAULT_TTL_MS, TimeUnit.MILLISECONDS)
                .build();
    }

    public void trackLocalStandbyPrediction(CacheMatchQuery query) {
        if (!canTrack(query)) {
            return;
        }
        if (query.blockCacheKeys() != null && query.blockCacheKeys().isEmpty()) {
            storePrediction(query, CompletableFuture.completedFuture(new StandbyPrediction(Collections.emptyMap(), 0)));
            return;
        }
        CompletableFuture<StandbyPrediction> localStandbyPredictionTask =
                localStandbyProvider.asyncLocalStandbyMatch(query)
                        .thenApply(matchResult -> {
                            if (!matchResult.querySucceeded()) {
                                throw new IllegalStateException("Local Standby prediction failed");
                            }
                            return new StandbyPrediction(matchResult.hostMatches(), matchResult.blockSize());
                        });
        storePrediction(query, localStandbyPredictionTask);
    }

    /**
     * Records a completed Local Standby prediction without issuing another cache match query.
     */
    public void trackResolvedLocalStandbyPrediction(CacheMatchQuery query, CacheMatchResult matchResult) {
        if (!canTrack(query)
                || matchResult == null
                || matchResult.source() != CacheMatchSource.LOCAL_STANDBY
                || !matchResult.querySucceeded()) {
            return;
        }
        storePrediction(query, CompletableFuture.completedFuture(
                new StandbyPrediction(matchResult.hostMatches(), matchResult.blockSize())));
    }

    public void recordSelectedWorker(String requestId, RoleType role, WorkerStatus worker, long inputTokens) {
        PendingPrediction pending = pendingLocalStandbyPredictions.getIfPresent(
                new LocalStandbyPredictionKey(requestId, role));
        if (pending == null) {
            return;
        }
        pending.selectedWorker().complete(new SelectedWorker(
                role, worker.getLogicalIpPort(), worker.getMetricIpPort(), inputTokens));
    }

    public Function<CacheHitFeedback, CompletableFuture<CacheHitComparisonResult>> captureComparison(String requestId,
                                                                                                     RoleType role) {
        PendingPrediction pending = enabled
                ? pendingLocalStandbyPredictions.asMap().remove(new LocalStandbyPredictionKey(requestId, role))
                : null;
        CompletableFuture<StandbyPrediction> prediction = pending == null ? null : pending.prediction();
        return feedback -> {
            if (prediction == null) {
                return CompletableFuture.completedFuture(withoutLocalStandbyPrediction(feedback));
            }
            return prediction.thenApply(value -> value).completeOnTimeout(null, 1, TimeUnit.SECONDS)
                    .handle((standbyPrediction, error) -> {
                        if (error != null || standbyPrediction == null) {
                            log.warn("Local Standby comparison unavailable, requestId={}", requestId, error);
                            return withoutLocalStandbyPrediction(feedback);
                        }
                        return withLocalStandbyPrediction(feedback, standbyPrediction);
                    });
        };
    }

    private CacheHitComparisonResult withLocalStandbyPrediction(CacheHitFeedback feedback,
                                                                StandbyPrediction standbyPrediction) {
        String workerIpPort = feedback.logicalWorkerId();
        HostCacheMatch match = standbyPrediction.matches().get(workerIpPort);
        long localStandbyPredictedHitTokens = match == null
                ? 0
                : CacheMatchResult.matchedTokens(
                        match.localMatchBlocks(),
                        standbyPrediction.blockSize(),
                        feedback.inputTokens());
        return result(
                feedback,
                new CacheHitComparisonResult.CachePrediction(localStandbyPredictedHitTokens, -1, -1));
    }

    private CacheHitComparisonResult withoutLocalStandbyPrediction(CacheHitFeedback feedback) {
        return feedback == null ? null : result(feedback, null);
    }

    private boolean canTrack(CacheMatchQuery query) {
        return enabled && query != null && query.blockSize() > 0;
    }

    private void storePrediction(CacheMatchQuery query, CompletableFuture<StandbyPrediction> prediction) {
        CompletableFuture<SelectedWorker> selectedWorker = new CompletableFuture<>();
        prediction.thenAcceptBoth(selectedWorker, (result, selected) -> {
            HostCacheMatch match = result.matches().get(selected.logicalWorkerId());
            long hitTokens = CacheMatchResult.matchedTokens(
                    match == null ? 0 : match.localMatchBlocks(), result.blockSize(), selected.inputTokens());
            cacheMetricsReporter.reportLocalStandbyPrediction(
                    selected.role(), selected.metricIpPort(), hitTokens, selected.inputTokens());
        }).exceptionally(error -> {
            log.warn("Local Standby prediction metrics unavailable, requestId={}", query.requestId(), error);
            return null;
        });
        pendingLocalStandbyPredictions.put(
                new LocalStandbyPredictionKey(query.requestId(), query.roleType()),
                new PendingPrediction(prediction, selectedWorker));
    }

    private CacheHitComparisonResult result(CacheHitFeedback feedback,
                                            CacheHitComparisonResult.CachePrediction localStandbyPrediction) {
        CacheHitComparisonResult.CachePrediction selectedSourcePrediction =
                new CacheHitComparisonResult.CachePrediction(feedback.predictedHitTokens(), -1, -1);
        CacheMatchSource source = CacheMatchSource.valueOf(feedback.cacheMatchSource());
        CacheHitComparisonResult.CachePrediction kvcmPrediction = source == CacheMatchSource.KVCM
                ? new CacheHitComparisonResult.CachePrediction(
                        selectedSourcePrediction.predictedHitTokens(),
                        feedback.kvcmMatchAvailable() ? feedback.kvcmLocalMatchTokens() : -1,
                        feedback.kvcmMatchAvailable() ? feedback.kvcmGlobalMatchTokens() : -1)
                : null;
        CacheHitComparisonResult.CachePrediction localSyncPrediction = source == CacheMatchSource.LOCAL_SYNC
                ? selectedSourcePrediction
                : null;
        CacheHitComparisonResult.CachePrediction effectiveLocalStandbyPrediction =
                source == CacheMatchSource.LOCAL_STANDBY
                        ? selectedSourcePrediction
                        : localStandbyPrediction;
        return new CacheHitComparisonResult(
                feedback.eventType(),
                feedback.requestId(),
                feedback.cacheMatchSource(),
                feedback.role(),
                feedback.group(),
                feedback.workerIdentity(),
                feedback.taskState(),
                feedback.inputTokens(),
                feedback.actualHitTokens(),
                kvcmPrediction,
                localSyncPrediction,
                effectiveLocalStandbyPrediction);
    }

    private record LocalStandbyPredictionKey(String requestId, RoleType roleType) {
    }

    private record PendingPrediction(CompletableFuture<StandbyPrediction> prediction,
                                     CompletableFuture<SelectedWorker> selectedWorker) {
    }

    private record SelectedWorker(RoleType role, String logicalWorkerId, String metricIpPort, long inputTokens) {
    }

    private record StandbyPrediction(Map<String, HostCacheMatch> matches, long blockSize) {

        private StandbyPrediction {
            matches = matches == null ? Collections.emptyMap() : Map.copyOf(matches);
        }
    }
}

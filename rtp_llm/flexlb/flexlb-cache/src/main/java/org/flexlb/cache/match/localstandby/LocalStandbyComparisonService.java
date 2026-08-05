package org.flexlb.cache.match.localstandby;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Compares local standby predictions with KVCM and eventual engine results off the routing path.
 */
@Slf4j
@Component
public class LocalStandbyComparisonService {

    private final boolean enabled;
    private final LocalStandbyCacheMatchProvider localStandbyProvider;
    private final Cache<LocalStandbyPredictionKey, CompletableFuture<StandbyPrediction>> pendingLocalStandbyPredictions;

    public LocalStandbyComparisonService(CacheMatchConfiguration configuration, LocalStandbyCacheMatchProvider localStandbyProvider) {
        LocalStandbyConfig config = configuration.getLocalStandbyConfig();
        this.enabled = configuration.isLocalStandbyEnabled();
        this.localStandbyProvider = localStandbyProvider;
        int queueCapacity = enabled
                ? config.getAsyncQueueCapacity()
                : LocalStandbyConfig.DEFAULT_ASYNC_QUEUE_CAPACITY;
        this.pendingLocalStandbyPredictions = Caffeine.newBuilder()
                .maximumSize(queueCapacity)
                .expireAfterWrite(enabled ? config.getEntryTtlMs() : LocalStandbyConfig.DEFAULT_ENTRY_TTL_MS, TimeUnit.MILLISECONDS)
                .build();
    }

    public void trackLocalStandbyPrediction(CacheMatchQuery query) {
        if (!enabled || query == null || query.localStandbyBlockSize() <= 0) {
            return;
        }
        CompletableFuture<StandbyPrediction> localStandbyPredictionTask =
                localStandbyProvider.asyncLocalStandbyMatch(query)
                        .thenApply(matchResult ->
                                new StandbyPrediction(matchResult.hostMatches(), matchResult.blockSize()));
        pendingLocalStandbyPredictions.put(
                new LocalStandbyPredictionKey(query.requestId(), query.roleType()), localStandbyPredictionTask);
    }

    public CompletableFuture<CacheHitComparisonResult> buildCacheHitComparison(CacheHitFeedback feedback) {
        if (!enabled || feedback == null || feedback.requestId() == null) {
            return CompletableFuture.completedFuture(withoutLocalStandbyPrediction(feedback));
        }
        RoleType roleType = resolveRoleType(feedback.role());
        if (roleType == null) {
            return CompletableFuture.completedFuture(withoutLocalStandbyPrediction(feedback));
        }
        CompletableFuture<StandbyPrediction> prediction = pendingLocalStandbyPredictions.asMap()
                .remove(new LocalStandbyPredictionKey(feedback.requestId(), roleType));
        if (prediction == null) {
            return CompletableFuture.completedFuture(withoutLocalStandbyPrediction(feedback));
        }

        return prediction.handle((standbyPrediction, error) -> {
            if (error != null || standbyPrediction == null) {
                log.warn("Failed to compare local standby cache prediction, requestId={}",
                        feedback.requestId(), error);
                return withoutLocalStandbyPrediction(feedback);
            }
            return withLocalStandbyPrediction(feedback, standbyPrediction);
        });
    }

    private CacheHitComparisonResult withLocalStandbyPrediction(CacheHitFeedback feedback, StandbyPrediction standbyPrediction) {
        String workerIpPort = feedback.workerIp() + ":" + feedback.workerPort();
        HostCacheMatch match = standbyPrediction.matches().get(workerIpPort);
        long localStandbyPredictedHitTokens = match == null
                ? 0
                : match.localMatchBlocks() * standbyPrediction.blockSize();
        return result(feedback, standbyPrediction.blockSize(), localStandbyPredictedHitTokens, true);
    }

    private CacheHitComparisonResult withoutLocalStandbyPrediction(CacheHitFeedback feedback) {
        return feedback == null ? null : result(feedback, 0, 0, false);
    }

    private CacheHitComparisonResult result(CacheHitFeedback feedback, long localStandbyBlockSize,
                                            long localStandbyPredictedHitTokens,
                                            boolean localStandbyPredictionAvailable) {
        long localStandbyDeltaHitTokens = localStandbyPredictionAvailable
                ? feedback.actualHitTokens() - localStandbyPredictedHitTokens
                : 0;
        long kvcmLocalDeltaHitTokens = feedback.kvcmMatchAvailable()
                ? feedback.actualHitTokens() - feedback.kvcmLocalMatchTokens()
                : 0;
        long kvcmP2pTotalMatchDeltaHitTokens = feedback.kvcmMatchAvailable()
                ? feedback.actualHitTokens() - feedback.kvcmP2pTotalMatchTokens()
                : 0;
        return new CacheHitComparisonResult(
                feedback.eventType(),
                feedback.requestId(),
                feedback.cacheMatchSource(),
                feedback.role(),
                feedback.group(),
                feedback.workerIp(),
                feedback.workerPort(),
                feedback.taskState(),
                feedback.inputTokens(),
                feedback.blockSize(),
                localStandbyBlockSize,
                feedback.predictedHitTokens(),
                feedback.kvcmMatchAvailable(),
                feedback.kvcmLocalMatchTokens(),
                feedback.kvcmP2pFetchTokens(),
                feedback.kvcmP2pTotalMatchTokens(),
                localStandbyPredictedHitTokens,
                localStandbyPredictionAvailable,
                feedback.actualHitTokens(),
                feedback.deltaHitTokens(),
                kvcmLocalDeltaHitTokens,
                kvcmP2pTotalMatchDeltaHitTokens,
                localStandbyDeltaHitTokens);
    }

    private RoleType resolveRoleType(String role) {
        for (RoleType roleType : RoleType.values()) {
            if (roleType.name().equals(role) || roleType.matches(role)) {
                return roleType;
            }
        }
        return null;
    }

    private record LocalStandbyPredictionKey(String requestId, RoleType roleType) {
    }

    private record StandbyPrediction(Map<String, HostCacheMatch> matches, long blockSize) {

        private StandbyPrediction {
            matches = matches == null ? Collections.emptyMap() : matches;
        }
    }
}

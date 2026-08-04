package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchFailoverAction;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.hash.LocalStandbyHashService;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

/**
 * Orchestrates cache matching across KVCM, local standby, and local cache-status sources.
 */
@Slf4j
@Component
public class CacheMatchQueryOrchestrator {

    private final LocalSyncCacheMatchProvider localSyncProvider;
    private final KvcmCacheMatchProvider kvcmProvider;
    private final LocalStandbyCacheMatchProvider localStandbyProvider;
    private final LocalStandbyCacheManager localStandbyCacheManager;
    private final CacheMatchFailoverManager failoverManager;
    private final LocalStandbyComparisonService comparisonService;
    private final LocalStandbyHashService localStandbyHashService;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchConfiguration configuration;

    public CacheMatchQueryOrchestrator(
            LocalSyncCacheMatchProvider localSyncProvider,
            KvcmCacheMatchProvider kvcmProvider,
            LocalStandbyCacheMatchProvider localStandbyProvider,
            LocalStandbyCacheManager localStandbyCacheManager,
            CacheMatchFailoverManager failoverManager,
            LocalStandbyComparisonService comparisonService,
            LocalStandbyHashService localStandbyHashService,
            CacheMetricsReporter cacheMetricsReporter,
            CacheMatchConfiguration configuration) {
        this.localSyncProvider = localSyncProvider;
        this.kvcmProvider = kvcmProvider;
        this.localStandbyProvider = localStandbyProvider;
        this.localStandbyCacheManager = localStandbyCacheManager;
        this.failoverManager = failoverManager;
        this.comparisonService = comparisonService;
        this.localStandbyHashService = localStandbyHashService;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.configuration = configuration;
        log.info("Cache match query orchestrator initialized: mode={}, autoSwitchEnabled={}, source={}",
                configuration.getConfiguredMode(),
                configuration.isAutoSwitchEnabled(),
                effectiveSource());
    }

    public Mono<CacheMatchResult> findMatchingEngines(CacheMatchQuery query) {
        return Mono.defer(() -> {
            long startTimeNs = System.nanoTime();
            if (!configuration.isKvcmEnabled()) {
                return queryLocalSync(query, startTimeNs);
            }
            CacheMatchSource source = failoverManager.activeSource();
            if (source == CacheMatchSource.LOCAL_STANDBY) {
                cacheMetricsReporter.reportStandbyFallback("active_source");
                return queryLocalStandby(query, startTimeNs);
            }
            if (query.blockCacheKeys() == null || query.blockCacheKeys().isEmpty()) {
                return Mono.just(CacheMatchResult.empty(CacheMatchSource.KVCM));
            }

            return kvcmProvider.findMatchingEngines(
                            query.requestId(), query.blockCacheKeys(), query.blockSize(), query.roleType(), query.group())
                    .doOnSuccess(ignored -> comparisonService.trackLocalStandbyPrediction(query))
                    .map(matches -> result(matches, CacheMatchSource.KVCM, startTimeNs, query.blockSize()))
                    .onErrorResume(error -> {
                        log.warn("KVCM cache query failed; requestId={}, action=LOCAL_STANDBY", query.requestId(), error);
                        cacheMetricsReporter.reportStandbyFallback("kvcm_query_failure");
                        return queryLocalStandby(query, startTimeNs);
                    });
        });
    }

    public void applyFailoverAction(CacheMatchFailoverAction action) {
        if (!configuration.isKvcmEnabled()) {
            throw new IllegalStateException("cache failover is unavailable in LOCAL_SYNC mode");
        }
        if (action == null) {
            throw new IllegalArgumentException("cache failover action must not be null");
        }
        switch (action) {
            case ACTIVATE_FALLBACK -> failoverManager.activateFallbackManually();
            case RECOVER_PRIMARY -> failoverManager.recoverPrimaryManually();
        }
    }

    public CacheMatchSource effectiveSource() {
        if (!configuration.isKvcmEnabled()) {
            return CacheMatchSource.LOCAL_SYNC;
        }
        return failoverManager.activeSource();
    }

    public CacheMatchStatus status() {
        KvcmHealthSnapshot health = failoverManager.healthSnapshot();
        CacheMatchSource source = effectiveSource();
        return new CacheMatchStatus(
                configuration.isKvcmEnabled(),
                configuration.isLocalStandbyEnabled(),
                configuration.getConfiguredMode(),
                configuration.isAutoSwitchEnabled(),
                source,
                health.state(),
                health.consecutiveQueryFailures(),
                health.consecutiveHeartbeatFailures(),
                health.consecutiveHeartbeatSuccesses(),
                health.lastHeartbeatSuccessTimeMs(),
                health.lastHeartbeatFailureTimeMs(),
                failoverManager.lastFailoverTimeMs(),
                failoverManager.lastFailoverReason(),
                localStandbyCacheManager.mappingCount(),
                localStandbyCacheManager.maximumEntryCount());
    }

    private Mono<CacheMatchResult> queryLocalSync(CacheMatchQuery query, long startTimeNs) {
        if (query.blockCacheKeys() == null || query.blockCacheKeys().isEmpty()) {
            return Mono.just(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
        }
        return localSyncProvider.findMatchingEngines(
                        query.requestId(), query.blockCacheKeys(), query.blockSize(), query.roleType(), query.group())
                .map(matches -> result(matches, CacheMatchSource.LOCAL_SYNC, startTimeNs, query.blockSize()));
    }

    private Mono<CacheMatchResult> queryLocalStandby(CacheMatchQuery query, long startTimeNs) {
        List<Long> blockCacheKeys = query.localStandbyBlockCacheKeys();
        if (blockCacheKeys != null && blockCacheKeys.isEmpty()) {
            return Mono.just(CacheMatchResult.empty(CacheMatchSource.LOCAL_STANDBY));
        }
        return Mono.fromFuture(localStandbyHashService.getHashResult(
                        query.requestId(), query.localStandbyBlockCacheKeys(), query.localStandbyBlockSize()))
                .flatMap(hashResult -> {
                    if (hashResult.blockCacheKeys().isEmpty()) {
                        return Mono.just(CacheMatchResult.empty(CacheMatchSource.LOCAL_STANDBY));
                    }
                    return localStandbyProvider.findMatchingEngines(
                                    query.requestId(),
                                    hashResult.blockCacheKeys(),
                                    hashResult.blockSize(),
                                    query.roleType(),
                                    query.group())
                            .map(matches -> result(
                                    matches, CacheMatchSource.LOCAL_STANDBY, startTimeNs, hashResult.blockSize()));
                });
    }

    private CacheMatchResult result(Map<String, Integer> matches, CacheMatchSource source, long startTimeNs, long blockSize) {
        return new CacheMatchResult(matches, source, (System.nanoTime() - startTimeNs) / 1_000, blockSize);
    }
}

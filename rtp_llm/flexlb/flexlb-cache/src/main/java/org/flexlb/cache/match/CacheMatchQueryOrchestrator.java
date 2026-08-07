package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchFailoverAction;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.cache.hash.LocalStandbyHashService;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.springframework.stereotype.Component;

import java.util.Collections;
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

    public CacheMatchResult findMatchingEngines(CacheMatchQuery query) {
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
            return emptyResult(CacheMatchSource.KVCM, startTimeNs);
        }

        try {
            Map<String, HostCacheMatch> kvcmMatches = kvcmProvider.findMatchingEngines(
                    query.requestId(), query.blockCacheKeys(), query.blockSize(), query.roleType(), query.group());
            comparisonService.trackLocalStandbyPrediction(query);
            return new CacheMatchResult(kvcmMatches, CacheMatchSource.KVCM, elapsedUs(startTimeNs), query.blockSize());
        } catch (RuntimeException e) {
            log.warn("KVCM cache query failed; requestId={}, action=LOCAL_STANDBY", query.requestId(), e);
            cacheMetricsReporter.reportStandbyFallback("kvcm_query_failure");
            return queryLocalStandby(query, startTimeNs);
        }
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

    private CacheMatchResult queryLocalSync(CacheMatchQuery query, long startTimeNs) {
        if (query.blockCacheKeys() == null || query.blockCacheKeys().isEmpty()) {
            return emptyResult(CacheMatchSource.LOCAL_SYNC, startTimeNs);
        }
        Map<String, HostCacheMatch> matches = localSyncProvider.findMatchingEngines(
                query.requestId(), query.blockCacheKeys(), query.blockSize(), query.roleType(), query.group());
        return result(matches, CacheMatchSource.LOCAL_SYNC, startTimeNs, query.blockSize());
    }

    private CacheMatchResult queryLocalStandby(CacheMatchQuery query, long startTimeNs) {
        List<Long> blockCacheKeys = query.localStandbyBlockCacheKeys();
        if (blockCacheKeys != null && blockCacheKeys.isEmpty()) {
            return emptyResult(CacheMatchSource.LOCAL_STANDBY, startTimeNs);
        }
        LocalStandbyHashResult hashResult = localStandbyHashService.getHashResult(
                query.requestId(), query.localStandbyBlockCacheKeys(), query.localStandbyBlockSize()).join();
        if (hashResult.blockCacheKeys().isEmpty()) {
            return emptyResult(CacheMatchSource.LOCAL_STANDBY, startTimeNs);
        }
        Map<String, HostCacheMatch> matches = localStandbyProvider.findMatchingEngines(
                query.requestId(), hashResult.blockCacheKeys(), hashResult.blockSize(), query.roleType(), query.group());
        return result(matches, CacheMatchSource.LOCAL_STANDBY, startTimeNs, hashResult.blockSize());
    }

    private CacheMatchResult result(Map<String, HostCacheMatch> matches, CacheMatchSource source, long startTimeNs, long blockSize) {
        return new CacheMatchResult(matches, source, elapsedUs(startTimeNs), blockSize);
    }

    private long elapsedUs(long startTimeNs) {
        return (System.nanoTime() - startTimeNs) / 1_000;
    }

    private CacheMatchResult emptyResult(CacheMatchSource source, long startTimeNs) {
        return new CacheMatchResult(
                Collections.emptyMap(), source, (System.nanoTime() - startTimeNs) / 1_000, 0);
    }
}

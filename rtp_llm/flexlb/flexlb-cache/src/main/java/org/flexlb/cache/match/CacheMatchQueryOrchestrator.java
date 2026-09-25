package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchFailoverAction;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Map;

/** Orchestrates the KVCM and complete local-snapshot cache matching sources. */
@Slf4j
@Component
public class CacheMatchQueryOrchestrator {

    private final LocalSyncCacheMatchProvider localSyncProvider;
    private final KvcmCacheMatchProvider kvcmProvider;
    private final LocalStandbyCacheMatchProvider localStandbyProvider;
    private final LocalStandbyCacheManager localStandbyCacheManager;
    private final CacheMatchFailoverManager failoverManager;
    private final LocalStandbyComparisonService comparisonService;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchConfiguration configuration;

    @Autowired
    public CacheMatchQueryOrchestrator(
            LocalSyncCacheMatchProvider localSyncProvider,
            KvcmCacheMatchProvider kvcmProvider,
            LocalStandbyCacheMatchProvider localStandbyProvider,
            LocalStandbyCacheManager localStandbyCacheManager,
            CacheMatchFailoverManager failoverManager,
            LocalStandbyComparisonService comparisonService,
            CacheMetricsReporter cacheMetricsReporter,
            CacheMatchConfiguration configuration) {
        this.localSyncProvider = localSyncProvider;
        this.kvcmProvider = kvcmProvider;
        this.localStandbyProvider = localStandbyProvider;
        this.localStandbyCacheManager = localStandbyCacheManager;
        this.failoverManager = failoverManager;
        this.comparisonService = comparisonService;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.configuration = configuration;
        log.info("Cache match query orchestrator initialized: source={}", effectiveSource());
    }

    public CacheMatchResult findMatchingEngines(CacheMatchQuery query) {
        long startTimeNs = System.nanoTime();
        if (!configuration.isKvcmEnabled()) {
            return queryLocalSync(query, startTimeNs);
        }
        CacheMatchSource source = failoverManager.activeSource();
        if (source == CacheMatchSource.LOCAL_STANDBY) {
            cacheMetricsReporter.reportStandbyFallback("active_source");
            return queryAndTrackLocalStandby(query, startTimeNs);
        }
        if (query.blockCacheKeys() == null || query.blockCacheKeys().isEmpty()) {
            return emptyResult(CacheMatchSource.KVCM, startTimeNs);
        }

        try {
            Map<String, HostCacheMatch> matches = kvcmProvider.findMatchingEngines(
                    query.requestId(), query.blockCacheKeys(), query.blockSize(),
                    query.roleType(), query.group());
            trackComparisonBestEffort(
                    query, () -> comparisonService.trackLocalStandbyPrediction(query));
            return new CacheMatchResult(
                    matches, CacheMatchSource.KVCM, elapsedUs(startTimeNs), query.blockSize());
        } catch (RuntimeException error) {
            log.warn("KVCM cache query failed; requestId={}, action=LOCAL_STANDBY",
                    query.requestId(), error);
            cacheMetricsReporter.reportStandbyFallback("kvcm_query_failure");
            return queryAndTrackLocalStandby(query, startTimeNs);
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
        return new CacheMatchStatus(
                configuration.isKvcmEnabled(),
                configuration.isLocalStandbyEnabled(),
                configuration.getConfiguredMode(),
                configuration.isAutoSwitchEnabled(),
                effectiveSource(),
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
                query.requestId(), query.blockCacheKeys(), query.blockSize(),
                query.roleType(), query.group());
        return new CacheMatchResult(
                matches, CacheMatchSource.LOCAL_SYNC, elapsedUs(startTimeNs), query.blockSize());
    }

    private CacheMatchResult queryLocalStandby(CacheMatchQuery query, long startTimeNs) {
        try {
            return localStandbyProvider.asyncLocalStandbyMatch(query).join();
        } catch (RuntimeException error) {
            log.warn("Local Standby cache query failed; requestId={}", query.requestId(), error);
            return CacheMatchResult.failed(CacheMatchSource.LOCAL_STANDBY, elapsedUs(startTimeNs));
        }
    }

    private CacheMatchResult queryAndTrackLocalStandby(
            CacheMatchQuery query, long startTimeNs) {
        CacheMatchResult result = queryLocalStandby(query, startTimeNs);
        trackComparisonBestEffort(
                query,
                () -> comparisonService.trackResolvedLocalStandbyPrediction(query, result));
        return result;
    }

    private void trackComparisonBestEffort(CacheMatchQuery query, Runnable tracker) {
        try {
            tracker.run();
        } catch (RuntimeException error) {
            log.warn("Local Standby comparison setup failed; requestId={}",
                    query.requestId(), error);
        }
    }

    private CacheMatchResult emptyResult(CacheMatchSource source, long startTimeNs) {
        return new CacheMatchResult(
                Collections.emptyMap(), source, elapsedUs(startTimeNs), 0);
    }

    private long elapsedUs(long startTimeNs) {
        return (System.nanoTime() - startTimeNs) / 1_000;
    }
}

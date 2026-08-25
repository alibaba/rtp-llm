package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Map;

/** Orchestrates the KVCM and complete local-snapshot cache matching sources. */
@Slf4j
@Component
public class CacheMatchQueryOrchestrator {

    private final LocalSyncCacheMatchProvider localSyncProvider;
    private final KvcmCacheMatchProvider kvcmProvider;
    private final CacheMatchConfiguration configuration;

    public CacheMatchQueryOrchestrator(
            LocalSyncCacheMatchProvider localSyncProvider,
            KvcmCacheMatchProvider kvcmProvider,
            CacheMatchConfiguration configuration) {
        this.localSyncProvider = localSyncProvider;
        this.kvcmProvider = kvcmProvider;
        this.configuration = configuration;
        log.info("Cache match query orchestrator initialized: source={}", effectiveSource());
    }

    public CacheMatchResult findMatchingEngines(CacheMatchQuery query) {
        long startTimeNs = System.nanoTime();
        CacheMatchSource source = effectiveSource();
        if (query.blockCacheKeys() == null || query.blockCacheKeys().isEmpty()) {
            return emptyResult(source, startTimeNs);
        }

        try {
            Map<String, HostCacheMatch> matches = source == CacheMatchSource.KVCM
                    ? kvcmProvider.findMatchingEngines(
                            query.requestId(), query.blockCacheKeys(), query.blockSize(),
                            query.roleType(), query.group())
                    : localSyncProvider.findMatchingEngines(
                            query.requestId(), query.blockCacheKeys(), query.blockSize(),
                            query.roleType(), query.group());
            return new CacheMatchResult(
                    matches, source, elapsedUs(startTimeNs), query.blockSize());
        } catch (RuntimeException error) {
            log.warn("Cache query failed; requestId={}, source={}",
                    query.requestId(), source, error);
            return CacheMatchResult.failed(source, elapsedUs(startTimeNs));
        }
    }

    public CacheMatchSource effectiveSource() {
        return configuration.isKvcmEnabled()
                ? CacheMatchSource.KVCM
                : CacheMatchSource.LOCAL_SYNC;
    }

    private CacheMatchResult emptyResult(CacheMatchSource source, long startTimeNs) {
        return new CacheMatchResult(
                Collections.emptyMap(), source, elapsedUs(startTimeNs), 0);
    }

    private long elapsedUs(long startTimeNs) {
        return (System.nanoTime() - startTimeNs) / 1_000;
    }
}

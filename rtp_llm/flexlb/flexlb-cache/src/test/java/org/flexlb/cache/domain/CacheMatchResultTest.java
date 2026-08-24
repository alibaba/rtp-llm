package org.flexlb.cache.domain;

import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class CacheMatchResultTest {

    @Test
    void shouldMatchLegacyPhysicalKvcmHostForSingleEngineWorker() {
        HostCacheMatch legacyMatch = HostCacheMatch.local(2);
        CacheMatchResult result = new CacheMatchResult(
                Map.of("10.0.0.1:8080", legacyMatch), CacheMatchSource.KVCM, 0, 1000);
        WorkerStatus worker = worker("10.0.0.1", 8080, 0, 1);

        assertSame(legacyMatch, result.hostMatch(worker));
    }

    @Test
    void shouldPreferLogicalKvcmHostForSingleEngineWorker() {
        HostCacheMatch logicalMatch = HostCacheMatch.local(3);
        CacheMatchResult result = new CacheMatchResult(
                Map.of(
                        "10.0.0.1:8080@0", logicalMatch,
                        "10.0.0.1:8080", HostCacheMatch.local(2)),
                CacheMatchSource.KVCM,
                0,
                1000);

        assertSame(logicalMatch, result.hostMatch(worker("10.0.0.1", 8080, 0, 1)));
    }

    @Test
    void shouldIgnoreLegacyPhysicalKvcmHostForMultiEngineWorker() {
        CacheMatchResult result = new CacheMatchResult(
                Map.of("10.0.0.1:8080", HostCacheMatch.local(2)), CacheMatchSource.KVCM, 0, 1000);

        assertNull(result.hostMatch(worker("10.0.0.1", 8080, 0, 2)));
    }

    @Test
    void shouldNotFallbackPhysicalHostOutsideKvcm() {
        WorkerStatus worker = worker("10.0.0.1", 8080, 0, 1);
        for (CacheMatchSource source : new CacheMatchSource[]{
                CacheMatchSource.LOCAL_SYNC, CacheMatchSource.LOCAL_STANDBY}) {
            CacheMatchResult result = new CacheMatchResult(
                    Map.of("10.0.0.1:8080", HostCacheMatch.local(2)), source, 0, 1000);

            assertNull(result.hostMatch(worker));
        }
    }

    @Test
    void shouldPreserveRawKvcmP2pMatchFields() {
        CacheMatchResult result = new CacheMatchResult(
                Map.of("10.0.0.1:8080", new HostCacheMatch(2, 8, 10)),
                CacheMatchSource.KVCM,
                0,
                1000);

        assertEquals(2, result.hostMatch("10.0.0.1:8080").localMatchBlocks());
        assertEquals(8, result.hostMatch("10.0.0.1:8080").p2pFetchBlocks());
        assertEquals(10, result.hostMatch("10.0.0.1:8080").p2pTotalMatchBlocks());
    }

    @Test
    void shouldUseLocalMatchWhenKVCMDoesNotReturnP2pDetails() {
        CacheMatchResult result = new CacheMatchResult(
                Map.of("10.0.0.1:8080", HostCacheMatch.local(2)), CacheMatchSource.KVCM, 0, 1000);

        assertEquals(2, result.hostMatch("10.0.0.1:8080").localMatchBlocks());
    }

    @Test
    void shouldCapMatchedTokensAtRequestLength() {
        assertEquals(5, CacheMatchResult.matchedTokens(2, 4, 5));
    }

    @Test
    void shouldKeepMatchedTokensWhenRequestLengthIsUnknown() {
        assertEquals(8, CacheMatchResult.matchedTokens(2, 4, 0));
    }

    private static WorkerStatus worker(String ip, int port, int engineIndex, int multiEngineNum) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(port);
        worker.setEngineIndex(engineIndex);
        worker.setMultiEngineNum(multiEngineNum);
        return worker;
    }
}

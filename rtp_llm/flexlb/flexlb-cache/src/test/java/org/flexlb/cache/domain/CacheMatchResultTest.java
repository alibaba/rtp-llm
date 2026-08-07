package org.flexlb.cache.domain;

import org.flexlb.dao.cache.HostCacheMatch;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CacheMatchResultTest {

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
}

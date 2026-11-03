package org.flexlb.mock.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mock.cache.CacheMetadata.Medium.*;

class CacheMetadataTest {
    @Test void onlyValidBlocksArePublishedAndTierMovesKeepOneLogicalIdentity() throws Exception {
        try (var cache = new KvCache(1, 4, 0, KvCache.EvictionPolicy.LRU, true, 2,
                KvCache.EvictionPolicy.LRU, new KvCache.Watermark(.25, .75), null)) {
            var changes = new ArrayList<CacheMetadata.Change>();
            try (var observation = cache.observeMetadata(changes::add)) {
                var lease = cache.acquire(1, List.of()).lease();
                assertTrue(cache.metadata().keys().isEmpty()); assertTrue(changes.isEmpty());
                lease.publish(List.of(11L), 1); lease.close();
                var initial = cache.metadata();
                assertEquals(Map.of(11L, HBM), initial.blocks());
                var keys = cache.metadataKeys(HBM);
                assertEquals(initial.version(), keys.version()); assertArrayEquals(new long[]{11L}, keys.keys());
                // Force device eviction to Host while the logical reusable prefix survives.
                try (var pressure = cache.acquire(3, List.of()).lease()) {
                    assertNotNull(pressure);
                    assertEquals(Map.of(11L, HOST), cache.metadata().blocks());
                    assertEquals(0, cache.metadataKeys(HBM).keys().length);
                    assertArrayEquals(new long[]{11L}, keys.keys());
                    assertEquals(Set.of(11L), cache.snapshot(-1, true).keys());
                    assertTrue(cache.metadata().version() > initial.version());
                }
                assertEquals(Map.of(11L, HBM), initial.blocks()); // immutable snapshot
                try (var reused = cache.acquire(1, List.of(11L)).lease()) {
                    assertEquals(1, reused.reusedTokens());
                    assertEquals(Map.of(11L, HBM), cache.metadata().blocks());
                }
                assertEquals(List.of(NONE, HBM, HOST), changes.stream().map(CacheMetadata.Change::before).toList());
                assertEquals(List.of(HBM, HOST, HBM), changes.stream().map(CacheMetadata.Change::after).toList());
                assertEquals(List.of(1L, 2L, 3L), changes.stream().map(CacheMetadata.Change::version).toList());
            }
        }
    }
    @org.junit.jupiter.api.Test void modelNamespaceDoesNotGuessBackendSpecificSeeds() throws Exception {
        var json = new com.fasterxml.jackson.databind.ObjectMapper();
        assertEquals(0, CacheSequence.cacheKeySeed(json.readTree("{\"model_type\":\"deepseek_v41\"}")));
        assertEquals(17, CacheSequence.cacheKeySeed(json.readTree("{\"cache_key_hash_seed\":17}")));
        assertThrows(IllegalArgumentException.class,
            () -> CacheSequence.cacheKeySeed(json.readTree("{\"cache_key_hash_seed\":1.5}")));
    }

}

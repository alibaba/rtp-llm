package org.flexlb.mock.engine.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mock.engine.cache.CacheMetadata.Medium.*;

class CacheMetadataTest {
    @Test void publishesOnlyValidBlocksAndReportsReclamation() throws Exception {
        try (var cache = new KvCache(1, 4, 0, KvCache.EvictionPolicy.LRU)) {
            var changes = new ArrayList<CacheMetadata.Change>();
            try (var observation = cache.observeMetadata(changes::add)) {
                var lease = cache.acquire(1, List.of()).lease();
                assertTrue(cache.metadata().keys().isEmpty()); assertTrue(changes.isEmpty());
                lease.publish(List.of(11L), 1); lease.close();
                var initial = cache.metadata();
                assertEquals(Map.of(11L, HBM), initial.blocks());
                var keys = cache.metadataKeys(HBM);
                assertEquals(initial.version(), keys.version()); assertArrayEquals(new long[]{11L}, keys.keys());
                // Force eviction from the single local pool.
                try (var pressure = cache.acquire(4, List.of()).lease()) {
                    assertNotNull(pressure);
                    assertTrue(cache.metadata().blocks().isEmpty());
                    assertEquals(0, cache.metadataKeys(HBM).keys().length);
                    assertArrayEquals(new long[]{11L}, keys.keys());
                    assertTrue(cache.snapshot(-1, true).keys().isEmpty());
                    assertTrue(cache.metadata().version() > initial.version());
                }
                assertEquals(Map.of(11L, HBM), initial.blocks()); // immutable snapshot
                assertEquals(List.of(NONE, HBM), changes.stream().map(CacheMetadata.Change::before).toList());
                assertEquals(List.of(HBM, NONE), changes.stream().map(CacheMetadata.Change::after).toList());
                assertEquals(List.of(1L, 2L), changes.stream().map(CacheMetadata.Change::version).toList());
            }
        }
    }
}

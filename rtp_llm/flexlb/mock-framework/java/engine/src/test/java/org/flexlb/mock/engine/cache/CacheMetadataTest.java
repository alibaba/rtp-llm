package org.flexlb.mock.engine.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CacheMetadataTest {
    @Test void publishesOnlyValidBlocksAndReportsReclamation() throws Exception {
        try (var cache = new KvCache(1, 4, 0)) {
            var changes = new ArrayList<CacheMetadata.Change>();
            try (var observation = cache.observeMetadata(changes::add)) {
                var lease = cache.acquire(1, List.of()).lease();
                assertEquals(0, cache.metadataKeys().keys().length); assertTrue(changes.isEmpty());
                lease.publish(List.of(11L), 1); lease.close();
                var initial = cache.metadataKeys();
                assertArrayEquals(new long[]{11L}, initial.keys());
                // Force eviction from the single local pool.
                try (var pressure = cache.acquire(4, List.of()).lease()) {
                    assertNotNull(pressure);
                    assertEquals(0, cache.metadataKeys().keys().length);
                    assertArrayEquals(new long[]{11L}, initial.keys());
                    assertEquals(0, cache.snapshot(-1, true).keys().length);
                    assertTrue(cache.metadataKeys().version() > initial.version());
                }
                assertEquals(List.of(true, false), changes.stream().map(CacheMetadata.Change::added).toList());
                assertEquals(List.of(1L, 2L), changes.stream().map(CacheMetadata.Change::version).toList());
            }
        }
    }
}

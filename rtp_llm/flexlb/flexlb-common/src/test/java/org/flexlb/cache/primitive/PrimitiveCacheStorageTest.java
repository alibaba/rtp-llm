package org.flexlb.cache.primitive;

import it.unimi.dsi.fastutil.HashCommon;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import org.junit.jupiter.api.Test;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrimitiveCacheStorageTest {
    @Test
    void randomMutationsAndDiffMatchReferenceIncludingExtremeKeys() {
        var actual = new LongKeySet();
        Set<Long> expected = new HashSet<>();
        Random random = new Random(817);
        for (long key : new long[]{0, Long.MIN_VALUE, Long.MAX_VALUE, -1}) {
            actual.add(key);
            expected.add(key);
        }
        for (int index = 0; index < 100_000; index++) {
            long key = random.nextInt(50_000);
            if (random.nextBoolean()) {
                assertEquals(expected.add(key), actual.add(key));
            } else {
                assertEquals(expected.remove(key), actual.remove(key));
            }
        }
        assertEquals(expected, actual);
        var retained = new LongKeySet(expected);
        retained.remove(0L);
        assertEquals(Set.of(0L), LongKeySet.difference(actual, retained));
        assertEquals(Set.of(0L), LongKeySet.difference(expected, retained));
        var clone = actual.clone();
        actual.clear();
        assertEquals(expected, clone);
    }

    @Test
    void concurrentReadersObserveExistingKeysDuringResize() throws Exception {
        var map = new LongObjectMap<String>();
        map.computeIfAbsent(Long.MIN_VALUE, key -> "retained");
        try (var executor = Executors.newFixedThreadPool(2)) {
            var writer = executor.submit(() -> {
                for (long key = 1; key <= 200_000; key++) {
                    map.computeIfAbsent(key, ignored -> "owner");
                }
            });
            var reader = executor.submit(() -> {
                for (int index = 0; index < 200_000; index++) {
                    assertEquals("retained", map.get(Long.MIN_VALUE));
                }
            });
            writer.get();
            reader.get();
        }
        assertEquals(200_001, map.mappingCount());
        assertEquals("retained", map.remove(Long.MIN_VALUE));
        assertNull(map.get(Long.MIN_VALUE));
    }

    @Test
    void shardsRetainDiverseInitialTableSlots() throws Exception {
        var map = new LongObjectMap<Boolean>();
        var random = new Random(817);
        for (int index = 0; index < 65_536; index++) {
            map.computeIfAbsent(random.nextLong(), ignored -> Boolean.TRUE);
        }
        var field = LongObjectMap.class.getDeclaredField("shards");
        field.setAccessible(true);
        var shards = (Object[]) field.get(map);
        for (Object storage : shards) {
            var shard = (Long2ObjectOpenHashMap<?>) storage;
            var slots = new BitSet(256);
            var keys = shard.keySet().iterator();
            while (keys.hasNext()) {
                slots.set((int) HashCommon.mix(keys.nextLong()) & 255);
            }
            assertTrue(shard.size() > 150 && shard.size() < 400,
                    "keys must spread across shards");
            assertTrue(slots.cardinality() > 100,
                    "shard selection must preserve diverse low hash bits for table slots");
        }
    }

    @Test
    void ownerIteratorsRetainPublishedSnapshot() {
        var owners = new SmallOwnerSet();
        owners.add("a");
        var snapshot = owners.iterator();
        owners.add("b");
        owners.remove("a");
        assertEquals("a", snapshot.next());
        assertFalse(snapshot.hasNext());
        assertEquals(Set.of("b"), owners);
    }
}

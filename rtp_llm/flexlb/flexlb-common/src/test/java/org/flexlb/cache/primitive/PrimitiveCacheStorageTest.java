package org.flexlb.cache.primitive;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

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
        map.computeIfAbsent(0, key -> "retained");
        try (var executor = Executors.newFixedThreadPool(2)) {
            var writer = executor.submit(() -> {
                for (long key = 1; key <= 200_000; key++) {
                    map.computeIfAbsent(key, ignored -> "owner");
                }
            });
            var reader = executor.submit(() -> {
                for (int index = 0; index < 200_000; index++) {
                    assertEquals("retained", map.get(0));
                }
            });
            writer.get();
            reader.get();
        }
        assertEquals(200_001, map.mappingCount());
        assertEquals("retained", map.remove(0));
        assertNull(map.get(0));
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

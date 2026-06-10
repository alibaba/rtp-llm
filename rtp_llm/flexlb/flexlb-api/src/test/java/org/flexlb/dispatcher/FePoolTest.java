package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.dispatcher.DispatcherTestSupport.fePool;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FePoolTest {

    @Test
    void roundRobinsAcrossAddresses() {
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals("http://a:8088", pool.next());
        assertEquals("http://b:8088", pool.next());
        assertEquals("http://a:8088", pool.next());
    }

    @Test
    void skipsDeadHostsPerPredicate() {
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088", "http://c:8088"),
                url -> !url.contains("b:"));
        for (int i = 0; i < 6; i++) {
            String picked = pool.next();
            assertNotEquals("http://b:8088", picked,
                    "host marked dead by predicate must never be returned");
        }
    }

    @Test
    void fallsBackToRoundRobinWhenAllDead() {
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088"),
                url -> false);
        String picked = pool.next();
        assertTrue(picked.startsWith("http://"),
                "all-dead fallback must still return a host, not refuse service");
    }

    @Test
    void readsDynamicSupplierOnEveryNext() {
        AtomicReference<List<String>> source = new AtomicReference<>(List.of("http://a:8088"));
        FePool pool = fePool(source::get, url -> true);
        assertEquals("http://a:8088", pool.next());

        source.set(List.of("http://b:8088", "http://c:8088"));
        // Pool must observe the new snapshot — not a cached copy from construction. Cursor is
        // shared, so the exact order across the swap depends on cumulative call count; only the
        // membership matters here.
        String first = pool.next();
        String second = pool.next();
        assertTrue(first.startsWith("http://b") || first.startsWith("http://c"),
                "post-swap call returned stale address: " + first);
        assertTrue(second.startsWith("http://b") || second.startsWith("http://c"),
                "post-swap call returned stale address: " + second);
        assertNotEquals(first, second, "two consecutive next() on a 2-host snapshot must alternate");
    }

    @Test
    void emptySupplierSnapshotThrowsOnNext() {
        FePool pool = fePool(List.of());
        assertThrows(IllegalStateException.class, pool::next);
    }
}

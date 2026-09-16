package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import static org.junit.jupiter.api.Assertions.*;

class WhaleCacheDiagnosticsTest {
    @Test void separatesLocalRetentionRoutingAndSharedHistory() {
        var diag = new MockCacheDiagnostics(120, 100, 1, () -> 0L);
        diag.completed("p0", List.of(1L));
        diag.completed("p1", List.of(1L, 2L));
        assertTrue(diag.select());
        diag.observe("p0", List.of(1L, 2L, 3L), 512, 1536, 0, 2, "p1");
        var s = diag.snapshot();
        assertEquals(512L, s.get("local_no_eviction_tokens"));
        assertEquals(1024L, s.get("best_resident_tokens"));
        assertEquals(1024L, s.get("global_no_eviction_tokens"));
        assertEquals(0L, s.get("actual_tokens"));
        // Observing alone never populates history, including a later failed request.
        diag.observe("p0", List.of(1L, 2L, 3L), 512, 1536, 0, 0, "");
        assertEquals(2048L, diag.snapshot().get("global_no_eviction_tokens"));
    }

    @Test void stopsAtBudgetInsteadOfSilentlyEvictingShadowHistory() {
        var diag = new MockCacheDiagnostics(120, 3, 1, () -> 0L);
        diag.completed("p0", List.of(1L, 2L));
        assertFalse(diag.select());
        assertEquals("key_limit", diag.snapshot().get("stop_reason"));
        assertEquals(2L, diag.snapshot().get("stored_key_entries"));
    }

    @Test void boundsDurationAndSamplingAndRequiresContinuousPrefix() {
        var clock = new AtomicLong();
        var diag = new MockCacheDiagnostics(1, 100, 2, clock::get);
        diag.completed("p0", List.of(1L, 3L));
        assertTrue(diag.select());
        assertFalse(diag.select());
        assertTrue(diag.select());
        diag.observe("p0", List.of(1L, 2L, 3L), 512, 1536, 0, 0, "");
        assertEquals(512L, diag.snapshot().get("local_no_eviction_tokens"));
        clock.set(1000);
        assertFalse(diag.select());
        assertEquals("duration_limit", diag.snapshot().get("stop_reason"));
    }

    @Test void diagnosticReadsDoNotRefreshMemoryLru() {
        var memory = new MockMemoryBlockCache(2, ignored -> {});
        memory.write(List.of(1L, 2L));
        assertEquals(1, memory.peekMatch(List.of(1L), 0));
        memory.write(List.of(3L));
        assertEquals(List.of(2L, 3L), memory.keys());
    }
}

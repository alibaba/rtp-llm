package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the java_mock_stats decode completion window
 * ({@link JavaMockEngineCluster.ClusterStats#recordDecodeDone} /
 * {@link JavaMockEngineCluster.ClusterStats#drainDecodeWindow}): per-sample
 * count and execution-time p50/p95/max, and the drain-reset semantics.
 */
class ClusterStatsDecodeWindowTest {

    @Test
    void drainOnEmptyWindowReturnsZeros() {
        JavaMockEngineCluster.ClusterStats stats = new JavaMockEngineCluster.ClusterStats();
        JavaMockEngineCluster.ClusterStats.DecodeWindow window = stats.drainDecodeWindow();
        assertEquals(0, window.count());
        assertEquals(0, window.p50Ms());
        assertEquals(0, window.p95Ms());
        assertEquals(0, window.maxMs());
    }

    @Test
    void drainSummarizesAndResetsWindow() {
        JavaMockEngineCluster.ClusterStats stats = new JavaMockEngineCluster.ClusterStats();
        // 1..100 ms: p50 = 50, p95 = 95, max = 100 with ceil-based percentile index.
        for (long ms = 1; ms <= 100; ms++) {
            stats.recordDecodeDone(ms);
        }
        JavaMockEngineCluster.ClusterStats.DecodeWindow window = stats.drainDecodeWindow();
        assertEquals(100, window.count());
        assertEquals(50, window.p50Ms());
        assertEquals(95, window.p95Ms());
        assertEquals(100, window.maxMs());

        // Drain resets: the next window starts from scratch.
        JavaMockEngineCluster.ClusterStats.DecodeWindow empty = stats.drainDecodeWindow();
        assertEquals(0, empty.count());
        assertEquals(0, empty.maxMs());
    }

    @Test
    void windowBeyondSampleCapKeepsExactCountAndMax() {
        JavaMockEngineCluster.ClusterStats stats = new JavaMockEngineCluster.ClusterStats();
        int total = 20_000; // > DECODE_WINDOW_SAMPLE_CAP (8192)
        for (int i = 1; i <= total; i++) {
            stats.recordDecodeDone(i);
        }
        JavaMockEngineCluster.ClusterStats.DecodeWindow window = stats.drainDecodeWindow();
        assertEquals(total, window.count(), "count is exact, not reservoir-bounded");
        assertEquals(total, window.maxMs(), "max is tracked exactly");
        // Percentiles are reservoir approximations; assert sane ordering/bounds.
        assertTrue(window.p50Ms() >= 1 && window.p50Ms() <= total);
        assertTrue(window.p95Ms() >= window.p50Ms());
        assertTrue(window.p95Ms() <= window.maxMs());
    }
}

package org.flexlb.mockengine;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class MetricAlignmentTest {
    @TempDir Path directory;

    @Test
    void realNamesMustHaveCurrentCppProducers() throws Exception {
        Path root = Path.of("").toAbsolutePath();
        while (root != null && !Files.exists(root.resolve("rtp_llm/cpp/metrics/RtpLLMMetrics.cc")))
            root = root.getParent();
        assertNotNull(root, "C++ source registry must be available to validate real names");
        String registry = Files.readString(root.resolve("rtp_llm/cpp/metrics/RtpLLMMetrics.cc"))
                + Files.readString(root.resolve("rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.cc"));
        for (var metric : MockMetricContract.ALL)
            if (metric.name().startsWith("rtp_llm_"))
                assertTrue(registry.contains("\"" + metric.name() + "\""), metric.name());
        assertEquals(MockMetricContract.Unit.TOKENS,
                MockMetricContract.require("rtp_llm_generate_tps").unit());
        assertEquals(MockMetricContract.Unit.TOKENS_PER_SECOND,
                MockMetricContract.require("mock_decode_wall_tps").unit());
    }

    @Test
    void cacheHitWindowWeightsTokensAndDoesNotInventIdleSamples() {
        var a = new CacheHitMetrics.Reader();
        var b = new CacheHitMetrics.Reader();
        var snapshot = new CacheHitMetrics.Snapshot(0, 1000, 100);
        // One fully hit 100-token request plus a cold 900-token request is 10%, not 50%.
        assertTrue(a.sample(snapshot, 59_000_000_000L).isEmpty());
        assertEquals(10, a.sample(snapshot, 60_000_000_000L).orElseThrow());
        assertEquals(10, b.sample(snapshot, 60_000_000_000L).orElseThrow());
        assertTrue(a.sample(snapshot, 120_000_000_000L).isEmpty());
        var restarted = new CacheHitMetrics.Snapshot(120_000_000_001L, 100, 100);
        assertTrue(a.sample(restarted, 120_000_000_002L).isEmpty());
        assertEquals(100, a.sample(restarted, 180_000_000_001L).orElseThrow());
    }

    @Test
    void reclaimableCacheStillOccupiesPool() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 63060, 1, 0)) {
            var engine = cluster.prefill(0);
            var field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
            field.setAccessible(true);
            var cache = (MockLruBlockCache) field.get(engine);
            cache.admit(java.util.List.of(1L, 2L));
            assertEquals(cache.totalBlocks(), cache.availableBlocks());
            assertEquals(200.0 / cache.totalBlocks(),
                    engine.whaleMetrics().get("rtp_llm_kv_cache_pool_used_ratio").doubleValue());
            assertFalse(engine.whaleMetrics().containsKey("mock_prefill_waiting_requests"));
        }
    }
}

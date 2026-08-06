package org.flexlb.cache.domain;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheHitComparisonResultTest {

    @Test
    void serializesNestedCacheHitComparison() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                new CacheHitComparisonResult.HitComparison(70, 50),
                new CacheHitComparisonResult.KvcmDetails(40, 30));

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertTrue(json.contains("\"event\":\"cache_hit_comparison\""));
        assertTrue(json.contains("\"source\":\"KVCM\""));
        assertTrue(json.contains("\"worker\":\"127.0.0.1\""));
        assertTrue(json.contains("\"state\":\"running\""));
        assertTrue(json.contains("\"actual\":{\"hit\":120}"));
        assertTrue(json.contains("\"kvcm\":{\"hit\":100,\"delta\":20}"));
        assertTrue(json.contains("\"localStandby\":{\"hit\":70,\"delta\":50}"));
        assertSame(comparison.routing(), comparison.kvcm());
        assertFalse(json.contains("\"routing\""));
        assertFalse(json.contains("\"kvcmDetails\""));
        assertTrue(json.indexOf("\"actual\"") < json.indexOf("\"kvcm\""));
        assertTrue(json.indexOf("\"kvcm\"") < json.indexOf("\"localStandby\""));
        assertFalse(json.contains("\"p2pFetch\""));
        assertFalse(json.contains("\"workerPort\""));
    }

    @Test
    void omitsUnavailableLocalStandbyPrediction() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                null,
                null);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"localStandby\""));
    }

    @Test
    void omitsKvcmForNonKvcmSource() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                null,
                null);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"kvcm\""));
        assertNull(comparison.kvcm());
    }
}

package org.flexlb.cache.domain;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheHitComparisonResultTest {

    @Test
    void serializesNestedCacheHitComparison() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                100, true, 70, true, 120, 20, 40, 30, 50);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertTrue(json.contains("\"event\":\"cache_hit_comparison\""));
        assertTrue(json.contains("\"source\":\"KVCM\""));
        assertTrue(json.contains("\"worker\":\"127.0.0.1\""));
        assertTrue(json.contains("\"state\":\"running\""));
        assertTrue(json.contains("\"actual\":{\"hit\":120}"));
        assertTrue(json.contains("\"kvcm\":{\"hit\":100,\"delta\":20}"));
        assertTrue(json.contains("\"localStandby\":{\"hit\":70,\"delta\":50}"));
        assertFalse(json.contains("\"routingPredictedHitTokens\""));
        assertFalse(json.contains("\"kvcmPredictionAvailable\""));
        assertFalse(json.contains("\"p2pFetch\""));
        assertFalse(json.contains("\"workerPort\""));
    }

    @Test
    void omitsUnavailableLocalStandbyPrediction() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                100, 0, false, 120, 20, 0);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"localStandby\""));
    }

    @Test
    void omitsKvcmForNonKvcmSource() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "default",
                "127.0.0.1", "running", 200,
                100, 0, false, 120, 20, 0);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"kvcm\""));
    }
}

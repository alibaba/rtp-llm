package org.flexlb.cache.domain;

import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheHitComparisonResultTest {

    @Test
    void serializesNestedCacheHitComparison() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                new WorkerIdentity("127.0.0.1", 8080, 0), "running", 200,
                120,
                new CacheHitComparisonResult.CachePrediction(100, 60, 110),
                null,
                new CacheHitComparisonResult.CachePrediction(70, -1, -1));

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertTrue(json.contains("\"event\":\"cache_hit_comparison\""));
        assertTrue(json.contains("\"source\":\"KVCM\""));
        assertTrue(json.contains("\"worker\":\"127.0.0.1:8080@0\""));
        assertTrue(json.contains("\"state\":\"running\""));
        assertTrue(json.contains("\"actualHitTokens\":120"));
        assertTrue(json.contains(
                "\"kvcmPrediction\":{\"predictedHitTokens\":100,"
                        + "\"localPredictionTokens\":60,\"globalPredictionTokens\":110}"));
        assertTrue(json.contains(
                "\"localStandbyPrediction\":{\"predictedHitTokens\":70,"
                        + "\"localPredictionTokens\":-1,\"globalPredictionTokens\":-1}"));
        assertEquals(100, comparison.kvcmPrediction().predictedHitTokens());
        assertEquals(20, comparison.actualHitTokens() - comparison.kvcmPrediction().predictedHitTokens());
        assertFalse(json.contains("\"routing\""));
        assertTrue(json.indexOf("\"actualHitTokens\"") < json.indexOf("\"kvcmPrediction\""));
        assertTrue(json.indexOf("\"kvcmPrediction\"") < json.indexOf("\"localStandbyPrediction\""));
        assertFalse(json.contains("p2p"));
        assertFalse(json.contains("\"workerPort\""));
        assertFalse(json.contains("\"ipIndex\""));
    }

    @Test
    void omitsUnavailableLocalStandbyPrediction() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                new WorkerIdentity("127.0.0.1", 8080, 0), "running", 200,
                120,
                new CacheHitComparisonResult.CachePrediction(100, -1, -1),
                null,
                null);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"localStandbyPrediction\""));
    }

    @Test
    void omitsKvcmForNonKvcmSource() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "default",
                new WorkerIdentity("127.0.0.1", 8080, 0), "running", 200,
                120,
                null,
                new CacheHitComparisonResult.CachePrediction(100, -1, -1),
                null);

        String json = JsonUtils.toStringOrEmpty(comparison);

        assertFalse(json.contains("\"kvcmPrediction\""));
        assertTrue(json.contains("\"localSyncPrediction\""));
    }
}

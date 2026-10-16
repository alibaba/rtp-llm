package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WhaleCacheFixtureMetricsTest {
    @Test
    void monitoringReadsDoNotProduceResponsesOrChangeCacheVersion() {
        var fixture = new CacheSyncFixture(0, 6, 176, 65, 2, true, 0, () -> 0L);
        var full = fixture.response(-1);
        fixture.response(2);
        var metrics = new java.util.HashMap<String, Number>();
        fixture.collectMetrics(metrics);
        fixture.collectMetrics(metrics);
        assertEquals(306, metrics.get("mock_cache_fixture_keys").intValue());
        assertEquals(2, metrics.get("mock_cache_fixture_responses_total").longValue());
        assertEquals(306, metrics.get("mock_cache_fixture_entries_total").longValue());
        assertEquals(full.getSerializedSize() + EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(2).build().getSerializedSize(),
                metrics.get("mock_cache_fixture_response_bytes_total").longValue());
        assertEquals(1, metrics.get("mock_cache_fixture_full_responses_total").longValue());
        assertEquals(1, metrics.get("mock_cache_fixture_unchanged_responses_total").longValue());
        assertEquals(2, fixture.version());
        fixture.grow(180, 66);
        fixture.collectMetrics(metrics);
        assertEquals(312, metrics.get("mock_cache_fixture_keys").intValue());
        assertEquals(6, metrics.get("mock_cache_fixture_added_keys_total").longValue());
        assertEquals(0, metrics.get("mock_cache_fixture_removed_keys_total").longValue());
    }
    @Test
    void fixtureRatesPreserveRolesAndNeverEmitDecodeThroughput() {
        var reports = new java.util.HashMap<String, Double>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report")) reports.put((String) args[0], ((Number) args[2]).doubleValue());
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var labels = java.util.Map.of("role", "ROLE_TYPE_PREFILL", "engine", "prefill-0");
        long now = System.nanoTime();
        monitor.sample(java.util.Map.of("mock_cache_fixture_responses_total", 10L,
                "mock_cache_fixture_response_bytes_total", 1000L), labels, now);
        reports.clear();
        monitor.sample(java.util.Map.of("mock_cache_fixture_responses_total", 14L,
                "mock_cache_fixture_response_bytes_total", 5000L), labels, now + 2_000_000_000L);
        assertEquals(2.0, reports.get("mock_cache_fixture_responses_per_second"));
        assertEquals(2000.0, reports.get("mock_cache_fixture_response_bytes_per_second"));
        assertFalse(reports.containsKey("rtp_llm_generate_tps"));
        reports.clear();
        monitor.sample(java.util.Map.of("mock_cache_fixture_keys", 306000),
                java.util.Map.of("role", "ROLE_TYPE_DECODE"), now);
        assertFalse(reports.containsKey("mock_cache_fixture_keys"));
    }
}

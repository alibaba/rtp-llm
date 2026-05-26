package org.flexlb.config;

import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigServiceTest {

    @Test
    void should_load_traffic_policy_from_standalone_env_config() {
        ConfigService configService = new ConfigService(Map.of(
                "TRAFFIC_POLICY_CONFIG", """
                        {
                          "default_group": "standalone-group"
                        }
                        """));

        assertEquals("standalone-group", configService.loadBalanceConfig()
                .getTrafficPolicy()
                .resolveTargetGroup(request())
                .orElseThrow());
    }

    @Test
    void standalone_traffic_policy_should_override_embedded_flexlb_config() {
        ConfigService configService = new ConfigService(Map.of(
                "FLEXLB_CONFIG", """
                        {
                          "trafficPolicy": {
                            "default_group": "embedded-group"
                          }
                        }
                        """,
                "TRAFFIC_POLICY_CONFIG", """
                        {
                          "default_group": "standalone-group"
                        }
                        """));

        assertEquals("standalone-group", configService.loadBalanceConfig()
                .getTrafficPolicy()
                .resolveTargetGroup(request())
                .orElseThrow());
    }

    @Test
    void should_load_traffic_policy_from_standalone_file(@TempDir Path tempDir) throws Exception {
        Path policyFile = tempDir.resolve("traffic-policy.json");
        Files.writeString(policyFile, """
                {
                  "default_group": "file-group"
                }
                """, StandardCharsets.UTF_8);

        ConfigService configService = new ConfigService(Map.of(
                "TRAFFIC_POLICY_CONFIG_FILE", policyFile.toString()));

        assertEquals("file-group", configService.loadBalanceConfig()
                .getTrafficPolicy()
                .resolveTargetGroup(request())
                .orElseThrow());
    }

    @Test
    void should_keep_scalar_env_overrides_with_injected_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "DECODE_CONCURRENCY_LIMIT", "32"));

        assertEquals(32, configService.loadBalanceConfig().getDecodeConcurrencyLimit());
    }

    @Test
    void should_override_cache_hit_time_window_ms_with_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "CACHE_HIT_TIME_WINDOW_MS", "600000"));

        assertEquals(600000L, configService.loadBalanceConfig().getCacheHitTimeWindowMs());
    }

    @Test
    void should_override_cache_hit_max_cache_keys_with_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "CACHE_HIT_MAX_CACHE_KEYS", "123456"));

        assertEquals(123456L, configService.loadBalanceConfig().getCacheHitMaxCacheKeys());
    }

    @Test
    void should_override_cache_hit_switches_with_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "CACHE_HIT_WINDOW_WRITE_ENABLED", "false",
                "CACHE_HIT_METRIC_REPORT_ENABLED", "false",
                "CACHE_HIT_TRACE_LOG_ENABLED", "true"));

        assertFalse(configService.loadBalanceConfig().isCacheHitWindowWriteEnabled());
        assertFalse(configService.loadBalanceConfig().isCacheHitMetricReportEnabled());
        assertTrue(configService.loadBalanceConfig().isCacheHitTraceLogEnabled());
    }

    private Request request() {
        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128L);
        return request;
    }
}

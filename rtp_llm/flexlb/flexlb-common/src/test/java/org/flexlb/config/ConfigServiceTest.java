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
                "CACHE_HIT_TRACE_LOG_ENABLED", "true",
                "CACHE_HIT_THEORY_LOG_ENABLED", "false"));

        assertFalse(configService.loadBalanceConfig().isCacheHitWindowWriteEnabled());
        assertFalse(configService.loadBalanceConfig().isCacheHitMetricReportEnabled());
        assertTrue(configService.loadBalanceConfig().isCacheHitTraceLogEnabled());
        assertFalse(configService.loadBalanceConfig().isCacheHitTheoryLogEnabled());
    }

    @Test
    void should_use_default_strategy_configs_without_environment() {
        ConfigService configService = new ConfigService(Map.of());

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.RATIO, candidatePool.getMode());
        assertEquals(0.3, candidatePool.getRatio());
        assertEquals(1, candidatePool.getMinSize());
        assertEquals(1, candidatePool.getSize());
        assertEquals(3, candidatePool.resolveCandidateCount(10));
    }

    @Test
    void should_load_shortest_ttft_strategy_configs_from_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", """
                        {
                          "shortestTtft": {
                            "candidatePool": {
                              "mode": "FIXED",
                              "size": 1
                            }
                          }
                        }
                        """));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.FIXED, candidatePool.getMode());
        assertEquals(1, candidatePool.getSize());
        assertEquals(0.3, candidatePool.getRatio());
        assertEquals(1, candidatePool.getMinSize());
    }

    @Test
    void should_load_strategy_config_enum_case_insensitively() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", """
                        {
                          "shortestTtft": {
                            "candidatePool": {
                              "mode": "fixed",
                              "size": 2
                            }
                          }
                        }
                        """));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.FIXED, candidatePool.getMode());
        assertEquals(2, candidatePool.getSize());
    }

    @Test
    void should_keep_strategy_config_defaults_for_missing_fields() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", """
                        {
                          "shortestTtft": {
                            "candidatePool": {
                              "ratio": 0.5
                            }
                          }
                        }
                        """));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.RATIO, candidatePool.getMode());
        assertEquals(0.5, candidatePool.getRatio());
        assertEquals(1, candidatePool.getMinSize());
        assertEquals(1, candidatePool.getSize());
        assertEquals(2, candidatePool.resolveCandidateCount(4));
    }

    @Test
    void should_normalize_invalid_strategy_candidate_pool_values() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", """
                        {
                          "shortestTtft": {
                            "candidatePool": {
                              "mode": "FIXED",
                              "size": 0,
                              "ratio": 2.0,
                              "minSize": 0
                            }
                          }
                        }
                        """));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.FIXED, candidatePool.getMode());
        assertEquals(0.3, candidatePool.getRatio());
        assertEquals(1, candidatePool.getMinSize());
        assertEquals(1, candidatePool.getSize());
    }

    @Test
    void should_normalize_invalid_strategy_candidate_pool_mode() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", """
                        {
                          "shortestTtft": {
                            "candidatePool": {
                              "mode": "BAD",
                              "size": 2
                            }
                          }
                        }
                        """));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.RATIO, candidatePool.getMode());
        assertEquals(2, candidatePool.getSize());
    }

    @Test
    void should_fallback_default_strategy_configs_when_environment_json_is_malformed() {
        ConfigService configService = new ConfigService(Map.of(
                "STRATEGY_CONFIGS", "{\"shortestTtft\":"));

        StrategyConfigs.CandidatePoolConfig candidatePool = configService.getStrategyConfigs()
                .getShortestTtft()
                .getCandidatePool();
        assertEquals(StrategyConfigs.CandidatePoolMode.RATIO, candidatePool.getMode());
        assertEquals(0.3, candidatePool.getRatio());
        assertEquals(1, candidatePool.getMinSize());
        assertEquals(1, candidatePool.getSize());
    }

    private Request request() {
        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128L);
        return request;
    }
}

package org.flexlb.config;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    void auto_tpm_defaults_are_all_off() {
        ConfigService configService = new ConfigService(Map.of());

        FlexlbConfig config = configService.loadBalanceConfig();
        assertFalse(config.isAutoTpmEnabled());
        assertFalse(config.isAutoTpmPrefillQueueEvictEnabled());
        assertFalse(config.isAutoTpmDecodeReservedEvictEnabled());
        assertEquals(50, config.getAutoTpmDefaultPriority());
        // §18 reserved fields (future phases, not wired yet): defaults
        assertEquals("30,40,50,60,70", config.getAutoTpmPriorityLevels());
        assertFalse(config.isAutoTpmDecodeAcceptedEvictEnabled());
        assertFalse(config.isAutoTpmDeadlineRescueEnabled());
        assertEquals(20L, config.getAutoTpmRescueScanIntervalMs());
        assertEquals(32, config.getAutoTpmMaxRescuePerTick());
        assertEquals(1, config.getAutoTpmMaxTransferCount());
        assertEquals(50L, config.getAutoTpmCommitWaitReleaseTimeoutMs());
    }

    @Test
    void should_override_auto_tpm_fields_with_environment() {
        ConfigService configService = new ConfigService(Map.of(
                "AUTO_TPM_ENABLED", "true",
                "AUTO_TPM_DEFAULT_PRIORITY", "60",
                "AUTO_TPM_SLO_LENGTH_BUCKETS", "512:200,*:3000",
                "AUTO_TPM_PRIORITY_SLO_MULTIPLIERS", "30:3.0,50:1.0",
                "AUTO_TPM_PREFILL_QUEUE_EVICT_ENABLED", "true",
                "AUTO_TPM_DECODE_RESERVED_EVICT_ENABLED", "true",
                "AUTO_TPM_DANGER_THRESHOLD_MS", "250"));

        FlexlbConfig config = configService.loadBalanceConfig();
        assertTrue(config.isAutoTpmEnabled());
        assertEquals(60, config.getAutoTpmDefaultPriority());
        assertEquals("512:200,*:3000", config.getAutoTpmSloLengthBuckets());
        assertEquals("30:3.0,50:1.0", config.getAutoTpmPrioritySloMultipliers());
        assertTrue(config.isAutoTpmPrefillQueueEvictEnabled());
        assertTrue(config.isAutoTpmDecodeReservedEvictEnabled());
        assertEquals(250L, config.getAutoTpmDangerThresholdMs());
    }

    @Test
    void should_override_auto_tpm_reserved_fields_with_environment() {
        // Spot-check of the §18 reserved fields via env override
        ConfigService configService = new ConfigService(Map.of(
                "AUTO_TPM_PRIORITY_LEVELS", "10,20,30",
                "AUTO_TPM_DECODE_ACCEPTED_EVICT_ENABLED", "true",
                "AUTO_TPM_DEADLINE_RESCUE_ENABLED", "true",
                "AUTO_TPM_RESCUE_SCAN_INTERVAL_MS", "40",
                "AUTO_TPM_MAX_RESCUE_PER_ENDPOINT_PER_TICK", "4",
                "AUTO_TPM_MAX_TRANSFER_COUNT", "2",
                "AUTO_TPM_COMMIT_WAIT_RELEASE_TIMEOUT_MS", "100"));

        FlexlbConfig config = configService.loadBalanceConfig();
        assertEquals("10,20,30", config.getAutoTpmPriorityLevels());
        assertTrue(config.isAutoTpmDecodeAcceptedEvictEnabled());
        assertTrue(config.isAutoTpmDeadlineRescueEnabled());
        assertEquals(40L, config.getAutoTpmRescueScanIntervalMs());
        assertEquals(4, config.getAutoTpmMaxRescuePerEndpointPerTick());
        assertEquals(2, config.getAutoTpmMaxTransferCount());
        assertEquals(100L, config.getAutoTpmCommitWaitReleaseTimeoutMs());
    }

    @Test
    void dump_effective_config_logs_auto_tpm_fields() {
        Logger logger = (Logger) LoggerFactory.getLogger(ConfigService.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            new ConfigService(Map.of("AUTO_TPM_ENABLED", "true"));
        } finally {
            logger.detachAppender(appender);
        }

        List<String> lines = appender.list.stream()
                .map(ILoggingEvent::getFormattedMessage)
                .toList();
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmEnabled=true")),
                "dumpEffectiveConfig should log autoTpmEnabled");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmSloLengthBuckets=")),
                "dumpEffectiveConfig should log autoTpmSloLengthBuckets");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmDangerThresholdMs=")),
                "dumpEffectiveConfig should log autoTpmDangerThresholdMs");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmPriorityLevels=")),
                "dumpEffectiveConfig should log autoTpmPriorityLevels");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmDeadlineRescueEnabled=")),
                "dumpEffectiveConfig should log autoTpmDeadlineRescueEnabled");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmMaxRescuePerEndpointPerTick=")),
                "dumpEffectiveConfig should log autoTpmMaxRescuePerEndpointPerTick");
        assertTrue(lines.stream().anyMatch(line -> line.contains("autoTpmCommitWaitReleaseTimeoutMs=")),
                "dumpEffectiveConfig should log autoTpmCommitWaitReleaseTimeoutMs");
    }

    // ---- F3 (P0-3): unmatched env var scan ----

    @Test
    void unmatched_env_scan_reports_only_prefixed_unknown_names() {
        Map<String, String> environment = Map.of(
                "AUTO_TPM_ENABLED", "true",                  // correct name → matched
                "FLEXLB_BATCH_QUEUE_MAX_SIZE", "2048",       // correct name → matched
                "COST_FORMULA", "1.0",                       // correct name → matched
                "AUTO_TPM_ENABLE", "true",                   // misspelled → warned
                "FLEXLB_BATCH_QUEUE_MAXSIZE", "2048",        // misspelled → warned
                "MAX_QUEUE_SIZE", "5000",                    // no scanned prefix → out of scope
                "PATH", "/usr/bin",                          // unrelated → ignored
                "FLEXLB_CONFIG", "{}",                       // special entry point → matched
                "FLEXLB_BATCH_ENABLED", "true");             // deprecated → dedicated warning only

        assertEquals(List.of("AUTO_TPM_ENABLE", "FLEXLB_BATCH_QUEUE_MAXSIZE"),
                ConfigService.findUnmatchedEnvVars(environment));
    }

    @Test
    void unmatched_env_scan_suggests_nearest_known_name() {
        assertEquals("AUTO_TPM_ENABLED", ConfigService.nearestKnownEnvName(
                "AUTO_TPM_ENABLE", ConfigService.knownEnvVarNames()));
    }

    @Test
    void unmatched_env_var_logs_warn_but_does_not_abort() {
        Logger logger = (Logger) LoggerFactory.getLogger(ConfigService.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            new ConfigService(Map.of("AUTO_TPM_ENABLE", "true"));
        } finally {
            logger.detachAppender(appender);
        }

        assertTrue(appender.list.stream()
                        .map(ILoggingEvent::getFormattedMessage)
                        .anyMatch(line -> line.contains("AUTO_TPM_ENABLE")
                                && line.contains("未匹配任何配置字段，将被忽略")),
                "unmatched env var must produce a warn log with the variable name");
    }

    // ---- F4 (P0-4): critical config expansion + SLO spec startup validation ----

    @Test
    void invalid_auto_tpm_enabled_aborts_startup() {
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of("AUTO_TPM_ENABLED", "notabool")));
    }

    @Test
    void invalid_flexlb_batch_queue_max_size_aborts_startup() {
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of("FLEXLB_BATCH_QUEUE_MAX_SIZE", "abc")));
    }

    @Test
    void invalid_slo_length_buckets_abort_startup_with_invalid_fragment() {
        ConfigValidationException e = assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(
                        "AUTO_TPM_SLO_LENGTH_BUCKETS", "256150,1024:300")));
        assertTrue(e.getMessage().contains("256150"),
                "abort message must name the invalid fragment: " + e.getMessage());
    }

    @Test
    void invalid_priority_slo_multipliers_abort_startup_with_invalid_fragment() {
        ConfigValidationException e = assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(
                        "AUTO_TPM_PRIORITY_SLO_MULTIPLIERS", "30:0,50:1.0")));
        assertTrue(e.getMessage().contains("30:0"),
                "abort message must name the invalid fragment: " + e.getMessage());
    }

    @Test
    void valid_slo_specs_pass_startup_validation() {
        ConfigService configService = new ConfigService(Map.of(
                "AUTO_TPM_SLO_LENGTH_BUCKETS", "512:200,*:3000",
                "AUTO_TPM_PRIORITY_SLO_MULTIPLIERS", "30:3.0,50:1.0"));

        assertEquals("512:200,*:3000",
                configService.loadBalanceConfig().getAutoTpmSloLengthBuckets());
    }

    @Test
    void blank_slo_specs_pass_startup_validation() {
        // Blank means "use built-in default" and must not abort.
        ConfigService configService = new ConfigService(Map.of(
                "FLEXLB_CONFIG", """
                        {
                          "autoTpmSloLengthBuckets": "",
                          "autoTpmPrioritySloMultipliers": ""
                        }
                        """));

        assertEquals("", configService.loadBalanceConfig().getAutoTpmSloLengthBuckets());
    }

    private Request request() {
        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128L);
        return request;
    }
}

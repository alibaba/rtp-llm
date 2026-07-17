package org.flexlb.config;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fail-fast validation tests for {@link ConfigService} and {@link FlexlbConfig}.
 *
 * <p>Covers startup pre-validation, strict boolean parsing, JSON parse errors,
 * critical vs non-critical field handling, and schedule-mode validation.
 */
class ConfigServiceFailFastTest {

    // ---- parseStrictBoolean ----

    @Test
    void parseStrictBoolean_accepts_on_off() {
        assertTrue(ConfigService.parseStrictBoolean("on", "testField"));
        assertFalse(ConfigService.parseStrictBoolean("off", "testField"));
    }

    @Test
    void parseStrictBoolean_accepts_enabled_disabled() {
        assertTrue(ConfigService.parseStrictBoolean("enabled", "testField"));
        assertFalse(ConfigService.parseStrictBoolean("disabled", "testField"));
    }

    @Test
    void parseStrictBoolean_rejects_unknown_value() {
        assertThrows(ConfigValidationException.class,
            () -> ConfigService.parseStrictBoolean("maybe", "testField"));
    }

    @Test
    void parseStrictBoolean_accepts_1_as_true() {
        assertTrue(ConfigService.parseStrictBoolean("1", "testField"));
    }

    @Test
    void parseStrictBoolean_accepts_0_as_false() {
        assertFalse(ConfigService.parseStrictBoolean("0", "testField"));
    }

    // ---- FLEXLB_CONFIG JSON parse error ----

    @Test
    void invalid_flexlb_config_json_throws_at_startup() {
        assertThrows(ConfigValidationException.class,
            () -> new ConfigService(Map.of("FLEXLB_CONFIG", "{not valid json")));
    }

    // ---- Critical field invalid value ----

    @Test
    void critical_field_default_schedule_mode_invalid_throws_at_startup() {
        assertThrows(ConfigValidationException.class,
            () -> new ConfigService(Map.of("DEFAULT_SCHEDULE_MODE", "INVALID")));
    }

    @Test
    void critical_field_default_schedule_mode_valid_does_not_throw() {
        assertDoesNotThrow(() -> new ConfigService(Map.of("DEFAULT_SCHEDULE_MODE", "BATCH")));
    }

    // ---- Non-critical field invalid value ----

    @Test
    void non_critical_field_invalid_value_does_not_throw() {
        // CACHE_HIT_TIME_WINDOW_MS is a non-critical long field; an invalid value
        // should be caught and logged, not propagated.
        assertDoesNotThrow(() -> new ConfigService(Map.of("CACHE_HIT_TIME_WINDOW_MS", "not-a-number")));
    }

    @Test
    void non_critical_boolean_field_invalid_value_does_not_throw() {
        // FLEXLB_BATCH_ENABLED is a non-critical boolean; invalid value should be
        // caught and logged, with the default value preserved.
        ConfigService configService = new ConfigService(Map.of("FLEXLB_BATCH_ENABLED", "maybe"));
        // Default is true, should remain true because override was rejected
        assertTrue(configService.loadBalanceConfig().isFlexlbBatchEnabled());
    }

    // ---- FlexlbConfig.getDefaultScheduleModeEnum ----

    @Test
    void getDefaultScheduleModeEnum_invalid_value_throws() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDefaultScheduleMode("INVALID");
        assertThrows(ConfigValidationException.class, config::getDefaultScheduleModeEnum);
    }

    @Test
    void getDefaultScheduleModeEnum_null_value_throws() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDefaultScheduleMode(null);
        assertThrows(ConfigValidationException.class, config::getDefaultScheduleModeEnum);
    }

    @Test
    void getDefaultScheduleModeEnum_valid_value_returns_enum() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDefaultScheduleMode("batch");
        assertEquals(org.flexlb.enums.ScheduleModeEnum.BATCH, config.getDefaultScheduleModeEnum());
    }

    // ---- dumpEffectiveConfig normal path ----

    @Test
    void default_config_construction_does_not_throw() {
        ConfigService configService = assertDoesNotThrow(() -> new ConfigService(Map.of()));
        // Verify the default schedule mode is AUTO
        assertEquals("AUTO", configService.loadBalanceConfig().getDefaultScheduleMode());
    }

    // ---- PREFILL_TIME_FORMULA blank skip ----

    @Test
    void blank_prefill_time_formula_does_not_throw() {
        assertDoesNotThrow(() -> new ConfigService(Map.of("PREFILL_TIME_FORMULA", "")));
    }

    @Test
    void non_blank_prefill_time_formula_sets_cost_formula() {
        ConfigService configService = new ConfigService(Map.of("PREFILL_TIME_FORMULA", "sum(computeTokens)"));
        assertEquals("sum(computeTokens)", configService.loadBalanceConfig().getCostFormula());
    }

    // ---- costSloBuckets pre-validation ----

    @Test
    void invalid_cost_slo_buckets_throws_at_startup() {
        assertThrows(ConfigValidationException.class,
            () -> new ConfigService(Map.of("COST_SLO_BUCKETS", "4096:not-a-number")));
    }

    @Test
    void valid_cost_slo_buckets_does_not_throw() {
        assertDoesNotThrow(() -> new ConfigService(Map.of("COST_SLO_BUCKETS", "4096:2000,8192:3000")));
    }

    // ---- CRITICAL_CONFIG_FIELDS reflection check ----

    @Test
    void criticalConfigFields_allMatchFlexlbConfigFields() throws Exception {
        // Access CRITICAL_CONFIG_FIELDS via reflection
        java.lang.reflect.Field fieldsField = ConfigService.class.getDeclaredField("CRITICAL_CONFIG_FIELDS");
        fieldsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Set<String> criticalFields = (Set<String>) fieldsField.get(null);

        // Verify each string in the set corresponds to an actual field in FlexlbConfig
        for (String fieldName : criticalFields) {
            assertNotNull(
                Arrays.stream(FlexlbConfig.class.getDeclaredFields())
                    .filter(f -> f.getName().equals(fieldName))
                    .findFirst()
                    .orElse(null),
                "CRITICAL_CONFIG_FIELDS contains '" + fieldName
                    + "' but no such field exists in FlexlbConfig. "
                    + "If a field was renamed, update CRITICAL_CONFIG_FIELDS accordingly."
            );
        }
    }
}

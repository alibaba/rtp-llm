package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

@Getter
@Slf4j
@Component
public class ConfigService {

    private static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    private static final String PREFILL_TIME_FORMULA_ENV = "PREFILL_TIME_FORMULA";
    private static final String TRAFFIC_POLICY_CONFIG_ENV = "TRAFFIC_POLICY_CONFIG";
    private static final String TRAFFIC_POLICY_CONFIG_FILE_ENV = "TRAFFIC_POLICY_CONFIG_FILE";

    /**
     * Critical config fields whose parse failures must abort startup (fail-fast)
     * instead of silently falling back to defaults.
     */
    private static final Set<String> CRITICAL_CONFIG_FIELDS = Set.of(
            "defaultScheduleMode",
            "flexlbBatchAlgorithm",
            "flexlbBatchMaxCapacity",
            "flexlbBatchMaxInflight",
            "flexlbBatchFixedMaxInflightBatches",
            "flexlbBatchSloMaxInflightBatches",
            "costFormula",
            "prefillPredictorType");

    private final FlexlbConfig flexlbConfig;

    public ConfigService() {
        this(System.getenv());
    }

    ConfigService(Map<String, String> environment) {
        String lbConfigStr = environment.get(FLEXLB_CONFIG_ENV);
        log.warn("FLEXLB_CONFIG = {}", lbConfigStr);
        FlexlbConfig config;
        if (lbConfigStr != null) {
            try {
                config = JsonUtils.toObject(lbConfigStr, FlexlbConfig.class);
            } catch (Exception e) {
                throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "Failed to parse FLEXLB_CONFIG JSON: " + e.getMessage(), e);
            }
        } else {
            config = new FlexlbConfig();
        }

        // If corresponding advanced environment variables exist, override and update
        applyEnvironmentOverrides(config, environment);
        applyTrafficPolicyOverride(config, environment);
        applyPrefillFormulaOverride(config, environment);

        // Pre-validate critical parsed config at startup (fail-fast).
        // If these throw ConfigValidationException, startup must abort rather
        // than letting every per-request call fail with a 500.
        config.getDefaultScheduleModeEnum();
        config.getParsedSloBuckets();

        dumpEffectiveConfig(config);
        this.flexlbConfig = config;
    }

    public FlexlbConfig loadBalanceConfig() {
        return flexlbConfig;
    }

    public synchronized void updateTrafficPolicy(TrafficPolicyConfig trafficPolicy) {
        if (trafficPolicy == null) {
            throw new IllegalArgumentException("trafficPolicy cannot be null");
        }
        flexlbConfig.setTrafficPolicy(trafficPolicy);
        log.warn("Traffic policy updated: {}", JsonUtils.toStringOrEmpty(trafficPolicy));
    }

    /**
     * Apply environment variable overrides to configuration
     * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
     * Example: enableQueueing -> ENABLE_QUEUEING
     */
    private void applyEnvironmentOverrides(FlexlbConfig config, Map<String, String> environment) {
        Field[] fields = FlexlbConfig.class.getDeclaredFields();
        for (Field field : fields) {
            // Only process primitive types and wrapper types
            Class<?> fieldType = field.getType();
            if (!isSupportedType(fieldType)) {
                continue;
            }

            String envVarName = camelToUpperSnakeCase(field.getName());
            String envValue = environment.get(envVarName);

            if (envValue != null && !envValue.trim().isEmpty()) {
                boolean isCritical = CRITICAL_CONFIG_FIELDS.contains(field.getName());
                try {
                    field.setAccessible(true);
                    Object parsedValue = parseValue(envValue.trim(), fieldType, envVarName);
                    Object oldValue = field.get(config);
                    field.set(config, parsedValue);
                    log.info(
                            "Environment variable override: {} = {} (field: {}, old value: {})",
                            envVarName,
                            parsedValue,
                            field.getName(),
                            oldValue);
                } catch (ConfigValidationException e) {
                    if (isCritical) {
                        throw e;
                    }
                    log.error(
                            "Failed to apply environment variable {}: {}",
                            envVarName,
                            e.getMessage(),
                            e);
                } catch (Exception e) {
                    if (isCritical) {
                        throw new ConfigValidationException(envVarName, e.getMessage(), e);
                    }
                    log.error(
                            "Failed to apply environment variable {}: {}",
                            envVarName,
                            e.getMessage(),
                            e);
                }
            }
        }
    }

    /**
     * Apply traffic policy from a standalone env var or file.
     * Priority: TRAFFIC_POLICY_CONFIG > TRAFFIC_POLICY_CONFIG_FILE > FLEXLB_CONFIG.trafficPolicy.
     */
    private void applyTrafficPolicyOverride(FlexlbConfig config, Map<String, String> environment) {
        String trafficPolicyConfig = environment.get(TRAFFIC_POLICY_CONFIG_ENV);
        String trafficPolicyConfigFile = environment.get(TRAFFIC_POLICY_CONFIG_FILE_ENV);

        if (StringUtils.isBlank(trafficPolicyConfig) && StringUtils.isNotBlank(trafficPolicyConfigFile)) {
            trafficPolicyConfig = readConfigFile(trafficPolicyConfigFile);
        }

        if (StringUtils.isBlank(trafficPolicyConfig)) {
            return;
        }

        try {
            TrafficPolicyConfig trafficPolicy = JsonUtils.toObject(trafficPolicyConfig, TrafficPolicyConfig.class);
            config.setTrafficPolicy(trafficPolicy);
            log.warn("Traffic policy loaded from standalone config: {}", JsonUtils.toStringOrEmpty(trafficPolicy));
        } catch (Exception e) {
            log.error("Failed to parse traffic policy config, skipping.", e);
        }
    }

    private void applyPrefillFormulaOverride(FlexlbConfig config, Map<String, String> environment) {
        String formula = environment.get(PREFILL_TIME_FORMULA_ENV);
        if (StringUtils.isBlank(formula)) {
            // Blank or unset formula means skip the override, preserving any formula
            // set via FLEXLB_CONFIG or COST_FORMULA. This maintains backward
            // compatibility with deployment scripts that set PREFILL_TIME_FORMULA=""
            // to cancel a formula override without clearing existing config.
            return;
        }
        config.setCostFormula(formula);
        log.warn("Prefill time formula loaded from {}: {}", PREFILL_TIME_FORMULA_ENV, formula);
    }

    private String readConfigFile(String filePath) {
        try {
            return Files.readString(Path.of(filePath), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read config file: " + filePath, e);
        }
    }

    /**
     * Check if the type is supported
     */
    private boolean isSupportedType(Class<?> type) {
        return type == int.class
                || type == Integer.class
                || type == long.class
                || type == Long.class
                || type == double.class
                || type == Double.class
                || type == boolean.class
                || type == Boolean.class
                || type == String.class
                || type.isEnum();
    }

    /**
     * Convert camel case to upper snake case
     * Example: enableQueueing -> ENABLE_QUEUEING
     */
    private String camelToUpperSnakeCase(String camelCase) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < camelCase.length(); i++) {
            char c = camelCase.charAt(i);
            if (Character.isUpperCase(c) && i > 0) {
                result.append('_');
            }
            result.append(Character.toUpperCase(c));
        }
        return result.toString();
    }

    /**
     * Parse string value based on target type
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private Object parseValue(String value, Class<?> targetType, String fieldName) {
        if (targetType == String.class) {
            return value;
        } else if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return parseStrictBoolean(value, fieldName);
        } else if (targetType.isEnum()) {
            return JsonUtils.toObject("\"" + value + "\"", targetType);
        }
        throw new IllegalArgumentException("Unsupported type: " + targetType);
    }

    /**
     * Strictly parse a boolean value, rejecting unrecognized strings.
     *
     * @throws ConfigValidationException if the value is not a recognized boolean literal.
     */
    static boolean parseStrictBoolean(String value, String fieldName) {
        String v = value.trim().toLowerCase();
        if ("true".equals(v) || "1".equals(v) || "yes".equals(v) || "on".equals(v) || "enabled".equals(v)) return true;
        if ("false".equals(v) || "0".equals(v) || "no".equals(v) || "off".equals(v) || "disabled".equals(v)) return false;
        throw new ConfigValidationException(fieldName,
            "Invalid boolean value '" + value + "'. Expected: true/false/1/0/yes/no/on/off/enabled/disabled");
    }

    /**
     * Log the effective configuration after all overrides have been applied.
     * Only dumps critical scheduling config — no sensitive information.
     */
    private void dumpEffectiveConfig(FlexlbConfig config) {
        log.info("===== FlexLB Effective Configuration =====");
        log.info("scheduleMode={}, batchAlgorithm={}, batchEnabled={}",
            config.getDefaultScheduleMode(), config.getFlexlbBatchAlgorithm(),
            config.isFlexlbBatchEnabled());
        log.info("batchMaxCapacity={}, batchMaxInflight={}",
            config.getFlexlbBatchMaxCapacity(), config.getFlexlbBatchMaxInflight());
        log.info("fixedMaxInflightBatches={}, sloMaxInflightBatches={}",
            config.getFlexlbBatchFixedMaxInflightBatches(),
            config.getFlexlbBatchSloMaxInflightBatches());
        log.info("prefillPredictorType={}", config.getPrefillPredictorType());
        log.info("==========================================");
    }
}

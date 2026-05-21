package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

@Getter
@Slf4j
@Component
public class ConfigService {

    private static final String FLEXLB_CONFIG_KEY = "FLEXLB_CONFIG";
    private static final String WHALE_WRAPPER_KEY = "WHALE_MASTER_CONFIG";
    private static final String TRAFFIC_POLICY_CONFIG_ENV = "TRAFFIC_POLICY_CONFIG";
    private static final String TRAFFIC_POLICY_CONFIG_FILE_ENV = "TRAFFIC_POLICY_CONFIG_FILE";

    private final FlexlbConfig flexlbConfig;

    @Autowired
    public ConfigService(FlexlbConfig flexlbConfig) {
        Map<String, String> environment = System.getenv();
        applyEnvironmentOverrides(flexlbConfig, environment);
        applyTrafficPolicyOverride(flexlbConfig, environment);
        this.flexlbConfig = flexlbConfig;
    }

    /**
     * Direct path for callers outside Spring (unit tests, ad-hoc tooling).
     * Mirrors what {@link FlexlbConfigJsonEnvironmentPostProcessor} +
     * {@code @ConfigurationProperties} binding produce for the Spring path:
     * parse {@code FLEXLB_CONFIG} JSON env (or extract it from
     * {@code WHALE_MASTER_CONFIG} when the platform wraps it), then apply
     * unprefixed per-field env overrides.
     */
    public ConfigService() {
        this(System.getenv());
    }

    ConfigService(Map<String, String> environment) {
        String lbConfigStr = resolveFlexlbConfigJson(environment);
        log.warn("FLEXLB_CONFIG = {}", lbConfigStr);
        FlexlbConfig config;
        if (lbConfigStr != null) {
            config = JsonUtils.toObject(lbConfigStr, FlexlbConfig.class);
        } else {
            config = new FlexlbConfig();
        }

        // If corresponding advanced environment variables exist, override and update
        applyEnvironmentOverrides(config, environment);
        applyTrafficPolicyOverride(config, environment);

        this.flexlbConfig = config;
    }

    private static String resolveFlexlbConfigJson(Map<String, String> environment) {
        String direct = environment.get(FLEXLB_CONFIG_KEY);
        if (direct != null && !direct.isBlank()) {
            return direct;
        }
        String wrapper = environment.get(WHALE_WRAPPER_KEY);
        if (wrapper == null || wrapper.isBlank()) {
            return null;
        }
        try {
            JsonNode envs = new ObjectMapper().readTree(wrapper)
                    .path("zone_process_setting")
                    .path("process_info")
                    .path("envs");
            if (!envs.isArray()) {
                return null;
            }
            for (JsonNode entry : envs) {
                if (entry.isArray() && entry.size() >= 2
                        && FLEXLB_CONFIG_KEY.equals(entry.get(0).asText())) {
                    return entry.get(1).asText();
                }
            }
            return null;
        } catch (Exception e) {
            throw new IllegalStateException("WHALE_MASTER_CONFIG env var is not valid JSON", e);
        }
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
                try {
                    field.setAccessible(true);
                    Object parsedValue = parseValue(envValue.trim(), fieldType);
                    Object oldValue = field.get(config);
                    field.set(config, parsedValue);
                    log.info(
                            "Environment variable override: {} = {} (field: {}, old value: {})",
                            envVarName,
                            parsedValue,
                            field.getName(),
                            oldValue);
                } catch (Exception e) {
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

        TrafficPolicyConfig trafficPolicy = JsonUtils.toObject(trafficPolicyConfig, TrafficPolicyConfig.class);
        config.setTrafficPolicy(trafficPolicy);
        log.warn("Traffic policy loaded from standalone config: {}", JsonUtils.toStringOrEmpty(trafficPolicy));
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
    private Object parseValue(String value, Class<?> targetType) {
        if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return Boolean.parseBoolean(value);
        } else if (targetType.isEnum()) {
            return Enum.valueOf((Class<Enum>) targetType, value);
        }
        throw new IllegalArgumentException("Unsupported type: " + targetType);
    }
}

package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.lang.reflect.Field;

@Getter
@Slf4j
@Component
public class ConfigService {

    private static final String FLEXLB_CONFIG_KEY = "FLEXLB_CONFIG";
    private static final String WHALE_WRAPPER_KEY = "WHALE_MASTER_CONFIG";

    private final FlexlbConfig flexlbConfig;

    @Autowired
    public ConfigService(FlexlbConfig flexlbConfig) {
        applyEnvironmentOverrides(flexlbConfig);
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
        String lbConfigStr = resolveFlexlbConfigJson();
        log.warn("FLEXLB_CONFIG = {}", lbConfigStr);
        FlexlbConfig config = lbConfigStr != null
                ? JsonUtils.toObject(lbConfigStr, FlexlbConfig.class)
                : new FlexlbConfig();
        applyEnvironmentOverrides(config);
        this.flexlbConfig = config;
    }

    private static String resolveFlexlbConfigJson() {
        String direct = System.getenv(FLEXLB_CONFIG_KEY);
        if (direct != null && !direct.isBlank()) {
            return direct;
        }
        String wrapper = System.getenv(WHALE_WRAPPER_KEY);
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

    /**
     * Apply environment variable overrides to configuration
     * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
     * Example: enableQueueing -> ENABLE_QUEUEING
     */
    private void applyEnvironmentOverrides(FlexlbConfig config) {
        Field[] fields = FlexlbConfig.class.getDeclaredFields();
        for (Field field : fields) {
            // Only process primitive types and wrapper types
            Class<?> fieldType = field.getType();
            if (!isSupportedType(fieldType)) {
                continue;
            }

            String envVarName = camelToUpperSnakeCase(field.getName());
            String envValue = System.getenv(envVarName);

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
    private Object parseValue(String value, Class<?> targetType) {
        if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return Boolean.parseBoolean(value);
        } else if (targetType == String.class) {
            return value;
        } else if (targetType.isEnum()) {
            return Enum.valueOf((Class<Enum>) targetType, value);
        }
        throw new IllegalArgumentException("Unsupported type: " + targetType);
    }
}

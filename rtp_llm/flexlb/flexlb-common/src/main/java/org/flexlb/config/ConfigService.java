package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.lang.reflect.Field;

@Getter
@Slf4j
@Component
public class ConfigService {

    private final WhaleMasterConfig whaleMasterConfig;

    public ConfigService() {
        String lbConfigStr = System.getenv("WHALE_MASTER_CONFIG");
        log.warn("WHALE_MASTER_CONFIG = {}", lbConfigStr);
        WhaleMasterConfig config;
        if (lbConfigStr != null) {
            config = JsonUtils.toObject(lbConfigStr, WhaleMasterConfig.class);
        } else {
            config = new WhaleMasterConfig();
        }

        // If corresponding advanced environment variables exist, override and update
        applyEnvironmentOverrides(config);

        this.whaleMasterConfig = config;
    }

    public WhaleMasterConfig loadBalanceConfig() {
        return whaleMasterConfig;
    }

    /**
     * Apply environment variable overrides to configuration
     * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
     * Example: enableQueueing -> ENABLE_QUEUEING
     */
    private void applyEnvironmentOverrides(WhaleMasterConfig config) {
        Field[] fields = WhaleMasterConfig.class.getDeclaredFields();
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
                || type == Boolean.class;
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
    private Object parseValue(String value, Class<?> targetType) {
        if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return Boolean.parseBoolean(value);
        }
        throw new IllegalArgumentException("Unsupported type: " + targetType);
    }
}

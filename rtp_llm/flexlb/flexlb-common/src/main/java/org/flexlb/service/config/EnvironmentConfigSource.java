package org.flexlb.service.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.function.Consumer;

@Slf4j
@Component
final class EnvironmentConfigSource implements ConfigSource {

    private static final int PRIORITY = 1;

    private String configContent;

    @Override
    public String name() {
        return "environment";
    }

    @Override
    public int priority() {
        return PRIORITY;
    }

    @PostConstruct
    void initialize() {
        String configJson = System.getenv("FLEXLB_CONFIG");
        log.info("Loading FLEXLB_CONFIG from environment: configured={}", configJson != null);
        FlexlbConfig config = configJson == null
                ? new FlexlbConfig()
                : JsonUtils.toObject(configJson, FlexlbConfig.class);
        applyFieldOverrides(config);
        configContent = JsonUtils.toString(config);
        ConfigService.register(this);
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {}

    @Override
    public String load() {
        return configContent;
    }

    private void applyFieldOverrides(FlexlbConfig config) {
        for (Field field : FlexlbConfig.class.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }

            String environmentName = camelToUpperSnakeCase(field.getName());
            String value = System.getenv(environmentName);
            if (value == null) {
                continue;
            }

            try {
                field.setAccessible(true);
                Object parsedValue = parseValue(value, field.getType());
                Object oldValue = field.get(config);
                field.set(config, parsedValue);
                log.info("Environment variable override: {} = {} (field: {}, old value: {})",
                        environmentName,
                        parsedValue,
                        field.getName(),
                        oldValue);
            } catch (Exception e) {
                log.error("Failed to apply environment variable {}: {}", environmentName, e.getMessage(), e);
            }
        }
    }

    private String camelToUpperSnakeCase(String camelCase) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < camelCase.length(); i++) {
            char character = camelCase.charAt(i);
            if (Character.isUpperCase(character) && i > 0) {
                result.append('_');
            }
            result.append(Character.toUpperCase(character));
        }
        return result.toString();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private Object parseValue(String value, Class<?> targetType) {
        if (targetType == String.class) {
            return value;
        }

        String normalizedValue = value.trim();
        if (targetType == byte.class || targetType == Byte.class) {
            return Byte.parseByte(normalizedValue);
        } else if (targetType == short.class || targetType == Short.class) {
            return Short.parseShort(normalizedValue);
        } else if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(normalizedValue);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(normalizedValue);
        } else if (targetType == float.class || targetType == Float.class) {
            return Float.parseFloat(normalizedValue);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(normalizedValue);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return Boolean.parseBoolean(normalizedValue);
        } else if (targetType == char.class || targetType == Character.class) {
            if (normalizedValue.length() != 1) {
                throw new IllegalArgumentException("Expected a single character");
            }
            return normalizedValue.charAt(0);
        } else if (targetType.isEnum()) {
            return Enum.valueOf((Class<Enum>) targetType, normalizedValue);
        }
        return JsonUtils.toObject(normalizedValue, targetType);
    }
}

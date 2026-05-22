package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Field;
import java.util.Locale;
import java.util.Map;

/**
 * Reflection-based env-variable override for plain config beans. For each declared field of
 * supported type, looks up an env var named {@code <prefix><FIELD_NAME_UPPER_SNAKE>} and, when
 * present and non-empty, writes the parsed value into the field. Convention matches the existing
 * {@code FLEXLB_CONFIG} contract: {@code enableQueueing} → {@code ENABLE_QUEUEING}.
 *
 * <p>Supports {@code int/Integer}, {@code long/Long}, {@code double/Double}, {@code boolean/Boolean},
 * {@code String}, and any {@code Enum}. Unsupported field types are silently skipped — they can't
 * be overridden through env, only through the bean's natural construction path.
 *
 * <p>Malformed values log an error and leave the field at its default; we never throw, so a typo
 * in a single env var cannot crash startup.
 */
@Slf4j
public final class EnvConfigOverrides {

    private EnvConfigOverrides() {
    }

    /**
     * Apply overrides from {@link System#getenv()} with the given prefix.
     */
    public static void apply(Object config, String prefix) {
        apply(config, prefix, System.getenv());
    }

    /**
     * Test-seam variant taking an explicit env map.
     */
    public static void apply(Object config, String prefix, Map<String, String> env) {
        for (Field field : config.getClass().getDeclaredFields()) {
            Class<?> type = field.getType();
            if (!isSupportedType(type)) {
                continue;
            }
            String envName = prefix + camelToUpperSnakeCase(field.getName());
            String raw = env.get(envName);
            if (raw == null || raw.trim().isEmpty()) {
                continue;
            }
            try {
                field.setAccessible(true);
                Object oldValue = field.get(config);
                Object parsed = parseValue(raw.trim(), type);
                field.set(config, parsed);
                log.info("env override: {} = {} (field: {}, old: {})",
                        envName, parsed, field.getName(), oldValue);
            } catch (Exception e) {
                log.error("env override failed for {} = {}: {}", envName, raw, e.getMessage(), e);
            }
        }
    }

    private static boolean isSupportedType(Class<?> type) {
        return type == int.class || type == Integer.class
                || type == long.class || type == Long.class
                || type == double.class || type == Double.class
                || type == boolean.class || type == Boolean.class
                || type == String.class
                || type.isEnum();
    }

    private static String camelToUpperSnakeCase(String camelCase) {
        StringBuilder out = new StringBuilder(camelCase.length() + 4);
        for (int i = 0; i < camelCase.length(); i++) {
            char c = camelCase.charAt(i);
            if (Character.isUpperCase(c) && i > 0) {
                out.append('_');
            }
            out.append(Character.toUpperCase(c));
        }
        return out.toString();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Object parseValue(String value, Class<?> targetType) {
        if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        }
        if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        }
        if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        }
        if (targetType == boolean.class || targetType == Boolean.class) {
            return switch (value.toLowerCase(Locale.ROOT)) {
                case "true", "1", "yes", "on"  -> true;
                case "false", "0", "no", "off" -> false;
                default -> throw new IllegalArgumentException(
                        "boolean must be one of true|false|1|0|yes|no|on|off, got: '" + value + "'");
            };
        }
        if (targetType == String.class) {
            return value;
        }
        if (targetType.isEnum()) {
            return Enum.valueOf((Class<Enum>) targetType, value.toUpperCase(Locale.ROOT));
        }
        throw new IllegalArgumentException("unsupported type: " + targetType);
    }
}

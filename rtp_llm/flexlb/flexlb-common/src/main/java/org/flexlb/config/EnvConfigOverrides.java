package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Field;
import java.util.Arrays;
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
 * <p>A malformed numeric value logs an error and leaves the field at its default, so a typo in a
 * tuning knob cannot crash startup. Booleans follow {@link Boolean#parseBoolean}: anything but a
 * case-insensitive {@code "true"} — including {@code "1"}/{@code "yes"}/typos — reads as
 * {@code false}, never errors (deliberate backward compatibility with live deployments). A
 * malformed <em>enum</em> override fails fast instead: an unknown categorical value (e.g. a
 * mistyped {@code ENGINE_TYPE}) must not silently fall back to the default and run the wrong mode.
 *
 * <p>On the empty-prefix path the bare names are dangerously generic ({@code ENGINE_TYPE},
 * {@code DEPLOY}, ...) and can collide with unrelated variables lingering in the container, so a
 * {@code FLEXLB_}-prefixed form of each name is consulted first and wins over the bare form.
 * Callers that already pass a namespace prefix (e.g. {@code DISPATCH_}) are collision-safe and
 * get no extra prefixing.
 */
@Slf4j
public final class EnvConfigOverrides {

    /** Namespaced fallback for the empty-prefix path; see the class javadoc. */
    private static final String GLOBAL_PREFIX = "FLEXLB_";

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
            if (prefix.isEmpty()) {
                String prefixedName = GLOBAL_PREFIX + envName;
                String prefixedRaw = env.get(prefixedName);
                if (prefixedRaw != null && !prefixedRaw.trim().isEmpty()) {
                    envName = prefixedName;
                    raw = prefixedRaw;
                }
            }
            if (raw == null) {
                continue;
            }
            String value = raw.trim();
            if (value.isEmpty()) {
                continue;
            }
            try {
                field.setAccessible(true);
                Object oldValue = field.get(config);
                Object parsed = parseValue(value, type);
                field.set(config, parsed);
                // WARN, not info: an override silently flipping behavior is exactly what an
                // upgrade postmortem needs to see, including which env name matched.
                log.warn("env override: {} = {} (field: {}, old: {})",
                        envName, parsed, field.getName(), oldValue);
            } catch (Exception e) {
                if (type.isEnum()) {
                    throw new IllegalArgumentException(
                            "invalid enum value for env override " + envName + "=" + raw
                                    + " (valid: " + Arrays.toString(type.getEnumConstants()) + ")", e);
                }
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
            return Boolean.parseBoolean(value);
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

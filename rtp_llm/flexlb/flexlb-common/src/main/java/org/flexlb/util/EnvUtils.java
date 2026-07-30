package org.flexlb.util;

public final class EnvUtils {

    private EnvUtils() {
    }

    public static long readPositiveLong(String name, long defaultValue) {
        return parsePositiveLong(name, System.getenv(name), defaultValue);
    }

    public static boolean readBoolean(String name, boolean defaultValue) {
        return parseBoolean(name, System.getenv(name), defaultValue);
    }

    public static long parsePositiveLong(String name, String value, long defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        try {
            long parsedValue = Long.parseLong(value.trim());
            if (parsedValue > 0) {
                return parsedValue;
            }
        } catch (NumberFormatException ignored) {
            // The warning below covers malformed and non-positive values uniformly.
        }
        Logger.warn(
                "Invalid {}='{}'; using default value {}",
                name,
                value,
                defaultValue);
        return defaultValue;
    }

    public static boolean parseBoolean(String name, String value, boolean defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        String normalized = value.trim();
        if ("true".equalsIgnoreCase(normalized)
                || "1".equals(normalized)
                || "yes".equalsIgnoreCase(normalized)
                || "on".equalsIgnoreCase(normalized)) {
            return true;
        }
        if ("false".equalsIgnoreCase(normalized)
                || "0".equals(normalized)
                || "no".equalsIgnoreCase(normalized)
                || "off".equalsIgnoreCase(normalized)) {
            return false;
        }
        Logger.warn(
                "Invalid {}='{}'; using default value {}",
                name,
                value,
                defaultValue);
        return defaultValue;
    }
}

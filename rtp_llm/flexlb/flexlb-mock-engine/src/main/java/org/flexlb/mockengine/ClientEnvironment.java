package org.flexlb.mockengine;

import java.io.IOException;
import java.util.Map;
import java.util.Properties;

/** Shared defaults; missing and cleared environment values have identical meaning. */
final class ClientEnvironment {
    private static final Properties DEFAULTS = loadDefaults();
    private final Map<String, String> values;

    ClientEnvironment(Map<String, String> values) { this.values = values; }

    private static Properties loadDefaults() {
        try (var input = ClientEnvironment.class.getResourceAsStream("/load-client-defaults.properties")) {
            if (input == null) throw new IllegalStateException("missing load client defaults");
            Properties defaults = new Properties();
            defaults.load(input);
            return defaults;
        } catch (IOException error) {
            throw new IllegalStateException("cannot read load client defaults", error);
        }
    }

    String text(String key) {
        String fallback = DEFAULTS.getProperty(key);
        if (fallback == null) throw new IllegalArgumentException("undefined client default: " + key);
        return text(key, fallback);
    }
    int integer(String key) { return Integer.parseInt(text(key)); }
    long longValue(String key) { return Long.parseLong(text(key)); }
    double number(String key) { return Double.parseDouble(text(key)); }
    boolean flag(String key) { return flag(key, Boolean.parseBoolean(DEFAULTS.getProperty(key))); }

    String text(String key, String fallback) {
        String value = values.get(key);
        return value == null || value.isEmpty() ? DEFAULTS.getProperty(key, fallback) : value;
    }
    int integer(String key, int fallback) { return Integer.parseInt(text(key, Integer.toString(fallback))); }
    long longValue(String key, long fallback) { return Long.parseLong(text(key, Long.toString(fallback))); }
    double number(String key, double fallback) { return Double.parseDouble(text(key, Double.toString(fallback))); }
    boolean flag(String key, boolean fallback) {
        String value = text(key, Boolean.toString(fallback));
        if (value.equals("1") || value.equalsIgnoreCase("true") || value.equalsIgnoreCase("yes")) return true;
        if (value.equals("0") || value.equalsIgnoreCase("false") || value.equalsIgnoreCase("no")) return false;
        throw new IllegalArgumentException("invalid boolean: " + key);
    }
}

package org.flexlb.config;

public class ConfigValidationException extends RuntimeException {
    public ConfigValidationException(String fieldName, String message) {
        super("Config validation failed for '" + fieldName + "': " + message);
    }

    public ConfigValidationException(String fieldName, String message, Throwable cause) {
        super("Config validation failed for '" + fieldName + "': " + message, cause);
    }
}

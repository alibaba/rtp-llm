package org.flexlb.config;

/**
 * Reports an invalid configuration field while retaining its validation failure detail.
 */
public class ConfigValidationException extends RuntimeException {
    public ConfigValidationException(String fieldName, String message) {
        super("Config validation failed for '" + fieldName + "': " + message);
    }

    public ConfigValidationException(String fieldName, String message, Throwable cause) {
        super("Config validation failed for '" + fieldName + "': " + message, cause);
    }
}

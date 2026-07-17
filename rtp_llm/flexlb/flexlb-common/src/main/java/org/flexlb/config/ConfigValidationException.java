package org.flexlb.config;

public class ConfigValidationException extends RuntimeException {
    private final String fieldName;

    public ConfigValidationException(String fieldName, String message) {
        super("Config validation failed for '" + fieldName + "': " + message);
        this.fieldName = fieldName;
    }

    public ConfigValidationException(String fieldName, String message, Throwable cause) {
        super("Config validation failed for '" + fieldName + "': " + message, cause);
        this.fieldName = fieldName;
    }

    public String getFieldName() {
        return fieldName;
    }
}

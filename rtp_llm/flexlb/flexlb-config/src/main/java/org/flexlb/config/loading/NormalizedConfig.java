package org.flexlb.config.loading;

/**
 * Configuration documents after parser normalization.
 *
 * @param sourceSchemaVersion schema version selected from the source document before normalization
 */
public record NormalizedConfig(
        String flexlbConfig, String modelServiceConfig, int sourceSchemaVersion) {}

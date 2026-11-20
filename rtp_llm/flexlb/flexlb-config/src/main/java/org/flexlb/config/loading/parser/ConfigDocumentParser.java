package org.flexlb.config.loading.parser;

import org.flexlb.config.loading.NormalizedConfig;
import org.flexlb.config.loading.source.ConfigSource;

/**
 * Converts a raw document obtained by a {@link ConfigSource} into the current
 * configuration contracts consumed by {@code ConfigService}.
 */
public interface ConfigDocumentParser {

    int schemaVersion();

    NormalizedConfig parse(String rawFlexlbConfig, String rawModelServiceConfig);
}

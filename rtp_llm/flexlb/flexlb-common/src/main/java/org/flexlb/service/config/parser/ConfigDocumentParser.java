package org.flexlb.service.config.parser;

import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;

/**
 * Converts a raw document obtained by a {@link ConfigSource} into the current
 * configuration contracts consumed by {@code ConfigService}.
 */
public interface ConfigDocumentParser {

    int schemaVersion();

    NormalizedConfig parse(String rawFlexlbConfig, String rawModelServiceConfig);

}

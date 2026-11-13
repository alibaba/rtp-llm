package org.flexlb.config.loading.parser;

import org.flexlb.config.loading.ConfigSchemaVersion;
import org.flexlb.config.loading.NormalizedConfig;
import org.springframework.stereotype.Component;

/**
 * Parses a document that already follows the current configuration contract.
 */
@Component
public final class StandardConfigDocumentParser implements ConfigDocumentParser {
    public StandardConfigDocumentParser() {
        ConfigDocumentParserResolver.register(this);
    }

    @Override
    public NormalizedConfig parse(String rawFlexlbConfig, String rawModelServiceConfig) {
        return new NormalizedConfig(rawFlexlbConfig, rawModelServiceConfig, schemaVersion());
    }

    @Override
    public int schemaVersion() {
        return ConfigSchemaVersion.STANDARD;
    }
}

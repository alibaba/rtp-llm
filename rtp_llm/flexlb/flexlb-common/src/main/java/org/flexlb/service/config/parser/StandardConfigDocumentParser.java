package org.flexlb.service.config.parser;

import org.flexlb.config.ConfigSchemaVersion;
import org.flexlb.service.config.NormalizedConfig;
import org.springframework.stereotype.Component;

/** Parses a document that already follows the current configuration contract. */
@Component
public final class StandardConfigDocumentParser implements ConfigDocumentParser {

    public StandardConfigDocumentParser() {
        ConfigDocumentParserResolver.register(this);
    }

    @Override
    public int schemaVersion() {
        return ConfigSchemaVersion.STANDARD;
    }

    @Override
    public NormalizedConfig parse(String rawFlexlbConfig, String rawModelServiceConfig) {
        return new NormalizedConfig(rawFlexlbConfig, rawModelServiceConfig);
    }
}

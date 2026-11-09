package org.flexlb.service.config;

import org.flexlb.service.config.parser.ConfigDocumentParser;
import org.flexlb.service.config.parser.ConfigDocumentParserResolver;

import java.util.function.Consumer;

/**
 * Loads a configuration document and optionally publishes later updates with an explicit source
 * priority.
 */
public interface ConfigSource extends AutoCloseable {
    String name();

    int priority();

    void setUpdateListener(Consumer<String> listener);

    default NormalizedConfig loadConfig() throws Exception {
        return normalize(load());
    }

    default NormalizedConfig normalize(String rawFlexlbConfig) {
        ConfigDocumentParser parser = ConfigDocumentParserResolver.resolve(rawFlexlbConfig);
        return parser.parse(rawFlexlbConfig, loadModelServiceConfig());
    }

    default String loadModelServiceConfig() {
        return null;
    }

    String load() throws Exception;

    @Override
    default void close() throws Exception {}
}
